"""
LCEL RAG chain — the core pipeline that turns a question into an answer.

Pipeline (LangChain Expression Language):
    query
      │
      ▼
    ChromaDB similarity_search  →  retrieved chunks
      │
      ▼
    format_context              →  formatted context string
      │
      ▼
    LLM (OpenAI / Groq)         →  raw AI message
      │
      ▼
    StrOutputParser             →  plain string answer

Provider is selected by LLM_PROVIDER in .env:
  LLM_PROVIDER=openai  →  ChatOpenAI  (requires OPENAI_API_KEY + billing)
  LLM_PROVIDER=groq    →  ChatGroq    (free tier, requires GROQ_API_KEY)

Returns a RAGResult dataclass with answer + source citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from config.settings import settings
from src.generation.prompt_loader import format_context, load_prompt
from src.vectorstore.chroma_store import similarity_search

console = Console()


@dataclass
class SourceChunk:
    """Metadata about a single retrieved chunk used in the answer."""
    filename: str
    chapter: str
    page_number: int
    chunk_id: str
    content_preview: str  # first 200 chars


@dataclass
class RAGResult:
    """Full result returned by the RAG chain."""
    question: str
    answer: str
    prompt_version: str
    model: str
    sources: list[SourceChunk] = field(default_factory=list)
    num_chunks_retrieved: int = 0


@lru_cache(maxsize=1)
def _get_llm():
    """Singleton LLM client — picks provider from LLM_PROVIDER setting."""
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at: https://console.groq.com/keys"
            )
        return ChatGroq(
            model=settings.groq_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            api_key=settings.groq_api_key,
        )
    else:  # default: openai
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example → .env and add your key."
            )
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            api_key=settings.openai_api_key,
        )


def _active_model_name() -> str:
    """Return the model name currently in use."""
    provider = settings.llm_provider.lower()
    return settings.groq_model if provider == "groq" else settings.openai_model


def _docs_to_sources(docs: list[Document]) -> list[SourceChunk]:
    """Convert retrieved Documents to serialisable SourceChunk objects."""
    sources = []
    for doc in docs:
        m = doc.metadata
        sources.append(
            SourceChunk(
                filename=m.get("filename", "unknown"),
                chapter=m.get("chapter", "unknown"),
                page_number=m.get("page_number", 0),
                chunk_id=m.get("chunk_id", ""),
                content_preview=doc.page_content[:200].replace("\n", " "),
            )
        )
    return sources


def run_rag_chain(
    question: str,
    top_k: int | None = None,
    chapter_filter: str | None = None,
    prompt_version: str | None = None,
) -> RAGResult:
    """
    Run the full RAG pipeline for a single question.

    Args:
        question:       The user's natural language question.
        top_k:          Number of chunks to retrieve (default: settings.retriever_top_k).
        chapter_filter: Optional filter to restrict to a specific chapter by name.
                        e.g. "chapter_03" matches the PDF filename stem.
        prompt_version: Override prompt version (e.g. "v2" for A/B testing).

    Returns:
        RAGResult with answer, sources, and metadata.
    """
    # ── 1. Retrieve ───────────────────────────────────────────
    # Ignore empty strings or Swagger's default 'string' placeholder
    if chapter_filter and chapter_filter.strip().lower() not in ("", "string"):
        filter_dict = {"chapter": chapter_filter.strip()}
    else:
        filter_dict = None
        
    k = top_k or settings.retriever_top_k

    console.print(f"\n[bold cyan]▶ Retrieving[/bold cyan] top-{k} chunks…")
    # Use Hybrid Search instead of pure Chroma similarity_search
    from src.vectorstore.hybrid_search import hybrid_search
    retrieved_docs = hybrid_search(question, top_k=k, filter_metadata=filter_dict)

    if not retrieved_docs:
        console.print("[yellow]No chunks retrieved. Is the vector store populated?[/yellow]")
        return RAGResult(
            question=question,
            answer="I don't have enough information in the provided chapters to answer this.",
            prompt_version=prompt_version or settings.prompt_version,
            model=_active_model_name(),
        )

    console.print(f"[green]✓ Retrieved {len(retrieved_docs)} chunks[/green]")

    # ── 2. Build Prompt ───────────────────────────────────────
    # Ignore Swagger's 'string' placeholder for prompt_version
    if prompt_version and prompt_version.strip().lower() not in ("", "string"):
        ver = prompt_version.strip()
    else:
        ver = settings.prompt_version

    system_prompt, user_template = load_prompt("rag_qa", version=ver)
    context_str = format_context(retrieved_docs)
    user_message = user_template.format(context=context_str, question=question)

    # ── 3. Generate ───────────────────────────────────────────
    llm = _get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    active_model = _active_model_name()
    console.print(f"[bold cyan]Generating[/bold cyan] answer with {active_model} ({settings.llm_provider})...")
    response = llm.invoke(messages)
    answer = response.content.strip()

    console.print("[green]Answer generated[/green]")

    return RAGResult(
        question=question,
        answer=answer,
        prompt_version=ver,
        model=active_model,
        sources=_docs_to_sources(retrieved_docs),
        num_chunks_retrieved=len(retrieved_docs),
    )
