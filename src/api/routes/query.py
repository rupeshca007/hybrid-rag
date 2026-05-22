"""
POST /query — ask a question against your indexed documents.
"""

from fastapi import APIRouter, HTTPException
from rich.console import Console

from src.generation.chain import run_rag_chain, SourceChunk
from src.api.schemas import QueryRequest, QueryResponse, SourceChunkSchema

console = Console()
router = APIRouter(prefix="/query", tags=["Query"])


def _source_to_schema(s: SourceChunk) -> SourceChunkSchema:
    return SourceChunkSchema(
        filename=s.filename,
        chapter=s.chapter,
        page_number=s.page_number,
        chunk_id=s.chunk_id,
        content_preview=s.content_preview,
    )


@router.post("", response_model=QueryResponse, summary="Ask a question about your documents")
async def query_documents(request: QueryRequest):
    """
    Ask a natural language question against all indexed PDF chapters.

    The pipeline:
    1. Embeds the question with sentence-transformers
    2. Retrieves top-k most similar chunks from ChromaDB
    3. Builds a prompt (from versioned YAML)
    4. Generates an answer with OpenAI GPT-4o
    5. Returns answer + source citations

    Optional `chapter_filter` restricts search to one chapter's chunks.
    Optional `prompt_version` allows A/B testing between prompt versions.
    """
    console.print(f"\n[bold blue]Query:[/bold blue] {request.question}")

    try:
        result = run_rag_chain(
            question=request.question,
            top_k=request.top_k,
            chapter_filter=request.chapter_filter,
            prompt_version=request.prompt_version,
        )
    except Exception as e:
        console.print(f"[red]✗ Query failed: {e}[/red]")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        prompt_version=result.prompt_version,
        model=result.model,
        num_chunks_retrieved=result.num_chunks_retrieved,
        sources=[_source_to_schema(s) for s in result.sources],
    )
