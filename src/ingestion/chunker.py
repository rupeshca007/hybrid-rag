"""
Chunking strategy for book chapters.

Why RecursiveCharacterTextSplitter?
- Tries to split on paragraph/sentence/word boundaries first
- Falls back to characters only when needed
- Preserves semantic coherence compared to fixed-size splitting

For chapter PDFs:
- chunk_size=800   → ~600-700 tokens, captures full paragraphs
- chunk_overlap=100 → prevents answer loss at chunk edges
- All original metadata is inherited by every chunk
- A unique `chunk_id` is added per chunk for traceability
"""

import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

from config.settings import settings

console = Console()


def build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter from settings (or overrides).

    Separator priority (highest → lowest):
      paragraph break → newline → sentence end → space → character
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def _make_chunk_id(content: str, metadata: dict, chunk_index: int = 0) -> str:
    """Stable deterministic ID for a chunk (SHA256 of source + page + index + content snippet)."""
    raw = (
        f"{metadata.get('source','')}::"
        f"{metadata.get('page_number',0)}::"
        f"{chunk_index}::"
        f"{content[:64]}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """
    Split a list of page-level Documents into smaller chunks.

    Each chunk inherits all metadata from its parent page and adds:
      - chunk_id     : stable 16-char hash
      - chunk_index  : position within the parent page's chunks

    Args:
        documents:     Output from loader.load_pdf() or load_pdfs_from_dir()
        chunk_size:    Override settings.chunk_size
        chunk_overlap: Override settings.chunk_overlap

    Returns:
        List of chunk-level Documents ready for embedding.
    """
    splitter = build_splitter(chunk_size, chunk_overlap)

    chunks: list[Document] = []
    for doc in documents:
        split_docs = splitter.split_documents([doc])
        for idx, chunk in enumerate(split_docs):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = _make_chunk_id(
                chunk.page_content, chunk.metadata, chunk_index=idx
            )
        chunks.extend(split_docs)

    console.print(
        f"[green]✓ Chunked {len(documents)} pages → {len(chunks)} chunks "
        f"(size={chunk_size or settings.chunk_size}, "
        f"overlap={chunk_overlap or settings.chunk_overlap})[/green]"
    )
    return chunks
