"""
ChromaDB persistent vector store wrapper.

Design decisions:
- Uses langchain-chroma (official LangChain integration)
- Embedding: local SentenceTransformer (all-MiniLM-L6-v2)
  → Zero API cost, runs on CPU, ~80MB model download on first use
- Deduplication: upsert by chunk_id avoids re-embedding the same chunk
- Metadata is stored alongside vectors for source attribution

Usage:
    store = get_vector_store()            # singleton
    store.add_chunks(chunks)              # from chunker.py
    results = store.similarity_search(query, k=6)
"""

from __future__ import annotations

import os
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rich.console import Console

from config.settings import settings

console = Console()


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load sentence-transformers model once and cache.
    Downloads ~80 MB on first call, then cached locally.
    """
    console.print(
        f"[cyan]Loading embedding model:[/cyan] {settings.embedding_model}"
    )
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_chroma_client() -> chromadb.PersistentClient:
    """Create a persistent ChromaDB client (file-backed)."""
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Return the singleton LangChain-Chroma vector store.
    Safe to call multiple times — returns the same instance.
    """
    embeddings = _get_embeddings()
    client = _get_chroma_client()

    store = Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )
    console.print(
        f"[green]✓ ChromaDB ready:[/green] "
        f"collection='{settings.chroma_collection_name}' "
        f"path='{settings.chroma_persist_dir}'"
    )
    return store


def add_chunks(chunks: list[Document]) -> int:
    """
    Upsert chunks into ChromaDB.

    Deduplication: existing chunks with the same chunk_id are not re-embedded.

    Args:
        chunks: Output from chunker.chunk_documents()

    Returns:
        Number of chunks actually added (excluding duplicates).
    """
    store = get_vector_store()

    # Get existing IDs to skip duplicates
    existing = store._collection.get(include=[])["ids"]
    existing_set = set(existing)

    new_chunks = [
        c for c in chunks if c.metadata.get("chunk_id") not in existing_set
    ]

    if not new_chunks:
        console.print("[yellow]⚠ All chunks already in store — skipping.[/yellow]")
        return 0

    ids = [c.metadata["chunk_id"] for c in new_chunks]
    store.add_documents(new_chunks, ids=ids)
    
    # Rebuild BM25 index to include the new chunks
    from src.vectorstore.bm25_store import rebuild_bm25_index
    rebuild_bm25_index()

    console.print(
        f"[green]✓ Added {len(new_chunks)} new chunks "
        f"({len(chunks) - len(new_chunks)} duplicates skipped)[/green]"
    )
    return len(new_chunks)


def similarity_search(
    query: str,
    k: int | None = None,
    filter_metadata: dict | None = None,
) -> list[Document]:
    """
    Retrieve top-k most similar chunks for a query.

    Args:
        query:           Natural language question.
        k:               Number of results (default: settings.retriever_top_k).
        filter_metadata: Optional ChromaDB `where` filter dict.
                         e.g. {"chapter": "chapter_03"}

    Returns:
        List of Documents sorted by relevance (most relevant first).
    """
    store = get_vector_store()
    top_k = k or settings.retriever_top_k

    results = store.similarity_search(
        query,
        k=top_k,
        filter=filter_metadata,
    )
    return results


def get_collection_stats() -> dict:
    """Return basic stats about the current ChromaDB collection."""
    client = _get_chroma_client()
    try:
        collection = client.get_collection(settings.chroma_collection_name)
        count = collection.count()
    except Exception:
        count = 0

    return {
        "collection": settings.chroma_collection_name,
        "total_chunks": count,
        "persist_dir": settings.chroma_persist_dir,
        "embedding_model": settings.embedding_model,
    }
