"""
BM25 Keyword Search Index (In-Memory).

Provides exact keyword matching to complement ChromaDB's semantic search.
Rebuilds the index automatically on startup or when new chunks are ingested.
"""

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from rich.console import Console

console = Console()

# Global state for the in-memory BM25 index
_bm25_index = None
_bm25_documents = []


def rebuild_bm25_index():
    """
    Fetch all documents from ChromaDB and rebuild the BM25 index.
    This runs entirely in-memory and is very fast for <10k chunks.
    """
    global _bm25_index, _bm25_documents
    
    from src.vectorstore.chroma_store import get_vector_store
    store = get_vector_store()
    
    # Get all documents from Chroma
    collection = store._collection
    result = collection.get(include=["documents", "metadatas"])
    
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    
    if not docs:
        console.print("[yellow]BM25 Index: No documents found in ChromaDB to index.[/yellow]")
        _bm25_index = None
        _bm25_documents = []
        return

    # Store corresponding LangChain documents for returning
    _bm25_documents = [
        Document(page_content=doc, metadata=meta) 
        for doc, meta in zip(docs, metas)
    ]
    
    # Tokenize corpus for BM25 (simple whitespace tokenization)
    tokenized_corpus = [doc.lower().split() for doc in docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    
    console.print(f"[green]✓ BM25 Index rebuilt with {len(docs)} chunks[/green]")


def bm25_search(query: str, k: int = 20, filter_metadata: dict | None = None) -> list[Document]:
    """
    Perform a keyword search using BM25.
    
    Args:
        query: Natural language query.
        k: Number of chunks to return.
        filter_metadata: Optional dictionary to filter by metadata (e.g. {"chapter": "..."})
        
    Returns:
        List of Document objects sorted by BM25 score.
    """
    global _bm25_index, _bm25_documents
    
    if _bm25_index is None:
        # Try to build it if it doesn't exist yet
        rebuild_bm25_index()
        if _bm25_index is None:
            return []

    tokenized_query = query.lower().split()
    
    # Get scores for all documents
    scores = _bm25_index.get_scores(tokenized_query)
    
    # Pair documents with scores
    scored_docs = list(zip(_bm25_documents, scores))
    
    # Filter if requested
    if filter_metadata:
        scored_docs = [
            (doc, score) for doc, score in scored_docs 
            if all(doc.metadata.get(key) == val for key, val in filter_metadata.items())
        ]
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out zero scores and take top k
    top_docs = [doc for doc, score in scored_docs if score > 0][:k]
    
    return top_docs
