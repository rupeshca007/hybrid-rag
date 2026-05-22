"""
Hybrid Search Implementation.

Combines:
1. Vector Semantic Search (ChromaDB)
2. Keyword Search (BM25)
3. Reranking (Cohere)

Uses Reciprocal Rank Fusion (RRF) to merge the semantic and keyword results
before sending the top candidate chunks to Cohere for final reranking.
"""

from typing import List, Dict
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from rich.console import Console

from config.settings import settings
from src.vectorstore.chroma_store import similarity_search as vector_search
from src.vectorstore.bm25_store import bm25_search

console = Console()


def reciprocal_rank_fusion(
    vector_results: List[Document],
    bm25_results: List[Document],
    k: int = 60
) -> List[Document]:
    """
    Fuse two ranked lists of documents using Reciprocal Rank Fusion (RRF).
    Score = 1 / (rank + k)
    """
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    def add_to_fusion(docs: List[Document]):
        for rank, doc in enumerate(docs):
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                continue
            
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                doc_map[chunk_id] = doc
                
            fused_scores[chunk_id] += 1.0 / (rank + k)

    add_to_fusion(vector_results)
    add_to_fusion(bm25_results)

    # Sort by RRF score descending
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return the Document objects in the fused order
    return [doc_map[chunk_id] for chunk_id, score in sorted_fused]


def hybrid_search(
    query: str,
    top_k: int | None = None,
    filter_metadata: dict | None = None,
) -> List[Document]:
    """
    Perform hybrid retrieval: Vector + BM25 -> RRF -> Cohere Rerank.
    """
    k = top_k or settings.retriever_top_k
    hybrid_k = settings.hybrid_search_k
    
    console.print(f"[cyan]Hybrid Search:[/cyan] Fetching top {hybrid_k} via Vector and BM25...")
    
    # 1. Fetch from both systems
    vector_docs = vector_search(query, k=hybrid_k, filter_metadata=filter_metadata)
    bm25_docs = bm25_search(query, k=hybrid_k, filter_metadata=filter_metadata)
    
    console.print(f"  - Vector found {len(vector_docs)} chunks")
    console.print(f"  - BM25 found {len(bm25_docs)} chunks")
    
    # 2. Fuse the results
    fused_docs = reciprocal_rank_fusion(vector_docs, bm25_docs)
    console.print(f"  - Fused into {len(fused_docs)} unique chunks")
    
    # If no results, return early
    if not fused_docs:
        return []
        
    # Trim to hybrid_k before sending to Cohere (to save API costs/time)
    fused_docs = fused_docs[:hybrid_k]
    
    # 3. Rerank using Cohere
    if not settings.cohere_api_key:
        console.print("[yellow]⚠ COHERE_API_KEY not set. Skipping reranking step and returning RRF top-k.[/yellow]")
        return fused_docs[:k]
        
    try:
        console.print("[cyan]Reranking[/cyan] with Cohere...")
        reranker = CohereRerank(
            cohere_api_key=settings.cohere_api_key,
            model="rerank-english-v3.0",
            top_n=k
        )
        # compress_documents applies the reranking model
        final_docs = reranker.compress_documents(documents=fused_docs, query=query)
        console.print(f"[green]✓ Final {len(final_docs)} chunks selected[/green]")
        return list(final_docs)
        
    except Exception as e:
        console.print(f"[red]✗ Cohere reranking failed: {e}[/red]")
        console.print("[yellow]Falling back to RRF top-k.[/yellow]")
        return fused_docs[:k]
