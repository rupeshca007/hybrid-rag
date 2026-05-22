"""
FastAPI application entry point.

Endpoints:
  GET  /            → health check
  GET  /stats       → ChromaDB collection stats
  POST /ingest      → upload & index a PDF chapter
  POST /query       → ask a question against indexed docs

Interactive docs at:
  http://localhost:8000/docs    (Swagger UI)
  http://localhost:8000/redoc   (ReDoc)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console

from config.settings import settings
from src.api.routes.ingest import router as ingest_router
from src.api.routes.query import router as query_router
from src.api.schemas import HealthResponse, StatsResponse
from src.vectorstore.chroma_store import get_collection_stats, get_vector_store

console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up the vector store and embedding model on startup."""
    console.print("\n[bold green]🚀 RAG System starting up…[/bold green]")
    # Pre-load ChromaDB + embeddings so first query is fast
    get_vector_store()
    
    # Initialize BM25 keyword index from stored chunks
    from src.vectorstore.bm25_store import rebuild_bm25_index
    rebuild_bm25_index()
    
    console.print("[bold green]✓ Ready! Docs at http://localhost:8000/docs[/bold green]\n")
    yield
    console.print("[yellow]RAG System shutting down.[/yellow]")


app = FastAPI(
    title="Domain-Specific RAG System",
    description=(
        "Production-grade Retrieval-Augmented Generation API. "
        "Upload PDF chapters via /ingest, then ask questions via /query."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS (open for local dev — restrict in production) ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────
app.include_router(ingest_router)
app.include_router(query_router)


# ── Root endpoints ─────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"], summary="Health check")
async def health():
    """Returns ok if the server is running."""
    return HealthResponse()


@app.get("/stats", response_model=StatsResponse, tags=["Health"], summary="ChromaDB stats")
async def stats():
    """Returns current collection stats: total chunks, embedding model, etc."""
    return StatsResponse(**get_collection_stats())


# ── Dev server entry point ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level,
    )
