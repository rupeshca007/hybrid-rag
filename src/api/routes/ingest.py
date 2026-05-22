"""
POST /ingest — upload a PDF and index it into ChromaDB.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from rich.console import Console

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import chunk_documents
from src.vectorstore.chroma_store import add_chunks
from src.api.schemas import IngestResponse

console = Console()
router = APIRouter(prefix="/ingest", tags=["Ingestion"])


@router.post("", response_model=IngestResponse, summary="Upload and index a PDF chapter")
async def ingest_pdf(file: UploadFile = File(..., description="PDF file to index")):
    """
    Upload a PDF chapter and index it into ChromaDB.

    - Loads each page with PyMuPDF
    - Splits into 800-char chunks with 100-char overlap
    - Embeds with local sentence-transformers
    - Stores in ChromaDB with deduplication

    Returns stats about pages loaded and chunks added.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    console.print(f"\n[bold blue]Ingest request:[/bold blue] {file.filename}")

    # Save upload to a temp file (UploadFile is a stream)
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Load → Chunk → Store
        docs = load_pdf(tmp_path)
        # Preserve original filename in metadata (not the temp path)
        for doc in docs:
            doc.metadata["filename"] = file.filename
            doc.metadata["chapter"] = Path(file.filename).stem
            doc.metadata["source"] = file.filename

        chunks = chunk_documents(docs)
        added = add_chunks(chunks)

    except Exception as e:
        console.print(f"[red]✗ Ingest failed: {e}[/red]")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        tmp_path.unlink(missing_ok=True)  # Clean up temp file

    return IngestResponse(
        status="success",
        filename=file.filename,
        pages_loaded=len(docs),
        chunks_added=added,
        message=f"Successfully indexed '{file.filename}': {len(docs)} pages → {added} chunks added.",
    )
