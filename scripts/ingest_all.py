"""
Bulk ingestion script — indexes all PDFs from data/pdfs/ into ChromaDB.

Usage (from project root):
    python scripts/ingest_all.py
    python scripts/ingest_all.py --pdf-dir /path/to/custom/folder
    python scripts/ingest_all.py --dry-run   (counts files without indexing)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from config.settings import settings
from src.ingestion.loader import load_pdfs_from_dir
from src.ingestion.chunker import chunk_documents
from src.vectorstore.chroma_store import add_chunks, get_collection_stats

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest PDFs into ChromaDB")
    parser.add_argument(
        "--pdf-dir",
        default=settings.pdf_dir,
        help=f"Directory containing PDFs (default: {settings.pdf_dir})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files without actually indexing them",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    console.rule("[bold blue]RAG Bulk Ingestion[/bold blue]")
    console.print(f"[cyan]Directory:[/cyan] {pdf_dir.resolve()}")

    if args.dry_run:
        pdfs = list(pdf_dir.rglob("*.pdf"))
        console.print(f"[yellow]DRY RUN — found {len(pdfs)} PDF(s):[/yellow]")
        for p in pdfs:
            console.print(f"  • {p.name}")
        return

    # ── Load ─────────────────────────────────────────────────
    console.rule("Step 1: Loading PDFs")
    docs = load_pdfs_from_dir(pdf_dir)

    if not docs:
        console.print("[red]No documents loaded. Exiting.[/red]")
        sys.exit(1)

    # ── Chunk ─────────────────────────────────────────────────
    console.rule("Step 2: Chunking")
    chunks = chunk_documents(docs)

    # ── Store ─────────────────────────────────────────────────
    console.rule("Step 3: Indexing into ChromaDB")
    added = add_chunks(chunks)

    # ── Summary ───────────────────────────────────────────────
    stats = get_collection_stats()
    console.rule("[bold green]Ingestion Complete[/bold green]")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Pages loaded", str(len(docs)))
    table.add_row("Chunks created", str(len(chunks)))
    table.add_row("New chunks added", str(added))
    table.add_row("Total chunks in store", str(stats["total_chunks"]))
    table.add_row("Collection", stats["collection"])
    table.add_row("Embedding model", stats["embedding_model"])
    console.print(table)


if __name__ == "__main__":
    main()
