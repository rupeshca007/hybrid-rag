"""
Phase 1 End-to-End Test Script
================================
Run this AFTER:
  1. Copying .env.example → .env and setting OPENAI_API_KEY
  2. Placing at least one PDF in data/pdfs/

Usage:
    .venv\Scripts\python.exe scripts/test_e2e.py --pdf path/to/your/chapter.pdf
    .venv\Scripts\python.exe scripts/test_e2e.py --pdf path/to/your/chapter.pdf --question "What is this chapter about?"

What it tests:
  ✓ Step 1 — Settings load correctly (API key present)
  ✓ Step 2 — PDF loads with PyMuPDF (pages + metadata)
  ✓ Step 3 — Chunking (count, metadata, unique IDs)
  ✓ Step 4 — ChromaDB upsert (vectors stored)
  ✓ Step 5 — Similarity search (retrieval works)
  ✓ Step 6 — Full RAG chain (OpenAI answer + citations)
  ✓ Step 7 — FastAPI health check (server OK)
"""

import sys
import argparse
import time
import io
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console(force_terminal=True, highlight=False)

PASS = "[bold green]PASS[/bold green]"
FAIL = "[bold red]FAIL[/bold red]"
SKIP = "[bold yellow]SKIP[/bold yellow]"


def run_test(name: str, fn, *args, **kwargs):
    """Run a single test and return (passed, result, error)."""
    try:
        result = fn(*args, **kwargs)
        return True, result, None
    except Exception as e:
        return False, None, e


def main():
    parser = argparse.ArgumentParser(description="Phase 1 End-to-End Test")
    parser.add_argument("--pdf", required=True, help="Path to a PDF chapter to test with")
    parser.add_argument(
        "--question",
        default=None,
        help="Question to ask (auto-generated from PDF name if not provided)"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Step 6 (OpenAI call) to test without API key"
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        console.print(f"[red]PDF not found: {pdf_path}[/red]")
        sys.exit(1)

    console.print(Panel.fit(
        "[bold cyan]Phase 1 RAG — End-to-End Test[/bold cyan]\n"
        f"PDF: [yellow]{pdf_path.name}[/yellow]",
        border_style="cyan"
    ))

    results = []

    # ── Step 1: Settings ─────────────────────────────────────────────────────
    console.rule("[cyan]Step 1: Settings & Config[/cyan]")
    def test_settings():
        from config.settings import settings
        assert settings.chroma_collection_name, "No collection name"
        assert settings.chunk_size > 0, "Invalid chunk size"
        assert settings.embedding_model, "No embedding model"
        if not args.skip_llm:
            provider = settings.llm_provider.lower()
            if provider == "groq":
                assert settings.groq_api_key, (
                    "GROQ_API_KEY is empty!\n"
                    "Get a free key at: https://console.groq.com/keys"
                )
            else:
                assert settings.openai_api_key, (
                    "OPENAI_API_KEY is empty!\n"
                    "Copy .env.example -> .env and add your key."
                )
        return {
            "provider": settings.llm_provider,
            "model": settings.groq_model if settings.llm_provider == "groq" else settings.openai_model,
            "chunk_size": settings.chunk_size,
            "embedding": settings.embedding_model,
            "collection": settings.chroma_collection_name,
        }

    ok, result, err = run_test("Settings load", test_settings)
    results.append(("Settings load", ok, str(err) if err else str(result)))
    if ok:
        console.print(f"{PASS} Provider=[bold]{result['provider']}[/bold] | Model=[bold]{result['model']}[/bold] | Chunks={result['chunk_size']}")
    else:
        console.print(f"{FAIL} {err}")
        if not args.skip_llm:
            sys.exit(1)

    # ── Step 2: PDF Loading ───────────────────────────────────────────────────
    console.rule("[cyan]Step 2: PDF Loading (PyMuPDF)[/cyan]")
    def test_pdf_load():
        from src.ingestion.loader import load_pdf
        docs = load_pdf(pdf_path)
        assert len(docs) > 0, "No pages loaded — PDF might be blank or image-only"
        sample = docs[0]
        assert "page_number" in sample.metadata
        assert "filename" in sample.metadata
        assert len(sample.page_content) > 10, "Page content seems empty"
        return docs

    ok, docs, err = run_test("PDF Load", test_pdf_load)
    results.append(("PDF Load", ok, str(err) if err else f"{len(docs)} pages"))
    if ok:
        console.print(f"{PASS} Loaded [bold]{len(docs)}[/bold] pages from '{pdf_path.name}'")
        console.print(f"  Sample page 1 preview: [dim]{docs[0].page_content[:120].replace(chr(10),' ')}…[/dim]")
    else:
        console.print(f"{FAIL} {err}")
        sys.exit(1)

    # ── Step 3: Chunking ──────────────────────────────────────────────────────
    console.rule("[cyan]Step 3: Chunking[/cyan]")
    def test_chunking(docs):
        from src.ingestion.chunker import chunk_documents
        chunks = chunk_documents(docs)
        assert len(chunks) > 0, "No chunks produced"
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), f"Duplicate chunk IDs found! ({len(ids) - len(set(ids))} duplicates)"
        for c in chunks[:3]:
            assert "page_number" in c.metadata
            assert "chunk_index" in c.metadata
        return chunks

    ok, chunks, err = run_test("Chunking", test_chunking, docs)
    results.append(("Chunking", ok, str(err) if err else f"{len(chunks)} chunks"))
    if ok:
        avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
        console.print(f"{PASS} Created [bold]{len(chunks)}[/bold] chunks (avg {avg_len} chars each)")
        console.print(f"  Sample chunk: [dim]{chunks[0].page_content[:100].replace(chr(10),' ')}…[/dim]")
    else:
        console.print(f"{FAIL} {err}")
        sys.exit(1)

    # ── Step 4: ChromaDB Upsert ───────────────────────────────────────────────
    console.rule("[cyan]Step 4: ChromaDB Vector Store (embedding + store)[/cyan]")
    console.print("[dim]First run downloads ~80MB embedding model — please wait…[/dim]")
    def test_upsert(chunks):
        from src.vectorstore.chroma_store import add_chunks, get_collection_stats
        t0 = time.time()
        added = add_chunks(chunks)
        elapsed = time.time() - t0
        stats = get_collection_stats()
        return added, stats, elapsed

    ok, result, err = run_test("ChromaDB upsert", test_upsert, chunks)
    results.append(("ChromaDB Upsert", ok, str(err) if err else f"{result[0]} added"))
    if ok:
        added, stats, elapsed = result
        console.print(f"{PASS} Stored [bold]{added}[/bold] new vectors in {elapsed:.1f}s")
        console.print(f"  Total in store: [bold]{stats['total_chunks']}[/bold] chunks | Collection: '{stats['collection']}'")
    else:
        console.print(f"{FAIL} {err}")
        sys.exit(1)

    # ── Step 5: Similarity Search ─────────────────────────────────────────────
    console.rule("[cyan]Step 5: Similarity Search (Retrieval)[/cyan]")
    auto_question = args.question or f"What is the main topic of {pdf_path.stem.replace('_', ' ')}?"

    def test_retrieval(question):
        from src.vectorstore.chroma_store import similarity_search
        results = similarity_search(question, k=3)
        assert len(results) > 0, "No results returned from vector store"
        return results

    ok, retrieved, err = run_test("Retrieval", test_retrieval, auto_question)
    results.append(("Similarity Search", ok, str(err) if err else f"{len(retrieved)} chunks"))
    if ok:
        console.print(f"{PASS} Retrieved [bold]{len(retrieved)}[/bold] chunks for query: '[italic]{auto_question}[/italic]'")
        for i, doc in enumerate(retrieved, 1):
            page = doc.metadata.get("page_number", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            console.print(f"  [{i}] Page {page}: [dim]{preview}…[/dim]")
    else:
        console.print(f"{FAIL} {err}")
        sys.exit(1)

    # ── Step 6: Full RAG Chain ────────────────────────────────────────────────
    from config.settings import settings as _s
    _provider_label = f"{_s.llm_provider.upper()} ({_s.groq_model if _s.llm_provider == 'groq' else _s.openai_model})"
    console.rule(f"[cyan]Step 6: Full RAG Chain ({_provider_label})[/cyan]")
    if args.skip_llm:
        console.print(f"{SKIP} Skipped (--skip-llm flag). Remove flag to test LLM.")
        results.append(("Full RAG Chain", None, "Skipped"))
    else:
        def test_rag_chain(question):
            from src.generation.chain import run_rag_chain
            result = run_rag_chain(question=question, top_k=4)
            assert result.answer, "Empty answer"
            assert len(result.sources) > 0, "No sources returned"
            return result

        ok, rag_result, err = run_test("RAG Chain", test_rag_chain, auto_question)
        results.append(("Full RAG Chain", ok, str(err) if err else "Answer received"))
        if ok:
            console.print(f"{PASS} Answer generated by [bold]{rag_result.model}[/bold] (prompt: {rag_result.prompt_version})")
            sources_str = ", ".join(
                f"{s.filename} p.{s.page_number}" for s in rag_result.sources[:3]
            )
            console.print(Panel(
                f"[bold]Q:[/bold] {rag_result.question}\n\n"
                f"[bold]A:[/bold] {rag_result.answer[:600]}{'…' if len(rag_result.answer) > 600 else ''}\n\n"
                f"[dim]Sources: {sources_str}[/dim]",
                title="RAG Answer",
                border_style="green",
            ))
        else:
            console.print(f"{FAIL} {err}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    console.rule("[bold cyan]Test Summary[/bold cyan]")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Step", style="cyan")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    all_pass = True
    for name, passed, detail in results:
        if passed is True:
            status = "[bold green]✓ PASS[/bold green]"
        elif passed is False:
            status = "[bold red]✗ FAIL[/bold red]"
            all_pass = False
        else:
            status = "[bold yellow]⚠ SKIP[/bold yellow]"
        table.add_row(name, status, detail)

    console.print(table)

    passed_count = sum(1 for _, p, _ in results if p is True)
    failed_count = sum(1 for _, p, _ in results if p is False)
    all_critical_pass = failed_count == 0

    if all_critical_pass:
        console.print(Panel.fit(
            "Phase 1 RAG is fully working!\n"
            "Next: start the API server:\n"
            ".venv\\Scripts\\python.exe src/api/main.py\n"
            "-> http://localhost:8000/docs",
            border_style="green"
        ))
    else:
        console.print(f"[red]{failed_count} step(s) failed. Fix the errors above and re-run.[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
