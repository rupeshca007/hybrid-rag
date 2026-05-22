"""
PDF Loader using PyMuPDF (fitz).

Why PyMuPDF over pypdf?
- Handles complex layouts, multi-column PDFs, scanned text with embedded fonts
- Preserves page numbers and section metadata accurately
- ~10x faster on large documents

Returns a list of LangChain Document objects with rich metadata:
  - source       : original file path
  - filename     : basename of the PDF
  - page_number  : 1-indexed page
  - total_pages  : total pages in document
  - chapter      : derived from filename (without extension)
"""

from pathlib import Path
import fitz  # PyMuPDF
from langchain_core.documents import Document
from rich.console import Console

console = Console()


def load_pdf(file_path: str | Path) -> list[Document]:
    """
    Load a single PDF and return a list of Documents (one per page).

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Documents with page-level metadata.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    documents: list[Document] = []
    chapter_name = path.stem  # filename without extension

    with fitz.open(str(path)) as pdf:
        total_pages = len(pdf)
        console.print(
            f"[cyan]Loading[/cyan] '{path.name}' — {total_pages} pages"
        )

        for page_idx in range(total_pages):
            page = pdf[page_idx]
            text = page.get_text("text").strip()

            # Skip blank pages
            if not text:
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(path.resolve()),
                    "filename": path.name,
                    "chapter": chapter_name,
                    "page_number": page_idx + 1,  # 1-indexed
                    "total_pages": total_pages,
                },
            )
            documents.append(doc)

    console.print(
        f"[green]✓[/green] Loaded {len(documents)} non-blank pages from '{path.name}'"
    )
    return documents


def load_pdfs_from_dir(directory: str | Path) -> list[Document]:
    """
    Load all PDFs from a directory recursively.

    Args:
        directory: Path to directory containing PDFs.

    Returns:
        Combined list of Documents from all PDFs.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    pdf_files = sorted(dir_path.rglob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]No PDFs found in {dir_path}[/yellow]")
        return []

    console.print(f"[blue]Found {len(pdf_files)} PDF(s) in {dir_path}[/blue]")

    all_docs: list[Document] = []
    for pdf_path in pdf_files:
        try:
            docs = load_pdf(pdf_path)
            all_docs.extend(docs)
        except Exception as e:
            console.print(f"[red]✗ Failed to load {pdf_path.name}: {e}[/red]")

    console.print(
        f"[green]✓ Total: {len(all_docs)} pages loaded from {len(pdf_files)} PDF(s)[/green]"
    )
    return all_docs
