"""Tests for PDF loading and chunking pipeline."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_documents, build_splitter
from langchain_core.documents import Document


class TestChunker:
    """Test the chunking pipeline without needing real PDFs."""

    def test_splitter_builds_correctly(self):
        splitter = build_splitter(chunk_size=200, chunk_overlap=20)
        assert splitter._chunk_size == 200
        assert splitter._chunk_overlap == 20

    def test_chunk_documents_splits_text(self):
        # Create a synthetic "page" document
        long_text = "This is a sentence about topic A. " * 50  # ~1700 chars
        doc = Document(
            page_content=long_text,
            metadata={
                "source": "test.pdf",
                "filename": "test.pdf",
                "chapter": "test",
                "page_number": 1,
                "total_pages": 1,
            },
        )
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

        # Should be split into multiple chunks
        assert len(chunks) > 1

    def test_chunks_inherit_metadata(self):
        doc = Document(
            page_content="A" * 1000,
            metadata={"source": "x.pdf", "filename": "x.pdf", "chapter": "x",
                       "page_number": 3, "total_pages": 10},
        )
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

        for chunk in chunks:
            assert chunk.metadata["page_number"] == 3
            assert chunk.metadata["chapter"] == "x"
            assert "chunk_id" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_ids_are_unique(self):
        doc = Document(
            page_content="Different content on every chunk " * 40,
            metadata={"source": "y.pdf", "filename": "y.pdf", "chapter": "y",
                       "page_number": 1, "total_pages": 1},
        )
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=0)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_empty_documents_returns_empty(self):
        chunks = chunk_documents([])
        assert chunks == []
