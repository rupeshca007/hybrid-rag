"""Tests for ChromaDB vector store operations."""
import sys
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document


class TestVectorStore:
    """
    Unit tests for vector store helpers.
    These mock ChromaDB so they run without a real DB.
    """

    def test_docs_to_sources_schema(self):
        """Verify SourceChunk is built correctly from Document metadata."""
        from src.generation.chain import _docs_to_sources

        doc = Document(
            page_content="Test content about neural networks.",
            metadata={
                "filename": "chapter_01.pdf",
                "chapter": "chapter_01",
                "page_number": 5,
                "chunk_id": "abc123",
            },
        )
        sources = _docs_to_sources([doc])
        assert len(sources) == 1
        s = sources[0]
        assert s.filename == "chapter_01.pdf"
        assert s.page_number == 5
        assert s.chunk_id == "abc123"
        assert "neural" in s.content_preview

    def test_source_preview_truncated(self):
        """Content preview must be ≤ 200 chars."""
        from src.generation.chain import _docs_to_sources

        doc = Document(
            page_content="X" * 500,
            metadata={"filename": "f.pdf", "chapter": "c",
                       "page_number": 1, "chunk_id": "id1"},
        )
        sources = _docs_to_sources([doc])
        assert len(sources[0].content_preview) <= 200
