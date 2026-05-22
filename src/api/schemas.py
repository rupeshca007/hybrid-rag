"""
Pydantic schemas for all API request/response models.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Ingest ──────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Response from POST /ingest"""
    status: str
    filename: str
    pages_loaded: int
    chunks_added: int
    message: str


# ── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="The question to ask your documents.",
        examples=["What is the main topic of this chapter?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of chunks to retrieve (default: 6). Increase for broader answers.",
    )
    chapter_filter: str | None = Field(
        default=None,
        description=(
            "OPTIONAL: Restrict search to one PDF by its filename stem (without .pdf). "
            "e.g. if you uploaded 'work.pdf', use 'work'. Leave null to search ALL documents."
        ),
    )
    prompt_version: str | None = Field(
        default=None,
        description="OPTIONAL: Prompt version to use ('v1', 'v2'). Leave null to use default from .env.",
    )


class SourceChunkSchema(BaseModel):
    """A single source chunk cited in the answer."""
    filename: str
    chapter: str
    page_number: int
    chunk_id: str
    content_preview: str


class QueryResponse(BaseModel):
    """Response from POST /query"""
    question: str
    answer: str
    prompt_version: str
    model: str
    num_chunks_retrieved: int
    sources: list[SourceChunkSchema]


# ── Health / Stats ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"


class StatsResponse(BaseModel):
    """Response from GET /stats"""
    collection: str
    total_chunks: int
    persist_dir: str
    embedding_model: str
