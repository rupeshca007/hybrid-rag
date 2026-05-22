"""
Pydantic-settings config — reads from .env automatically.
All downstream modules import `settings` from here.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM Provider ─────────────────────────────────────────
    # "openai" or "groq"
    llm_provider: str = "openai"

    # ── OpenAI ──────────────────────────────────────────────
    openai_api_key: str = ""  # Set in .env — validated at runtime when needed
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.0
    openai_max_tokens: int = 1024

    # ── Groq (free tier) ─────────────────────────────────────
    groq_api_key: str = ""   # Get free key at https://console.groq.com/keys
    groq_model: str = "llama-3.3-70b-versatile"

    # ── Embeddings ──────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ── ChromaDB ────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "chapters"

    # ── Chunking ────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100

    # ── Cohere (Reranking) ──────────────────────────────────
    cohere_api_key: str = "" # Set in .env

    # ── Retrieval & Hybrid Search ───────────────────────────
    retriever_top_k: int = 6
    hybrid_search_k: int = 20
    rerank_top_k: int = 5

    # ── Prompt Versioning ───────────────────────────────────
    prompt_version: str = "v1"
    prompts_dir: str = "./prompts"

    # ── API ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    log_level: str = "info"

    # ── Data ────────────────────────────────────────────────
    pdf_dir: str = "./data/pdfs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — import this everywhere."""
    return Settings()


# Convenient top-level import alias
settings = get_settings()
