"""Configuration management for Robinhood RAG project."""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    LLM_BASE_URL: str = Field(
        default="https://api.deepseek.com",
        description="Base URL for OpenAI-compatible API"
    )
    LLM_API_KEY: str = Field(
        default="",
        description="API key for LLM service"
    )
    LLM_MODEL: str = Field(
        default="deepseek-reasoner",
        description="Model name to use for reasoning"
    )
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # Chunking Configuration
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Size of text chunks in characters"
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between chunks in characters"
    )
    
    # Search Configuration
    SEARCH_TOP_K: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    RRF_K: int = Field(
        default=60,
        description="Constant k for Reciprocal Rank Fusion"
    )
    
    # Agent Configuration
    MAX_AGENT_ITERATIONS: int = Field(
        default=5,
        description="Maximum iterations for agent loop"
    )
    
    # Path Configuration
    DATA_DIR: str = Field(
        default="data",
        description="Directory for data storage"
    )
    INDEX_DIR: str = Field(
        default="indexes",
        description="Directory for index storage"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global settings instance
settings = Settings()

# Get project root directory (where config.py is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def get_raw_data_path() -> str:
    """Get path for raw scraped data."""
    path = os.path.join(PROJECT_ROOT, settings.DATA_DIR, "raw")
    os.makedirs(path, exist_ok=True)
    return path


def get_processed_data_path() -> str:
    """Get path for processed chunks."""
    path = os.path.join(PROJECT_ROOT, settings.DATA_DIR, "processed")
    os.makedirs(path, exist_ok=True)
    return path


def get_chroma_path() -> str:
    """Get path for ChromaDB storage."""
    path = os.path.join(PROJECT_ROOT, settings.INDEX_DIR, "chroma")
    os.makedirs(path, exist_ok=True)
    return path


def get_bm25_path() -> str:
    """Get path for BM25 index storage."""
    path = os.path.join(PROJECT_ROOT, settings.INDEX_DIR, "bm25")
    os.makedirs(path, exist_ok=True)
    return path
