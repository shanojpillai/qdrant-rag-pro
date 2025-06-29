"""
Configuration settings for QdrantRAG-Pro.

This module provides centralized configuration management using Pydantic settings
with environment variable support and validation.
"""

from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    
    # Qdrant Configuration
    qdrant_host: str = Field("localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field("qdrant_rag_collection", env="QDRANT_COLLECTION_NAME")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")
    
    # Redis Configuration (Optional)
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")
    
    # Application Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_search_results: int = Field(10, env="MAX_SEARCH_RESULTS")
    default_vector_weight: float = Field(0.7, env="DEFAULT_VECTOR_WEIGHT")
    default_keyword_weight: float = Field(0.3, env="DEFAULT_KEYWORD_WEIGHT")
    min_search_score: float = Field(0.6, env="MIN_SEARCH_SCORE")
    
    # Embedding Configuration
    max_tokens_per_chunk: int = Field(8192, env="MAX_TOKENS_PER_CHUNK")
    chunk_overlap_tokens: int = Field(200, env="CHUNK_OVERLAP_TOKENS")
    batch_size: int = Field(100, env="BATCH_SIZE")
    
    # Response Generation
    max_response_tokens: int = Field(1000, env="MAX_RESPONSE_TOKENS")
    response_temperature: float = Field(0.1, env="RESPONSE_TEMPERATURE")
    max_sources_per_response: int = Field(5, env="MAX_SOURCES_PER_RESPONSE")
    
    # Development Settings
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("development", env="ENVIRONMENT")
    
    # Optional: Alternative LLM Providers
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    huggingface_api_key: Optional[str] = Field(None, env="HUGGINGFACE_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @field_validator("default_vector_weight", "default_keyword_weight")
    @classmethod
    def validate_weights(cls, v):
        """Ensure weights are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Ensure log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ["development", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    @property
    def qdrant_url(self) -> str:
        """Get the full Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def redis_url(self) -> str:
        """Get the full Redis URL."""
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}@{self.redis_host}:"
                f"{self.redis_port}/{self.redis_db}"
            )
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    settings = Settings()
    return settings
