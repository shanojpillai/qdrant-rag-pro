"""
Document models for QdrantRAG-Pro.

This module defines the data structures for documents and their metadata
used throughout the RAG system.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class DocumentMetadata(BaseModel):
    """Metadata associated with a document."""
    
    # Basic metadata
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Source of the document")
    url: Optional[str] = Field(None, description="Original URL if applicable")
    
    # Classification
    category: Optional[str] = Field(None, description="Document category")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    language: str = Field("en", description="Document language code")
    
    # Technical metadata
    document_type: DocumentType = Field(DocumentType.TEXT, description="Type of document")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    word_count: Optional[int] = Field(None, description="Number of words")
    
    # Timestamps
    created_at: Optional[datetime] = Field(None, description="Document creation time")
    modified_at: Optional[datetime] = Field(None, description="Last modification time")
    ingested_at: Optional[datetime] = Field(None, description="Ingestion timestamp")
    
    # Processing metadata
    status: DocumentStatus = Field(DocumentStatus.PENDING, description="Processing status")
    processing_version: str = Field("1.0", description="Processing pipeline version")
    
    # Custom metadata
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
    
    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Ensure tags are non-empty strings."""
        return [tag.strip() for tag in v if tag.strip()]

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Ensure language code is valid."""
        # Basic validation for common language codes
        valid_codes = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar"]
        if v not in valid_codes:
            # Allow any 2-letter code for flexibility
            if len(v) != 2:
                raise ValueError("Language code must be a 2-letter ISO code")
        return v.lower()
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class Document(BaseModel):
    """Main document model for the RAG system."""
    
    id: Optional[str] = Field(None, description="Unique document identifier")
    content: str = Field(..., description="Main document content")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    
    # Chunking information
    chunks: Optional[List[str]] = Field(None, description="Document chunks for processing")
    chunk_metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each chunk")
    
    # Embedding information
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    chunk_embeddings: Optional[List[List[float]]] = Field(None, description="Chunk embedding vectors")
    
    # Processing information
    token_count: Optional[int] = Field(None, description="Total token count")
    chunk_token_counts: Optional[List[int]] = Field(None, description="Token count per chunk")
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Document content cannot be empty")
        return v.strip()

    @model_validator(mode='after')
    def validate_chunks_consistency(self):
        """Validate chunks consistency."""
        if self.chunks is not None:
            # Ensure chunks are non-empty
            self.chunks = [chunk.strip() for chunk in self.chunks if chunk.strip()]
            if not self.chunks:
                self.chunks = None

        # Validate chunk embeddings match chunks
        if self.chunk_embeddings is not None and self.chunks is not None:
            if len(self.chunk_embeddings) != len(self.chunks):
                raise ValueError("Number of chunk embeddings must match number of chunks")

        # Validate chunk token counts match chunks
        if self.chunk_token_counts is not None and self.chunks is not None:
            if len(self.chunk_token_counts) != len(self.chunks):
                raise ValueError("Number of chunk token counts must match number of chunks")

        return self
    
    @property
    def has_chunks(self) -> bool:
        """Check if document has chunks."""
        return self.chunks is not None and len(self.chunks) > 0
    
    @property
    def has_embeddings(self) -> bool:
        """Check if document has embeddings."""
        return self.embedding is not None
    
    @property
    def has_chunk_embeddings(self) -> bool:
        """Check if document has chunk embeddings."""
        return self.chunk_embeddings is not None and len(self.chunk_embeddings) > 0
    
    def get_chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks) if self.chunks else 0
    
    def get_total_tokens(self) -> int:
        """Get total token count."""
        if self.token_count:
            return self.token_count
        elif self.chunk_token_counts:
            return sum(self.chunk_token_counts)
        else:
            # Rough estimate
            return len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return self.dict(exclude_none=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create document from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_text(
        cls, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None
    ) -> "Document":
        """Create document from plain text."""
        doc_metadata = DocumentMetadata(**(metadata or {}))
        return cls(
            id=document_id,
            content=content,
            metadata=doc_metadata
        )
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
