"""
Search result models for QdrantRAG-Pro.

This module defines the data structures for search results and response analysis
used in the hybrid search and response generation systems.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum


class SearchResultType(str, Enum):
    """Type of search result."""
    DOCUMENT = "document"
    CHUNK = "chunk"
    HYBRID = "hybrid"


class ConfidenceLevel(str, Enum):
    """Confidence levels for responses."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SearchResult(BaseModel):
    """Comprehensive search result with scoring details."""
    
    # Basic information
    id: str = Field(..., description="Unique identifier of the result")
    content: str = Field(..., description="Content of the search result")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")
    
    # Scoring information
    vector_score: float = Field(..., description="Vector similarity score")
    keyword_score: float = Field(0.0, description="Keyword matching score")
    combined_score: float = Field(..., description="Final combined score")
    
    # Additional scoring details
    explanation: str = Field("", description="Explanation of scoring calculation")
    rank: Optional[int] = Field(None, description="Rank in search results")
    
    # Result type and source
    result_type: SearchResultType = Field(SearchResultType.DOCUMENT, description="Type of search result")
    parent_document_id: Optional[str] = Field(None, description="Parent document ID for chunks")
    chunk_index: Optional[int] = Field(None, description="Chunk index within parent document")
    
    # Processing metadata
    retrieved_at: datetime = Field(default_factory=datetime.now, description="Retrieval timestamp")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    
    @field_validator("vector_score", "keyword_score", "combined_score")
    @classmethod
    def validate_scores(cls, v):
        """Ensure scores are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Scores must be between 0 and 1")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Ensure content is not empty."""
        if not v or not v.strip():
            raise ValueError("Search result content cannot be empty")
        return v.strip()
    
    @property
    def is_chunk(self) -> bool:
        """Check if this is a chunk result."""
        return self.result_type == SearchResultType.CHUNK
    
    @property
    def is_document(self) -> bool:
        """Check if this is a document result."""
        return self.result_type == SearchResultType.DOCUMENT
    
    def get_source_info(self) -> Dict[str, Any]:
        """Get source information for citation."""
        source_info = {
            "id": self.id,
            "type": self.result_type,
            "score": self.combined_score
        }
        
        if self.parent_document_id:
            source_info["parent_document_id"] = self.parent_document_id
        
        if self.chunk_index is not None:
            source_info["chunk_index"] = self.chunk_index
        
        # Add relevant metadata
        if "title" in self.metadata:
            source_info["title"] = self.metadata["title"]
        if "author" in self.metadata:
            source_info["author"] = self.metadata["author"]
        if "source" in self.metadata:
            source_info["source"] = self.metadata["source"]
        
        return source_info
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ResponseAnalysis(BaseModel):
    """Structured analysis of response generation."""
    
    # Core response
    answer: str = Field(..., description="The synthesized answer")
    
    # Quality metrics
    confidence_score: float = Field(..., description="Confidence in answer accuracy (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Categorical confidence level")
    source_coverage: float = Field(..., description="Percentage of sources used in response")
    
    # Reasoning and sources
    reasoning_steps: List[str] = Field(default_factory=list, description="Step-by-step reasoning process")
    sources_used: List[str] = Field(default_factory=list, description="IDs of sources referenced")
    source_details: List[Dict[str, Any]] = Field(default_factory=list, description="Detailed source information")
    
    # Quality indicators
    limitations: Optional[str] = Field(None, description="Limitations or gaps in available information")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made in the response")
    alternative_interpretations: List[str] = Field(default_factory=list, description="Alternative ways to interpret the query")
    
    # Processing metadata
    processing_time: Optional[float] = Field(None, description="Response generation time in seconds")
    token_count: Optional[int] = Field(None, description="Number of tokens in response")
    model_used: Optional[str] = Field(None, description="LLM model used for generation")
    
    # Search context
    query: Optional[str] = Field(None, description="Original query")
    search_results_count: Optional[int] = Field(None, description="Number of search results used")
    
    @field_validator("confidence_score", "source_coverage")
    @classmethod
    def validate_percentages(cls, v):
        """Ensure percentage values are between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v):
        """Ensure answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()

    @model_validator(mode='after')
    def set_confidence_level(self):
        """Automatically set confidence level based on confidence score."""
        if isinstance(self.confidence_level, str):
            return self

        if self.confidence_score >= 0.9:
            self.confidence_level = ConfidenceLevel.VERY_HIGH
        elif self.confidence_score >= 0.75:
            self.confidence_level = ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.5:
            self.confidence_level = ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 0.25:
            self.confidence_level = ConfidenceLevel.LOW
        else:
            self.confidence_level = ConfidenceLevel.VERY_LOW

        return self
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if response has high confidence."""
        return self.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
    
    @property
    def needs_review(self) -> bool:
        """Check if response needs human review."""
        return (
            self.confidence_level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW] or
            self.source_coverage < 0.3 or
            len(self.sources_used) < 2
        )
    
    def get_citation_text(self) -> str:
        """Generate citation text for the response."""
        if not self.sources_used:
            return "No sources available."
        
        citations = []
        for i, source_id in enumerate(self.sources_used, 1):
            # Find source details
            source_detail = next(
                (s for s in self.source_details if s.get("id") == source_id),
                {"id": source_id}
            )
            
            title = source_detail.get("title", f"Source {i}")
            citations.append(f"[{i}] {title}")
        
        return "Sources: " + "; ".join(citations)
    
    def to_summary(self) -> Dict[str, Any]:
        """Create a summary of the response analysis."""
        return {
            "answer": self.answer,
            "confidence": {
                "score": self.confidence_score,
                "level": self.confidence_level
            },
            "sources_count": len(self.sources_used),
            "source_coverage": self.source_coverage,
            "needs_review": self.needs_review,
            "limitations": self.limitations
        }
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
