"""
High-performance Qdrant vector database manager with advanced search capabilities.

This module provides a comprehensive wrapper around the Qdrant client with
optimized configurations for production workloads.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    Range, MatchValue, OptimizersConfig, HnswConfig, CollectionInfo
)
from typing import List, Dict, Any, Optional, Union
import logging
import uuid
from dataclasses import dataclass
import time

from ..config.settings import Settings


@dataclass
class SearchPoint:
    """Represents a search result point from Qdrant."""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class QdrantManager:
    """High-performance Qdrant vector database manager with advanced search capabilities."""
    
    def __init__(self, settings: Settings):
        """Initialize Qdrant manager with settings."""
        self.settings = settings
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = 1536  # OpenAI text-embedding-3-small dimensions
        self.logger = logging.getLogger(__name__)
        
    def initialize_collection(self) -> None:
        """Create collection with optimized vector configuration."""
        try:
            # Check if collection already exists
            collections = self.client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if self.collection_name in existing_collections:
                self.logger.info(f"Collection '{self.collection_name}' already exists")
                return
            
            # Create collection with optimized settings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    on_disk=True  # Enable disk storage for large datasets
                ),
                optimizers_config=OptimizersConfig(
                    default_segment_number=2,
                    max_segment_size=20000,
                    memmap_threshold=20000,
                    indexing_threshold=20000,
                    flush_interval_sec=5,
                    max_optimization_threads=2
                ),
                hnsw_config=HnswConfig(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                    max_indexing_threads=2,
                    on_disk=True
                )
            )
            self.logger.info(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get information about the collection."""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return None
    
    def upsert_points(
        self, 
        points: List[PointStruct], 
        batch_size: int = 100
    ) -> bool:
        """Upsert points in batches for optimal performance."""
        try:
            total_points = len(points)
            self.logger.info(f"Upserting {total_points} points in batches of {batch_size}")
            
            for i in range(0, total_points, batch_size):
                batch = points[i:i + batch_size]
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
                
                self.logger.debug(f"Batch {i//batch_size + 1}: {operation_info}")
            
            self.logger.info(f"Successfully upserted {total_points} points")
            return True
            
        except Exception as e:
            self.logger.error(f"Error upserting points: {e}")
            return False
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Filter] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: Optional[float] = None
    ) -> List[SearchPoint]:
        """Perform vector search with optional filtering."""
        try:
            start_time = time.time()
            
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=with_payload,
                with_vectors=with_vectors,
                score_threshold=score_threshold
            )
            
            search_time = time.time() - start_time
            self.logger.debug(f"Search completed in {search_time:.3f}s, found {len(search_result)} results")
            
            # Convert to SearchPoint objects
            points = []
            for result in search_result:
                points.append(SearchPoint(
                    id=str(result.id),
                    score=result.score,
                    payload=result.payload or {},
                    vector=result.vector if with_vectors else None
                ))
            
            return points
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def delete_points(self, point_ids: List[str]) -> bool:
        """Delete points by their IDs."""
        try:
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
                wait=True
            )
            
            self.logger.info(f"Deleted {len(point_ids)} points: {operation_info}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting points: {e}")
            return False
    
    def count_points(self, count_filter: Optional[Filter] = None) -> int:
        """Count points in the collection with optional filtering."""
        try:
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=count_filter,
                exact=True
            )
            return count_result.count
            
        except Exception as e:
            self.logger.error(f"Error counting points: {e}")
            return 0
    
    def scroll_points(
        self,
        limit: int = 100,
        offset: Optional[str] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> tuple[List[SearchPoint], Optional[str]]:
        """Scroll through points in the collection."""
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=with_payload,
                with_vectors=with_vectors
            )
            
            points = []
            for point in scroll_result[0]:
                points.append(SearchPoint(
                    id=str(point.id),
                    score=1.0,  # Scroll doesn't provide scores
                    payload=point.payload or {},
                    vector=point.vector if with_vectors else None
                ))
            
            next_offset = scroll_result[1]
            return points, next_offset
            
        except Exception as e:
            self.logger.error(f"Error scrolling points: {e}")
            return [], None
    
    def create_payload_index(self, field_name: str, field_type: str = "keyword") -> bool:
        """Create an index on a payload field for faster filtering."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_type
            )
            
            self.logger.info(f"Created payload index for field '{field_name}' with type '{field_type}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating payload index: {e}")
            return False
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.logger.info(f"Collection '{self.collection_name}' deleted successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return False
    
    def health_check(self) -> bool:
        """Check if Qdrant is healthy and accessible."""
        try:
            collections = self.client.get_collections()
            self.logger.debug(f"Health check passed, found {len(collections.collections)} collections")
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
