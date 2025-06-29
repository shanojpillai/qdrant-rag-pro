"""
QdrantRAG-Pro: Production-Ready RAG System with Qdrant Vector Database

A comprehensive Retrieval-Augmented Generation system that combines Qdrant's 
powerful vector database with advanced search capabilities, hybrid search, 
metadata filtering, and intelligent response synthesis.
"""

__version__ = "1.0.0"
__author__ = "QdrantRAG-Pro Team"
__email__ = "contact@qdrantrag-pro.com"

from .config.settings import Settings
from .database.qdrant_client import QdrantManager
from .database.document_store import DocumentStore
from .services.embedding_service import EmbeddingService
from .services.search_engine import HybridSearchEngine
from .services.response_generator import ResponseGenerator

__all__ = [
    "Settings",
    "QdrantManager", 
    "DocumentStore",
    "EmbeddingService",
    "HybridSearchEngine", 
    "ResponseGenerator"
]
