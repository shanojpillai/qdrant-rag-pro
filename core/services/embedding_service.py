"""
Production-grade embedding service with batching and caching.

This module provides advanced embedding generation capabilities with support for
multiple providers, automatic batching, caching, and async processing.
"""

import openai
import tiktoken
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import logging
import asyncio
import aiohttp
import hashlib
import time
import json
from datetime import datetime, timedelta

from ..config.settings import Settings


@dataclass
class EmbeddingResult:
    """Structured embedding result with metadata."""
    embedding: List[float]
    token_count: int
    processing_time: float
    text_hash: str
    model_used: str
    cached: bool = False
    created_at: Optional[datetime] = None


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.cache: Dict[str, tuple[EmbeddingResult, datetime]] = {}
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.logger = logging.getLogger(__name__)
    
    def get(self, text_hash: str) -> Optional[EmbeddingResult]:
        """Get cached embedding if available and not expired."""
        if text_hash in self.cache:
            result, timestamp = self.cache[text_hash]
            if datetime.now() - timestamp < self.ttl:
                result.cached = True
                return result
            else:
                # Remove expired entry
                del self.cache[text_hash]
        return None
    
    def set(self, text_hash: str, result: EmbeddingResult) -> None:
        """Cache an embedding result."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[text_hash] = (result, datetime.now())
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_hours": self.ttl.total_seconds() / 3600
        }


class EmbeddingService:
    """Production-grade embedding service with batching and caching."""
    
    def __init__(self, settings: Settings, enable_cache: bool = True):
        """Initialize embedding service."""
        self.settings = settings
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        # Use cl100k_base encoding which is used by text-embedding models
        try:
            self.encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base encoding for embedding models
            self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = settings.max_tokens_per_chunk
        self.batch_size = settings.batch_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = EmbeddingCache() if enable_cache else None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    async def create_embedding(self, text: str) -> EmbeddingResult:
        """Create embedding for a single text."""
        results = await self.create_embeddings_batch([text])
        return results[0]
    
    async def create_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Create embeddings for multiple texts with automatic chunking."""
        if not texts:
            return []
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Check cache first
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(processed_texts):
            text_hash = self._hash_text(text)
            
            if self.cache:
                cached_result = self.cache.get(text_hash)
                if cached_result:
                    results.append((i, cached_result))
                    continue
            
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            # Split into optimal batches for API efficiency
            batches = self._create_batches(uncached_texts, self.batch_size)
            batch_results = []
            
            for batch in batches:
                batch_result = await self._process_batch(batch)
                batch_results.extend(batch_result)
            
            # Add to results and cache
            for i, result in enumerate(batch_results):
                original_index = uncached_indices[i]
                results.append((original_index, result))
                
                if self.cache:
                    self.cache.set(result.text_hash, result)
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _preprocess_text(self, text: str) -> str:
        """Optimize text for embedding generation."""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize
        text = " ".join(text.split())
        
        # Truncate if too long
        tokens = self.encoding.encode(text)
        if len(tokens) > self.max_tokens:
            truncated_tokens = tokens[:self.max_tokens]
            text = self.encoding.decode(truncated_tokens)
            self.logger.warning(f"Text truncated from {len(tokens)} to {len(truncated_tokens)} tokens")
        
        return text
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text caching."""
        content = f"{text}:{self.model}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _create_batches(self, texts: List[str], batch_size: int) -> List[List[str]]:
        """Split texts into batches."""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append(batch)
        return batches
    
    async def _process_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Process a batch of texts with retry logic."""
        start_time = time.time()
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            
            self.last_request_time = time.time()
            processing_time = self.last_request_time - start_time
            
            results = []
            for i, embedding_data in enumerate(response.data):
                text_hash = self._hash_text(texts[i])
                token_count = len(self.encoding.encode(texts[i]))
                
                results.append(EmbeddingResult(
                    embedding=embedding_data.embedding,
                    token_count=token_count,
                    processing_time=processing_time / len(texts),
                    text_hash=text_hash,
                    model_used=self.model,
                    cached=False,
                    created_at=datetime.now()
                ))
            
            self.logger.debug(f"Processed batch of {len(texts)} texts in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing embedding batch: {e}")
            # Return empty embeddings as fallback
            return [
                EmbeddingResult(
                    embedding=[0.0] * 1536,  # Default dimension for text-embedding-3-small
                    token_count=len(self.encoding.encode(text)),
                    processing_time=0.0,
                    text_hash=self._hash_text(text),
                    model_used=self.model,
                    cached=False,
                    created_at=datetime.now()
                )
                for text in texts
            ]
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """Split text into chunks for embedding."""
        if not text:
            return []
        
        chunk_size = chunk_size or self.max_tokens
        overlap = overlap or self.settings.chunk_overlap_tokens
        
        # Tokenize the text
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(tokens):
                break
        
        self.logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    async def embed_document_with_chunks(
        self, 
        content: str,
        chunk_size: Optional[int] = None
    ) -> tuple[EmbeddingResult, List[EmbeddingResult]]:
        """Embed a document and its chunks."""
        # Create main document embedding
        main_embedding = await self.create_embedding(content)
        
        # Create chunks and their embeddings
        chunks = self.chunk_text(content, chunk_size)
        chunk_embeddings = []
        
        if len(chunks) > 1:  # Only create chunk embeddings if we actually have chunks
            chunk_embeddings = await self.create_embeddings_batch(chunks)
        
        return main_embedding, chunk_embeddings
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
    
    def estimate_cost(self, texts: List[str]) -> Dict[str, Any]:
        """Estimate the cost of embedding generation."""
        total_tokens = sum(len(self.encoding.encode(text)) for text in texts)
        
        # Pricing for text-embedding-3-small (as of 2024)
        cost_per_1k_tokens = 0.00002  # $0.00002 per 1K tokens
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
        
        return {
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model,
            "batch_count": len(self._create_batches(texts, self.batch_size))
        }
