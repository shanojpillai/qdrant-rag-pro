"""
Tests for the embedding service functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from core.services.embedding_service import EmbeddingService, EmbeddingResult, EmbeddingCache
from core.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.openai_api_key = "test-api-key"
    settings.embedding_model = "text-embedding-3-small"
    settings.max_tokens_per_chunk = 8192
    settings.chunk_overlap_tokens = 200
    settings.batch_size = 100
    return settings


@pytest.fixture
def embedding_service(mock_settings):
    """Create embedding service for testing."""
    with patch('core.services.embedding_service.openai.OpenAI'):
        service = EmbeddingService(mock_settings, enable_cache=False)
        return service


@pytest.fixture
def embedding_service_with_cache(mock_settings):
    """Create embedding service with cache for testing."""
    with patch('core.services.embedding_service.openai.OpenAI'):
        service = EmbeddingService(mock_settings, enable_cache=True)
        return service


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = EmbeddingCache(max_size=100, ttl_hours=24)
        assert cache.max_size == 100
        assert len(cache.cache) == 0
    
    def test_cache_set_and_get(self):
        """Test setting and getting cache entries."""
        cache = EmbeddingCache()
        
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            token_count=10,
            processing_time=0.1,
            text_hash="test_hash",
            model_used="test-model"
        )
        
        # Set cache entry
        cache.set("test_hash", result)
        
        # Get cache entry
        cached_result = cache.get("test_hash")
        assert cached_result is not None
        assert cached_result.embedding == [0.1, 0.2, 0.3]
        assert cached_result.cached is True
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = EmbeddingCache()
        result = cache.get("nonexistent_hash")
        assert result is None
    
    def test_cache_max_size(self):
        """Test cache size limit."""
        cache = EmbeddingCache(max_size=2)
        
        result1 = EmbeddingResult([0.1], 10, 0.1, "hash1", "model")
        result2 = EmbeddingResult([0.2], 10, 0.1, "hash2", "model")
        result3 = EmbeddingResult([0.3], 10, 0.1, "hash3", "model")
        
        cache.set("hash1", result1)
        cache.set("hash2", result2)
        cache.set("hash3", result3)  # Should evict hash1
        
        assert len(cache.cache) == 2
        assert cache.get("hash1") is None
        assert cache.get("hash2") is not None
        assert cache.get("hash3") is not None


class TestEmbeddingService:
    """Test embedding service functionality."""
    
    def test_initialization(self, embedding_service):
        """Test service initialization."""
        assert embedding_service.model == "text-embedding-3-small"
        assert embedding_service.max_tokens == 8192
        assert embedding_service.batch_size == 100
    
    def test_preprocess_text(self, embedding_service):
        """Test text preprocessing."""
        # Test whitespace normalization
        text = "  This   has    extra   spaces  "
        processed = embedding_service._preprocess_text(text)
        assert processed == "This has extra spaces"
        
        # Test empty text
        assert embedding_service._preprocess_text("") == ""
        assert embedding_service._preprocess_text("   ") == ""
    
    def test_hash_text(self, embedding_service):
        """Test text hashing."""
        text1 = "Hello world"
        text2 = "Hello world"
        text3 = "Different text"
        
        hash1 = embedding_service._hash_text(text1)
        hash2 = embedding_service._hash_text(text2)
        hash3 = embedding_service._hash_text(text3)
        
        assert hash1 == hash2  # Same text should produce same hash
        assert hash1 != hash3  # Different text should produce different hash
        assert len(hash1) == 64  # SHA256 produces 64-character hex string
    
    def test_create_batches(self, embedding_service):
        """Test batch creation."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        batches = embedding_service._create_batches(texts, batch_size=2)
        
        assert len(batches) == 3
        assert batches[0] == ["text1", "text2"]
        assert batches[1] == ["text3", "text4"]
        assert batches[2] == ["text5"]
    
    def test_chunk_text(self, embedding_service):
        """Test text chunking."""
        # Mock the encoding
        with patch.object(embedding_service.encoding, 'encode') as mock_encode, \
             patch.object(embedding_service.encoding, 'decode') as mock_decode:
            
            # Simulate a text that needs chunking
            mock_encode.return_value = list(range(1000))  # 1000 tokens
            mock_decode.side_effect = lambda tokens: f"chunk_{len(tokens)}"
            
            chunks = embedding_service.chunk_text("long text", chunk_size=500, overlap=100)
            
            assert len(chunks) > 1
            assert all(chunk.startswith("chunk_") for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_create_embeddings_batch_mock(self, embedding_service):
        """Test batch embedding creation with mocked API."""
        texts = ["text1", "text2"]
        
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        with patch.object(embedding_service.client.embeddings, 'create', return_value=mock_response):
            results = await embedding_service.create_embeddings_batch(texts)
            
            assert len(results) == 2
            assert results[0].embedding == [0.1, 0.2, 0.3]
            assert results[1].embedding == [0.4, 0.5, 0.6]
            assert all(not result.cached for result in results)
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, embedding_service_with_cache):
        """Test caching functionality."""
        texts = ["cached text"]
        
        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        with patch.object(embedding_service_with_cache.client.embeddings, 'create', return_value=mock_response) as mock_create:
            # First call should hit the API
            results1 = await embedding_service_with_cache.create_embeddings_batch(texts)
            assert mock_create.call_count == 1
            assert not results1[0].cached
            
            # Second call should use cache
            results2 = await embedding_service_with_cache.create_embeddings_batch(texts)
            assert mock_create.call_count == 1  # No additional API calls
            assert results2[0].cached
    
    def test_estimate_cost(self, embedding_service):
        """Test cost estimation."""
        texts = ["short text", "another short text"]
        
        with patch.object(embedding_service.encoding, 'encode') as mock_encode:
            mock_encode.side_effect = [
                list(range(100)),  # 100 tokens
                list(range(150))   # 150 tokens
            ]
            
            cost_estimate = embedding_service.estimate_cost(texts)
            
            assert cost_estimate["total_tokens"] == 250
            assert cost_estimate["model"] == "text-embedding-3-small"
            assert cost_estimate["estimated_cost_usd"] > 0
            assert cost_estimate["batch_count"] == 1
    
    @pytest.mark.asyncio
    async def test_embed_document_with_chunks(self, embedding_service):
        """Test document embedding with chunking."""
        content = "This is a long document that should be chunked into smaller pieces for processing."
        
        # Mock responses
        mock_main_response = Mock()
        mock_main_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        
        mock_chunk_response = Mock()
        mock_chunk_response.data = [
            Mock(embedding=[0.4, 0.5, 0.6]),
            Mock(embedding=[0.7, 0.8, 0.9])
        ]
        
        with patch.object(embedding_service.client.embeddings, 'create') as mock_create, \
             patch.object(embedding_service, 'chunk_text', return_value=["chunk1", "chunk2"]):
            
            mock_create.side_effect = [mock_main_response, mock_chunk_response]
            
            main_embedding, chunk_embeddings = await embedding_service.embed_document_with_chunks(content)
            
            assert main_embedding.embedding == [0.1, 0.2, 0.3]
            assert len(chunk_embeddings) == 2
            assert chunk_embeddings[0].embedding == [0.4, 0.5, 0.6]
            assert chunk_embeddings[1].embedding == [0.7, 0.8, 0.9]


if __name__ == "__main__":
    pytest.main([__file__])
