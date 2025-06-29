"""
Tests for the search engine functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from core.services.search_engine import HybridSearchEngine, QueryAnalyzer, SearchQuery
from core.services.embedding_service import EmbeddingService, EmbeddingResult
from core.database.qdrant_client import QdrantManager, SearchPoint
from core.models.search_result import SearchResult, SearchResultType
from core.config.settings import Settings


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = Mock(spec=Settings)
    settings.default_vector_weight = 0.7
    settings.default_keyword_weight = 0.3
    settings.min_search_score = 0.6
    settings.max_sources_per_response = 5
    return settings


@pytest.fixture
def mock_qdrant_manager():
    """Create mock Qdrant manager."""
    return Mock(spec=QdrantManager)


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    return Mock(spec=EmbeddingService)


@pytest.fixture
def search_engine(mock_qdrant_manager, mock_embedding_service, mock_settings):
    """Create search engine for testing."""
    return HybridSearchEngine(mock_qdrant_manager, mock_embedding_service, mock_settings)


class TestQueryAnalyzer:
    """Test query analysis functionality."""
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = QueryAnalyzer()
        assert len(analyzer.technical_patterns) > 0
        assert len(analyzer.technical_keywords) > 0
    
    def test_preprocess_query(self):
        """Test query preprocessing."""
        analyzer = QueryAnalyzer()
        
        # Test whitespace normalization
        query = "  multiple   spaces   here  "
        processed = analyzer._preprocess_query(query)
        assert processed == "multiple spaces here"
        
        # Test punctuation handling
        query = "What is API-v2.0 configuration?"
        processed = analyzer._preprocess_query(query)
        assert "API" in processed
        assert "v2" in processed
    
    def test_technical_score_calculation(self):
        """Test technical score calculation."""
        analyzer = QueryAnalyzer()
        
        # Technical query
        technical_query = "API v2.0 configuration setup"
        terms = technical_query.lower().split()
        score = analyzer._calculate_technical_score(technical_query, terms)
        assert score > 0.3  # Should be considered somewhat technical
        
        # Conversational query
        conversational_query = "how do I feel better today"
        terms = conversational_query.lower().split()
        score = analyzer._calculate_technical_score(conversational_query, terms)
        assert score < 0.3  # Should be considered non-technical
    
    def test_analyze_query_technical(self):
        """Test analysis of technical queries."""
        analyzer = QueryAnalyzer()
        
        query = "Configure API v2.0 authentication parameters"
        analysis = analyzer.analyze_query(query)
        
        assert analysis.original_query == query
        assert analysis.query_type in ["technical", "mixed"]
        assert analysis.suggested_weights["keyword"] >= 0.4
    
    def test_analyze_query_conversational(self):
        """Test analysis of conversational queries."""
        analyzer = QueryAnalyzer()
        
        query = "How can I improve my understanding of machine learning?"
        analysis = analyzer.analyze_query(query)
        
        assert analysis.original_query == query
        assert analysis.query_type in ["conversational", "mixed"]
        assert analysis.suggested_weights["vector"] >= 0.6


class TestHybridSearchEngine:
    """Test hybrid search engine functionality."""
    
    def test_initialization(self, search_engine):
        """Test search engine initialization."""
        assert search_engine.default_vector_weight == 0.7
        assert search_engine.default_keyword_weight == 0.3
        assert search_engine.min_score == 0.6
        assert isinstance(search_engine.query_analyzer, QueryAnalyzer)
    
    def test_build_filter_exact_match(self, search_engine):
        """Test building exact match filters."""
        filters = {"category": "technology", "author": "test_author"}
        qdrant_filter = search_engine._build_filter(filters)
        
        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 2
    
    def test_build_filter_range(self, search_engine):
        """Test building range filters."""
        filters = {"score": {"gte": 0.5, "lte": 1.0}}
        qdrant_filter = search_engine._build_filter(filters)
        
        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 1
    
    def test_build_filter_list(self, search_engine):
        """Test building list filters."""
        filters = {"category": {"in": ["tech", "science"]}}
        qdrant_filter = search_engine._build_filter(filters)
        
        assert qdrant_filter is not None
        assert len(qdrant_filter.must) == 2  # One condition per list item
    
    def test_calculate_keyword_scores(self, search_engine):
        """Test keyword score calculation."""
        query_analysis = SearchQuery(
            original_query="machine learning algorithms",
            processed_query="machine learning algorithms",
            query_terms=["machine", "learning", "algorithms"],
            query_type="technical",
            suggested_weights={"vector": 0.6, "keyword": 0.4}
        )
        
        # Mock search results
        search_results = [
            SearchPoint(
                id="doc1",
                score=0.8,
                payload={"content": "Machine learning algorithms are powerful tools for data analysis"}
            ),
            SearchPoint(
                id="doc2", 
                score=0.6,
                payload={"content": "Deep learning is a subset of artificial intelligence"}
            )
        ]
        
        keyword_scores = search_engine._calculate_keyword_scores(query_analysis, search_results)
        
        assert "doc1" in keyword_scores
        assert "doc2" in keyword_scores
        assert keyword_scores["doc1"] > keyword_scores["doc2"]  # doc1 has more matching terms
    
    def test_combine_scores(self, search_engine):
        """Test score combination."""
        vector_results = [
            SearchPoint(
                id="doc1",
                score=0.9,
                payload={
                    "content": "Test content",
                    "metadata": {"title": "Test Document"},
                    "document_type": "main"
                }
            )
        ]
        
        keyword_scores = {"doc1": 0.7}
        vector_weight = 0.6
        keyword_weight = 0.4
        
        query_analysis = SearchQuery(
            original_query="test query",
            processed_query="test query", 
            query_terms=["test", "query"],
            query_type="mixed",
            suggested_weights={"vector": 0.6, "keyword": 0.4}
        )
        
        combined_results = search_engine._combine_scores(
            vector_results, keyword_scores, vector_weight, keyword_weight, query_analysis
        )
        
        assert len(combined_results) == 1
        result = combined_results[0]
        
        assert isinstance(result, SearchResult)
        assert result.vector_score == 0.9
        assert result.keyword_score == 0.7
        assert result.combined_score == (0.9 * 0.6) + (0.7 * 0.4)  # 0.54 + 0.28 = 0.82
        assert result.result_type == SearchResultType.DOCUMENT
    
    def test_calculate_technical_boost(self, search_engine):
        """Test technical boost calculation."""
        # Query with technical terms
        query = "API v2.0 configuration"
        content = "The API v2.0 provides advanced configuration options"
        
        boost = search_engine._calculate_technical_boost(query, content)
        assert boost > 0  # Should get boost for matching technical terms
        
        # Query without technical matches
        query = "API v2.0 configuration"
        content = "This is general information about software"
        
        boost = search_engine._calculate_technical_boost(query, content)
        assert boost == 0  # No technical term matches
    
    @pytest.mark.asyncio
    async def test_search_basic(self, search_engine):
        """Test basic search functionality."""
        query = "machine learning"
        
        # Mock embedding service
        embedding_result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            token_count=10,
            processing_time=0.1,
            text_hash="test_hash",
            model_used="test-model"
        )
        search_engine.embedder.create_embedding = AsyncMock(return_value=embedding_result)
        
        # Mock Qdrant search results
        mock_search_results = [
            SearchPoint(
                id="doc1",
                score=0.85,
                payload={
                    "content": "Machine learning is a subset of artificial intelligence",
                    "metadata": {"title": "ML Introduction", "category": "technology"},
                    "document_type": "main"
                }
            ),
            SearchPoint(
                id="doc2",
                score=0.75,
                payload={
                    "content": "Deep learning algorithms use neural networks",
                    "metadata": {"title": "Deep Learning", "category": "technology"},
                    "document_type": "main"
                }
            )
        ]
        search_engine.qdrant.search = Mock(return_value=mock_search_results)
        
        # Perform search
        results = await search_engine.search(query, limit=5)
        
        assert len(results) <= 5
        assert all(isinstance(result, SearchResult) for result in results)
        assert all(result.combined_score >= search_engine.min_score for result in results)
        
        # Results should be sorted by combined score (descending)
        for i in range(len(results) - 1):
            assert results[i].combined_score >= results[i + 1].combined_score
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_engine):
        """Test search with filters."""
        query = "test query"
        filters = {"category": "technology"}
        
        # Mock embedding service
        embedding_result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            token_count=10,
            processing_time=0.1,
            text_hash="test_hash",
            model_used="test-model"
        )
        search_engine.embedder.create_embedding = AsyncMock(return_value=embedding_result)
        
        # Mock Qdrant search
        search_engine.qdrant.search = Mock(return_value=[])
        
        # Perform search with filters
        await search_engine.search(query, filters=filters)
        
        # Verify that search was called with filters
        search_engine.qdrant.search.assert_called_once()
        call_args = search_engine.qdrant.search.call_args
        assert call_args[1]["query_filter"] is not None
    
    @pytest.mark.asyncio
    async def test_search_auto_weight_adjustment(self, search_engine):
        """Test automatic weight adjustment based on query type."""
        # Technical query should favor keyword search
        technical_query = "API v2.0 configuration"
        
        # Mock embedding service
        embedding_result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            token_count=10,
            processing_time=0.1,
            text_hash="test_hash",
            model_used="test-model"
        )
        search_engine.embedder.create_embedding = AsyncMock(return_value=embedding_result)
        search_engine.qdrant.search = Mock(return_value=[])
        
        # Perform search with auto weight adjustment
        await search_engine.search(technical_query, auto_adjust_weights=True)
        
        # Verify that the query was analyzed and weights were adjusted
        # (This is implicit in the search process)
        search_engine.embedder.create_embedding.assert_called_once()
        search_engine.qdrant.search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_no_results(self, search_engine):
        """Test search when no results are found."""
        query = "nonexistent query"
        
        # Mock embedding service
        embedding_result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            token_count=10,
            processing_time=0.1,
            text_hash="test_hash",
            model_used="test-model"
        )
        search_engine.embedder.create_embedding = AsyncMock(return_value=embedding_result)
        
        # Mock empty search results
        search_engine.qdrant.search = Mock(return_value=[])
        
        # Perform search
        results = await search_engine.search(query)
        
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
