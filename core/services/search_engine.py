"""
Advanced search engine combining vector and keyword search with intelligent ranking.

This module implements hybrid search capabilities that combine semantic vector search
with traditional keyword matching for superior retrieval accuracy.
"""

from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import time
import re
import math
from collections import Counter
from dataclasses import dataclass

from ..models.search_result import SearchResult, SearchResultType
from ..database.qdrant_client import QdrantManager, SearchPoint
from .embedding_service import EmbeddingService
from ..config.settings import Settings


@dataclass
class SearchQuery:
    """Structured search query with analysis."""
    original_query: str
    processed_query: str
    query_terms: List[str]
    query_type: str  # "technical", "conversational", "mixed"
    suggested_weights: Dict[str, float]


class QueryAnalyzer:
    """Analyzes queries to determine optimal search strategy."""
    
    def __init__(self):
        self.technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\d+\.\d+\b',   # Version numbers
            r'\b[a-zA-Z]+\d+\b',  # Product codes
            r'\b[A-Z][a-z]+[A-Z][a-z]+\b',  # CamelCase
            r'\b\w+\.\w+\b',   # Dotted notation
        ]
        self.technical_keywords = {
            'api', 'sdk', 'framework', 'library', 'function', 'method', 'class',
            'variable', 'parameter', 'configuration', 'installation', 'setup',
            'error', 'exception', 'debug', 'log', 'database', 'query', 'schema'
        }
    
    def analyze_query(self, query: str) -> SearchQuery:
        """Analyze query and suggest optimal search strategy."""
        processed_query = self._preprocess_query(query)
        query_terms = processed_query.lower().split()
        
        # Determine query type
        technical_score = self._calculate_technical_score(query, query_terms)
        
        if technical_score > 0.6:
            query_type = "technical"
            suggested_weights = {"vector": 0.4, "keyword": 0.6}
        elif technical_score < 0.2:
            query_type = "conversational"
            suggested_weights = {"vector": 0.8, "keyword": 0.2}
        else:
            query_type = "mixed"
            suggested_weights = {"vector": 0.6, "keyword": 0.4}
        
        return SearchQuery(
            original_query=query,
            processed_query=processed_query,
            query_terms=query_terms,
            query_type=query_type,
            suggested_weights=suggested_weights
        )
    
    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize query text."""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Preserve important punctuation but remove noise
        query = re.sub(r'[^\w\s\.\-_]', ' ', query)
        
        return query
    
    def _calculate_technical_score(self, query: str, terms: List[str]) -> float:
        """Calculate how technical a query is."""
        score = 0.0
        total_factors = 0
        
        # Check for technical patterns
        for pattern in self.technical_patterns:
            matches = len(re.findall(pattern, query))
            score += min(matches * 0.2, 0.4)
            total_factors += 1
        
        # Check for technical keywords
        technical_term_count = sum(1 for term in terms if term in self.technical_keywords)
        if terms:
            score += (technical_term_count / len(terms)) * 0.6
            total_factors += 1
        
        return score / max(total_factors, 1)


class HybridSearchEngine:
    """Advanced search engine combining vector and keyword search with intelligent ranking."""
    
    def __init__(
        self, 
        qdrant_manager: QdrantManager, 
        embedding_service: EmbeddingService,
        settings: Settings
    ):
        """Initialize hybrid search engine."""
        self.qdrant = qdrant_manager
        self.embedder = embedding_service
        self.settings = settings
        self.query_analyzer = QueryAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Search configuration
        self.default_vector_weight = settings.default_vector_weight
        self.default_keyword_weight = settings.default_keyword_weight
        self.min_score = settings.min_search_score
    
    async def search(
        self, 
        query: str, 
        limit: int = 10,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_score: Optional[float] = None,
        auto_adjust_weights: bool = True
    ) -> List[SearchResult]:
        """Perform hybrid search with weighted score combination."""
        start_time = time.time()
        
        # Analyze query
        query_analysis = self.query_analyzer.analyze_query(query)
        
        # Determine weights
        if auto_adjust_weights and vector_weight is None and keyword_weight is None:
            vector_weight = query_analysis.suggested_weights["vector"]
            keyword_weight = query_analysis.suggested_weights["keyword"]
        else:
            vector_weight = vector_weight or self.default_vector_weight
            keyword_weight = keyword_weight or self.default_keyword_weight
        
        min_score = min_score or self.min_score
        
        self.logger.info(
            f"Searching with query type: {query_analysis.query_type}, "
            f"weights: vector={vector_weight:.2f}, keyword={keyword_weight:.2f}"
        )
        
        # Generate query embedding
        query_embedding_result = await self.embedder.create_embedding(query_analysis.processed_query)
        query_vector = query_embedding_result.embedding
        
        # Prepare Qdrant filter
        qdrant_filter = self._build_filter(filters) if filters else None
        
        # Perform vector search with expanded limit for reranking
        vector_results = self.qdrant.search(
            query_vector=query_vector,
            limit=limit * 3,  # Get more results for reranking
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=False,
            score_threshold=0.1  # Low threshold for initial retrieval
        )
        
        if not vector_results:
            self.logger.warning("No vector search results found")
            return []
        
        # Calculate keyword scores
        keyword_scores = self._calculate_keyword_scores(query_analysis, vector_results)
        
        # Combine and rank results
        hybrid_results = self._combine_scores(
            vector_results, 
            keyword_scores, 
            vector_weight, 
            keyword_weight,
            query_analysis
        )
        
        # Filter by minimum score and return top results
        filtered_results = [r for r in hybrid_results if r.combined_score >= min_score]
        final_results = sorted(filtered_results, key=lambda x: x.combined_score, reverse=True)[:limit]
        
        search_time = time.time() - start_time
        self.logger.info(
            f"Search completed in {search_time:.3f}s: "
            f"{len(vector_results)} initial → {len(filtered_results)} filtered → {len(final_results)} final"
        )
        
        # Add rank information
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        return final_results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build complex Qdrant filters from simple dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, dict):
                # Range filter: {"price": {"gte": 100, "lte": 500}}
                if "gte" in value or "lte" in value:
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
                        )
                    )
                # List filter: {"category": {"in": ["tech", "business"]}}
                elif "in" in value:
                    # Create OR conditions for list items
                    list_conditions = []
                    for item in value["in"]:
                        list_conditions.append(
                            FieldCondition(
                                key=f"metadata.{key}",
                                match=MatchValue(value=item)
                            )
                        )
                    # Add as OR conditions (should)
                    if list_conditions:
                        if len(list_conditions) == 1:
                            conditions.append(list_conditions[0])
                        else:
                            conditions.append(Filter(should=list_conditions))
            else:
                # Exact match filter
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    def _calculate_keyword_scores(
        self, 
        query_analysis: SearchQuery, 
        vector_results: List[SearchPoint]
    ) -> Dict[str, float]:
        """Calculate keyword relevance scores using TF-IDF principles."""
        query_terms = set(term.lower() for term in query_analysis.query_terms)
        scores = {}
        
        # Calculate document frequencies for IDF
        doc_frequencies = Counter()
        all_documents = []
        
        for result in vector_results:
            content = result.payload.get("content", "").lower()
            content_terms = set(content.split())
            all_documents.append(content_terms)
            
            for term in query_terms:
                if term in content_terms:
                    doc_frequencies[term] += 1
        
        total_docs = len(all_documents)
        
        # Calculate scores for each document
        for i, result in enumerate(vector_results):
            content = result.payload.get("content", "").lower()
            content_terms = content.split()
            content_term_counts = Counter(content_terms)
            
            tf_idf_score = 0.0
            
            for term in query_terms:
                if term in content_term_counts:
                    # Term frequency
                    tf = content_term_counts[term] / len(content_terms)
                    
                    # Inverse document frequency
                    df = doc_frequencies[term]
                    idf = math.log(total_docs / (df + 1)) if df > 0 else 0
                    
                    tf_idf_score += tf * idf
            
            # Normalize by query length
            normalized_score = tf_idf_score / len(query_terms) if query_terms else 0
            
            # Apply additional scoring factors
            exact_matches = sum(1 for term in query_terms if term in content.lower())
            exact_match_bonus = exact_matches / len(query_terms) if query_terms else 0
            
            # Combine TF-IDF with exact match bonus
            final_score = (normalized_score * 0.7) + (exact_match_bonus * 0.3)
            
            scores[result.id] = min(final_score, 1.0)  # Cap at 1.0
        
        return scores
    
    def _combine_scores(
        self, 
        vector_results: List[SearchPoint], 
        keyword_scores: Dict[str, float],
        vector_weight: float,
        keyword_weight: float,
        query_analysis: SearchQuery
    ) -> List[SearchResult]:
        """Combine vector and keyword scores with weighted ranking."""
        combined_results = []
        
        for result in vector_results:
            vector_score = result.score
            keyword_score = keyword_scores.get(result.id, 0.0)
            
            # Weighted combination
            combined_score = (vector_score * vector_weight) + (keyword_score * keyword_weight)
            
            # Apply query-specific boosts
            if query_analysis.query_type == "technical":
                # Boost results with exact technical term matches
                technical_boost = self._calculate_technical_boost(
                    query_analysis.original_query, 
                    result.payload.get("content", "")
                )
                combined_score *= (1 + technical_boost * 0.1)
            
            explanation = (
                f"Vector: {vector_score:.3f} (w={vector_weight:.2f}) + "
                f"Keyword: {keyword_score:.3f} (w={keyword_weight:.2f}) = "
                f"{combined_score:.3f}"
            )
            
            # Determine result type
            result_type = SearchResultType.CHUNK if result.payload.get("document_type") == "chunk" else SearchResultType.DOCUMENT
            
            combined_results.append(SearchResult(
                id=result.id,
                content=result.payload.get("content", ""),
                metadata=result.payload.get("metadata", {}),
                vector_score=vector_score,
                keyword_score=keyword_score,
                combined_score=min(combined_score, 1.0),  # Cap at 1.0
                explanation=explanation,
                result_type=result_type,
                parent_document_id=result.payload.get("parent_document_id"),
                chunk_index=result.payload.get("chunk_index")
            ))
        
        return combined_results
    
    def _calculate_technical_boost(self, query: str, content: str) -> float:
        """Calculate boost for technical term matches."""
        # Look for exact matches of technical patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\d+\.\d+\b',   # Version numbers
            r'\b[a-zA-Z]+\d+\b',  # Product codes
        ]
        
        boost = 0.0
        for pattern in technical_patterns:
            query_matches = set(re.findall(pattern, query))
            content_matches = set(re.findall(pattern, content))
            
            common_matches = query_matches.intersection(content_matches)
            if common_matches:
                boost += len(common_matches) * 0.1
        
        return min(boost, 0.5)  # Cap boost at 50%
