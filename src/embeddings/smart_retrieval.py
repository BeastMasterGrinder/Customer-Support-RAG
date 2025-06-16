import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from .embedding_manager import EmbeddingManager
from config.retrieval_config import (
    DOC_TYPE_PRIORITIES,
    QUERY_PATTERNS,
    NEGATION_PATTERNS,
    RECENCY_SETTINGS,
    RELEVANCE_WEIGHTS,
    KEYWORD_SETTINGS,
    RERANK_TOP_K,
    FINAL_RESULTS_K
)

@dataclass
class SearchResult:
    document: Document
    semantic_score: float
    keyword_score: float
    doc_priority: float
    recency_score: float
    final_score: float = 0.0

class SmartRetrieval:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        
    def _classify_query(self, query: str) -> List[str]:
        """Classify query into predefined categories."""
        categories = []
        for category, patterns in QUERY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    categories.append(category)
                    break
        return list(set(categories))
    
    def _has_negation(self, query: str) -> bool:
        """Check if query contains negation."""
        for pattern in NEGATION_PATTERNS:
            if re.search(fr'\b{pattern}\b', query, re.IGNORECASE):
                return True
        return False
    
    def _calculate_keyword_score(self, query: str, document: Document) -> float:
        """Calculate keyword matching score using n-grams."""
        query_terms = query.lower().split()
        doc_text = document.page_content.lower()
        
        # Generate n-grams from query
        ngrams = []
        for n in range(KEYWORD_SETTINGS['min_ngram'], KEYWORD_SETTINGS['max_ngram'] + 1):
            for i in range(len(query_terms) - n + 1):
                ngrams.append(' '.join(query_terms[i:i+n]))
        
        # Calculate matches
        matches = 0
        total_ngrams = len(ngrams)
        
        for ngram in ngrams:
            if ngram in doc_text:
                matches += 1
        
        return matches / total_ngrams if total_ngrams > 0 else 0.0
    
    def _calculate_doc_priority(self, document: Document) -> float:
        """Calculate document priority score based on type."""
        doc_type = document.metadata.get('source', '')
        if doc_type == 'support_ticket':
            status = document.metadata.get('status', '').lower()
            doc_type = 'resolved_ticket' if status == 'resolved' else 'pending_ticket'
            
        priority = DOC_TYPE_PRIORITIES.get(doc_type, 0)
        max_priority = max(DOC_TYPE_PRIORITIES.values())
        return priority / max_priority
    
    def _calculate_recency_score(self, document: Document) -> float:
        """Calculate recency score based on document age."""
        date_field = None
        if document.metadata.get('source') == 'support_ticket':
            date_field = document.metadata.get('resolved_date') or document.metadata.get('created_date')
        else:
            date_field = document.metadata.get('last_updated')
            
        if not date_field:
            return 0.0
            
        doc_date = datetime.fromisoformat(date_field)
        age = datetime.now() - doc_date
        
        if age <= RECENCY_SETTINGS['recent_threshold']:
            return 1.0
        elif age >= RECENCY_SETTINGS['max_age']:
            return 0.0
        else:
            # Linear interpolation between recent_threshold and max_age
            age_range = (RECENCY_SETTINGS['max_age'] - RECENCY_SETTINGS['recent_threshold']).days
            doc_age = (age - RECENCY_SETTINGS['recent_threshold']).days
            return 1.0 - (doc_age / age_range)
    
    def _calculate_final_score(self, result: SearchResult) -> float:
        """Calculate final score using weighted components."""
        return (
            result.semantic_score * RELEVANCE_WEIGHTS['semantic_score'] +
            result.keyword_score * RELEVANCE_WEIGHTS['keyword_score'] +
            result.doc_priority * RELEVANCE_WEIGHTS['doc_priority'] +
            result.recency_score * RELEVANCE_WEIGHTS['recency']
        )
    
    def _filter_by_version(self, documents: List[Document], version: str = None) -> List[Document]:
        """Filter documents by version if specified."""
        if not version:
            return documents
            
        filtered_docs = []
        for doc in documents:
            doc_version = doc.metadata.get('version') or doc.metadata.get('user_version')
            if doc_version and doc_version == version:
                filtered_docs.append(doc)
        return filtered_docs
    
    def _boost_similar_cases(self, query: str, documents: List[Document]) -> List[Document]:
        """Boost results from similar previous support cases."""
        query_categories = self._classify_query(query)
        
        # Skip if no categories identified
        if not query_categories:
            return documents
            
        for doc in documents:
            if doc.metadata.get('source') == 'support_ticket':
                doc_categories = doc.metadata.get('category', '').split(',')
                matching_categories = set(query_categories) & set(doc_categories)
                if matching_categories:
                    # Boost score by 20% for each matching category
                    boost = 1 + (0.2 * len(matching_categories))
                    doc.metadata['_score_boost'] = boost
                    
        return documents
    
    def search(
        self,
        query: str,
        version: str = None,
        k: int = FINAL_RESULTS_K
    ) -> List[Tuple[Document, float]]:
        """
        Perform smart retrieval combining semantic search, keyword matching,
        and advanced ranking features.
        """
        # Get initial results from semantic search
        initial_results = self.embedding_manager.similarity_search(
            query,
            k=RERANK_TOP_K
        )
        
        # Filter by version if specified
        initial_results = self._filter_by_version(initial_results, version)
        
        # Boost similar support cases
        initial_results = self._boost_similar_cases(query, initial_results)
        
        # Calculate scores for each result
        search_results = []
        has_negation = self._has_negation(query)
        
        for doc in initial_results:
            # Base semantic score (normalized to 0-1)
            semantic_score = 1.0 - (initial_results.index(doc) / len(initial_results))
            
            # Calculate other scores
            keyword_score = self._calculate_keyword_score(query, doc)
            doc_priority = self._calculate_doc_priority(doc)
            recency_score = self._calculate_recency_score(doc)
            
            # Handle negation by inverting semantic and keyword scores
            if has_negation:
                semantic_score = 1.0 - semantic_score
                keyword_score = 1.0 - keyword_score
            
            # Apply any boosts from similar case matching
            boost = doc.metadata.pop('_score_boost', 1.0)
            
            result = SearchResult(
                document=doc,
                semantic_score=semantic_score * boost,
                keyword_score=keyword_score * boost,
                doc_priority=doc_priority,
                recency_score=recency_score
            )
            
            # Calculate final score
            result.final_score = self._calculate_final_score(result)
            search_results.append(result)
        
        # Sort by final score and return top k
        search_results.sort(key=lambda x: x.final_score, reverse=True)
        return [(result.document, result.final_score) for result in search_results[:k]] 