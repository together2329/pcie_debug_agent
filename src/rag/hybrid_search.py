"""
Hybrid Search implementation combining BM25 keyword search with FAISS semantic search
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class HybridSearchResult:
    """Result from hybrid search"""
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank: int


class HybridSearchEngine:
    """
    Combines BM25 keyword search with FAISS semantic search for improved retrieval
    """
    
    def __init__(self, 
                 vector_store,
                 index_path: Optional[str] = None,
                 alpha: float = 0.7):
        """
        Initialize hybrid search engine
        
        Args:
            vector_store: FAISS vector store instance
            index_path: Path to save/load BM25 index
            alpha: Weight for semantic search (1-alpha for keyword search)
        """
        self.vector_store = vector_store
        self.index_path = Path(index_path) if index_path else None
        self.alpha = alpha
        
        # BM25 components
        self.bm25 = None
        self.tokenized_docs = []
        self.doc_mapping = []  # Maps BM25 index to vector store index
        
        # Initialize or load BM25 index
        if self.index_path and self.index_path.exists():
            self.load_bm25_index()
        else:
            self.build_bm25_index()
    
    def build_bm25_index(self):
        """Build BM25 index from vector store documents"""
        logger.info("Building BM25 index...")
        
        # Get all documents from vector store
        documents = self.vector_store.documents
        metadata = self.vector_store.metadata
        
        # Tokenize documents for BM25
        self.tokenized_docs = []
        self.doc_mapping = []
        
        for idx, (doc, meta) in enumerate(zip(documents, metadata)):
            # Simple tokenization - can be improved with better NLP
            tokens = self._tokenize(doc)
            self.tokenized_docs.append(tokens)
            self.doc_mapping.append(idx)
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Save index if path provided
        if self.index_path:
            self.save_bm25_index()
        
        logger.info(f"BM25 index built with {len(self.tokenized_docs)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25
        
        Simple tokenization - can be enhanced with:
        - Stemming/lemmatization
        - Stop word removal
        - N-grams
        - Domain-specific tokenization
        """
        # Convert to lowercase and split
        tokens = text.lower().split()
        
        # Remove punctuation from tokens
        import string
        tokens = [token.strip(string.punctuation) for token in tokens]
        
        # Filter empty tokens
        tokens = [token for token in tokens if token]
        
        return tokens
    
    def save_bm25_index(self):
        """Save BM25 index to disk"""
        if not self.index_path:
            return
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'tokenized_docs': self.tokenized_docs,
            'doc_mapping': self.doc_mapping,
            'alpha': self.alpha
        }
        
        with open(self.index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"BM25 index saved to {self.index_path}")
    
    def load_bm25_index(self):
        """Load BM25 index from disk"""
        try:
            with open(self.index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.tokenized_docs = index_data['tokenized_docs']
            self.doc_mapping = index_data['doc_mapping']
            self.alpha = index_data.get('alpha', 0.7)
            
            # Rebuild BM25 from tokenized docs
            self.bm25 = BM25Okapi(self.tokenized_docs)
            
            logger.info(f"BM25 index loaded from {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            logger.info("Building new BM25 index...")
            self.build_bm25_index()
    
    def update_document(self, doc_idx: int, new_content: str):
        """Update a document in BM25 index"""
        if doc_idx < len(self.tokenized_docs):
            # Update tokenized version
            self.tokenized_docs[doc_idx] = self._tokenize(new_content)
            
            # Rebuild BM25 index
            self.bm25 = BM25Okapi(self.tokenized_docs)
            
            # Save if path provided
            if self.index_path:
                self.save_bm25_index()
    
    def add_documents(self, contents: List[str], start_idx: int):
        """Add new documents to BM25 index"""
        for i, content in enumerate(contents):
            tokens = self._tokenize(content)
            self.tokenized_docs.append(tokens)
            self.doc_mapping.append(start_idx + i)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        
        # Save if path provided
        if self.index_path:
            self.save_bm25_index()
    
    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword search
        
        Returns:
            List of (doc_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Filter out zero scores and map to original indices
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                original_idx = self.doc_mapping[idx]
                results.append((original_idx, float(scores[idx])))
        
        return results
    
    def semantic_search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
        """
        Perform semantic search using vector store
        
        Returns:
            List of (doc_index, score) tuples
        """
        # Convert numpy array to list for vector store
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Use vector store search
        results = self.vector_store.search(query_list, k=k)
        
        # Convert to index-score pairs
        indexed_results = []
        for i, (content, metadata, score) in enumerate(results):
            # Find index in documents
            try:
                idx = self.vector_store.documents.index(content)
                indexed_results.append((idx, float(score)))
            except ValueError:
                logger.warning(f"Document not found in index: {content[:50]}...")
        
        return indexed_results
    
    def hybrid_search(self, 
                     query: str, 
                     query_embedding: np.ndarray,
                     k: int = 10,
                     alpha: Optional[float] = None) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining keyword and semantic search
        
        Args:
            query: Text query
            query_embedding: Query embedding vector
            k: Number of results to return
            alpha: Weight for semantic search (overrides default)
            
        Returns:
            List of HybridSearchResult objects
        """
        if alpha is None:
            alpha = self.alpha
        
        # Get results from both methods
        keyword_results = self.keyword_search(query, k=k*2)
        semantic_results = self.semantic_search(query_embedding, k=k*2)
        
        # Create dictionaries for easy lookup
        keyword_dict = {idx: score for idx, score in keyword_results}
        semantic_dict = {idx: score for idx, score in semantic_results}
        
        # Get all unique document indices
        all_indices = set(keyword_dict.keys()) | set(semantic_dict.keys())
        
        # Calculate combined scores
        combined_results = []
        
        for idx in all_indices:
            # Get scores (0 if not found)
            keyword_score = keyword_dict.get(idx, 0.0)
            semantic_score = semantic_dict.get(idx, 0.0)
            
            # Normalize scores
            keyword_score_norm = self._normalize_score(keyword_score, keyword_results)
            semantic_score_norm = self._normalize_score(semantic_score, semantic_results)
            
            # Calculate combined score
            combined_score = alpha * semantic_score_norm + (1 - alpha) * keyword_score_norm
            
            # Get document content and metadata
            if idx < len(self.vector_store.documents):
                content = self.vector_store.documents[idx]
                metadata = self.vector_store.metadata[idx] if idx < len(self.vector_store.metadata) else {}
                
                result = HybridSearchResult(
                    content=content,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    combined_score=combined_score,
                    rank=0  # Will be set after sorting
                )
                combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Set ranks and return top k
        for i, result in enumerate(combined_results[:k]):
            result.rank = i + 1
        
        return combined_results[:k]
    
    def _normalize_score(self, score: float, all_scores: List[Tuple[int, float]]) -> float:
        """Normalize score to [0, 1] range"""
        if not all_scores:
            return 0.0
        
        scores_only = [s for _, s in all_scores]
        min_score = min(scores_only)
        max_score = max(scores_only)
        
        if max_score == min_score:
            return 0.5
        
        return (score - min_score) / (max_score - min_score)
    
    def reciprocal_rank_fusion(self,
                              query: str,
                              query_embedding: np.ndarray,
                              k: int = 10,
                              rrf_k: int = 60) -> List[HybridSearchResult]:
        """
        Alternative fusion using Reciprocal Rank Fusion (RRF)
        
        RRF is often more robust than linear combination
        """
        # Get results from both methods
        keyword_results = self.keyword_search(query, k=k*2)
        semantic_results = self.semantic_search(query_embedding, k=k*2)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Add keyword search contributions
        for rank, (idx, _) in enumerate(keyword_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        
        # Add semantic search contributions
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        
        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create result objects
        results = []
        for rank, (idx, rrf_score) in enumerate(sorted_indices[:k]):
            if idx < len(self.vector_store.documents):
                content = self.vector_store.documents[idx]
                metadata = self.vector_store.metadata[idx] if idx < len(self.vector_store.metadata) else {}
                
                # Get original scores
                keyword_score = next((s for i, s in keyword_results if i == idx), 0.0)
                semantic_score = next((s for i, s in semantic_results if i == idx), 0.0)
                
                result = HybridSearchResult(
                    content=content,
                    metadata=metadata,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    combined_score=rrf_score,
                    rank=rank + 1
                )
                results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            'total_documents': len(self.tokenized_docs),
            'bm25_vocabulary_size': len(set(token for doc in self.tokenized_docs for token in doc)),
            'average_doc_length': np.mean([len(doc) for doc in self.tokenized_docs]) if self.tokenized_docs else 0,
            'alpha_weight': self.alpha,
            'index_path': str(self.index_path) if self.index_path else None
        }