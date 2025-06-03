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
import re

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
    Enhanced Hybrid Search combining BM25 keyword search with FAISS semantic search
    Features phrase matching boost and PCIe technical term recognition
    """
    
    def __init__(self, 
                 vector_store,
                 index_path: Optional[str] = None,
                 alpha: float = 0.7,
                 phrase_boost: float = 2.0,
                 technical_term_boost: float = 1.5):
        """
        Initialize enhanced hybrid search engine
        
        Args:
            vector_store: FAISS vector store instance
            index_path: Path to save/load BM25 index
            alpha: Weight for semantic search (1-alpha for keyword search)
            phrase_boost: Multiplier for exact phrase matches
            technical_term_boost: Multiplier for technical term matches
        """
        self.vector_store = vector_store
        self.index_path = Path(index_path) if index_path else None
        self.alpha = alpha
        self.phrase_boost = phrase_boost
        self.technical_term_boost = technical_term_boost
        
        # BM25 components
        self.bm25 = None
        self.tokenized_docs = []
        self.doc_mapping = []  # Maps BM25 index to vector store index
        
        # PCIe technical terms for enhanced matching
        self.pcie_technical_terms = self._initialize_pcie_terms()
        
        # Initialize or load BM25 index
        if self.index_path and self.index_path.exists():
            self.load_bm25_index()
        else:
            self.build_bm25_index()
    
    def _initialize_pcie_terms(self) -> Dict[str, float]:
        """Initialize PCIe technical terms with their importance weights"""
        return {
            # Core PCIe terms
            'pcie': 2.0, 'pci express': 2.0, 'pci-express': 2.0,
            
            # Generations and speeds
            'gen1': 1.8, 'gen2': 1.8, 'gen3': 1.8, 'gen4': 1.8, 'gen5': 1.8, 'gen6': 1.8,
            '2.5 gt/s': 1.8, '5.0 gt/s': 1.8, '8.0 gt/s': 1.8, '16.0 gt/s': 1.8, '32.0 gt/s': 1.8,
            
            # Technical acronyms (high importance)
            'flr': 2.5, 'function level reset': 2.5,
            'crs': 2.5, 'configuration request retry status': 2.5,
            'ltssm': 2.0, 'link training state machine': 2.0,
            'aer': 1.8, 'advanced error reporting': 1.8,
            'msi': 1.8, 'msi-x': 1.8, 'message signaled interrupt': 1.8,
            'tlp': 1.8, 'transaction layer packet': 1.8,
            'dllp': 1.8, 'data link layer packet': 1.8,
            
            # Error handling terms
            'completion timeout': 2.2,
            'malformed tlp': 2.0,
            'poisoned tlp': 2.0,
            'ecrc error': 1.8,
            'data link protocol error': 1.8,
            'surprise down': 1.8,
            'completer abort': 1.8,
            'unexpected completion': 1.8,
            'receiver overflow': 1.8,
            
            # Configuration and capabilities
            'configuration space': 1.5,
            'capability': 1.5, 'extended capability': 1.6,
            'bars': 1.5, 'base address register': 1.5,
            'vendor id': 1.4, 'device id': 1.4,
            
            # Power management
            'aspm': 1.6, 'active state power management': 1.6,
            'l0s': 1.5, 'l1': 1.5, 'l2': 1.5, 'l3': 1.5,
            'd0': 1.4, 'd1': 1.4, 'd2': 1.4, 'd3': 1.4,
            
            # Architecture components
            'root complex': 1.6, 'endpoint': 1.5, 'switch': 1.5,
            'upstream port': 1.5, 'downstream port': 1.5,
            'bridge': 1.4,
            
            # Link training
            'link training': 1.8,
            'polling': 1.5, 'configuration': 1.4,
            'detect': 1.4, 'recovery': 1.6,
            
            # Debugging terms
            'timeout': 1.8, 'error': 1.6, 'debug': 1.5,
            'troubleshooting': 1.5, 'analysis': 1.4
        }
    
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
        Enhanced tokenization with PCIe domain awareness
        
        Features:
        - Technical term preservation
        - Phrase detection
        - Domain-specific patterns
        """
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # First, extract and preserve important technical phrases
        preserved_phrases = self._extract_technical_phrases(text_lower)
        
        # Replace phrases with single tokens to preserve them
        phrase_map = {}
        for i, phrase in enumerate(preserved_phrases):
            placeholder = f"__PHRASE_{i}__"
            phrase_map[placeholder] = phrase
            text_lower = text_lower.replace(phrase, placeholder)
        
        # Standard tokenization
        import string
        tokens = text_lower.split()
        tokens = [token.strip(string.punctuation) for token in tokens]
        tokens = [token for token in tokens if token]
        
        # Restore preserved phrases and add individual words
        final_tokens = []
        for token in tokens:
            if token in phrase_map:
                # Add the phrase as a single token
                final_tokens.append(phrase_map[token].replace(' ', '_'))
                # Also add individual words from the phrase
                phrase_words = phrase_map[token].split()
                final_tokens.extend(phrase_words)
            else:
                final_tokens.append(token)
        
        return final_tokens
    
    def _extract_technical_phrases(self, text: str) -> List[str]:
        """Extract PCIe technical phrases that should be preserved as units"""
        phrases = []
        
        # Multi-word technical terms
        phrase_patterns = [
            r'function level reset',
            r'configuration request retry status',
            r'link training state machine',
            r'advanced error reporting',
            r'message signaled interrupt',
            r'transaction layer packet',
            r'data link layer packet',
            r'completion timeout',
            r'malformed tlp',
            r'poisoned tlp',
            r'ecrc error',
            r'data link protocol error',
            r'surprise down',
            r'completer abort',
            r'unexpected completion',
            r'receiver overflow',
            r'configuration space',
            r'extended capability',
            r'base address register',
            r'active state power management',
            r'root complex',
            r'upstream port',
            r'downstream port',
            r'link training',
            r'pci express',
            r'pci-express',
            # Speed patterns
            r'\d+\.\d+\s*gt/s',
            r'\d+\s*gbps',
            # Hex patterns
            r'0x[0-9a-f]+',
            # Gen patterns
            r'gen\d+',
            r'pcie\s*gen\d+',
        ]
        
        for pattern in phrase_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        return phrases
    
    def save_bm25_index(self):
        """Save BM25 index to disk"""
        if not self.index_path:
            return
        
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_data = {
            'tokenized_docs': self.tokenized_docs,
            'doc_mapping': self.doc_mapping,
            'alpha': self.alpha,
            'phrase_boost': self.phrase_boost,
            'technical_term_boost': self.technical_term_boost
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
            self.phrase_boost = index_data.get('phrase_boost', 2.0)
            self.technical_term_boost = index_data.get('technical_term_boost', 1.5)
            
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
        Enhanced BM25 keyword search with phrase boost and technical term recognition
        
        Returns:
            List of (doc_index, score) tuples
        """
        query_tokens = self._tokenize(query)
        base_scores = self.bm25.get_scores(query_tokens)
        
        # Apply enhancements
        enhanced_scores = self._apply_search_enhancements(query, base_scores)
        
        # Get top k indices
        top_indices = np.argsort(enhanced_scores)[::-1][:k]
        
        # Filter out zero scores and map to original indices
        results = []
        for idx in top_indices:
            if enhanced_scores[idx] > 0:
                original_idx = self.doc_mapping[idx]
                results.append((original_idx, float(enhanced_scores[idx])))
        
        return results
    
    def _apply_search_enhancements(self, query: str, base_scores: np.ndarray) -> np.ndarray:
        """Apply phrase matching boost and technical term recognition"""
        enhanced_scores = base_scores.copy()
        query_lower = query.lower()
        
        # Get documents for analysis
        documents = self.vector_store.documents
        
        for idx, base_score in enumerate(base_scores):
            if base_score <= 0 or idx >= len(self.doc_mapping):
                continue
                
            doc_idx = self.doc_mapping[idx]
            if doc_idx >= len(documents):
                continue
                
            doc_content = documents[doc_idx].lower()
            boost_factor = 1.0
            
            # 1. Exact phrase matching boost
            phrase_boost = self._calculate_phrase_boost(query_lower, doc_content)
            boost_factor *= phrase_boost
            
            # 2. Technical term boost
            tech_boost = self._calculate_technical_term_boost(query_lower, doc_content)
            boost_factor *= tech_boost
            
            # 3. PCIe domain context boost
            context_boost = self._calculate_context_boost(query_lower, doc_content)
            boost_factor *= context_boost
            
            # Apply boost (capped to prevent excessive inflation)
            enhanced_scores[idx] = base_score * min(boost_factor, 5.0)
        
        return enhanced_scores
    
    def _calculate_phrase_boost(self, query: str, doc_content: str) -> float:
        """Calculate boost for exact phrase matches"""
        boost = 1.0
        
        # Extract meaningful phrases from query (2+ words)
        words = query.split()
        
        # Check for exact phrase matches
        for i in range(len(words) - 1):
            for j in range(i + 2, min(i + 6, len(words) + 1)):  # phrases up to 5 words
                phrase = ' '.join(words[i:j])
                if len(phrase) > 3 and phrase in doc_content:  # meaningful phrases only
                    boost *= self.phrase_boost
                    break  # Avoid double-counting overlapping phrases
        
        return boost
    
    def _calculate_technical_term_boost(self, query: str, doc_content: str) -> float:
        """Calculate boost for PCIe technical terms"""
        boost = 1.0
        
        for term, weight in self.pcie_technical_terms.items():
            # Check if term appears in both query and document
            if term in query and term in doc_content:
                # Count occurrences for stronger boost
                query_count = query.count(term)
                doc_count = doc_content.count(term)
                
                # Apply boost based on term importance and frequency
                term_boost = 1.0 + (weight - 1.0) * min(query_count * doc_count, 3) / 3
                boost *= term_boost
        
        return min(boost, self.technical_term_boost * 2)  # Cap the boost
    
    def _calculate_context_boost(self, query: str, doc_content: str) -> float:
        """Calculate boost based on PCIe domain context"""
        boost = 1.0
        
        # PCIe error context indicators
        error_indicators = ['error', 'timeout', 'failed', 'abort', 'violation', 'malformed', 'poisoned']
        debug_indicators = ['debug', 'troubleshoot', 'analyze', 'diagnose', 'trace']
        spec_indicators = ['specification', 'compliance', 'standard', 'protocol']
        
        # Boost for error-related queries in error-related documents
        if any(indicator in query for indicator in error_indicators):
            if any(indicator in doc_content for indicator in error_indicators):
                boost *= 1.2
        
        # Boost for debug-related queries in debug-related documents
        if any(indicator in query for indicator in debug_indicators):
            if any(indicator in doc_content for indicator in debug_indicators):
                boost *= 1.15
        
        # Boost for spec-related queries in spec-related documents
        if any(indicator in query for indicator in spec_indicators):
            if any(indicator in doc_content for indicator in spec_indicators):
                boost *= 1.1
        
        return boost
    
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
            
            # Calculate combined score with technical term consideration
            tech_bonus = self._calculate_technical_bonus(query, content)
            combined_score = alpha * semantic_score_norm + (1 - alpha) * keyword_score_norm + tech_bonus
            
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
    
    def _calculate_technical_bonus(self, query: str, content: str) -> float:
        """Calculate additional bonus for technical term alignment"""
        bonus = 0.0
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Small bonus for technical term matches (additive, not multiplicative)
        for term, weight in self.pcie_technical_terms.items():
            if term in query_lower and term in content_lower:
                bonus += (weight - 1.0) * 0.05  # Small additive bonus
        
        return min(bonus, 0.3)  # Cap bonus at 0.3
    
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
        """Get enhanced search engine statistics"""
        return {
            'total_documents': len(self.tokenized_docs),
            'bm25_vocabulary_size': len(set(token for doc in self.tokenized_docs for token in doc)),
            'average_doc_length': np.mean([len(doc) for doc in self.tokenized_docs]) if self.tokenized_docs else 0,
            'alpha_weight': self.alpha,
            'phrase_boost': self.phrase_boost,
            'technical_term_boost': self.technical_term_boost,
            'pcie_terms_count': len(self.pcie_technical_terms),
            'index_path': str(self.index_path) if self.index_path else None,
            'enhancements': {
                'phrase_matching': True,
                'technical_term_recognition': True,
                'context_awareness': True
            }
        }