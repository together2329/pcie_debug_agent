#!/usr/bin/env python3
"""
Demonstration of Hybrid Search (Keyword + Semantic) for enhanced RAG
This shows how to combine BM25 keyword search with FAISS semantic search
"""

import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from dataclasses import dataclass
import logging

# Example implementation of hybrid search
@dataclass
class SearchResult:
    content: str
    score: float
    source: str
    search_type: str  # 'semantic' or 'keyword'


class HybridSearchEngine:
    """
    Combines keyword-based BM25 search with semantic FAISS search
    for improved retrieval accuracy
    """
    
    def __init__(self, documents: List[Dict[str, str]]):
        self.documents = documents
        self.setup_keyword_index()
        self.logger = logging.getLogger(__name__)
        
    def setup_keyword_index(self):
        """Initialize BM25 keyword search index"""
        # Tokenize documents for BM25
        tokenized_docs = [doc['content'].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def keyword_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append(SearchResult(
                    content=self.documents[idx]['content'],
                    score=float(scores[idx]),
                    source=self.documents[idx].get('source', f'doc_{idx}'),
                    search_type='keyword'
                ))
        
        return results
    
    def semantic_search(self, query: str, query_embedding: np.ndarray, 
                       doc_embeddings: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Perform semantic similarity search"""
        # Calculate cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding)
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append(SearchResult(
                content=self.documents[idx]['content'],
                score=float(similarities[idx]),
                source=self.documents[idx].get('source', f'doc_{idx}'),
                search_type='semantic'
            ))
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray,
                     doc_embeddings: np.ndarray, k: int = 10, 
                     alpha: float = 0.7) -> List[SearchResult]:
        """
        Combine keyword and semantic search results
        
        Args:
            alpha: Weight for semantic search (1-alpha for keyword search)
        """
        # Get results from both methods
        keyword_results = self.keyword_search(query, k=k*2)
        semantic_results = self.semantic_search(query, query_embedding, doc_embeddings, k=k*2)
        
        # Normalize scores
        keyword_scores = self._normalize_scores([r.score for r in keyword_results])
        semantic_scores = self._normalize_scores([r.score for r in semantic_results])
        
        # Create score dictionaries
        keyword_dict = {r.source: (r, score) for r, score in zip(keyword_results, keyword_scores)}
        semantic_dict = {r.source: (r, score) for r, score in zip(semantic_results, semantic_scores)}
        
        # Combine scores
        combined_scores = {}
        all_sources = set(keyword_dict.keys()) | set(semantic_dict.keys())
        
        for source in all_sources:
            keyword_score = keyword_dict.get(source, (None, 0))[1]
            semantic_score = semantic_dict.get(source, (None, 0))[1]
            
            # Weighted combination
            combined_score = alpha * semantic_score + (1 - alpha) * keyword_score
            
            # Get the result object (prefer semantic if available)
            result = semantic_dict.get(source, keyword_dict.get(source))[0]
            combined_scores[source] = (result, combined_score)
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1][1], reverse=True)
        
        # Return top k results
        return [result for _, (result, _) in sorted_results[:k]]
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


def demonstrate_hybrid_search():
    """Demo showing hybrid search in action"""
    # Sample PCIe knowledge base
    documents = [
        {
            "content": "PCIe Gen 4 provides 16 GT/s data rate per lane with 128b/130b encoding",
            "source": "pcie_speeds.md"
        },
        {
            "content": "LTSSM recovery state handles link errors and retraining without full reset",
            "source": "ltssm_guide.md"
        },
        {
            "content": "PCIe Advanced Error Reporting (AER) enables detailed error logging",
            "source": "aer_guide.md"
        },
        {
            "content": "The recovery state in LTSSM can be entered from L0, L0s, or L1 states",
            "source": "ltssm_transitions.md"
        },
        {
            "content": "Gen 4 PCIe doubles bandwidth compared to Gen 3 with improved encoding",
            "source": "pcie_evolution.md"
        }
    ]
    
    # Initialize hybrid search
    search_engine = HybridSearchEngine(documents)
    
    # Simulate embeddings (in real implementation, use actual embedding model)
    np.random.seed(42)
    doc_embeddings = np.random.randn(len(documents), 384)  # 384-dim embeddings
    doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    
    # Test queries
    test_queries = [
        "PCIe Gen 4 speed",
        "LTSSM recovery error handling",
        "bandwidth improvement Gen 4"
    ]
    
    print("🔍 Hybrid Search Demonstration")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 60)
        
        # Simulate query embedding
        query_embedding = np.random.randn(384)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Perform searches
        print("\n1️⃣ Keyword Search (BM25):")
        keyword_results = search_engine.keyword_search(query, k=3)
        for i, result in enumerate(keyword_results, 1):
            print(f"   [{i}] {result.source} (score: {result.score:.3f})")
        
        print("\n2️⃣ Semantic Search (Embeddings):")
        semantic_results = search_engine.semantic_search(query, query_embedding, doc_embeddings, k=3)
        for i, result in enumerate(semantic_results, 1):
            print(f"   [{i}] {result.source} (score: {result.score:.3f})")
        
        print("\n3️⃣ Hybrid Search (α=0.7):")
        hybrid_results = search_engine.hybrid_search(query, query_embedding, doc_embeddings, k=3, alpha=0.7)
        for i, result in enumerate(hybrid_results, 1):
            print(f"   [{i}] {result.source}")
            print(f"       {result.content[:80]}...")


def show_fusion_strategies():
    """Demonstrate different result fusion strategies"""
    print("\n\n🔄 Result Fusion Strategies")
    print("=" * 60)
    
    # Example scores from two search methods
    keyword_scores = {"doc1": 0.8, "doc2": 0.6, "doc3": 0.4}
    semantic_scores = {"doc1": 0.5, "doc3": 0.9, "doc4": 0.7}
    
    print("\nKeyword scores:", keyword_scores)
    print("Semantic scores:", semantic_scores)
    
    # Strategy 1: Weighted Linear Combination
    print("\n1. Weighted Linear (α=0.7):")
    alpha = 0.7
    combined = {}
    for doc in set(keyword_scores.keys()) | set(semantic_scores.keys()):
        k_score = keyword_scores.get(doc, 0)
        s_score = semantic_scores.get(doc, 0)
        combined[doc] = alpha * s_score + (1 - alpha) * k_score
    print("   ", dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)))
    
    # Strategy 2: Reciprocal Rank Fusion
    print("\n2. Reciprocal Rank Fusion (k=60):")
    k = 60
    rrf_scores = {}
    
    # Convert to ranks
    keyword_ranks = {doc: i+1 for i, doc in enumerate(sorted(keyword_scores, key=keyword_scores.get, reverse=True))}
    semantic_ranks = {doc: i+1 for i, doc in enumerate(sorted(semantic_scores, key=semantic_scores.get, reverse=True))}
    
    for doc in set(keyword_scores.keys()) | set(semantic_scores.keys()):
        k_rank = keyword_ranks.get(doc, k)
        s_rank = semantic_ranks.get(doc, k)
        rrf_scores[doc] = 1/(k + k_rank) + 1/(k + s_rank)
    print("   ", dict(sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)))
    
    # Strategy 3: Max pooling
    print("\n3. Max Pooling:")
    max_scores = {}
    for doc in set(keyword_scores.keys()) | set(semantic_scores.keys()):
        max_scores[doc] = max(keyword_scores.get(doc, 0), semantic_scores.get(doc, 0))
    print("   ", dict(sorted(max_scores.items(), key=lambda x: x[1], reverse=True)))


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demonstrations
    demonstrate_hybrid_search()
    show_fusion_strategies()
    
    print("\n\n💡 Key Insights:")
    print("1. Keyword search excels at exact term matching")
    print("2. Semantic search captures meaning and synonyms")
    print("3. Hybrid search combines both strengths")
    print("4. Different fusion strategies suit different use cases")
    print("5. Tuning α parameter balances keyword vs semantic importance")