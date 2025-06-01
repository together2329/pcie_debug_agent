"""
Enhanced RAG Engine V2 with Hybrid Search support
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

from src.vectorstore.faiss_store import FAISSVectorStore
from src.models.model_manager import ModelManager
from src.rag.retriever import Retriever
from src.rag.analyzer import Analyzer
from src.rag.hybrid_search import HybridSearchEngine, HybridSearchResult
from src.models.pcie_prompts import PCIePromptTemplates

logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """Enhanced RAG query object with hybrid search options"""
    query: str
    context_window: int = 5
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.5
    include_metadata: bool = True
    use_hybrid_search: bool = True  # New: Enable hybrid search
    alpha: float = 0.7  # New: Weight for semantic vs keyword search
    fusion_method: str = "weighted"  # New: "weighted" or "rrf"

@dataclass
class RAGResponse:
    """Enhanced RAG response object"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    search_method: str = "hybrid"  # New: Track search method used

class EnhancedRAGEngineV2:
    """Enhanced RAG Engine with Hybrid Search Integration"""
    
    def __init__(self,
                 vector_store: FAISSVectorStore,
                 model_manager: ModelManager,
                 llm_provider: str = "openai",
                 llm_model: str = "gpt-4",
                 temperature: float = 0.1,
                 max_tokens: int = 2000,
                 enable_hybrid_search: bool = True,
                 bm25_index_path: Optional[str] = None):
        
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_hybrid_search = enable_hybrid_search
        
        # Initialize retrievers
        self.retriever = Retriever(vector_store, model_manager)
        
        # Initialize hybrid search if enabled
        self.hybrid_search = None
        if enable_hybrid_search:
            try:
                self.hybrid_search = HybridSearchEngine(
                    vector_store=vector_store,
                    index_path=bm25_index_path or "data/vectorstore/bm25_index.pkl"
                )
                logger.info("Hybrid search engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize hybrid search: {e}")
                logger.info("Falling back to semantic search only")
                self.enable_hybrid_search = False
        
        # Initialize analyzer
        self.analyzer = Analyzer(
            llm_provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "cache_hits": 0,
            "hybrid_searches": 0,
            "semantic_searches": 0
        }
        
        # Query cache
        self.query_cache = {}
        self.cache_size = 100
        
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Execute RAG query with hybrid search support
        
        Args:
            rag_query: RAG query object
            
        Returns:
            RAG response object
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(rag_query)
            if cache_key in self.query_cache:
                self.metrics["cache_hits"] += 1
                cached_response = self.query_cache[cache_key]
                cached_response.metadata["from_cache"] = True
                return cached_response
            
            # Generate query embedding
            query_embedding = self._generate_query_embedding(rag_query.query)
            
            # Retrieve documents using appropriate method
            if self.enable_hybrid_search and rag_query.use_hybrid_search and self.hybrid_search:
                retrieved_docs = self._hybrid_retrieve(
                    query=rag_query.query,
                    query_embedding=query_embedding,
                    k=rag_query.context_window * 2,  # Get more candidates
                    alpha=rag_query.alpha,
                    fusion_method=rag_query.fusion_method
                )
                search_method = "hybrid"
                self.metrics["hybrid_searches"] += 1
            else:
                retrieved_docs = self._semantic_retrieve(
                    query_embedding=query_embedding,
                    k=rag_query.context_window,
                    filters=rag_query.filters,
                    min_similarity=rag_query.min_similarity
                )
                search_method = "semantic"
                self.metrics["semantic_searches"] += 1
            
            # Filter by similarity threshold
            filtered_docs = self._filter_by_similarity(retrieved_docs, rag_query.min_similarity)
            
            # Take top k documents
            final_docs = filtered_docs[:rag_query.context_window]
            
            # Format sources
            sources = self._format_sources(final_docs, rag_query.include_metadata)
            
            # Generate answer
            if final_docs:
                answer, confidence = self._generate_answer(
                    query=rag_query.query,
                    sources=sources,
                    search_method=search_method
                )
            else:
                answer = "I couldn't find relevant information to answer your question."
                confidence = 0.0
            
            # Create response
            response_time = (datetime.now() - start_time).total_seconds()
            response = RAGResponse(
                answer=answer,
                sources=sources,
                confidence=confidence,
                search_method=search_method,
                metadata={
                    "response_time": response_time,
                    "num_sources": len(sources),
                    "search_method": search_method,
                    "query_embedding_norm": float(np.linalg.norm(query_embedding))
                }
            )
            
            # Update cache
            self._update_cache(cache_key, response)
            
            # Update metrics
            self._update_metrics(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return RAGResponse(
                answer=f"Sorry, an error occurred while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                search_method="error",
                metadata={"error": str(e)}
            )
    
    def _hybrid_retrieve(self, 
                        query: str, 
                        query_embedding: np.ndarray,
                        k: int,
                        alpha: float,
                        fusion_method: str) -> List[Dict[str, Any]]:
        """Retrieve documents using hybrid search"""
        if fusion_method == "rrf":
            results = self.hybrid_search.reciprocal_rank_fusion(
                query=query,
                query_embedding=query_embedding,
                k=k
            )
        else:  # weighted
            results = self.hybrid_search.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                k=k,
                alpha=alpha
            )
        
        # Convert HybridSearchResult to standard format
        formatted_results = []
        for result in results:
            formatted_results.append({
                'content': result.content,
                'metadata': result.metadata,
                'score': result.combined_score,
                'semantic_score': result.semantic_score,
                'keyword_score': result.keyword_score,
                'rank': result.rank
            })
        
        return formatted_results
    
    def _semantic_retrieve(self,
                          query_embedding: np.ndarray,
                          k: int,
                          filters: Optional[Dict[str, Any]],
                          min_similarity: float) -> List[Dict[str, Any]]:
        """Retrieve documents using semantic search only"""
        results = self.vector_store.search(query_embedding, k=k)
        
        # Format results
        formatted_results = []
        for i, (content, metadata, score) in enumerate(results):
            formatted_results.append({
                'content': content,
                'metadata': metadata,
                'score': score,
                'semantic_score': score,
                'keyword_score': 0.0,
                'rank': i + 1
            })
        
        return formatted_results
    
    def _filter_by_similarity(self, docs: List[Dict[str, Any]], min_similarity: float) -> List[Dict[str, Any]]:
        """Filter documents by minimum similarity score"""
        return [doc for doc in docs if doc['score'] >= min_similarity]
    
    def _format_sources(self, docs: List[Dict[str, Any]], include_metadata: bool) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        sources = []
        for doc in docs:
            source = {
                'content': doc['content'],
                'score': doc['score']
            }
            
            if include_metadata:
                source['metadata'] = doc.get('metadata', {})
                source['semantic_score'] = doc.get('semantic_score', doc['score'])
                source['keyword_score'] = doc.get('keyword_score', 0.0)
                source['rank'] = doc.get('rank', 0)
            
            sources.append(source)
        
        return sources
    
    def _generate_answer(self, query: str, sources: List[Dict[str, Any]], search_method: str) -> Tuple[str, float]:
        """Generate answer using LLM"""
        # Prepare context from sources
        context = "\n\n".join([
            f"Source {i+1} (relevance: {source['score']:.3f}):\n{source['content']}"
            for i, source in enumerate(sources)
        ])
        
        # Create prompt
        prompt = f"""You are a PCIe debugging expert. Based on the following context, please answer the user's question.

Search method used: {search_method}

Context:
{context}

Question: {query}

Please provide a detailed and accurate answer based on the context provided. If the context doesn't contain enough information, say so clearly.

Answer:"""
        
        # Generate response
        try:
            answer = self.model_manager.generate_completion(
                provider=self.llm_provider,
                model=self.llm_model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Calculate confidence based on source scores
            if sources:
                avg_score = np.mean([s['score'] for s in sources])
                confidence = min(avg_score * 1.2, 1.0)  # Scale up slightly but cap at 1.0
            else:
                confidence = 0.0
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}", 0.0
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate query embedding"""
        embeddings = self.model_manager.generate_embeddings([query])
        return embeddings[0]
    
    def _get_cache_key(self, rag_query: RAGQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            rag_query.query,
            str(rag_query.context_window),
            str(rag_query.use_hybrid_search),
            str(rag_query.alpha),
            rag_query.fusion_method
        ]
        return "|".join(key_parts)
    
    def _update_cache(self, key: str, response: RAGResponse):
        """Update query cache"""
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[key] = response
    
    def _update_metrics(self, response: RAGResponse):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1
        
        # Update average response time
        if "response_time" in response.metadata:
            n = self.metrics["queries_processed"]
            prev_avg = self.metrics["average_response_time"]
            new_time = response.metadata["response_time"]
            self.metrics["average_response_time"] = (prev_avg * (n-1) + new_time) / n
        
        # Update average confidence
        n = self.metrics["queries_processed"]
        prev_avg = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (prev_avg * (n-1) + response.confidence) / n
    
    def update_bm25_index(self):
        """Update BM25 index after vector store changes"""
        if self.hybrid_search:
            logger.info("Rebuilding BM25 index...")
            self.hybrid_search.build_bm25_index()
            logger.info("BM25 index rebuilt successfully")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "metrics": self.metrics,
            "cache_size": len(self.query_cache),
            "hybrid_search_enabled": self.enable_hybrid_search,
            "vector_store_size": self.vector_store.index.ntotal if self.vector_store else 0
        }
        
        if self.hybrid_search:
            stats["hybrid_search_stats"] = self.hybrid_search.get_statistics()
        
        return stats
    
    async def aquery(self, rag_query: RAGQuery) -> RAGResponse:
        """Async RAG query execution"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            response = await loop.run_in_executor(executor, self.query, rag_query)
        return response
    
    def batch_query(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """Batch query processing with shared embeddings"""
        responses = []
        
        # Generate all query embeddings at once
        query_texts = [q.query for q in queries]
        query_embeddings = self.model_manager.generate_embeddings(query_texts)
        
        for query, embedding in zip(queries, query_embeddings):
            # Process each query with pre-computed embedding
            response = self._process_with_embedding(query, embedding)
            responses.append(response)
            
        return responses
    
    def _process_with_embedding(self, rag_query: RAGQuery, query_embedding: np.ndarray) -> RAGResponse:
        """Process query with pre-computed embedding"""
        # Similar to query() but skips embedding generation
        # Implementation details omitted for brevity
        pass