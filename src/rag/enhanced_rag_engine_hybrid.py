"""
Enhanced RAG Engine with Hybrid LLM Integration
Supports both quick and detailed analysis using Llama 3.2 3B + DeepSeek
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

from src.rag.vector_store import FAISSVectorStore
from src.models.model_manager import ModelManager
from src.models.hybrid_llm_provider import HybridLLMProvider, AnalysisRequest
from src.rag.retriever import Retriever
from src.models.pcie_prompts import PCIePromptTemplates

logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """Enhanced RAG query with hybrid analysis support"""
    query: str
    context_window: int = 5
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.5
    include_metadata: bool = True
    analysis_type: Literal["quick", "detailed", "auto"] = "auto"
    max_response_time: float = 30.0

@dataclass
class RAGResponse:
    """Enhanced RAG response with hybrid model info"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    model_used: str = ""
    response_time: float = 0.0
    analysis_type: str = ""

class EnhancedRAGEngineHybrid:
    """Enhanced RAG Engine with Hybrid LLM Support"""
    
    def __init__(self,
                 vector_store: FAISSVectorStore,
                 model_manager: ModelManager,
                 models_dir: str = "models",
                 enable_cache: bool = True):
        
        self.vector_store = vector_store
        self.model_manager = model_manager
        
        # Initialize hybrid LLM provider
        self.hybrid_provider = HybridLLMProvider(models_dir=models_dir)
        
        # Check model availability
        status = self.hybrid_provider.get_model_status()
        logger.info(f"Hybrid LLM Status - Llama: {status['llama']['available']}, "
                   f"DeepSeek: {status['deepseek']['available']}")
        
        # Initialize retriever
        self.retriever = Retriever(vector_store, model_manager)
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "cache_hits": 0,
            "model_usage": {"llama": 0, "deepseek": 0},
            "analysis_types": {"quick": 0, "detailed": 0, "auto": 0}
        }
        
        # Query cache
        self.enable_cache = enable_cache
        self.query_cache = {}
        self.cache_size = 100
        
    def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Execute RAG query with hybrid LLM analysis
        
        Args:
            rag_query: RAG query object with analysis preferences
            
        Returns:
            RAG response with hybrid analysis results
        """
        start_time = datetime.now()
        
        try:
            # Check cache
            if self.enable_cache:
                cache_key = self._get_cache_key(rag_query)
                if cache_key in self.query_cache:
                    self.metrics["cache_hits"] += 1
                    cached_response = self.query_cache[cache_key]
                    cached_response.metadata["from_cache"] = True
                    return cached_response
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(
                query=rag_query.query,
                k=rag_query.context_window,
                filters=rag_query.filters,
                min_similarity=rag_query.min_similarity
            )
            
            # Rerank if requested
            if rag_query.rerank and len(retrieved_docs) > 1:
                retrieved_docs = self._rerank_documents(
                    query=rag_query.query,
                    documents=retrieved_docs
                )
            
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Create analysis request for hybrid provider
            analysis_request = AnalysisRequest(
                query=rag_query.query,
                error_log=context,
                analysis_type=rag_query.analysis_type,
                max_response_time=rag_query.max_response_time,
                context="Retrieved from PCIe knowledge base"
            )
            
            # Get hybrid analysis
            hybrid_response = self.hybrid_provider.analyze_pcie_error(analysis_request)
            
            # Build RAG response
            response = RAGResponse(
                answer=hybrid_response.response if hybrid_response.response else "Analysis failed",
                sources=[{
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "similarity": doc.get("similarity", 0.0)
                } for doc in retrieved_docs],
                confidence=hybrid_response.confidence_score,
                reasoning=self._extract_reasoning(hybrid_response.response),
                metadata={
                    "model_used": hybrid_response.model_used,
                    "analysis_type": hybrid_response.analysis_type,
                    "fallback_used": hybrid_response.fallback_used,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "retrieved_docs": len(retrieved_docs),
                    "error": hybrid_response.error
                },
                model_used=hybrid_response.model_used,
                response_time=hybrid_response.response_time,
                analysis_type=hybrid_response.analysis_type
            )
            
            # Update metrics
            self._update_metrics(response, hybrid_response)
            
            # Cache successful response
            if self.enable_cache and hybrid_response.error is None:
                self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return RAGResponse(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    def quick_pcie_analysis(self, query: str, error_log: str = "") -> RAGResponse:
        """
        Quick PCIe error analysis using Llama 3.2 3B
        
        Args:
            query: Analysis query
            error_log: Optional error log content
            
        Returns:
            Quick analysis response
        """
        rag_query = RAGQuery(
            query=f"{query}\n\nError Log:\n{error_log}" if error_log else query,
            analysis_type="quick",
            max_response_time=15.0,
            context_window=3
        )
        return self.query(rag_query)
    
    def detailed_pcie_analysis(self, query: str, error_log: str = "") -> RAGResponse:
        """
        Detailed PCIe error analysis using DeepSeek (or fallback)
        
        Args:
            query: Analysis query
            error_log: Optional error log content
            
        Returns:
            Detailed analysis response
        """
        rag_query = RAGQuery(
            query=f"{query}\n\nError Log:\n{error_log}" if error_log else query,
            analysis_type="detailed",
            max_response_time=180.0,
            context_window=10
        )
        return self.query(rag_query)
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents[:5]):  # Top 5 documents
            source = doc.get("metadata", {}).get("source", "Unknown")
            content = doc.get("content", "")[:500]  # First 500 chars
            similarity = doc.get("similarity", 0.0)
            
            context_parts.append(
                f"[Source {i+1}] {source} (similarity: {similarity:.2f})\n{content}"
            )
        
        return "\n\n".join(context_parts)
    
    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents based on PCIe-specific relevance"""
        # Simple reranking based on PCIe keywords
        pcie_keywords = [
            "PCIe", "TLP", "LTSSM", "link training", "recovery",
            "malformed", "error", "completion", "timeout", "enumeration"
        ]
        
        for doc in documents:
            content = doc.get("content", "").lower()
            keyword_score = sum(1 for kw in pcie_keywords if kw.lower() in content)
            doc["rerank_score"] = doc.get("similarity", 0.0) + (keyword_score * 0.1)
        
        # Sort by rerank score
        return sorted(documents, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    
    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning from LLM response"""
        if not response:
            return None
            
        # Look for reasoning patterns
        reasoning_markers = [
            "Root cause:", "Analysis:", "Reasoning:", "Because", "Due to"
        ]
        
        for marker in reasoning_markers:
            if marker in response:
                idx = response.find(marker)
                reasoning = response[idx:idx+200]  # Extract snippet
                return reasoning.strip()
        
        return None
    
    def _update_metrics(self, response: RAGResponse, hybrid_response):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1
        
        # Update average response time
        n = self.metrics["queries_processed"]
        avg_time = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = ((n-1) * avg_time + response.response_time) / n
        
        # Update average confidence
        avg_conf = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = ((n-1) * avg_conf + response.confidence) / n
        
        # Update model usage
        if "llama" in hybrid_response.model_used:
            self.metrics["model_usage"]["llama"] += 1
        elif "deepseek" in hybrid_response.model_used:
            self.metrics["model_usage"]["deepseek"] += 1
        
        # Update analysis type counts
        self.metrics["analysis_types"][hybrid_response.analysis_type] += 1
    
    def _get_cache_key(self, rag_query: RAGQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            rag_query.query,
            str(rag_query.context_window),
            str(rag_query.analysis_type),
            str(rag_query.filters)
        ]
        return hash("|".join(key_parts))
    
    def _cache_response(self, key: str, response: RAGResponse):
        """Cache response with LRU eviction"""
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]
        
        self.query_cache[key] = response
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            "cache_hit_rate": self.metrics["cache_hits"] / max(self.metrics["queries_processed"], 1),
            "model_status": self.hybrid_provider.get_model_status()
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")