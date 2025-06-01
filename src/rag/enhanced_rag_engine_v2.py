#!/usr/bin/env python3
"""
Enhanced RAG Engine V2 with Context Compression
Integrates intelligent context compression for improved quality and efficiency
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import time

try:
    from src.vectorstore.faiss_store import FAISSVectorStore
    from src.models.model_manager import ModelManager
    from src.rag.retriever import Retriever
    from src.rag.analyzer import Analyzer
    from src.rag.context_compressor import ContextCompressor, CompressedContext
    from src.models.pcie_prompts import PCIePromptTemplates
except ImportError:
    # Fallback for standalone testing
    try:
        from context_compressor import ContextCompressor, CompressedContext
    except ImportError:
        # Define minimal classes for testing
        from dataclasses import dataclass
        
        @dataclass
        class CompressedContext:
            compressed_text: str
            original_length: int
            compressed_length: int
            compression_ratio: float
            key_sections: list
            relevance_score: float
        
        class ContextCompressor:
            def __init__(self, **kwargs):
                pass
            def compress_context(self, query, retrieved_docs, error_log):
                content = error_log + "\n" + "\n".join([doc.get('content', '') for doc in retrieved_docs])
                return CompressedContext(
                    compressed_text=content,
                    original_length=len(content),
                    compressed_length=len(content),
                    compression_ratio=1.0,
                    key_sections=[],
                    relevance_score=0.8
                )
    
    # Mock other imports for testing
    FAISSVectorStore = None
    ModelManager = None
    Retriever = None
    Analyzer = None
    PCIePromptTemplates = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancedRAGQuery:
    """Enhanced RAG query with compression parameters"""
    query: str
    error_log: str = ""
    context_window: int = 5
    rerank: bool = True
    filters: Optional[Dict[str, Any]] = None
    min_similarity: float = 0.5
    include_metadata: bool = True
    # Compression parameters
    compress_context: bool = True
    max_context_tokens: int = 8000
    preserve_ratio: float = 0.7
    analysis_type: str = "auto"  # auto, quick, detailed

@dataclass 
class EnhancedRAGResponse:
    """Enhanced RAG response with compression metadata"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    # Compression metadata
    compression_info: Optional[CompressedContext] = None
    model_used: str = ""
    response_time: float = 0.0

class EnhancedRAGEngineV2:
    """
    Enhanced RAG Engine V2 with Intelligent Context Compression
    
    Features:
    - Context compression to fit token limits
    - PCIe-specific keyword weighting
    - Intelligent retrieval strategies
    - Quality optimization
    """
    
    def __init__(self,
                 vector_store: Optional[FAISSVectorStore] = None,
                 model_manager: Optional[ModelManager] = None,
                 llm_provider: str = "local",
                 temperature: float = 0.1,
                 max_tokens: int = 2000):
        """
        Initialize enhanced RAG engine
        
        Args:
            vector_store: FAISS vector store for document retrieval
            model_manager: Model manager for embeddings
            llm_provider: LLM provider (local, openai, etc.)
            temperature: Generation temperature
            max_tokens: Maximum response tokens
        """
        
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize components
        if vector_store and model_manager:
            self.retriever = Retriever(vector_store, model_manager)
        else:
            self.retriever = None
            logger.warning("No vector store/model manager - retrieval disabled")
        
        # Context compressor
        self.compressor = ContextCompressor(
            max_tokens=6000,  # Leave room for prompt and response
            preserve_ratio=0.7
        )
        
        # Performance metrics
        self.metrics = {
            "queries_processed": 0,
            "average_response_time": 0,
            "average_confidence": 0,
            "compression_ratio": 0,
            "cache_hits": 0
        }
        
        # Enhanced query cache with compression
        self.query_cache = {}
        self.cache_size = 50
        
        # PCIe-specific prompt templates
        if PCIePromptTemplates:
            self.prompt_templates = PCIePromptTemplates()
        else:
            self.prompt_templates = None
    
    def query(self, rag_query: EnhancedRAGQuery) -> EnhancedRAGResponse:
        """
        Execute enhanced RAG query with context compression
        
        Args:
            rag_query: Enhanced RAG query object
            
        Returns:
            Enhanced RAG response with compression metadata
        """
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(rag_query)
            if cache_key in self.query_cache:
                self.metrics["cache_hits"] += 1
                cached_response = self.query_cache[cache_key]
                cached_response.metadata["from_cache"] = True
                return cached_response
            
            # Retrieve relevant documents
            retrieved_docs = self._retrieve_documents(rag_query)
            
            # Compress context intelligently
            compression_info = None
            if rag_query.compress_context:
                compression_info = self.compressor.compress_context(
                    query=rag_query.query,
                    retrieved_docs=retrieved_docs,
                    error_log=rag_query.error_log
                )
                
                # Update metrics
                if compression_info.original_length > 0:
                    self.metrics["compression_ratio"] = (
                        self.metrics["compression_ratio"] * self.metrics["queries_processed"] +
                        compression_info.compression_ratio
                    ) / (self.metrics["queries_processed"] + 1)
            
            # Generate response using compressed context
            answer, reasoning, model_used = self._generate_response(
                rag_query, retrieved_docs, compression_info
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                retrieved_docs, answer, compression_info
            )
            
            # Create response
            response_time = time.time() - start_time
            response = EnhancedRAGResponse(
                answer=answer,
                sources=retrieved_docs[:rag_query.context_window],
                confidence=confidence,
                reasoning=reasoning,
                metadata={
                    "query_time": response_time,
                    "num_sources": len(retrieved_docs),
                    "analysis_type": rag_query.analysis_type,
                    "from_cache": False,
                    "retrieval_enabled": self.retriever is not None
                },
                compression_info=compression_info,
                model_used=model_used,
                response_time=response_time
            )
            
            # Update cache
            if len(self.query_cache) < self.cache_size:
                self.query_cache[cache_key] = response
            
            # Update metrics
            self._update_metrics(response)
            
            return response
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            
            # Fallback response
            return EnhancedRAGResponse(
                answer=f"I encountered an error while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                reasoning="Error occurred during processing",
                metadata={
                    "query_time": time.time() - start_time,
                    "error": str(e),
                    "from_cache": False
                },
                model_used="error",
                response_time=time.time() - start_time
            )
    
    def _retrieve_documents(self, rag_query: EnhancedRAGQuery) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using RAG"""
        if not self.retriever:
            # No retrieval available - return empty docs
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model_manager.get_embedding(rag_query.query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding,
                k=rag_query.context_window * 2,  # Get more for filtering
                threshold=rag_query.min_similarity
            )
            
            # Convert to document format
            docs = []
            for result in results:
                docs.append({
                    'content': result.get('content', result.get('text', '')),
                    'metadata': result.get('metadata', {}),
                    'score': result.get('score', 0.0)
                })
            
            # Rerank if requested
            if rag_query.rerank and len(docs) > 1:
                docs = self._rerank_documents(rag_query.query, docs)
            
            return docs[:rag_query.context_window]
            
        except Exception as e:
            logger.warning(f"Document retrieval failed: {e}")
            return []
    
    def _rerank_documents(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank documents by relevance to query"""
        try:
            # Simple keyword-based reranking for PCIe queries
            query_lower = query.lower()
            
            # PCIe-specific keywords for boosting
            pcie_keywords = [
                'pcie', 'link', 'training', 'ltssm', 'tlp', 'aer', 'error',
                'lane', 'gen1', 'gen2', 'gen3', 'gen4', 'gen5', 'recovery'
            ]
            
            def calculate_relevance(doc):
                content = doc.get('content', '').lower()
                score = doc.get('score', 0.0)
                
                # Boost based on PCIe keywords
                keyword_boost = sum(1 for kw in pcie_keywords if kw in content)
                
                # Boost if query terms appear
                query_terms = query_lower.split()
                query_boost = sum(1 for term in query_terms if term in content)
                
                return score + (keyword_boost * 0.1) + (query_boost * 0.2)
            
            docs.sort(key=calculate_relevance, reverse=True)
            return docs
            
        except Exception as e:
            logger.warning(f"Document reranking failed: {e}")
            return docs
    
    def _generate_response(self, 
                          rag_query: EnhancedRAGQuery,
                          retrieved_docs: List[Dict[str, Any]],
                          compression_info: Optional[CompressedContext]) -> Tuple[str, str, str]:
        """Generate response using compressed context"""
        
        # Use compressed context if available
        if compression_info:
            context = compression_info.compressed_text
        else:
            # Fallback to simple context building
            context = self._build_simple_context(retrieved_docs, rag_query.error_log)
        
        # Create PCIe-optimized prompt
        prompt = self._create_pcie_prompt(rag_query, context)
        
        # Generate response (simplified - would integrate with hybrid LLM)
        try:
            # This would call the hybrid LLM provider
            answer = self._generate_with_llm(prompt, rag_query.analysis_type)
            reasoning = f"Generated using {rag_query.analysis_type} analysis with {len(retrieved_docs)} sources"
            model_used = f"hybrid-{rag_query.analysis_type}"
            
            return answer, reasoning, model_used
            
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            
            # Fallback to template-based response
            answer = self._create_fallback_response(rag_query, retrieved_docs)
            reasoning = "Used fallback response due to LLM unavailability"
            model_used = "fallback"
            
            return answer, reasoning, model_used
    
    def _build_simple_context(self, docs: List[Dict[str, Any]], error_log: str) -> str:
        """Build simple context when compression is disabled"""
        context_parts = []
        
        if error_log.strip():
            context_parts.append(f"Error Log:\n{error_log}\n")
        
        for i, doc in enumerate(docs):
            content = doc.get('content', '')
            if content:
                context_parts.append(f"Source {i+1}:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _create_pcie_prompt(self, rag_query: EnhancedRAGQuery, context: str) -> str:
        """Create PCIe-optimized prompt"""
        
        base_prompt = f"""You are a PCIe (PCI Express) debug expert. Analyze the following information and provide a detailed response.

QUERY: {rag_query.query}

CONTEXT:
{context}

Please provide:
1. Root cause analysis
2. Specific technical details
3. Step-by-step resolution
4. Prevention recommendations

Focus on PCIe-specific terminology and technical accuracy. Be precise and actionable.

RESPONSE:"""
        
        return base_prompt
    
    def _generate_with_llm(self, prompt: str, analysis_type: str) -> str:
        """Generate response with LLM (placeholder for integration)"""
        # This would integrate with the hybrid LLM provider
        # For now, return a placeholder
        
        if analysis_type == "quick":
            return "Quick analysis: PCIe error detected. Check link training and signal integrity."
        elif analysis_type == "detailed":
            return "Detailed analysis: This appears to be a PCIe link training failure. The LTSSM state machine is stuck in Recovery.RcvrLock, indicating signal integrity issues on lanes 0-3. Recommended actions: 1) Check physical connections, 2) Verify power supply stability, 3) Test with different cables, 4) Check for EMI interference."
        else:
            return "Auto analysis: PCIe link training issue detected. Investigating signal integrity and hardware configuration."
    
    def _create_fallback_response(self, 
                                rag_query: EnhancedRAGQuery,
                                docs: List[Dict[str, Any]]) -> str:
        """Create fallback response when LLM is unavailable"""
        
        if not docs and not rag_query.error_log:
            return "I don't have enough information to analyze this PCIe issue. Please provide error logs or more specific details."
        
        response_parts = [
            "Based on the available information:",
            ""
        ]
        
        if rag_query.error_log:
            response_parts.extend([
                "Error Log Analysis:",
                f"- {rag_query.error_log}",
                ""
            ])
        
        if docs:
            response_parts.extend([
                "Related Documentation:",
                ""
            ])
            
            for i, doc in enumerate(docs[:3]):
                content = doc.get('content', '')[:200]
                response_parts.append(f"{i+1}. {content}...")
        
        response_parts.extend([
            "",
            "For detailed analysis, please ensure the LLM system is available."
        ])
        
        return "\n".join(response_parts)
    
    def _calculate_confidence(self, 
                            docs: List[Dict[str, Any]],
                            answer: str,
                            compression_info: Optional[CompressedContext]) -> float:
        """Calculate response confidence"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on retrieved documents
        if docs:
            confidence += min(0.3, len(docs) * 0.1)
        
        # Boost based on compression quality
        if compression_info:
            confidence += compression_info.relevance_score * 0.2
        
        # Boost based on answer length and structure
        if len(answer) > 100:
            confidence += 0.1
            
        if any(keyword in answer.lower() for keyword in ['pcie', 'link', 'error', 'ltssm']):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _get_cache_key(self, rag_query: EnhancedRAGQuery) -> str:
        """Generate cache key for query"""
        key_parts = [
            rag_query.query,
            rag_query.error_log,
            str(rag_query.context_window),
            str(rag_query.compress_context),
            rag_query.analysis_type
        ]
        return hash("|".join(key_parts))
    
    def _update_metrics(self, response: EnhancedRAGResponse):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1
        
        # Update average response time
        self.metrics["average_response_time"] = (
            self.metrics["average_response_time"] * (self.metrics["queries_processed"] - 1) +
            response.response_time
        ) / self.metrics["queries_processed"]
        
        # Update average confidence
        self.metrics["average_confidence"] = (
            self.metrics["average_confidence"] * (self.metrics["queries_processed"] - 1) +
            response.confidence
        ) / self.metrics["queries_processed"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.metrics.copy()
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()

def main():
    """Test the enhanced RAG engine"""
    engine = EnhancedRAGEngineV2()
    
    # Test query
    query = EnhancedRAGQuery(
        query="Why is PCIe link training failing?",
        error_log="[10:15:30] PCIe: Link training failed on device 0000:01:00.0",
        analysis_type="detailed",
        compress_context=True
    )
    
    # Execute query
    response = engine.query(query)
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    print(f"Model: {response.model_used}")
    print(f"Response time: {response.response_time:.2f}s")
    
    if response.compression_info:
        print(f"Compression ratio: {response.compression_info.compression_ratio:.2f}")
        print(f"Relevance score: {response.compression_info.relevance_score:.2f}")

if __name__ == "__main__":
    main()