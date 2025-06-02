"""
Unified RAG Engine - Intelligently combines all RAG methods
Uses adaptive query routing, multi-stage processing, and result fusion
"""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery, RAGResponse
from src.rag.metadata_enhanced_rag import MetadataEnhancedRAGEngine, MetadataRAGQuery
from src.rag.hybrid_search import HybridSearchEngine
from src.vectorstore.faiss_store import FAISSVectorStore
from src.models.model_manager import ModelManager

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for adaptive routing"""
    TECHNICAL_SPECIFIC = "technical_specific"  # "PCIe 4.0 Gen4 x16 error 0x1234"
    CONCEPTUAL = "conceptual"                  # "How does link training work?"
    TROUBLESHOOTING = "troubleshooting"        # "Debug LTSSM timeout issues"
    EXPLORATORY = "exploratory"               # "PCIe overview"
    REAL_TIME = "real_time"                   # Fast monitoring queries
    COMPARATIVE = "comparative"               # "PCIe 3.0 vs 4.0 differences"


class ProcessingStrategy(Enum):
    """Processing strategies for different scenarios"""
    FAST_ONLY = "fast_only"                  # Keyword only
    BALANCED = "balanced"                     # Hybrid approach
    COMPREHENSIVE = "comprehensive"           # All methods
    ADAPTIVE = "adaptive"                     # Smart selection
    CASCADING = "cascading"                   # Progressive enhancement


@dataclass
class UnifiedRAGQuery:
    """Unified query supporting all RAG methods"""
    query: str
    
    # Strategy configuration
    strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    max_response_time: Optional[float] = None  # Time budget in seconds
    min_confidence: float = 0.7
    
    # Method weights (will be auto-adjusted if None)
    keyword_weight: Optional[float] = None
    semantic_weight: Optional[float] = None
    hybrid_weight: Optional[float] = None
    metadata_weight: Optional[float] = None
    
    # Context
    context_window: int = 5
    user_expertise: str = "intermediate"  # beginner, intermediate, expert
    priority: str = "accuracy"            # speed, accuracy, balance
    
    # Metadata filters (optional)
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class MethodResult:
    """Result from a single RAG method"""
    method: str
    documents: List[Tuple[str, Dict[str, Any], float]]
    processing_time: float
    confidence: float
    method_specific_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedRAGResponse:
    """Enhanced response with method fusion details"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    
    # Method contribution details
    methods_used: List[str]
    method_contributions: Dict[str, float]
    processing_breakdown: Dict[str, float]
    
    # Quality metrics
    result_diversity: float
    consensus_score: float
    total_processing_time: float
    
    # Recommendations
    query_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """Analyzes queries to determine optimal processing strategy"""
    
    def __init__(self):
        # Technical term patterns
        self.technical_patterns = [
            r'PCIe?\s*[1-6]\.[0-9]',  # PCIe versions
            r'Gen[1-6]',              # Generation specs
            r'x[1-9][0-9]*',          # Lane widths
            r'0x[0-9A-Fa-f]+',        # Error codes
            r'GT/s',                  # Speed specs
            r'LTSSM',                 # Technical acronyms
            r'TLP|DLLP|PLP',         # Protocol layers
        ]
        
        # Conceptual indicators
        self.conceptual_indicators = [
            'how', 'what', 'why', 'explain', 'understand', 'concept',
            'overview', 'introduction', 'basics', 'fundamentals'
        ]
        
        # Troubleshooting indicators
        self.troubleshooting_indicators = [
            'debug', 'fix', 'solve', 'problem', 'issue', 'error', 'fail',
            'troubleshoot', 'diagnose', 'repair', 'resolve'
        ]
    
    def analyze_query(self, query: str) -> Tuple[QueryType, Dict[str, Any]]:
        """Analyze query to determine type and characteristics"""
        query_lower = query.lower()
        analysis = {
            'technical_terms': 0,
            'conceptual_score': 0,
            'troubleshooting_score': 0,
            'complexity': 'medium',
            'specificity': 'medium'
        }
        
        # Count technical patterns
        import re
        for pattern in self.technical_patterns:
            matches = len(re.findall(pattern, query, re.IGNORECASE))
            analysis['technical_terms'] += matches
        
        # Score conceptual indicators
        for indicator in self.conceptual_indicators:
            if indicator in query_lower:
                analysis['conceptual_score'] += 1
        
        # Score troubleshooting indicators
        for indicator in self.troubleshooting_indicators:
            if indicator in query_lower:
                analysis['troubleshooting_score'] += 1
        
        # Determine query type
        if analysis['technical_terms'] >= 3:
            query_type = QueryType.TECHNICAL_SPECIFIC
            analysis['specificity'] = 'high'
        elif analysis['troubleshooting_score'] >= 2:
            query_type = QueryType.TROUBLESHOOTING
        elif analysis['conceptual_score'] >= 2:
            query_type = QueryType.CONCEPTUAL
        elif 'vs' in query_lower or 'compare' in query_lower:
            query_type = QueryType.COMPARATIVE
        elif len(query.split()) <= 3:
            query_type = QueryType.REAL_TIME
        else:
            query_type = QueryType.EXPLORATORY
        
        # Determine complexity
        word_count = len(query.split())
        if word_count <= 3:
            analysis['complexity'] = 'low'
        elif word_count >= 8:
            analysis['complexity'] = 'high'
        
        return query_type, analysis


class ResultFusion:
    """Fuses results from multiple RAG methods"""
    
    def __init__(self):
        self.similarity_threshold = 0.8
        
    def fuse_results(self, 
                    method_results: List[MethodResult],
                    method_weights: Dict[str, float]) -> List[Tuple[str, Dict[str, Any], float]]:
        """Fuse results from multiple methods using weighted ranking"""
        
        # Collect all unique documents
        all_documents = {}
        method_scores = {}
        
        for result in method_results:
            method = result.method
            weight = method_weights.get(method, 0.25)
            
            for doc, metadata, score in result.documents:
                doc_key = self._get_document_key(doc, metadata)
                
                if doc_key not in all_documents:
                    all_documents[doc_key] = (doc, metadata)
                    method_scores[doc_key] = {}
                
                # Store weighted score for this method
                method_scores[doc_key][method] = score * weight
        
        # Calculate final scores
        final_results = []
        for doc_key, (doc, metadata) in all_documents.items():
            scores = method_scores[doc_key]
            
            # Weighted average of available method scores
            total_weight = sum(method_weights.get(method, 0.25) 
                             for method in scores.keys())
            
            if total_weight > 0:
                final_score = sum(scores.values()) / total_weight
                
                # Boost documents found by multiple methods
                method_consensus = len(scores) / len(method_results)
                final_score *= (0.8 + 0.4 * method_consensus)  # Max 20% boost
                
                # Add method contribution info to metadata
                metadata['method_contributions'] = scores
                metadata['consensus_score'] = method_consensus
                
                final_results.append((doc, metadata, final_score))
        
        # Sort by final score
        final_results.sort(key=lambda x: x[2], reverse=True)
        return final_results
    
    def _get_document_key(self, doc: str, metadata: Dict[str, Any]) -> str:
        """Generate unique key for document deduplication"""
        # Use document hash or source identifier
        source = metadata.get('source', '')
        return f"{hash(doc[:100])}_{source}"
    
    def calculate_diversity(self, results: List[Tuple[str, Dict[str, Any], float]]) -> float:
        """Calculate result diversity score"""
        if len(results) <= 1:
            return 0.0
        
        # Simple diversity based on source variety
        sources = set()
        doc_types = set()
        
        for _, metadata, _ in results:
            sources.add(metadata.get('source', 'unknown'))
            doc_types.add(metadata.get('document_type', 'unknown'))
        
        source_diversity = len(sources) / len(results)
        type_diversity = len(doc_types) / len(results)
        
        return (source_diversity + type_diversity) / 2


class UnifiedRAGEngine:
    """Unified RAG engine combining all methods intelligently"""
    
    def __init__(self,
                 vector_store: FAISSVectorStore,
                 model_manager: ModelManager,
                 **kwargs):
        
        self.vector_store = vector_store
        self.model_manager = model_manager
        
        # Initialize individual engines
        self.enhanced_rag = EnhancedRAGEngine(vector_store, model_manager, **kwargs)
        self.metadata_rag = MetadataEnhancedRAGEngine(vector_store, model_manager, **kwargs)
        self.hybrid_search = HybridSearchEngine(vector_store)
        
        # Initialize analyzers
        self.query_analyzer = QueryAnalyzer()
        self.result_fusion = ResultFusion()
        
        # Performance tracking
        self.metrics = {
            "queries_processed": 0,
            "avg_processing_time": 0,
            "method_usage_stats": {},
            "strategy_effectiveness": {}
        }
    
    async def query(self, unified_query: UnifiedRAGQuery) -> UnifiedRAGResponse:
        """Process unified query with intelligent method combination"""
        start_time = time.time()
        
        try:
            # Analyze query
            query_type, analysis = self.query_analyzer.analyze_query(unified_query.query)
            
            # Determine processing strategy
            strategy = self._determine_strategy(unified_query, query_type, analysis)
            
            # Execute strategy
            method_results = await self._execute_strategy(strategy, unified_query, query_type)
            
            # Determine method weights
            method_weights = self._calculate_method_weights(unified_query, query_type, analysis)
            
            # Fuse results
            fused_results = self.result_fusion.fuse_results(method_results, method_weights)
            
            # Generate final response
            response = await self._generate_unified_response(
                unified_query, fused_results, method_results, method_weights, query_type
            )
            
            # Update metrics
            self._update_metrics(strategy, time.time() - start_time, method_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Unified RAG query failed: {str(e)}")
            # Fallback to basic semantic search
            return await self._fallback_query(unified_query)
    
    def _determine_strategy(self,
                          unified_query: UnifiedRAGQuery,
                          query_type: QueryType,
                          analysis: Dict[str, Any]) -> ProcessingStrategy:
        """Determine optimal processing strategy"""
        
        # User specified strategy
        if unified_query.strategy != ProcessingStrategy.ADAPTIVE:
            return unified_query.strategy
        
        # Time budget constraints
        if unified_query.max_response_time and unified_query.max_response_time < 1.0:
            return ProcessingStrategy.FAST_ONLY
        
        # Query type based routing
        strategy_map = {
            QueryType.REAL_TIME: ProcessingStrategy.FAST_ONLY,
            QueryType.TECHNICAL_SPECIFIC: ProcessingStrategy.COMPREHENSIVE,
            QueryType.TROUBLESHOOTING: ProcessingStrategy.COMPREHENSIVE,
            QueryType.CONCEPTUAL: ProcessingStrategy.BALANCED,
            QueryType.EXPLORATORY: ProcessingStrategy.BALANCED,
            QueryType.COMPARATIVE: ProcessingStrategy.COMPREHENSIVE
        }
        
        base_strategy = strategy_map.get(query_type, ProcessingStrategy.BALANCED)
        
        # Adjust based on priority
        if unified_query.priority == "speed":
            if base_strategy == ProcessingStrategy.COMPREHENSIVE:
                return ProcessingStrategy.BALANCED
            elif base_strategy == ProcessingStrategy.BALANCED:
                return ProcessingStrategy.FAST_ONLY
        elif unified_query.priority == "accuracy":
            if base_strategy == ProcessingStrategy.FAST_ONLY:
                return ProcessingStrategy.BALANCED
            elif base_strategy == ProcessingStrategy.BALANCED:
                return ProcessingStrategy.COMPREHENSIVE
        
        return base_strategy
    
    async def _execute_strategy(self,
                              strategy: ProcessingStrategy,
                              unified_query: UnifiedRAGQuery,
                              query_type: QueryType) -> List[MethodResult]:
        """Execute the determined strategy"""
        
        method_results = []
        
        if strategy == ProcessingStrategy.FAST_ONLY:
            # Only keyword search
            result = await self._run_keyword_search(unified_query)
            method_results.append(result)
            
        elif strategy == ProcessingStrategy.BALANCED:
            # Hybrid + semantic
            tasks = [
                self._run_hybrid_search(unified_query),
                self._run_semantic_search(unified_query)
            ]
            results = await asyncio.gather(*tasks)
            method_results.extend(results)
            
        elif strategy == ProcessingStrategy.COMPREHENSIVE:
            # All methods
            tasks = [
                self._run_keyword_search(unified_query),
                self._run_semantic_search(unified_query),
                self._run_hybrid_search(unified_query),
                self._run_metadata_search(unified_query)
            ]
            results = await asyncio.gather(*tasks)
            method_results.extend(results)
            
        elif strategy == ProcessingStrategy.CASCADING:
            # Progressive enhancement
            method_results = await self._run_cascading_search(unified_query)
        
        return method_results
    
    async def _run_keyword_search(self, query: UnifiedRAGQuery) -> MethodResult:
        """Run keyword-based search"""
        start_time = time.time()
        
        try:
            # Use BM25 search from hybrid engine
            results = self.hybrid_search.search_bm25(query.query, k=query.context_window)
            
            return MethodResult(
                method="keyword",
                documents=results,
                processing_time=time.time() - start_time,
                confidence=0.8,  # Keyword search confidence
                method_specific_metrics={"exact_matches": len(results)}
            )
        except Exception as e:
            logger.error(f"Keyword search failed: {str(e)}")
            return MethodResult("keyword", [], time.time() - start_time, 0.0)
    
    async def _run_semantic_search(self, query: UnifiedRAGQuery) -> MethodResult:
        """Run semantic search"""
        start_time = time.time()
        
        try:
            rag_query = RAGQuery(
                query=query.query,
                context_window=query.context_window,
                min_similarity=0.1
            )
            
            # Use enhanced RAG for semantic search
            embedding = self.model_manager.embed([query.query])[0]
            results = self.vector_store.search(embedding, k=query.context_window)
            
            return MethodResult(
                method="semantic",
                documents=results,
                processing_time=time.time() - start_time,
                confidence=0.75,
                method_specific_metrics={"avg_similarity": np.mean([s for _, _, s in results]) if results else 0}
            )
        except Exception as e:
            logger.error(f"Semantic search failed: {str(e)}")
            return MethodResult("semantic", [], time.time() - start_time, 0.0)
    
    async def _run_hybrid_search(self, query: UnifiedRAGQuery) -> MethodResult:
        """Run hybrid search"""
        start_time = time.time()
        
        try:
            results = self.hybrid_search.search(query.query, k=query.context_window)
            
            return MethodResult(
                method="hybrid",
                documents=results,
                processing_time=time.time() - start_time,
                confidence=0.85,
                method_specific_metrics={"hybrid_score": "balanced"}
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            return MethodResult("hybrid", [], time.time() - start_time, 0.0)
    
    async def _run_metadata_search(self, query: UnifiedRAGQuery) -> MethodResult:
        """Run metadata-enhanced search"""
        start_time = time.time()
        
        try:
            metadata_query = MetadataRAGQuery(
                query=query.query,
                context_window=query.context_window,
                **query.metadata_filters if query.metadata_filters else {}
            )
            
            response = self.metadata_rag.query_with_metadata(metadata_query)
            
            # Convert response to method result format
            documents = [(src.get('content', ''), src.get('metadata', {}), src.get('relevance_score', 0.0))
                        for src in response.sources]
            
            return MethodResult(
                method="metadata",
                documents=documents,
                processing_time=time.time() - start_time,
                confidence=response.confidence,
                method_specific_metrics={"metadata_filtered": True}
            )
        except Exception as e:
            logger.error(f"Metadata search failed: {str(e)}")
            return MethodResult("metadata", [], time.time() - start_time, 0.0)
    
    async def _run_cascading_search(self, query: UnifiedRAGQuery) -> List[MethodResult]:
        """Run cascading search with progressive enhancement"""
        results = []
        
        # Start with fastest method
        keyword_result = await self._run_keyword_search(query)
        results.append(keyword_result)
        
        # If keyword results are insufficient, add semantic
        if keyword_result.confidence < query.min_confidence:
            semantic_result = await self._run_semantic_search(query)
            results.append(semantic_result)
            
            # If still insufficient, add metadata
            if semantic_result.confidence < query.min_confidence:
                metadata_result = await self._run_metadata_search(query)
                results.append(metadata_result)
        
        return results
    
    def _calculate_method_weights(self,
                                unified_query: UnifiedRAGQuery,
                                query_type: QueryType,
                                analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights for different methods based on query characteristics"""
        
        # Default weights
        weights = {
            "keyword": 0.2,
            "semantic": 0.25,
            "hybrid": 0.3,
            "metadata": 0.25
        }
        
        # Adjust based on query type
        if query_type == QueryType.TECHNICAL_SPECIFIC:
            weights.update({"keyword": 0.3, "metadata": 0.4, "semantic": 0.15, "hybrid": 0.15})
        elif query_type == QueryType.CONCEPTUAL:
            weights.update({"semantic": 0.4, "hybrid": 0.3, "metadata": 0.2, "keyword": 0.1})
        elif query_type == QueryType.TROUBLESHOOTING:
            weights.update({"metadata": 0.35, "hybrid": 0.3, "semantic": 0.2, "keyword": 0.15})
        elif query_type == QueryType.REAL_TIME:
            weights.update({"keyword": 0.6, "hybrid": 0.25, "semantic": 0.1, "metadata": 0.05})
        
        # Apply user-specified weights if provided
        if unified_query.keyword_weight is not None:
            weights["keyword"] = unified_query.keyword_weight
        if unified_query.semantic_weight is not None:
            weights["semantic"] = unified_query.semantic_weight
        if unified_query.hybrid_weight is not None:
            weights["hybrid"] = unified_query.hybrid_weight
        if unified_query.metadata_weight is not None:
            weights["metadata"] = unified_query.metadata_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    async def _generate_unified_response(self,
                                       query: UnifiedRAGQuery,
                                       fused_results: List[Tuple[str, Dict[str, Any], float]],
                                       method_results: List[MethodResult],
                                       method_weights: Dict[str, float],
                                       query_type: QueryType) -> UnifiedRAGResponse:
        """Generate unified response from fused results"""
        
        # Prepare context for LLM
        context_parts = []
        sources = []
        
        for i, (doc, metadata, score) in enumerate(fused_results[:query.context_window]):
            context_parts.append(f"[Document {i+1}]")
            context_parts.append(f"Content: {doc[:500]}...")
            context_parts.append("")
            
            # Enhanced source information
            source_info = {
                "document_id": i + 1,
                "content_preview": doc[:200],
                "metadata": metadata,
                "final_score": float(score),
                "method_contributions": metadata.get('method_contributions', {}),
                "consensus_score": metadata.get('consensus_score', 0.0)
            }
            sources.append(source_info)
        
        context = "\n".join(context_parts)
        
        # Generate answer using the enhanced RAG analyzer
        answer = self.enhanced_rag.analyzer.analyze(
            query=query.query,
            context=context,
            analysis_type="comprehensive_answer"
        )
        
        # Calculate metrics
        methods_used = [result.method for result in method_results if result.documents]
        processing_breakdown = {result.method: result.processing_time for result in method_results}
        total_time = sum(processing_breakdown.values())
        
        # Calculate method contributions
        method_contributions = {}
        for method in methods_used:
            method_contributions[method] = method_weights.get(method, 0.0)
        
        # Calculate result diversity
        diversity = self.result_fusion.calculate_diversity(fused_results)
        
        # Calculate consensus score
        consensus_scores = [metadata.get('consensus_score', 0.0) 
                          for _, metadata, _ in fused_results[:5]]
        avg_consensus = np.mean(consensus_scores) if consensus_scores else 0.0
        
        # Calculate overall confidence
        if fused_results:
            top_scores = [score for _, _, score in fused_results[:3]]
            confidence = np.mean(top_scores)
        else:
            confidence = 0.0
        
        return UnifiedRAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            methods_used=methods_used,
            method_contributions=method_contributions,
            processing_breakdown=processing_breakdown,
            result_diversity=diversity,
            consensus_score=avg_consensus,
            total_processing_time=total_time,
            metadata={
                "query_type": query_type.value,
                "strategy_used": "adaptive",
                "total_documents_considered": sum(len(r.documents) for r in method_results)
            }
        )
    
    async def _fallback_query(self, query: UnifiedRAGQuery) -> UnifiedRAGResponse:
        """Fallback to basic semantic search if unified approach fails"""
        try:
            semantic_result = await self._run_semantic_search(query)
            
            return UnifiedRAGResponse(
                answer="Fallback response due to processing error.",
                sources=[],
                confidence=0.3,
                methods_used=["semantic_fallback"],
                method_contributions={"semantic": 1.0},
                processing_breakdown={"semantic": semantic_result.processing_time},
                result_diversity=0.0,
                consensus_score=0.0,
                total_processing_time=semantic_result.processing_time,
                metadata={"fallback": True}
            )
        except Exception as e:
            logger.error(f"Fallback query also failed: {str(e)}")
            return UnifiedRAGResponse(
                answer="Unable to process query due to system error.",
                sources=[],
                confidence=0.0,
                methods_used=[],
                method_contributions={},
                processing_breakdown={},
                result_diversity=0.0,
                consensus_score=0.0,
                total_processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def _update_metrics(self, 
                       strategy: ProcessingStrategy,
                       processing_time: float,
                       method_results: List[MethodResult]):
        """Update performance metrics"""
        self.metrics["queries_processed"] += 1
        
        # Update average processing time
        prev_avg = self.metrics["avg_processing_time"]
        count = self.metrics["queries_processed"]
        self.metrics["avg_processing_time"] = (prev_avg * (count - 1) + processing_time) / count
        
        # Update method usage stats
        for result in method_results:
            method = result.method
            if method not in self.metrics["method_usage_stats"]:
                self.metrics["method_usage_stats"][method] = {"count": 0, "avg_time": 0}
            
            stats = self.metrics["method_usage_stats"][method]
            stats["count"] += 1
            prev_avg = stats["avg_time"]
            stats["avg_time"] = (prev_avg * (stats["count"] - 1) + result.processing_time) / stats["count"]
        
        # Update strategy effectiveness
        strategy_key = strategy.value if hasattr(strategy, 'value') else str(strategy)
        if strategy_key not in self.metrics["strategy_effectiveness"]:
            self.metrics["strategy_effectiveness"][strategy_key] = {
                "count": 0, "avg_time": 0, "avg_methods": 0
            }
        
        strategy_stats = self.metrics["strategy_effectiveness"][strategy_key]
        strategy_stats["count"] += 1
        prev_avg_time = strategy_stats["avg_time"]
        strategy_stats["avg_time"] = (prev_avg_time * (strategy_stats["count"] - 1) + processing_time) / strategy_stats["count"]
        
        prev_avg_methods = strategy_stats["avg_methods"]
        method_count = len(method_results)
        strategy_stats["avg_methods"] = (prev_avg_methods * (strategy_stats["count"] - 1) + method_count) / strategy_stats["count"]