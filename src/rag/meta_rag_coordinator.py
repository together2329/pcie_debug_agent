#!/usr/bin/env python3
"""
Meta-RAG Coordination System for Phase 3 Intelligence Layer

Orchestrates multiple RAG engines, provides intelligent routing,
and coordinates between different specialized RAG components.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from collections import defaultdict, deque

from .enhanced_rag_engine import EnhancedRAGEngine
from .pcie_knowledge_classifier import PCIeCategory, QueryIntent
from .specialized_routing import SpecializedRouter, ProcessingTier
from .model_ensemble import ModelEnsemble
from .quality_monitor import ContinuousQualityMonitor, QualityMetric

logger = logging.getLogger(__name__)

class RAGEngineType(Enum):
    """Types of RAG engines"""
    PRODUCTION = "production"
    ENHANCED_V2 = "enhanced_v2"
    ENHANCED_V3 = "enhanced_v3"
    STANDARD = "standard"
    SPECIALIZED = "specialized"

class CoordinationStrategy(Enum):
    """Meta-RAG coordination strategies"""
    SINGLE_BEST = "single_best"           # Route to single best engine
    PARALLEL_ENSEMBLE = "parallel_ensemble"  # Run multiple engines in parallel
    SEQUENTIAL_FALLBACK = "sequential_fallback"  # Try engines in sequence
    ADAPTIVE_ROUTING = "adaptive_routing"     # Adapt based on performance
    QUALITY_OPTIMIZED = "quality_optimized"  # Optimize for quality metrics

@dataclass
class RAGEngineConfig:
    """Configuration for a RAG engine"""
    engine_type: RAGEngineType
    engine_instance: Any
    priority: int  # Higher number = higher priority
    strengths: List[str]
    weaknesses: List[str]
    optimal_categories: List[PCIeCategory]
    optimal_intents: List[QueryIntent]
    max_concurrent: int = 1
    timeout_seconds: float = 30.0
    enabled: bool = True
    
@dataclass
class RoutingDecision:
    """Meta-RAG routing decision"""
    selected_engines: List[RAGEngineType]
    strategy: CoordinationStrategy
    reasoning: str
    estimated_time: float
    confidence_target: float
    fallback_engines: List[RAGEngineType]

@dataclass
class QueryExecution:
    """Query execution tracking"""
    query_id: str
    query: str
    category: PCIeCategory
    intent: QueryIntent
    routing_decision: RoutingDecision
    start_time: datetime
    end_time: Optional[datetime] = None
    results: Dict[RAGEngineType, Any] = None
    final_result: Any = None
    execution_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None

class MetaRAGCoordinator:
    """Coordinates multiple RAG engines with intelligent routing"""
    
    def __init__(self, quality_monitor: ContinuousQualityMonitor = None):
        self.quality_monitor = quality_monitor or ContinuousQualityMonitor()
        
        # Engine registry
        self.engines: Dict[RAGEngineType, RAGEngineConfig] = {}
        self.specialized_router = None
        self.model_ensemble = None
        
        # Performance tracking
        self.engine_performance = defaultdict(lambda: {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_response_time': 0.0,
            'avg_confidence': 0.0,
            'recent_scores': deque(maxlen=100)
        })
        
        # Coordination strategy
        self.default_strategy = CoordinationStrategy.ADAPTIVE_ROUTING
        self.adaptive_thresholds = {
            'confidence_threshold': 0.7,
            'response_time_threshold': 5.0,
            'success_rate_threshold': 0.9
        }
        
        # Execution tracking
        self.active_executions: Dict[str, QueryExecution] = {}
        self.execution_history = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.Lock()
        
    def register_engine(self, engine_type: RAGEngineType, engine_instance: Any,
                       priority: int = 1, strengths: List[str] = None,
                       weaknesses: List[str] = None,
                       optimal_categories: List[PCIeCategory] = None,
                       optimal_intents: List[QueryIntent] = None,
                       max_concurrent: int = 1, timeout_seconds: float = 30.0):
        """Register a RAG engine with the coordinator"""
        
        config = RAGEngineConfig(
            engine_type=engine_type,
            engine_instance=engine_instance,
            priority=priority,
            strengths=strengths or [],
            weaknesses=weaknesses or [],
            optimal_categories=optimal_categories or [],
            optimal_intents=optimal_intents or [],
            max_concurrent=max_concurrent,
            timeout_seconds=timeout_seconds
        )
        
        with self._lock:
            self.engines[engine_type] = config
            
        logger.info(f"Registered RAG engine: {engine_type.value} (priority: {priority})")
    
    def set_specialized_components(self, router: SpecializedRouter, 
                                 ensemble: ModelEnsemble):
        """Set specialized routing and ensemble components"""
        self.specialized_router = router
        self.model_ensemble = ensemble
    
    def execute_query(self, query: str, session_id: str = None,
                     preferred_strategy: CoordinationStrategy = None,
                     context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute query using meta-RAG coordination"""
        
        query_id = f"meta_query_{int(time.time() * 1000)}"
        start_time = datetime.now()
        
        try:
            # Analyze query characteristics
            category, intent = self._analyze_query(query)
            
            # Make routing decision
            routing_decision = self._make_routing_decision(
                query, category, intent, preferred_strategy
            )
            
            # Create execution tracking
            execution = QueryExecution(
                query_id=query_id,
                query=query,
                category=category,
                intent=intent,
                routing_decision=routing_decision,
                start_time=start_time,
                results={}
            )
            
            with self._lock:
                self.active_executions[query_id] = execution
            
            # Execute based on strategy
            result = self._execute_with_strategy(execution, context)
            
            # Record successful completion
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            execution.end_time = end_time
            execution.execution_time = execution_time
            execution.final_result = result
            execution.success = True
            
            # Update performance metrics
            self._update_engine_performance(execution)
            
            # Record quality metrics
            if self.quality_monitor:
                confidence = result.get('confidence', 0.0)
                self.quality_monitor.record_query_execution(
                    execution_time, confidence, True, context, session_id, query_id
                )
            
            # Clean up
            with self._lock:
                if query_id in self.active_executions:
                    del self.active_executions[query_id]
                self.execution_history.append(execution)
            
            # Add coordination metadata
            result['meta_coordination'] = {
                'query_id': query_id,
                'strategy': routing_decision.strategy.value,
                'engines_used': [e.value for e in routing_decision.selected_engines],
                'execution_time': execution_time,
                'routing_reasoning': routing_decision.reasoning
            }
            
            return result
            
        except Exception as e:
            # Record failure
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            if 'execution' in locals():
                execution.end_time = end_time
                execution.execution_time = execution_time
                execution.success = False
                execution.error_message = str(e)
                
                # Clean up
                with self._lock:
                    if query_id in self.active_executions:
                        del self.active_executions[query_id]
                    self.execution_history.append(execution)
            
            # Record quality metrics for failure
            if self.quality_monitor:
                self.quality_monitor.record_query_execution(
                    execution_time, 0.0, False, context, session_id, query_id
                )
            
            logger.error(f"Meta-RAG execution failed for query {query_id}: {e}")
            
            return {
                'error': str(e),
                'query_id': query_id,
                'execution_time': execution_time,
                'meta_coordination': {
                    'strategy': 'failed',
                    'engines_used': [],
                    'error': str(e)
                }
            }
    
    def _analyze_query(self, query: str) -> Tuple[PCIeCategory, QueryIntent]:
        """Analyze query to determine category and intent"""
        if self.specialized_router:
            # Use specialized router's query expansion
            expanded = self.specialized_router.query_expander.expand_query(query)
            return expanded.category, expanded.intent
        else:
            # Fallback to basic analysis
            from .pcie_knowledge_classifier import PCIeKnowledgeClassifier
            classifier = PCIeKnowledgeClassifier()
            category = classifier._categorize_content(query)
            intent = classifier.classify_query_intent(query)
            return category, intent
    
    def _make_routing_decision(self, query: str, category: PCIeCategory, 
                             intent: QueryIntent, 
                             preferred_strategy: CoordinationStrategy = None) -> RoutingDecision:
        """Make intelligent routing decision"""
        
        strategy = preferred_strategy or self._determine_optimal_strategy(query, category, intent)
        
        # Select engines based on strategy and performance
        selected_engines = self._select_engines_for_strategy(strategy, category, intent)
        
        # Determine fallback engines
        fallback_engines = self._determine_fallback_engines(selected_engines, category, intent)
        
        # Estimate execution parameters
        estimated_time = self._estimate_execution_time(selected_engines, strategy)
        confidence_target = self._estimate_confidence_target(selected_engines, category, intent)
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(strategy, selected_engines, category, intent)
        
        return RoutingDecision(
            selected_engines=selected_engines,
            strategy=strategy,
            reasoning=reasoning,
            estimated_time=estimated_time,
            confidence_target=confidence_target,
            fallback_engines=fallback_engines
        )
    
    def _determine_optimal_strategy(self, query: str, category: PCIeCategory, 
                                  intent: QueryIntent) -> CoordinationStrategy:
        """Determine optimal coordination strategy"""
        
        # High-priority queries benefit from parallel processing
        if intent in [QueryIntent.TROUBLESHOOT, QueryIntent.VERIFY] and category == PCIeCategory.ERROR_HANDLING:
            return CoordinationStrategy.PARALLEL_ENSEMBLE
        
        # Simple reference queries can use single best engine
        if intent == QueryIntent.REFERENCE and len(query.split()) < 10:
            return CoordinationStrategy.SINGLE_BEST
        
        # Complex technical queries benefit from adaptive routing
        if category in [PCIeCategory.COMPLIANCE, PCIeCategory.DEBUGGING] and len(query.split()) > 15:
            return CoordinationStrategy.ADAPTIVE_ROUTING
        
        # Default to adaptive routing
        return CoordinationStrategy.ADAPTIVE_ROUTING
    
    def _select_engines_for_strategy(self, strategy: CoordinationStrategy,
                                   category: PCIeCategory, intent: QueryIntent) -> List[RAGEngineType]:
        """Select engines based on strategy and query characteristics"""
        
        available_engines = [e for e, config in self.engines.items() if config.enabled]
        
        if strategy == CoordinationStrategy.SINGLE_BEST:
            # Select single best engine based on performance and suitability
            best_engine = self._find_best_engine(category, intent)
            return [best_engine] if best_engine else available_engines[:1]
        
        elif strategy == CoordinationStrategy.PARALLEL_ENSEMBLE:
            # Select top 2-3 engines for parallel execution
            ranked_engines = self._rank_engines_by_suitability(category, intent)
            return ranked_engines[:3]
        
        elif strategy == CoordinationStrategy.SEQUENTIAL_FALLBACK:
            # Return engines in order of preference
            return self._rank_engines_by_suitability(category, intent)
        
        elif strategy == CoordinationStrategy.ADAPTIVE_ROUTING:
            # Select based on current performance metrics
            return self._adaptive_engine_selection(category, intent)
        
        elif strategy == CoordinationStrategy.QUALITY_OPTIMIZED:
            # Select engines optimized for quality
            return self._quality_optimized_selection(category, intent)
        
        return available_engines[:1]  # Fallback
    
    def _find_best_engine(self, category: PCIeCategory, intent: QueryIntent) -> Optional[RAGEngineType]:
        """Find the single best engine for given category/intent"""
        best_engine = None
        best_score = -1
        
        for engine_type, config in self.engines.items():
            if not config.enabled:
                continue
            
            score = self._calculate_engine_suitability_score(config, category, intent)
            performance = self.engine_performance[engine_type]
            
            # Combine suitability with recent performance
            if performance['total_queries'] > 0:
                success_rate = performance['successful_queries'] / performance['total_queries']
                avg_confidence = performance['avg_confidence']
                response_time_factor = max(0, 1 - performance['avg_response_time'] / 10.0)
                
                performance_score = (success_rate + avg_confidence + response_time_factor) / 3
                total_score = score * 0.6 + performance_score * 0.4
            else:
                total_score = score
            
            if total_score > best_score:
                best_score = total_score
                best_engine = engine_type
        
        return best_engine
    
    def _rank_engines_by_suitability(self, category: PCIeCategory, 
                                   intent: QueryIntent) -> List[RAGEngineType]:
        """Rank engines by suitability for category/intent"""
        engine_scores = []
        
        for engine_type, config in self.engines.items():
            if not config.enabled:
                continue
            
            score = self._calculate_engine_suitability_score(config, category, intent)
            engine_scores.append((engine_type, score))
        
        # Sort by score (descending)
        engine_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [engine_type for engine_type, _ in engine_scores]
    
    def _calculate_engine_suitability_score(self, config: RAGEngineConfig,
                                          category: PCIeCategory, intent: QueryIntent) -> float:
        """Calculate suitability score for engine given category/intent"""
        score = config.priority  # Base score from priority
        
        # Category suitability bonus
        if category in config.optimal_categories:
            score += 2.0
        
        # Intent suitability bonus
        if intent in config.optimal_intents:
            score += 1.5
        
        # Strength/weakness adjustments
        category_strength_map = {
            PCIeCategory.ERROR_HANDLING: ['error analysis', 'troubleshooting', 'debugging'],
            PCIeCategory.COMPLIANCE: ['compliance checking', 'specification', 'validation'],
            PCIeCategory.DEBUGGING: ['debugging', 'analysis', 'diagnostics'],
            PCIeCategory.PERFORMANCE: ['optimization', 'performance', 'efficiency']
        }
        
        intent_strength_map = {
            QueryIntent.TROUBLESHOOT: ['troubleshooting', 'debugging', 'problem solving'],
            QueryIntent.VERIFY: ['compliance', 'validation', 'verification'],
            QueryIntent.IMPLEMENT: ['implementation', 'configuration', 'setup'],
            QueryIntent.OPTIMIZE: ['optimization', 'performance', 'tuning']
        }
        
        # Check if engine strengths match category/intent requirements
        relevant_strengths = (category_strength_map.get(category, []) + 
                            intent_strength_map.get(intent, []))
        
        for strength in config.strengths:
            if any(req in strength.lower() for req in relevant_strengths):
                score += 0.5
        
        # Penalty for weaknesses in relevant areas
        for weakness in config.weaknesses:
            if any(req in weakness.lower() for req in relevant_strengths):
                score -= 0.3
        
        return max(0, score)
    
    def _adaptive_engine_selection(self, category: PCIeCategory, 
                                 intent: QueryIntent) -> List[RAGEngineType]:
        """Adaptively select engines based on recent performance"""
        
        # Get performance-ranked engines
        performance_ranked = self._rank_engines_by_performance()
        
        # Get suitability-ranked engines
        suitability_ranked = self._rank_engines_by_suitability(category, intent)
        
        # Combine rankings with weighted average
        combined_scores = {}
        
        for i, engine in enumerate(performance_ranked):
            combined_scores[engine] = combined_scores.get(engine, 0) + (len(performance_ranked) - i) * 0.4
        
        for i, engine in enumerate(suitability_ranked):
            combined_scores[engine] = combined_scores.get(engine, 0) + (len(suitability_ranked) - i) * 0.6
        
        # Sort by combined score
        sorted_engines = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 2 engines for adaptive strategy
        return [engine for engine, _ in sorted_engines[:2]]
    
    def _rank_engines_by_performance(self) -> List[RAGEngineType]:
        """Rank engines by recent performance"""
        engine_scores = []
        
        for engine_type, performance in self.engine_performance.items():
            if engine_type not in self.engines or not self.engines[engine_type].enabled:
                continue
            
            if performance['total_queries'] == 0:
                score = 0.5  # Neutral score for untested engines
            else:
                success_rate = performance['successful_queries'] / performance['total_queries']
                confidence_score = performance['avg_confidence']
                speed_score = max(0, 1 - performance['avg_response_time'] / 10.0)
                
                # Recent performance trend
                if performance['recent_scores']:
                    recent_trend = sum(performance['recent_scores'][-10:]) / min(10, len(performance['recent_scores']))
                else:
                    recent_trend = 0.5
                
                score = (success_rate + confidence_score + speed_score + recent_trend) / 4
            
            engine_scores.append((engine_type, score))
        
        engine_scores.sort(key=lambda x: x[1], reverse=True)
        return [engine_type for engine_type, _ in engine_scores]
    
    def _quality_optimized_selection(self, category: PCIeCategory, 
                                   intent: QueryIntent) -> List[RAGEngineType]:
        """Select engines optimized for quality over speed"""
        
        # Prioritize engines with highest quality potential
        quality_engines = []
        
        for engine_type, config in self.engines.items():
            if not config.enabled:
                continue
            
            # Quality factors
            quality_score = 0
            
            # Higher priority engines typically have better quality
            quality_score += config.priority
            
            # Engines with matching strengths
            if category == PCIeCategory.COMPLIANCE and 'compliance' in str(config.strengths).lower():
                quality_score += 2
            if intent == QueryIntent.VERIFY and 'verification' in str(config.strengths).lower():
                quality_score += 2
            
            # Performance history
            performance = self.engine_performance[engine_type]
            if performance['total_queries'] > 0:
                quality_score += performance['avg_confidence'] * 2
            
            quality_engines.append((engine_type, quality_score))
        
        quality_engines.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 2 quality engines
        return [engine_type for engine_type, _ in quality_engines[:2]]
    
    def _determine_fallback_engines(self, selected_engines: List[RAGEngineType],
                                  category: PCIeCategory, intent: QueryIntent) -> List[RAGEngineType]:
        """Determine fallback engines if primary selection fails"""
        all_suitable = self._rank_engines_by_suitability(category, intent)
        
        # Return engines not in selected list
        fallbacks = [e for e in all_suitable if e not in selected_engines]
        return fallbacks[:2]  # Limit to 2 fallback engines
    
    def _estimate_execution_time(self, selected_engines: List[RAGEngineType],
                               strategy: CoordinationStrategy) -> float:
        """Estimate total execution time"""
        if not selected_engines:
            return 5.0  # Default estimate
        
        engine_times = []
        for engine_type in selected_engines:
            performance = self.engine_performance[engine_type]
            if performance['total_queries'] > 0:
                engine_times.append(performance['avg_response_time'])
            else:
                engine_times.append(3.0)  # Default for untested engines
        
        if strategy == CoordinationStrategy.PARALLEL_ENSEMBLE:
            # Parallel execution - use max time
            return max(engine_times)
        elif strategy == CoordinationStrategy.SEQUENTIAL_FALLBACK:
            # Sequential - sum all times (worst case)
            return sum(engine_times)
        else:
            # Single or adaptive - use average
            return sum(engine_times) / len(engine_times)
    
    def _estimate_confidence_target(self, selected_engines: List[RAGEngineType],
                                  category: PCIeCategory, intent: QueryIntent) -> float:
        """Estimate target confidence level"""
        base_confidence = 0.7
        
        # Adjust based on engine quality
        for engine_type in selected_engines:
            performance = self.engine_performance[engine_type]
            if performance['total_queries'] > 0:
                engine_confidence = performance['avg_confidence']
                base_confidence = max(base_confidence, engine_confidence)
        
        # Adjust based on category criticality
        if category == PCIeCategory.COMPLIANCE:
            base_confidence += 0.1  # Higher confidence needed for compliance
        if intent == QueryIntent.VERIFY:
            base_confidence += 0.05  # Higher confidence for verification
        
        return min(0.95, base_confidence)
    
    def _generate_routing_reasoning(self, strategy: CoordinationStrategy,
                                  selected_engines: List[RAGEngineType],
                                  category: PCIeCategory, intent: QueryIntent) -> str:
        """Generate human-readable routing reasoning"""
        
        reasons = []
        
        # Strategy reasoning
        if strategy == CoordinationStrategy.PARALLEL_ENSEMBLE:
            reasons.append("Using parallel ensemble for higher quality and reliability")
        elif strategy == CoordinationStrategy.SINGLE_BEST:
            reasons.append("Using single best engine for optimal efficiency")
        elif strategy == CoordinationStrategy.ADAPTIVE_ROUTING:
            reasons.append("Using adaptive routing based on performance metrics")
        
        # Engine selection reasoning
        if len(selected_engines) > 1:
            reasons.append(f"Selected {len(selected_engines)} engines for redundancy")
        
        # Category/intent specific reasoning
        if category == PCIeCategory.ERROR_HANDLING:
            reasons.append("Optimized for error handling and troubleshooting")
        if intent == QueryIntent.TROUBLESHOOT:
            reasons.append("Configured for troubleshooting workflow")
        
        return "; ".join(reasons)
    
    def _execute_with_strategy(self, execution: QueryExecution, 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute query based on selected strategy"""
        
        strategy = execution.routing_decision.strategy
        selected_engines = execution.routing_decision.selected_engines
        
        if strategy == CoordinationStrategy.SINGLE_BEST:
            return self._execute_single_best(execution, context)
        elif strategy == CoordinationStrategy.PARALLEL_ENSEMBLE:
            return self._execute_parallel_ensemble(execution, context)
        elif strategy == CoordinationStrategy.SEQUENTIAL_FALLBACK:
            return self._execute_sequential_fallback(execution, context)
        elif strategy in [CoordinationStrategy.ADAPTIVE_ROUTING, CoordinationStrategy.QUALITY_OPTIMIZED]:
            return self._execute_adaptive(execution, context)
        else:
            # Fallback to single best
            return self._execute_single_best(execution, context)
    
    def _execute_single_best(self, execution: QueryExecution, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute using single best engine"""
        engine_type = execution.routing_decision.selected_engines[0]
        config = self.engines[engine_type]
        
        try:
            result = self._call_engine(config.engine_instance, execution.query, context)
            execution.results[engine_type] = result
            return result
        except Exception as e:
            # Try fallback engines
            for fallback_engine in execution.routing_decision.fallback_engines:
                try:
                    fallback_config = self.engines[fallback_engine]
                    result = self._call_engine(fallback_config.engine_instance, execution.query, context)
                    execution.results[fallback_engine] = result
                    return result
                except Exception:
                    continue
            
            raise e
    
    def _execute_parallel_ensemble(self, execution: QueryExecution,
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute using parallel ensemble"""
        import concurrent.futures
        
        results = {}
        
        # Execute engines in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(execution.routing_decision.selected_engines)) as executor:
            future_to_engine = {}
            
            for engine_type in execution.routing_decision.selected_engines:
                config = self.engines[engine_type]
                future = executor.submit(self._call_engine, config.engine_instance, execution.query, context)
                future_to_engine[future] = engine_type
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_engine, timeout=30):
                engine_type = future_to_engine[future]
                try:
                    result = future.result()
                    results[engine_type] = result
                except Exception as e:
                    logger.warning(f"Engine {engine_type.value} failed: {e}")
                    results[engine_type] = {'error': str(e)}
        
        execution.results = results
        
        # Combine results intelligently
        return self._combine_parallel_results(results)
    
    def _execute_sequential_fallback(self, execution: QueryExecution,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute using sequential fallback"""
        for engine_type in execution.routing_decision.selected_engines:
            config = self.engines[engine_type]
            
            try:
                result = self._call_engine(config.engine_instance, execution.query, context)
                execution.results[engine_type] = result
                
                # Check if result meets quality threshold
                confidence = result.get('confidence', 0.0)
                if confidence >= self.adaptive_thresholds['confidence_threshold']:
                    return result
                
            except Exception as e:
                logger.warning(f"Engine {engine_type.value} failed, trying next: {e}")
                execution.results[engine_type] = {'error': str(e)}
                continue
        
        # If all engines failed or didn't meet threshold, return best available result
        if execution.results:
            best_result = max(execution.results.values(), 
                            key=lambda x: x.get('confidence', 0.0) if 'error' not in x else 0.0)
            return best_result
        
        raise Exception("All engines failed in sequential fallback")
    
    def _execute_adaptive(self, execution: QueryExecution,
                        context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute using adaptive strategy"""
        
        # Start with best engine
        primary_engine = execution.routing_decision.selected_engines[0]
        config = self.engines[primary_engine]
        
        try:
            result = self._call_engine(config.engine_instance, execution.query, context)
            execution.results[primary_engine] = result
            
            # Check if result meets adaptive thresholds
            confidence = result.get('confidence', 0.0)
            response_time = execution.execution_time
            
            # If quality is sufficient, return result
            if (confidence >= self.adaptive_thresholds['confidence_threshold'] and
                response_time <= self.adaptive_thresholds['response_time_threshold']):
                return result
            
            # If quality is insufficient, try ensemble approach
            if len(execution.routing_decision.selected_engines) > 1:
                logger.info(f"Primary engine result insufficient (confidence: {confidence:.2f}), trying ensemble")
                return self._execute_parallel_ensemble(execution, context)
            
            return result
            
        except Exception as e:
            # Fallback to sequential approach
            logger.warning(f"Adaptive primary engine failed: {e}, falling back to sequential")
            return self._execute_sequential_fallback(execution, context)
    
    def _call_engine(self, engine_instance: Any, query: str, 
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a specific RAG engine instance"""
        
        # Different engine types may have different interfaces
        if hasattr(engine_instance, 'query'):
            return engine_instance.query(query, context)
        elif hasattr(engine_instance, 'process_query'):
            return engine_instance.process_query(query, context)
        elif hasattr(engine_instance, 'ask'):
            return engine_instance.ask(query, context)
        elif callable(engine_instance):
            return engine_instance(query, context)
        else:
            raise ValueError(f"Unknown engine interface: {type(engine_instance)}")
    
    def _combine_parallel_results(self, results: Dict[RAGEngineType, Any]) -> Dict[str, Any]:
        """Intelligently combine results from parallel execution"""
        
        # Filter out error results
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_results:
            # Return the first error if all failed
            return list(results.values())[0]
        
        # Find result with highest confidence
        best_result = max(valid_results.values(), 
                         key=lambda x: x.get('confidence', 0.0))
        
        # Enhance result with ensemble information
        best_result['ensemble_info'] = {
            'engines_used': list(valid_results.keys()),
            'total_engines': len(results),
            'successful_engines': len(valid_results),
            'confidence_scores': {k.value: v.get('confidence', 0.0) for k, v in valid_results.items()}
        }
        
        # Boost confidence if multiple engines agree
        if len(valid_results) > 1:
            confidences = [r.get('confidence', 0.0) for r in valid_results.values()]
            confidence_agreement = 1.0 - (max(confidences) - min(confidences))
            boost_factor = 1.0 + (confidence_agreement * 0.1)  # Up to 10% boost
            best_result['confidence'] = min(1.0, best_result.get('confidence', 0.0) * boost_factor)
        
        return best_result
    
    def _update_engine_performance(self, execution: QueryExecution):
        """Update performance metrics for engines used"""
        
        for engine_type, result in execution.results.items():
            performance = self.engine_performance[engine_type]
            
            performance['total_queries'] += 1
            
            if 'error' not in result:
                performance['successful_queries'] += 1
                confidence = result.get('confidence', 0.0)
                
                # Update moving averages
                total = performance['total_queries']
                performance['avg_response_time'] = (
                    (performance['avg_response_time'] * (total - 1) + execution.execution_time) / total
                )
                performance['avg_confidence'] = (
                    (performance['avg_confidence'] * (total - 1) + confidence) / total
                )
                
                # Calculate quality score for recent tracking
                quality_score = (confidence + max(0, 1 - execution.execution_time / 10.0)) / 2
                performance['recent_scores'].append(quality_score)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        
        with self._lock:
            active_count = len(self.active_executions)
            
        engine_status = {}
        for engine_type, config in self.engines.items():
            performance = self.engine_performance[engine_type]
            engine_status[engine_type.value] = {
                'enabled': config.enabled,
                'priority': config.priority,
                'total_queries': performance['total_queries'],
                'success_rate': (performance['successful_queries'] / performance['total_queries'] 
                               if performance['total_queries'] > 0 else 0.0),
                'avg_response_time': performance['avg_response_time'],
                'avg_confidence': performance['avg_confidence']
            }
        
        recent_executions = list(self.execution_history)[-10:]
        
        return {
            'active_executions': active_count,
            'total_engines': len(self.engines),
            'enabled_engines': len([e for e in self.engines.values() if e.enabled]),
            'engine_status': engine_status,
            'default_strategy': self.default_strategy.value,
            'adaptive_thresholds': self.adaptive_thresholds,
            'recent_executions': [
                {
                    'query_id': ex.query_id,
                    'strategy': ex.routing_decision.strategy.value,
                    'engines_used': [e.value for e in ex.routing_decision.selected_engines],
                    'execution_time': ex.execution_time,
                    'success': ex.success
                } for ex in recent_executions
            ]
        }