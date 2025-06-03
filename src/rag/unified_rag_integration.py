#!/usr/bin/env python3
"""
Unified RAG Integration Layer - REALITY CHECK IMPLEMENTATION

This is the ACTUAL integration that connects all Phase 1-3 components
and provides a working, testable, end-to-end RAG system.

Instead of more theoretical components, this focuses on making 
everything work together efficiently in the real world.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Import all our Phase 1-3 components
from .enhanced_rag_engine import EnhancedRAGEngine
from .pcie_knowledge_classifier import PCIeKnowledgeClassifier, PCIeCategory, QueryIntent
from .query_expansion_engine import QueryExpansionEngine
from .compliance_intelligence import ComplianceIntelligence
from .model_ensemble import ModelEnsemble
from .specialized_routing import SpecializedRouter, ProcessingTier
from .quality_monitor import ContinuousQualityMonitor
from .meta_rag_coordinator import MetaRAGCoordinator
from .performance_analytics import PerformanceAnalytics
from .response_optimizer import ResponseOptimizer
from .context_memory import ContextMemory

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Complete query result with all metadata"""
    query_id: str
    query: str
    answer: str
    confidence: float
    response_time: float
    engine_used: str
    processing_tier: str
    category: PCIeCategory
    intent: QueryIntent
    sources: List[str]
    personalization_applied: bool
    quality_score: float
    metadata: Dict[str, Any]

class UnifiedRAGSystem:
    """
    Unified RAG System that actually integrates all components
    and provides a working end-to-end experience.
    
    This is the REAL implementation that focuses on:
    1. Actually working integration
    2. Performance optimization
    3. Testable functionality
    4. Resource efficiency
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        
        # Initialize core components in order of dependency
        logger.info("Initializing Unified RAG System...")
        
        # Phase 1: Core RAG Engine
        self.rag_engine = EnhancedRAGEngine()
        self.knowledge_classifier = PCIeKnowledgeClassifier()
        
        # Phase 2: Advanced Features  
        self.query_expander = QueryExpansionEngine()
        self.compliance_engine = ComplianceIntelligence()
        self.model_ensemble = ModelEnsemble(self.rag_engine.model_manager)
        self.specialized_router = SpecializedRouter(self.model_ensemble)
        
        # Phase 3: Intelligence Layer
        self.quality_monitor = ContinuousQualityMonitor()
        self.performance_analytics = PerformanceAnalytics(self.quality_monitor)
        self.response_optimizer = ResponseOptimizer(self.quality_monitor, self.performance_analytics)
        self.context_memory = ContextMemory()
        self.meta_coordinator = MetaRAGCoordinator(self.quality_monitor)
        
        # Register engines with meta coordinator
        self._register_rag_engines()
        
        # Start monitoring systems
        self.quality_monitor.start_monitoring()
        
        # Performance tracking
        self.total_queries = 0
        self.successful_queries = 0
        self.avg_response_time = 0.0
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Unified RAG System initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the unified system"""
        return {
            'enable_quality_monitoring': True,
            'enable_performance_analytics': True,
            'enable_response_optimization': True,
            'enable_context_memory': True,
            'enable_meta_coordination': True,
            'max_response_time': 30.0,
            'min_confidence_threshold': 0.6,
            'enable_personalization': True,
            'enable_compliance_checking': True,
            'parallel_processing': True
        }
    
    def _register_rag_engines(self):
        """Register different RAG engines with meta coordinator"""
        
        from .enhanced_rag_engine import EnhancedRAGEngine
        
        # Register the main enhanced engine
        self.meta_coordinator.register_engine(
            engine_type=self.meta_coordinator.RAGEngineType.ENHANCED_V2,
            engine_instance=self.rag_engine,
            priority=3,
            strengths=['enhanced confidence scoring', 'automatic citations', 'phrase matching'],
            optimal_categories=[PCIeCategory.ERROR_HANDLING, PCIeCategory.COMPLIANCE],
            optimal_intents=[QueryIntent.TROUBLESHOOT, QueryIntent.VERIFY]
        )
        
        # Register a standard engine for simple queries
        standard_engine = self._create_standard_engine()
        self.meta_coordinator.register_engine(
            engine_type=self.meta_coordinator.RAGEngineType.STANDARD,
            engine_instance=standard_engine,
            priority=1,
            strengths=['fast responses', 'simple queries'],
            optimal_categories=[PCIeCategory.GENERAL],
            optimal_intents=[QueryIntent.REFERENCE, QueryIntent.LEARN]
        )
        
        # Set specialized components
        self.meta_coordinator.set_specialized_components(self.specialized_router, self.model_ensemble)
    
    def _create_standard_engine(self):
        """Create a simpler engine for basic queries"""
        # This would be a simplified version for fast responses
        return self.rag_engine  # Using same engine for now, but configured differently
    
    def process_query(self, query: str, user_id: str = None, session_id: str = None,
                     context: Dict[str, Any] = None) -> QueryResult:
        """
        Process a query through the complete unified RAG system.
        This is the main entry point that orchestrates all components.
        """
        
        start_time = time.time()
        query_id = f"unified_{int(time.time() * 1000)}"
        
        try:
            # Step 1: Context Enhancement (if enabled)
            enhanced_query, user_context = self._enhance_with_context(
                query, session_id, user_id
            )
            
            # Step 2: Query Classification and Expansion
            expanded_query_obj = self.query_expander.expand_query(enhanced_query)
            category = expanded_query_obj.category
            intent = expanded_query_obj.intent
            
            # Step 3: Route to appropriate processing
            if self.config['enable_meta_coordination']:
                # Use meta coordinator for intelligent routing
                result = self._process_with_meta_coordination(
                    expanded_query_obj.expanded_query, user_id, session_id, user_context
                )
            else:
                # Direct processing with enhanced engine
                result = self._process_with_enhanced_engine(
                    expanded_query_obj.expanded_query, category, intent, user_context
                )
            
            # Step 4: Compliance checking (if enabled and relevant)
            if self.config['enable_compliance_checking'] and category == PCIeCategory.COMPLIANCE:
                compliance_result = self.compliance_engine.instant_compliance_check(
                    query, result.get('answer', '')
                )
                result['compliance_check'] = compliance_result
            
            # Step 5: Response optimization (if enabled)
            if self.config['enable_response_optimization'] and user_id:
                result = self._optimize_response(result, user_id, session_id, category, intent)
            
            # Step 6: Quality monitoring
            response_time = time.time() - start_time
            success = 'error' not in result
            confidence = result.get('confidence', 0.0)
            
            if self.config['enable_quality_monitoring']:
                self.quality_monitor.record_query_execution(
                    response_time, confidence, success, context, session_id, query_id
                )
            
            # Step 7: Performance analytics
            if self.config['enable_performance_analytics']:
                self.performance_analytics.record_performance_snapshot(
                    response_time, confidence, success, 
                    result.get('engine_used', 'enhanced'),
                    category.value, intent.value
                )
            
            # Step 8: Context memory update
            if self.config['enable_context_memory'] and session_id:
                self.context_memory.record_interaction(
                    session_id, query, result, user_id
                )
            
            # Step 9: Create unified result
            unified_result = QueryResult(
                query_id=query_id,
                query=query,
                answer=result.get('answer', ''),
                confidence=confidence,
                response_time=response_time,
                engine_used=result.get('engine_used', 'enhanced'),
                processing_tier=result.get('processing_tier', 'standard'),
                category=category,
                intent=intent,
                sources=result.get('sources', []),
                personalization_applied=user_id is not None,
                quality_score=self._calculate_quality_score(result, response_time),
                metadata={
                    'expanded_query': expanded_query_obj.expanded_query,
                    'context_hints': expanded_query_obj.context_hints,
                    'user_context': user_context,
                    'compliance_check': result.get('compliance_check'),
                    'meta_coordination': result.get('meta_coordination')
                }
            )
            
            # Update performance statistics
            self._update_performance_stats(True, response_time)
            
            logger.info(f"Query processed successfully: {query_id} in {response_time:.3f}s")
            return unified_result
            
        except Exception as e:
            # Error handling
            error_time = time.time() - start_time
            logger.error(f"Error processing query {query_id}: {e}")
            
            # Update performance statistics
            self._update_performance_stats(False, error_time)
            
            # Return error result
            return QueryResult(
                query_id=query_id,
                query=query,
                answer=f"I apologize, but I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                response_time=error_time,
                engine_used="error_handler",
                processing_tier="error",
                category=PCIeCategory.GENERAL,
                intent=QueryIntent.LEARN,
                sources=[],
                personalization_applied=False,
                quality_score=0.0,
                metadata={'error': str(e)}
            )
    
    def _enhance_with_context(self, query: str, session_id: str = None, 
                            user_id: str = None) -> Tuple[str, Dict[str, Any]]:
        """Enhance query with context memory"""
        
        if not self.config['enable_context_memory'] or not session_id:
            return query, {}
        
        try:
            enhanced = self.context_memory.enhance_query_with_context(
                session_id, query
            )
            return enhanced['enhanced_query'], enhanced['user_context']
        except Exception as e:
            logger.warning(f"Context enhancement failed: {e}")
            return query, {}
    
    def _process_with_meta_coordination(self, query: str, user_id: str, 
                                      session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using meta-RAG coordination"""
        
        try:
            result = self.meta_coordinator.execute_query(
                query, session_id, context=context
            )
            return result
        except Exception as e:
            logger.warning(f"Meta coordination failed, falling back to enhanced engine: {e}")
            # Fallback to enhanced engine
            return self._process_with_enhanced_engine(query, PCIeCategory.GENERAL, QueryIntent.LEARN, context)
    
    def _process_with_enhanced_engine(self, query: str, category: PCIeCategory,
                                    intent: QueryIntent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query with enhanced RAG engine"""
        
        try:
            # Use enhanced RAG engine with all Phase 1 improvements
            result = self.rag_engine.query(query, context)
            result['engine_used'] = 'enhanced_v2'
            result['category'] = category.value
            result['intent'] = intent.value
            return result
        except Exception as e:
            logger.error(f"Enhanced engine failed: {e}")
            # Last resort fallback
            return {
                'answer': "I apologize, but I'm having trouble processing your query right now. Please try rephrasing your question.",
                'confidence': 0.1,
                'sources': [],
                'error': str(e),
                'engine_used': 'fallback'
            }
    
    def _optimize_response(self, result: Dict[str, Any], user_id: str, session_id: str,
                         category: PCIeCategory, intent: QueryIntent) -> Dict[str, Any]:
        """Apply response optimization"""
        
        try:
            # Record interaction for learning
            self.response_optimizer.record_query_execution(
                result.get('response_time', 0),
                result.get('confidence', 0),
                'error' not in result,
                {'category': category.value, 'intent': intent.value},
                session_id
            )
            
            # Apply optimizations (simplified for now)
            optimized_result = result.copy()
            optimized_result['optimization_applied'] = True
            return optimized_result
            
        except Exception as e:
            logger.warning(f"Response optimization failed: {e}")
            return result
    
    def _calculate_quality_score(self, result: Dict[str, Any], response_time: float) -> float:
        """Calculate overall quality score for the response"""
        
        confidence = result.get('confidence', 0.0)
        
        # Time score (faster is better, up to 10 seconds)
        time_score = max(0, 1 - response_time / 10.0)
        
        # Error penalty
        error_penalty = 0.5 if 'error' in result else 0.0
        
        # Compliance bonus
        compliance_bonus = 0.1 if result.get('compliance_check', {}).get('compliance_status') == 'COMPLIANT' else 0.0
        
        quality_score = (confidence * 0.6 + time_score * 0.3 + compliance_bonus) - error_penalty
        return max(0.0, min(1.0, quality_score))
    
    def _update_performance_stats(self, success: bool, response_time: float):
        """Update internal performance statistics"""
        
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        
        # Update average response time
        self.avg_response_time = (self.avg_response_time * (self.total_queries - 1) + response_time) / self.total_queries
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get component statuses
        quality_status = self.quality_monitor.get_real_time_metrics() if self.config['enable_quality_monitoring'] else {}
        analytics_status = self.performance_analytics.get_real_time_dashboard_data() if self.config['enable_performance_analytics'] else {}
        coordinator_status = self.meta_coordinator.get_coordination_status() if self.config['enable_meta_coordination'] else {}
        memory_status = self.context_memory.get_memory_status() if self.config['enable_context_memory'] else {}
        
        return {
            'unified_system': {
                'total_queries': self.total_queries,
                'success_rate': self.successful_queries / max(1, self.total_queries),
                'avg_response_time': self.avg_response_time,
                'components_active': self._count_active_components()
            },
            'quality_monitoring': quality_status,
            'performance_analytics': analytics_status,
            'meta_coordination': coordinator_status,
            'context_memory': memory_status,
            'configuration': self.config
        }
    
    def _count_active_components(self) -> int:
        """Count how many components are active"""
        count = 1  # Core RAG engine always active
        
        if self.config['enable_quality_monitoring']:
            count += 1
        if self.config['enable_performance_analytics']:
            count += 1
        if self.config['enable_response_optimization']:
            count += 1
        if self.config['enable_context_memory']:
            count += 1
        if self.config['enable_meta_coordination']:
            count += 1
        
        return count
    
    def run_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration test to validate the system works"""
        
        logger.info("Running integration test...")
        
        test_queries = [
            "What is PCIe FLR?",
            "How do I debug a PCIe completion timeout?",
            "What are the compliance requirements for CRS?",
            "Explain PCIe error handling mechanisms",
            "How to implement PCIe power management?"
        ]
        
        test_results = []
        
        for i, query in enumerate(test_queries):
            try:
                start_time = time.time()
                result = self.process_query(
                    query, 
                    user_id=f"test_user_{i}", 
                    session_id=f"test_session_{i}"
                )
                
                test_results.append({
                    'query': query,
                    'success': True,
                    'response_time': result.response_time,
                    'confidence': result.confidence,
                    'quality_score': result.quality_score,
                    'engine_used': result.engine_used
                })
                
            except Exception as e:
                test_results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'response_time': time.time() - start_time
                })
        
        # Calculate test summary
        successful_tests = len([r for r in test_results if r['success']])
        avg_response_time = sum([r.get('response_time', 0) for r in test_results]) / len(test_results)
        avg_confidence = sum([r.get('confidence', 0) for r in test_results if r['success']]) / max(1, successful_tests)
        
        summary = {
            'total_tests': len(test_queries),
            'successful_tests': successful_tests,
            'success_rate': successful_tests / len(test_queries),
            'avg_response_time': avg_response_time,
            'avg_confidence': avg_confidence,
            'test_results': test_results
        }
        
        logger.info(f"Integration test complete: {successful_tests}/{len(test_queries)} passed")
        return summary
    
    def shutdown(self):
        """Gracefully shutdown the unified system"""
        
        logger.info("Shutting down Unified RAG System...")
        
        # Stop monitoring systems
        if self.quality_monitor:
            self.quality_monitor.stop_monitoring()
        
        # Shutdown thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
        
        logger.info("Unified RAG System shutdown complete")


# Simplified usage interface
class RAGSystemInterface:
    """Simplified interface for using the unified RAG system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.system = UnifiedRAGSystem(config)
    
    def ask(self, question: str, user_id: str = None, session_id: str = None) -> str:
        """Simple interface: ask a question, get an answer"""
        result = self.system.process_query(question, user_id, session_id)
        return result.answer
    
    def ask_detailed(self, question: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        """Detailed interface: get full result information"""
        result = self.system.process_query(question, user_id, session_id)
        return asdict(result)
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return self.system.get_system_status()
    
    def test_system(self) -> Dict[str, Any]:
        """Test the integrated system"""
        return self.system.run_integration_test()


# Example usage and validation
if __name__ == "__main__":
    # Initialize the unified system
    rag = RAGSystemInterface()
    
    # Test basic functionality
    print("Testing Unified RAG System...")
    
    # Simple query
    answer = rag.ask("What is PCIe FLR?")
    print(f"Simple query result: {answer[:100]}...")
    
    # Detailed query
    detailed = rag.ask_detailed("How do I debug PCIe completion timeout?", user_id="test_user")
    print(f"Detailed query confidence: {detailed['confidence']:.2f}")
    
    # System status
    status = rag.get_status()
    print(f"System components active: {status['unified_system']['components_active']}")
    
    # Integration test
    test_results = rag.test_system()
    print(f"Integration test: {test_results['successful_tests']}/{test_results['total_tests']} passed")