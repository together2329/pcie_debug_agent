#!/usr/bin/env python3
"""
Unified RAG Integration for PCIe Debug Agent
Merges all RAG capabilities into a single /rag command with auto-testing and quality monitoring
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import queue
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class UnifiedRAGConfig:
    """Configuration for unified RAG system"""
    auto_test_enabled: bool = True
    auto_test_interval: int = 3600  # 1 hour
    quality_threshold: float = 0.7
    max_response_time: float = 5.0
    enable_caching: bool = True
    enable_learning: bool = True
    fallback_engines: List[str] = None

class UnifiedRAGEngine:
    """Unified RAG engine that combines all approaches with auto-testing"""
    
    def __init__(self, interactive_shell, config: UnifiedRAGConfig = None):
        self.shell = interactive_shell
        self.config = config or UnifiedRAGConfig()
        
        # Available engines
        self.engines = {}
        self._initialize_engines()
        
        # Quality monitoring
        self.quality_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'avg_confidence': 0.0,
            'avg_response_time': 0.0,
            'last_test_score': 0.0,
            'last_test_time': None
        }
        
        # Auto-testing
        self.auto_tester = None
        if self.config.auto_test_enabled:
            self._start_auto_testing()
    
    def _initialize_engines(self):
        """Initialize all available RAG engines"""
        try:
            # Production engine
            from production_rag_fix import ProductionRAGEngine
            self.engines['production'] = ProductionRAGEngine(
                vector_store=getattr(self.shell, 'vector_store', None),
                model_manager=getattr(self.shell, 'model_manager', None)
            )
            logger.info("Production RAG engine initialized")
        except Exception as e:
            logger.warning(f"Production engine failed: {e}")
        
        try:
            # Enhanced RAG V3
            from src.rag.enhanced_rag_engine_v3 import EnhancedRAGEngineV3
            if hasattr(self.shell, 'vector_store') and self.shell.vector_store:
                self.engines['v3'] = EnhancedRAGEngineV3(
                    vector_store=self.shell.vector_store,
                    model_manager=getattr(self.shell, 'model_manager', None)
                )
                logger.info("Enhanced RAG v3 engine initialized")
        except Exception as e:
            logger.warning(f"V3 engine failed: {e}")
        
        try:
            # Standard RAG engine
            if hasattr(self.shell, 'rag_engine') and self.shell.rag_engine:
                self.engines['standard'] = self.shell.rag_engine
                logger.info("Standard RAG engine available")
        except Exception as e:
            logger.warning(f"Standard engine failed: {e}")
        
        # Set default engine priority
        self.engine_priority = ['production', 'v3', 'standard']
        self.current_engine = self._get_best_engine()
    
    def _get_best_engine(self) -> str:
        """Get best available engine based on quality metrics"""
        for engine_name in self.engine_priority:
            if engine_name in self.engines:
                return engine_name
        return None
    
    def _start_auto_testing(self):
        """Start background auto-testing thread"""
        def auto_test_worker():
            while True:
                try:
                    time.sleep(self.config.auto_test_interval)
                    self._run_background_test()
                except Exception as e:
                    logger.error(f"Auto-test failed: {e}")
        
        self.auto_tester = threading.Thread(target=auto_test_worker, daemon=True)
        self.auto_tester.start()
        logger.info("Auto-testing started")
    
    def _run_background_test(self):
        """Run background quality test"""
        try:
            from automated_rag_test_suite import PCIeRAGTestSuite
            
            print("\n🔄 Running background quality test...")
            
            test_suite = PCIeRAGTestSuite()
            # Run subset of critical tests
            critical_tests = [tc for tc in test_suite.test_cases if tc.compliance_critical]
            
            if critical_tests:
                results = []
                for test_case in critical_tests[:3]:  # Quick test
                    result = test_suite._run_single_test(test_case, self)
                    results.append(result)
                
                # Update metrics
                avg_score = sum(r.score for r in results) / len(results)
                self.quality_metrics['last_test_score'] = avg_score
                self.quality_metrics['last_test_time'] = datetime.now().isoformat()
                
                # Switch engines if quality drops
                if avg_score < self.config.quality_threshold:
                    self._try_engine_switch()
                
                print(f"   Background test score: {avg_score:.2f}")
        
        except Exception as e:
            logger.error(f"Background test failed: {e}")
    
    def _try_engine_switch(self):
        """Try switching to a better engine"""
        current_index = self.engine_priority.index(self.current_engine) if self.current_engine in self.engine_priority else -1
        
        # Try next engine in priority list
        for i in range(current_index + 1, len(self.engine_priority)):
            engine_name = self.engine_priority[i]
            if engine_name in self.engines:
                self.current_engine = engine_name
                print(f"🔄 Switched to {engine_name} engine for better quality")
                break
    
    def query(self, question: str, mode: str = "auto") -> Dict[str, Any]:
        """Unified query interface with auto-testing and fallback"""
        start_time = time.time()
        
        try:
            # Choose engine based on mode
            engine_name = mode if mode in self.engines else self.current_engine
            if not engine_name or engine_name not in self.engines:
                raise Exception("No suitable RAG engine available")
            
            engine = self.engines[engine_name]
            
            # Execute query based on engine type
            if engine_name == 'production':
                from production_rag_fix import ProductionRAGQuery
                query_obj = ProductionRAGQuery(query=question)
                response = engine.query(query_obj)
                
                result = {
                    'answer': response.answer,
                    'confidence': response.confidence,
                    'sources': response.sources,
                    'engine': engine_name,
                    'analysis_type': response.analysis_type,
                    'debugging_hints': response.debugging_hints,
                    'spec_references': response.spec_references,
                    'metadata': response.metadata
                }
            
            elif engine_name == 'v3':
                from src.rag.enhanced_rag_engine_v3 import EnhancedRAGQuery
                query_obj = EnhancedRAGQuery(query=question)
                response = engine.query(query_obj)
                
                result = {
                    'answer': response.answer,
                    'confidence': response.confidence,
                    'sources': response.sources,
                    'engine': engine_name,
                    'verification': response.verification,
                    'normalized_question': response.normalized_question,
                    'knowledge_items': response.knowledge_items,
                    'metadata': response.metadata
                }
            
            else:  # standard engine
                from src.rag.enhanced_rag_engine import RAGQuery
                query_obj = RAGQuery(query=question)
                response = engine.query(query_obj)
                
                result = {
                    'answer': response.answer,
                    'confidence': response.confidence,
                    'sources': response.sources,
                    'engine': engine_name,
                    'reasoning': response.reasoning,
                    'metadata': response.metadata
                }
            
            # Update metrics
            response_time = time.time() - start_time
            self._update_metrics(result['confidence'], response_time, True)
            result['response_time'] = response_time
            
            return result
        
        except Exception as e:
            # Fallback to next available engine
            response_time = time.time() - start_time
            self._update_metrics(0.0, response_time, False)
            
            if self.config.fallback_engines:
                for fallback_engine in self.config.fallback_engines:
                    if fallback_engine in self.engines and fallback_engine != engine_name:
                        try:
                            return self.query(question, fallback_engine)
                        except:
                            continue
            
            return {
                'answer': f"Query failed: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'engine': 'error',
                'error': str(e),
                'response_time': response_time
            }
    
    def _update_metrics(self, confidence: float, response_time: float, success: bool):
        """Update quality metrics"""
        self.quality_metrics['total_queries'] += 1
        if success:
            self.quality_metrics['successful_queries'] += 1
        
        # Running averages
        total = self.quality_metrics['total_queries']
        self.quality_metrics['avg_confidence'] = (
            (self.quality_metrics['avg_confidence'] * (total - 1) + confidence) / total
        )
        self.quality_metrics['avg_response_time'] = (
            (self.quality_metrics['avg_response_time'] * (total - 1) + response_time) / total
        )
    
    def run_full_test(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        try:
            from automated_rag_test_suite import PCIeRAGTestSuite
            
            test_suite = PCIeRAGTestSuite()
            report = test_suite.run_full_suite(self)
            
            # Update metrics
            self.quality_metrics['last_test_score'] = report.overall_score
            self.quality_metrics['last_test_time'] = report.timestamp
            
            return asdict(report)
        
        except Exception as e:
            return {'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get unified RAG system status"""
        return {
            'engines_available': list(self.engines.keys()),
            'current_engine': self.current_engine,
            'engine_priority': self.engine_priority,
            'quality_metrics': self.quality_metrics,
            'config': asdict(self.config),
            'auto_testing': self.auto_tester is not None and self.auto_tester.is_alive()
        }

def integrate_unified_rag(interactive_shell):
    """Integrate unified RAG system into interactive shell"""
    
    # Initialize unified RAG
    shell.unified_rag = UnifiedRAGEngine(interactive_shell)
    
    def do_rag(self, arg):
        """Unified RAG command with auto-testing and quality monitoring
        
        Usage:
            /rag <question>                    - Query with best available engine
            /rag <question> --engine production - Query with specific engine
            /rag --test                        - Run quality test suite
            /rag --status                      - Show system status
            /rag --engines                     - List available engines
            /rag --config                      - Show configuration
        
        Examples:
            /rag "why dut send successful return during flr?"
            /rag "PCIe completion timeout debug" --engine v3
            /rag --test
        """
        if not arg:
            print("Usage: /rag <question> [--engine ENGINE] or /rag --status/--test/--engines")
            return
        
        # Parse arguments
        parts = arg.split()
        if not parts:
            return
        
        # Handle special commands
        if parts[0] == '--test':
            print("🧪 Running comprehensive RAG test suite...")
            result = self.unified_rag.run_full_test()
            if 'error' in result:
                print(f"❌ Test failed: {result['error']}")
            else:
                print(f"✅ Test completed - Overall score: {result.get('overall_score', 0):.2f}")
            return
        
        if parts[0] == '--status':
            status = self.unified_rag.get_status()
            print("📊 Unified RAG System Status")
            print("=" * 40)
            print(f"Available Engines: {', '.join(status['engines_available'])}")
            print(f"Current Engine: {status['current_engine']}")
            print(f"Total Queries: {status['quality_metrics']['total_queries']}")
            print(f"Success Rate: {status['quality_metrics']['successful_queries']/max(status['quality_metrics']['total_queries'], 1)*100:.1f}%")
            print(f"Avg Confidence: {status['quality_metrics']['avg_confidence']:.1%}")
            print(f"Avg Response Time: {status['quality_metrics']['avg_response_time']:.2f}s")
            print(f"Last Test Score: {status['quality_metrics']['last_test_score']:.2f}")
            print(f"Auto-testing: {'ON' if status['auto_testing'] else 'OFF'}")
            return
        
        if parts[0] == '--engines':
            status = self.unified_rag.get_status()
            print("🔧 Available RAG Engines")
            print("=" * 30)
            for i, engine in enumerate(status['engine_priority'], 1):
                current = " (CURRENT)" if engine == status['current_engine'] else ""
                available = "✅" if engine in status['engines_available'] else "❌"
                print(f"{i}. {engine}{current} {available}")
            return
        
        if parts[0] == '--config':
            status = self.unified_rag.get_status()
            print("⚙️  Unified RAG Configuration")
            print("=" * 35)
            for key, value in status['config'].items():
                print(f"{key}: {value}")
            return
        
        # Parse query and engine
        engine = "auto"
        question = arg
        
        if '--engine' in parts:
            engine_index = parts.index('--engine')
            if engine_index + 1 < len(parts):
                engine = parts[engine_index + 1]
                # Remove engine args from question
                parts = parts[:engine_index] + parts[engine_index + 2:]
                question = ' '.join(parts)
        
        # Execute query
        print(f"🔍 Unified RAG Query (Engine: {engine})")
        print("-" * 50)
        
        result = self.unified_rag.query(question, engine)
        
        # Display results
        print(f"\n📝 Answer (Confidence: {result['confidence']:.1%}):")
        print(result['answer'])
        
        # Show additional info based on engine
        if 'debugging_hints' in result and result['debugging_hints']:
            print(f"\n🔧 Debug Hints:")
            for hint in result['debugging_hints']:
                print(f"• {hint}")
        
        if 'spec_references' in result and result['spec_references']:
            print(f"\n📋 Spec References: {', '.join(result['spec_references'])}")
        
        if 'verification' in result and result['verification']:
            print(f"\n✅ Verification: {result['verification'].confidence:.1%} ({result['verification'].verification_method})")
        
        # Performance info
        print(f"\n⚡ Engine: {result['engine']} | Time: {result.get('response_time', 0):.2f}s")
        
        # Auto-suggest improvements if quality is low
        if result['confidence'] < 0.6:
            print(f"\n💡 Suggestion: Try '/rag --test' to check system quality")
    
    # Add method to shell
    import types
    interactive_shell.do_rag = types.MethodType(do_rag, interactive_shell)
    
    # Also keep the old rag_analyze as alias
    interactive_shell.do_rag_analyze = interactive_shell.do_rag
    
    print("✅ Unified RAG system integrated!")
    print("   Use '/rag <question>' for intelligent queries")
    print("   Use '/rag --status' to monitor quality")
    print("   Use '/rag --test' for quality testing")
    
    return interactive_shell

if __name__ == "__main__":
    print("Unified RAG Integration - Ready for deployment")
    print("Features:")
    print("• Auto-testing and quality monitoring")  
    print("• Multi-engine fallback")
    print("• Performance optimization")
    print("• Unified command interface")