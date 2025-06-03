"""
Enhanced Interactive CLI with RAG v3 Integration
Integrates all RAG improvements into the interactive shell
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging

# Import enhanced RAG components
from src.rag.enhanced_rag_engine_v3 import EnhancedRAGEngineV3, EnhancedRAGQuery, EnhancedRAGResponse
from src.rag.pcie_knowledge_classifier import PCIeCategory
from src.rag.question_normalizer import QuestionIntent

# Import existing components
from src.cli.interactive import InteractiveShell  # Inherit from existing shell
from src.vectorstore.faiss_store import FAISSVectorStore
from src.config.settings import Settings

logger = logging.getLogger(__name__)

class EnhancedInteractiveShell(InteractiveShell):
    """Enhanced interactive shell with RAG v3 capabilities"""
    
    def __init__(self, args):
        """Initialize enhanced shell"""
        super().__init__(args)
        
        # Initialize enhanced RAG engine
        self.enhanced_rag = None
        self._init_enhanced_rag()
        
        # Enhanced command mappings
        self.enhanced_commands = {
            '/rag_analyze': self._cmd_rag_analyze,
            '/rag_verify': self._cmd_rag_verify,
            '/rag_normalize': self._cmd_rag_normalize,
            '/rag_categories': self._cmd_rag_categories,
            '/rag_benchmark': self._cmd_rag_benchmark,
            '/rag_confidence': self._cmd_rag_confidence,
            '/rag_v3_status': self._cmd_rag_v3_status,
            '/suggest': self._cmd_suggest_questions
        }
        
        # Update command registry
        self.commands.update(self.enhanced_commands)
        
        print("🚀 Enhanced RAG Engine v3 Active")
        print("   New commands: /rag_analyze, /rag_verify, /rag_normalize")
        print("   Type /rag_v3_status for system status")
    
    def _init_enhanced_rag(self):
        """Initialize enhanced RAG engine"""
        try:
            if hasattr(self, 'vector_store') and self.vector_store:
                self.enhanced_rag = EnhancedRAGEngineV3(
                    vector_store=self.vector_store,
                    model_manager=getattr(self, 'model_manager', None),
                    llm_provider=getattr(self, 'llm_provider', 'openai'),
                    llm_model=getattr(self, 'current_model', 'gpt-4o-mini'),
                    use_hybrid_search=True
                )
                print("✅ Enhanced RAG Engine v3 initialized")
            else:
                print("⚠️  Vector store not available, enhanced RAG disabled")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG: {e}")
            print(f"❌ Enhanced RAG initialization failed: {e}")
    
    def _process_enhanced_query(self, query: str, **kwargs) -> EnhancedRAGResponse:
        """Process query using enhanced RAG engine"""
        if not self.enhanced_rag:
            # Fallback to standard processing
            return self._fallback_query(query)
        
        try:
            rag_query = EnhancedRAGQuery(
                query=query,
                use_hybrid_search=kwargs.get('use_hybrid', True),
                verify_answer=kwargs.get('verify', True),
                normalize_question=kwargs.get('normalize', True),
                min_confidence=kwargs.get('min_confidence', 0.3),
                max_results=kwargs.get('max_results', 10)
            )
            
            response = self.enhanced_rag.query(rag_query)
            return response
            
        except Exception as e:
            logger.error(f"Enhanced query processing failed: {e}")
            return self._fallback_query(query)
    
    def _fallback_query(self, query: str) -> EnhancedRAGResponse:
        """Fallback to basic query processing"""
        # Create minimal response structure
        from src.rag.enhanced_rag_engine_v3 import EnhancedRAGResponse
        
        return EnhancedRAGResponse(
            answer="Enhanced RAG not available, using basic processing.",
            confidence=0.5,
            verification=None,
            normalized_question=None,
            sources=[],
            knowledge_items=[],
            metadata={"fallback": True}
        )
    
    def _cmd_rag_analyze(self, args: str):
        """Analyze query with enhanced RAG components"""
        if not args:
            print("Usage: /rag_analyze <query>")
            return
        
        print(f"🔍 Enhanced RAG Analysis: {args}")
        print("-" * 50)
        
        response = self._process_enhanced_query(args)
        
        # Display comprehensive analysis
        print(f"\n📝 Answer (Confidence: {response.confidence:.1%}):")
        print(response.answer)
        
        if response.normalized_question:
            print(f"\n🔄 Question Normalization:")
            print(f"   Original: {args}")
            print(f"   Normalized: {response.normalized_question.normalized_form}")
            print(f"   Intent: {response.normalized_question.intent.value}")
            print(f"   Key Concepts: {response.normalized_question.key_concepts}")
        
        if response.verification:
            print(f"\n✅ Answer Verification:")
            print(f"   Confidence: {response.verification.confidence:.1%}")
            print(f"   Method: {response.verification.verification_method.value}")
            print(f"   Explanation: {response.verification.explanation}")
        
        if response.knowledge_items:
            print(f"\n🧠 Knowledge Classification:")
            for i, item in enumerate(response.knowledge_items[:3]):
                print(f"   {i+1}. Category: {item.category.value}")
                print(f"      Facts: {len(item.facts)}")
                print(f"      Difficulty: {item.difficulty}")
        
        if response.suggestions:
            print(f"\n💡 Related Questions:")
            for suggestion in response.suggestions[:3]:
                print(f"   • {suggestion}")
    
    def _cmd_rag_verify(self, args: str):
        """Test answer verification on a specific query"""
        if not args:
            print("Usage: /rag_verify <query>")
            return
        
        print(f"✅ Answer Verification Test: {args}")
        print("-" * 40)
        
        response = self._process_enhanced_query(args, verify=True)
        
        if response.verification:
            v = response.verification
            print(f"Confidence: {v.confidence:.1%}")
            print(f"Method: {v.verification_method.value}")
            print(f"Matched Keywords: {v.matched_keywords}")
            print(f"Missing Keywords: {v.missing_keywords}")
            print(f"Fact Accuracy: {v.fact_accuracy:.1%}")
            print(f"Explanation: {v.explanation}")
        else:
            print("No verification data available")
    
    def _cmd_rag_normalize(self, args: str):
        """Test question normalization"""
        if not args:
            print("Usage: /rag_normalize <question>")
            return
        
        if not self.enhanced_rag:
            print("Enhanced RAG not available")
            return
        
        print(f"🔄 Question Normalization: {args}")
        print("-" * 40)
        
        normalizer = self.enhanced_rag.question_normalizer
        normalized = normalizer.normalize_question(args)
        
        print(f"Original: {args}")
        print(f"Normalized: {normalized.normalized_form}")
        print(f"Intent: {normalized.intent.value}")
        print(f"Key Concepts: {normalized.key_concepts}")
        print(f"Confidence: {normalized.confidence:.2f}")
        print(f"Type: {normalized.question_type}")
        
        # Show variations
        variations = normalizer.get_question_variations(normalized.normalized_form)
        if variations:
            print(f"\nKnown Variations:")
            for var in variations[:5]:
                print(f"  • {var}")
    
    def _cmd_rag_categories(self, args: str):
        """Show PCIe knowledge categories"""
        print("🧠 PCIe Knowledge Categories:")
        print("-" * 30)
        
        categories = [
            ("error_handling", "Error handling, AER, correctable/uncorrectable errors"),
            ("ltssm", "Link Training State Machine, LTSSM states"),
            ("tlp", "Transaction Layer Packets, TLP headers and formats"),
            ("power_management", "Power states, ASPM, D-states, L-states"),
            ("physical_layer", "Physical layer, signaling, lanes, speeds"),
            ("architecture", "PCIe topology, root complex, endpoints, switches"),
            ("configuration", "Configuration space, registers, capabilities"),
            ("flow_control", "Credit-based flow control, DLLPs"),
            ("advanced_features", "SR-IOV, MSI/MSI-X, ATS, etc.")
        ]
        
        for cat, desc in categories:
            print(f"{cat:18} - {desc}")
    
    def _cmd_rag_benchmark(self, args: str):
        """Run RAG system benchmark"""
        if not self.enhanced_rag:
            print("Enhanced RAG not available")
            return
        
        print("📊 Running RAG System Benchmark...")
        print("-" * 35)
        
        test_queries = [
            "What is the 3DW Memory Request TLP header format?",
            "List all PCIe LTSSM states with timeouts",
            "How does PCIe flow control work?",
            "What are PCIe power states?",
            "Explain AER capability registers"
        ]
        
        try:
            results = self.enhanced_rag.benchmark_performance(test_queries)
            
            print(f"Total Queries: {results['total_queries']}")
            print(f"Successful: {results['successful_queries']}")
            print(f"High Confidence: {results['high_confidence_answers']}")
            print(f"Verified Answers: {results['verified_answers']}")
            print(f"Avg Response Time: {results['average_response_time']:.2f}s")
            print(f"Avg Confidence: {results['average_confidence']:.1%}")
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    def _cmd_rag_confidence(self, args: str):
        """Set minimum confidence threshold"""
        if not args:
            print("Usage: /rag_confidence <threshold>")
            print("Current: Use default threshold (0.3)")
            return
        
        try:
            threshold = float(args)
            if 0.0 <= threshold <= 1.0:
                self.rag_min_confidence = threshold
                print(f"✅ Confidence threshold set to {threshold:.1%}")
            else:
                print("❌ Threshold must be between 0.0 and 1.0")
        except ValueError:
            print("❌ Invalid threshold value")
    
    def _cmd_rag_v3_status(self, args: str):
        """Show enhanced RAG engine status"""
        print("🚀 Enhanced RAG Engine v3 Status")
        print("-" * 35)
        
        if self.enhanced_rag:
            status = self.enhanced_rag.get_system_status()
            
            print("📊 Metrics:")
            metrics = status["metrics"]
            print(f"   Queries Processed: {metrics['queries_processed']}")
            print(f"   Avg Response Time: {metrics['average_response_time']:.2f}s")
            print(f"   Avg Confidence: {metrics['average_confidence']:.1%}")
            print(f"   Cache Hit Rate: {status['cache_stats']['hit_rate']:.1%}")
            
            print("\n🔧 Components:")
            for comp, state in status["components"].items():
                print(f"   {comp}: {state}")
            
            print("\n💾 Cache Stats:")
            cache = status["cache_stats"]
            print(f"   Size: {cache['size']}/{cache['max_size']}")
            print(f"   Hit Rate: {cache['hit_rate']:.1%}")
        else:
            print("❌ Enhanced RAG Engine not available")
    
    def _cmd_suggest_questions(self, args: str):
        """Get question suggestions based on concepts"""
        if not args:
            print("Usage: /suggest <concepts>")
            print("Example: /suggest ltssm power")
            return
        
        if not self.enhanced_rag:
            print("Enhanced RAG not available")
            return
        
        concepts = args.split()
        normalizer = self.enhanced_rag.question_normalizer
        suggestions = normalizer.suggest_related_questions(concepts)
        
        print(f"💡 Question Suggestions for: {', '.join(concepts)}")
        print("-" * 40)
        
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    
    def process_query(self, query: str) -> str:
        """Override main query processing to use enhanced RAG"""
        if query.startswith('/'):
            return super().process_query(query)
        
        # Use enhanced RAG for regular queries
        if self.enhanced_rag:
            try:
                response = self._process_enhanced_query(query)
                
                # Format enhanced response
                output = []
                output.append(f"📝 Answer (Confidence: {response.confidence:.1%}):")
                output.append(response.answer)
                
                if self.verbose and response.verification:
                    output.append(f"\n🔍 Verification: {response.verification.explanation}")
                
                if response.suggestions and len(response.suggestions) > 0:
                    output.append(f"\n💡 Related: {response.suggestions[0]}")
                
                return "\n".join(output)
                
            except Exception as e:
                logger.error(f"Enhanced query failed: {e}")
                return f"Enhanced processing failed: {e}"
        
        # Fallback to standard processing
        return super().process_query(query)

def main():
    """Main entry point for enhanced interactive shell"""
    parser = argparse.ArgumentParser(description="Enhanced PCIe Debug Agent with RAG v3")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument("--hybrid-search", action="store_true", default=True, help="Enable hybrid search")
    
    args = parser.parse_args()
    
    # Initialize and run enhanced shell
    shell = EnhancedInteractiveShell(args)
    shell.run()

if __name__ == "__main__":
    main()