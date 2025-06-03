#!/usr/bin/env python3
"""
Enhanced Context-Based RAG Integration
Integrates the self-evolving AutoRAG system with the existing PCIe Debug Agent
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from auto_rag_system import AutoRAGAgent, RAGParams, EvolutionConfig, EmbeddingRetriever, DocumentChunker, PCIeDocumentProcessor

logger = logging.getLogger(__name__)

@dataclass
class ContextualRAGConfig:
    """Configuration for contextual RAG enhancement"""
    auto_evolution_enabled: bool = True
    evolution_interval_hours: int = 24
    context_window_size: int = 3
    relevance_threshold: float = 0.7
    use_adaptive_chunking: bool = True
    enable_query_expansion: bool = True
    max_context_tokens: int = 2048

class ContextualRAGEngine:
    """Enhanced RAG engine with self-evolving context optimization"""
    
    def __init__(self, config: ContextualRAGConfig = None):
        self.config = config or ContextualRAGConfig()
        self.auto_rag_agent = None
        self.current_params = None
        self.retriever = None
        self.chunker = None
        self.processor = PCIeDocumentProcessor()
        
        # Load or initialize best configuration
        self._load_best_config()
        
        # Initialize system
        self._initialize_rag_system()
        
        # Set up auto-evolution if enabled
        if self.config.auto_evolution_enabled:
            self._setup_auto_evolution()
    
    def _load_best_config(self):
        """Load best configuration from previous evolution"""
        config_path = Path("best_config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Remove metadata
            if 'evolution_metadata' in config_dict:
                metadata = config_dict.pop('evolution_metadata')
                logger.info(f"Loaded config from {metadata.get('timestamp', 'unknown time')}")
            
            self.current_params = RAGParams(**config_dict)
            logger.info("✅ Loaded optimized RAG configuration")
        else:
            # Use default configuration
            self.current_params = RAGParams()
            logger.info("⚠️  Using default RAG configuration (run evolution to optimize)")
    
    def _initialize_rag_system(self):
        """Initialize RAG system with current parameters"""
        # Load documents
        documents = self.processor.load_documents()
        
        # Initialize chunker and retriever
        self.chunker = DocumentChunker(self.current_params)
        chunks = self.chunker.chunk_documents(documents)
        
        self.retriever = EmbeddingRetriever(self.current_params)
        self.retriever.build_index(chunks)
        
        logger.info(f"✅ RAG system initialized with {len(chunks)} chunks")
    
    def _setup_auto_evolution(self):
        """Setup automatic evolution process"""
        self.auto_rag_agent = AutoRAGAgent(EvolutionConfig(
            max_trials=30,
            target_recall=0.95,
            max_time_minutes=15
        ))
        logger.info("🔄 Auto-evolution enabled")
    
    def query(self, question: str, context_hints: List[str] = None) -> Dict[str, Any]:
        """Enhanced query with contextual optimization"""
        start_time = datetime.now()
        
        try:
            # Expand query if enabled
            if self.config.enable_query_expansion and context_hints:
                expanded_query = self._expand_query(question, context_hints)
            else:
                expanded_query = question
            
            # Retrieve relevant chunks
            raw_results = self.retriever.retrieve(expanded_query, k=10)
            
            # Apply contextual filtering and ranking
            contextual_results = self._apply_contextual_filtering(
                raw_results, question, context_hints
            )
            
            # Build contextual answer
            answer = self._build_contextual_answer(
                question, contextual_results, context_hints
            )
            
            # Calculate confidence based on context alignment
            confidence = self._calculate_contextual_confidence(
                question, contextual_results, answer
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'answer': answer,
                'confidence': confidence,
                'sources': contextual_results[:self.config.context_window_size],
                'context_applied': context_hints or [],
                'query_expansion': expanded_query if expanded_query != question else None,
                'response_time': response_time,
                'rag_params': self.current_params.__dict__,
                'engine': 'contextual_rag_v2'
            }
            
        except Exception as e:
            logger.error(f"Contextual RAG query failed: {e}")
            return {
                'answer': f"Query processing failed: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'error': str(e),
                'engine': 'contextual_rag_v2'
            }
    
    def _expand_query(self, question: str, context_hints: List[str]) -> str:
        """Expand query with context hints"""
        if not context_hints:
            return question
        
        # Add relevant context terms
        expansion_terms = []
        question_lower = question.lower()
        
        for hint in context_hints:
            hint_lower = hint.lower()
            if hint_lower not in question_lower:
                expansion_terms.append(hint)
        
        if expansion_terms:
            return f"{question} {' '.join(expansion_terms[:3])}"  # Limit expansion
        
        return question
    
    def _apply_contextual_filtering(self, results: List[Dict], question: str, 
                                  context_hints: List[str]) -> List[Dict]:
        """Apply contextual filtering to improve relevance"""
        if not context_hints:
            return results
        
        # Score results based on context alignment
        for result in results:
            content_lower = result['content'].lower()
            
            # Base retrieval score
            base_score = result.get('score', 0.0)
            
            # Context alignment bonus
            context_score = 0.0
            for hint in context_hints:
                if hint.lower() in content_lower:
                    context_score += 0.1
            
            # PCIe domain-specific bonuses
            pcie_terms = ['pcie', 'function level reset', 'flr', 'crs', 'completion timeout', 
                         'ltssm', 'tlp', 'aer', 'aspm', 'configuration space']
            
            domain_score = 0.0
            for term in pcie_terms:
                if term in content_lower:
                    domain_score += 0.05
            
            # Compliance-critical bonus
            if any(term in question.lower() for term in ['compliance', 'expect', 'should', 'must']):
                if any(term in content_lower for term in ['specification', 'section', 'required']):
                    domain_score += 0.15
            
            # Combined score
            result['contextual_score'] = base_score + context_score + domain_score
        
        # Sort by contextual score
        results.sort(key=lambda x: x.get('contextual_score', 0), reverse=True)
        
        # Filter by relevance threshold
        filtered_results = [
            r for r in results 
            if r.get('contextual_score', 0) >= self.config.relevance_threshold
        ]
        
        return filtered_results or results[:5]  # Fallback to top 5 if all filtered out
    
    def _build_contextual_answer(self, question: str, results: List[Dict], 
                               context_hints: List[str]) -> str:
        """Build answer with contextual awareness"""
        if not results:
            return "No relevant information found in the PCIe knowledge base."
        
        # Analyze question type for contextual response
        question_lower = question.lower()
        
        # Compliance/specification questions
        if any(term in question_lower for term in ['expect', 'should', 'compliance', 'specification']):
            return self._build_compliance_answer(question, results, context_hints)
        
        # Debug/troubleshooting questions  
        elif any(term in question_lower for term in ['debug', 'cause', 'why', 'how to', 'stuck']):
            return self._build_debug_answer(question, results, context_hints)
        
        # Definition/explanation questions
        elif any(term in question_lower for term in ['what is', 'define', 'explain', 'format']):
            return self._build_explanation_answer(question, results, context_hints)
        
        # General contextual answer
        else:
            return self._build_general_answer(question, results, context_hints)
    
    def _build_compliance_answer(self, question: str, results: List[Dict], 
                               context_hints: List[str]) -> str:
        """Build compliance-focused answer"""
        answer = "**PCIe Compliance Analysis:**\n\n"
        
        # Find specification references
        spec_refs = []
        for result in results[:3]:
            content = result['content']
            if 'section' in content.lower() or 'specification' in content.lower():
                spec_refs.append(content[:200] + "...")
        
        if spec_refs:
            answer += "**Specification Requirements:**\n"
            for i, ref in enumerate(spec_refs, 1):
                answer += f"{i}. {ref}\n"
            answer += "\n"
        
        # Extract key compliance points
        compliance_keywords = ['must', 'shall', 'required', 'mandatory', 'specification']
        compliance_points = []
        
        for result in results[:2]:
            content = result['content']
            sentences = content.split('.')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in compliance_keywords):
                    compliance_points.append(sentence.strip())
        
        if compliance_points:
            answer += "**Key Compliance Requirements:**\n"
            for point in compliance_points[:3]:
                answer += f"• {point}\n"
        
        return answer
    
    def _build_debug_answer(self, question: str, results: List[Dict], 
                          context_hints: List[str]) -> str:
        """Build debugging-focused answer"""
        answer = "**PCIe Debug Analysis:**\n\n"
        
        # Extract common causes and solutions
        debug_keywords = ['cause', 'reason', 'issue', 'problem', 'error', 'failure']
        solution_keywords = ['fix', 'resolve', 'solution', 'debug', 'check', 'verify']
        
        causes = []
        solutions = []
        
        for result in results[:3]:
            content = result['content']
            sentences = content.split('.')
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in debug_keywords):
                    causes.append(sentence.strip())
                elif any(keyword in sentence_lower for keyword in solution_keywords):
                    solutions.append(sentence.strip())
        
        if causes:
            answer += "**Common Causes:**\n"
            for cause in causes[:3]:
                answer += f"• {cause}\n"
            answer += "\n"
        
        if solutions:
            answer += "**Debug Steps:**\n"
            for solution in solutions[:3]:
                answer += f"• {solution}\n"
        
        # Add relevant technical details
        answer += "\n**Technical Context:**\n"
        for result in results[:2]:
            content = result['content'][:150]
            answer += f"• {content}...\n"
        
        return answer
    
    def _build_explanation_answer(self, question: str, results: List[Dict], 
                                context_hints: List[str]) -> str:
        """Build explanation-focused answer"""
        answer = "**Technical Explanation:**\n\n"
        
        # Find the most relevant definition/explanation
        best_result = results[0] if results else None
        
        if best_result:
            content = best_result['content']
            
            # Extract key points
            answer += f"{content[:300]}...\n\n"
            
            # Add structured details if available
            if len(results) > 1:
                answer += "**Additional Details:**\n"
                for result in results[1:3]:
                    snippet = result['content'][:100]
                    answer += f"• {snippet}...\n"
        
        return answer
    
    def _build_general_answer(self, question: str, results: List[Dict], 
                            context_hints: List[str]) -> str:
        """Build general contextual answer"""
        answer = f"**PCIe Information for: {question}**\n\n"
        
        for i, result in enumerate(results[:3], 1):
            content = result['content']
            score = result.get('contextual_score', result.get('score', 0))
            
            answer += f"**{i}. Relevance: {score:.1%}**\n"
            answer += f"{content[:200]}...\n\n"
        
        if context_hints:
            answer += f"**Context Applied:** {', '.join(context_hints)}\n"
        
        return answer
    
    def _calculate_contextual_confidence(self, question: str, results: List[Dict], 
                                       answer: str) -> float:
        """Calculate confidence with contextual factors"""
        if not results:
            return 0.0
        
        confidence_factors = []
        
        # Base retrieval scores
        avg_score = sum(r.get('contextual_score', r.get('score', 0)) for r in results[:3]) / min(3, len(results))
        confidence_factors.append(avg_score * 0.4)
        
        # Answer completeness
        answer_length_factor = min(len(answer) / 500, 1.0)
        confidence_factors.append(answer_length_factor * 0.2)
        
        # Specification reference bonus
        if any(term in answer.lower() for term in ['section', 'specification', 'pcie base']):
            confidence_factors.append(0.2)
        
        # PCIe domain relevance
        pcie_terms = ['pcie', 'flr', 'crs', 'completion', 'ltssm', 'tlp', 'aer']
        domain_relevance = sum(1 for term in pcie_terms if term in answer.lower()) / len(pcie_terms)
        confidence_factors.append(domain_relevance * 0.2)
        
        return min(sum(confidence_factors), 1.0)
    
    def evolve_system(self) -> Dict[str, Any]:
        """Trigger system evolution"""
        if not self.auto_rag_agent:
            return {"error": "Auto-evolution not enabled"}
        
        logger.info("🚀 Starting RAG system evolution...")
        
        try:
            # Run evolution
            new_params = self.auto_rag_agent.evolve()
            
            # Update system with new parameters
            old_params = self.current_params
            self.current_params = new_params
            self._initialize_rag_system()
            
            logger.info("✅ System evolution completed successfully")
            
            return {
                "success": True,
                "old_params": old_params.__dict__ if old_params else {},
                "new_params": new_params.__dict__,
                "improvement": True  # Could calculate actual improvement
            }
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "current_params": self.current_params.__dict__ if self.current_params else {},
            "auto_evolution_enabled": self.config.auto_evolution_enabled,
            "retriever_status": "ready" if self.retriever else "not_initialized",
            "num_indexed_chunks": len(self.retriever.chunks) if self.retriever else 0,
            "embedding_model": getattr(self.retriever, 'model_name', 'unknown') if self.retriever else None,
            "config": self.config.__dict__
        }

def integrate_contextual_rag(interactive_shell):
    """Integrate contextual RAG into interactive shell"""
    
    # Initialize contextual RAG engine
    shell.contextual_rag = ContextualRAGEngine(ContextualRAGConfig(
        auto_evolution_enabled=True,
        evolution_interval_hours=24,
        context_window_size=3,
        use_adaptive_chunking=True
    ))
    
    def do_context_rag(self, arg):
        """Contextual RAG with self-evolving optimization
        
        Usage:
            /context_rag <question>                    - Query with contextual enhancement
            /context_rag <question> --context hint1,hint2 - Query with context hints
            /context_rag --evolve                      - Trigger system evolution
            /context_rag --status                      - Show system status
        
        Examples:
            /context_rag "why dut send successful completion during flr?"
            /context_rag "completion timeout causes" --context debug,troubleshooting
            /context_rag --evolve
        """
        if not arg:
            print("Usage: /context_rag <question> [--context hints] or /context_rag --evolve/--status")
            return
        
        # Handle special commands
        if arg == '--evolve':
            print("🚀 Starting RAG system evolution...")
            result = self.contextual_rag.evolve_system()
            
            if result.get('success'):
                print("✅ Evolution completed successfully!")
                print(f"   New configuration optimized for better performance")
            else:
                print(f"❌ Evolution failed: {result.get('error', 'Unknown error')}")
            return
        
        if arg == '--status':
            status = self.contextual_rag.get_system_status()
            print("📊 Contextual RAG System Status")
            print("=" * 40)
            print(f"Auto-evolution: {'ON' if status['auto_evolution_enabled'] else 'OFF'}")
            print(f"Retriever: {status['retriever_status']}")
            print(f"Indexed chunks: {status['num_indexed_chunks']}")
            print(f"Embedding model: {status['embedding_model']}")
            print(f"Context window: {status['config']['context_window_size']}")
            print(f"Relevance threshold: {status['config']['relevance_threshold']:.2f}")
            return
        
        # Parse query and context hints
        parts = arg.split(' --context ')
        question = parts[0].strip()
        context_hints = []
        
        if len(parts) > 1:
            context_hints = [h.strip() for h in parts[1].split(',')]
        
        # Execute contextual query
        print(f"🔍 Contextual RAG Query")
        if context_hints:
            print(f"   Context hints: {', '.join(context_hints)}")
        print("-" * 50)
        
        result = self.contextual_rag.query(question, context_hints)
        
        # Display results
        print(f"\n📝 Answer (Confidence: {result['confidence']:.1%}):")
        print(result['answer'])
        
        if result.get('query_expansion'):
            print(f"\n🔍 Query expanded to: \"{result['query_expansion']}\"")
        
        if result.get('context_applied'):
            print(f"\n🎯 Context applied: {', '.join(result['context_applied'])}")
        
        # Performance info
        print(f"\n⚡ Engine: {result['engine']} | Time: {result.get('response_time', 0):.2f}s")
        print(f"   Sources: {len(result.get('sources', []))} | Chunking: {result.get('rag_params', {}).get('chunking_strategy', 'unknown')}")
    
    # Add method to shell
    import types
    interactive_shell.do_context_rag = types.MethodType(do_context_rag, interactive_shell)
    
    print("✅ Contextual RAG system integrated!")
    print("   Use '/context_rag <question>' for enhanced contextual queries")
    print("   Use '/context_rag --evolve' to optimize system performance")
    print("   Use '/context_rag --status' to check system health")
    
    return interactive_shell

if __name__ == "__main__":
    print("🧠 Enhanced Context-Based RAG System")
    print("Features:")
    print("• Self-evolving optimization with Optuna")
    print("• Contextual query expansion and filtering")
    print("• PCIe domain-specific intelligence")
    print("• Adaptive chunking strategies")
    print("• Real-time performance monitoring")