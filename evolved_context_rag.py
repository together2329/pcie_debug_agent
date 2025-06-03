#!/usr/bin/env python3
"""
Evolved Context RAG Interface
Integrates the evolved RAG system with contextual query processing
"""

import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

from auto_evolving_rag import AutoEvolvingRAG, EvolutionConfig, RAGParams

class EvolvedContextRAG:
    """Enhanced RAG with evolved parameters and context processing"""
    
    def __init__(self):
        # Load evolved system
        self.rag_system = AutoEvolvingRAG(EvolutionConfig(
            max_trials=5,  # Quick evolution for demo
            target_recall=0.80
        ))
        
        # Evolution status
        self.evolution_status = "initialized"
        self.query_count = 0
        self.total_evolution_time = 0
    
    def context_rag_query(self, question: str, context_hints: List[str] = None) -> Dict[str, Any]:
        """Process contextual RAG query"""
        start_time = time.time()
        self.query_count += 1
        
        print(f"🔍 Processing Query {self.query_count}: {question}")
        if context_hints:
            print(f"   Context Hints: {', '.join(context_hints)}")
        
        # Expand query with context hints
        expanded_query = self._expand_query_with_context(question, context_hints)
        
        # Get response from evolved system
        result = self.rag_system.query_with_best_params(expanded_query)
        
        # Enhance response with context
        enhanced_result = self._enhance_response_with_context(
            result, question, context_hints
        )
        
        response_time = time.time() - start_time
        enhanced_result['response_time'] = response_time
        enhanced_result['query_expansion'] = expanded_query if expanded_query != question else None
        enhanced_result['context_applied'] = context_hints or []
        
        return enhanced_result
    
    def _expand_query_with_context(self, question: str, context_hints: List[str]) -> str:
        """Expand query with contextual hints"""
        if not context_hints:
            return question
        
        # Add relevant context terms that aren't already in the question
        question_lower = question.lower()
        expansion_terms = []
        
        for hint in context_hints:
            hint_lower = hint.lower()
            if hint_lower not in question_lower:
                expansion_terms.append(hint)
        
        if expansion_terms:
            return f"{question} {' '.join(expansion_terms[:2])}"  # Limit to 2 terms
        
        return question
    
    def _enhance_response_with_context(self, result: Dict[str, Any], 
                                     question: str, context_hints: List[str]) -> Dict[str, Any]:
        """Enhance response with contextual information"""
        
        # Detect question type for context-specific enhancement
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['expect', 'should', 'compliance']):
            result['analysis_type'] = 'compliance'
            result['answer'] = self._enhance_compliance_answer(result['answer'], context_hints)
        
        elif any(term in question_lower for term in ['debug', 'cause', 'why', 'timeout']):
            result['analysis_type'] = 'debugging'
            result['answer'] = self._enhance_debug_answer(result['answer'], context_hints)
        
        elif any(term in question_lower for term in ['what is', 'format', 'structure']):
            result['analysis_type'] = 'technical'
            result['answer'] = self._enhance_technical_answer(result['answer'], context_hints)
        
        else:
            result['analysis_type'] = 'general'
        
        # Add contextual recommendations
        result['recommendations'] = self._generate_context_recommendations(
            question, context_hints, result['analysis_type']
        )
        
        return result
    
    def _enhance_compliance_answer(self, answer: str, context_hints: List[str]) -> str:
        """Enhance compliance-focused answers"""
        enhancement = "\n\n**Compliance Focus:**\n"
        
        if context_hints:
            for hint in context_hints:
                if 'compliance' in hint.lower():
                    enhancement += f"• Checking {hint} requirements\n"
                elif 'spec' in hint.lower():
                    enhancement += f"• Referencing {hint} documentation\n"
        
        enhancement += "• Verify implementation against PCIe Base Specification\n"
        enhancement += "• Test for compliance violations with appropriate tools\n"
        
        return answer + enhancement
    
    def _enhance_debug_answer(self, answer: str, context_hints: List[str]) -> str:
        """Enhance debugging-focused answers"""
        enhancement = "\n\n**Debug Recommendations:**\n"
        
        if context_hints:
            if 'troubleshooting' in ' '.join(context_hints).lower():
                enhancement += "• Follow systematic troubleshooting methodology\n"
            if 'debug' in ' '.join(context_hints).lower():
                enhancement += "• Enable detailed logging and monitoring\n"
        
        enhancement += "• Check system logs for related error patterns\n"
        enhancement += "• Verify hardware configuration and connections\n"
        enhancement += "• Test with known-good configurations\n"
        
        return answer + enhancement
    
    def _enhance_technical_answer(self, answer: str, context_hints: List[str]) -> str:
        """Enhance technical explanations"""
        enhancement = "\n\n**Technical Details:**\n"
        
        if context_hints:
            for hint in context_hints:
                if 'format' in hint.lower():
                    enhancement += f"• Detailed {hint} specifications available\n"
                elif 'implementation' in hint.lower():
                    enhancement += f"• {hint} considerations for design\n"
        
        enhancement += "• Refer to latest PCIe specification for complete details\n"
        enhancement += "• Consider backward compatibility requirements\n"
        
        return answer + enhancement
    
    def _generate_context_recommendations(self, question: str, context_hints: List[str], 
                                        analysis_type: str) -> List[str]:
        """Generate contextual recommendations"""
        recommendations = []
        
        # Base recommendations by analysis type
        if analysis_type == 'compliance':
            recommendations.extend([
                "Review PCIe Base Specification Section 6.6.1.2 for FLR requirements",
                "Test with PCIe compliance tools",
                "Verify timing requirements are met"
            ])
        elif analysis_type == 'debugging':
            recommendations.extend([
                "Enable verbose logging for detailed analysis",
                "Check hardware connections and power states",
                "Monitor PCIe traffic with protocol analyzer"
            ])
        elif analysis_type == 'technical':
            recommendations.extend([
                "Consult PCIe specification for complete technical details",
                "Review reference implementations",
                "Consider performance implications"
            ])
        
        # Context-specific recommendations
        if context_hints:
            for hint in context_hints:
                if 'flr' in hint.lower():
                    recommendations.append("Focus on Function Level Reset implementation details")
                elif 'timeout' in hint.lower():
                    recommendations.append("Analyze timeout configuration and handling")
                elif 'ltssm' in hint.lower():
                    recommendations.append("Review LTSSM state machine operation")
        
        return recommendations[:5]  # Limit to top 5
    
    def evolve_system(self) -> Dict[str, Any]:
        """Trigger system evolution"""
        print("🚀 Triggering RAG system evolution...")
        
        evolution_start = time.time()
        
        try:
            # Run evolution
            best_params = self.rag_system.evolve()
            
            evolution_time = time.time() - evolution_start
            self.total_evolution_time += evolution_time
            
            if best_params:
                self.evolution_status = "evolved"
                status = self.rag_system.get_evolution_status()
                
                print(f"✅ Evolution completed in {evolution_time:.2f}s")
                print(f"   Best Score: {status['best_score']:.4f}")
                print(f"   Generation: {status['current_generation']}")
                
                return {
                    'success': True,
                    'evolution_time': evolution_time,
                    'best_score': status['best_score'],
                    'generation': status['current_generation'],
                    'best_params': status['best_params']
                }
            else:
                return {'success': False, 'error': 'Evolution failed'}
                
        except Exception as e:
            print(f"❌ Evolution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        rag_status = self.rag_system.get_evolution_status()
        
        return {
            'evolution_status': self.evolution_status,
            'current_generation': rag_status['current_generation'],
            'total_queries': self.query_count,
            'total_evolution_time': self.total_evolution_time,
            'best_score': rag_status['best_score'],
            'total_trials': rag_status['total_trials'],
            'best_config': rag_status['best_params']
        }

def simulate_context_rag_session():
    """Simulate a context RAG session"""
    print("🧠 Evolved Context RAG Session")
    print("=" * 50)
    
    # Initialize system
    context_rag = EvolvedContextRAG()
    
    # Test queries with different contexts
    test_queries = [
        {
            "query": "why dut send successful completion during flr?",
            "context": ["compliance", "reset"],
            "description": "FLR compliance issue"
        },
        {
            "query": "completion timeout debug",
            "context": ["troubleshooting", "debug"],
            "description": "Timeout debugging"
        },
        {
            "query": "PCIe TLP header format",
            "context": ["specification", "format"],
            "description": "Technical specification"
        },
        {
            "query": "LTSSM stuck in polling state",
            "context": ["debug", "ltssm"],
            "description": "State machine issue"
        }
    ]
    
    print(f"\n📋 Running {len(test_queries)} test queries...")
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {test['description']}")
        print("-" * 30)
        
        result = context_rag.context_rag_query(
            test['query'], 
            test['context']
        )
        
        print(f"✅ Analysis Type: {result.get('analysis_type', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Response Time: {result.get('response_time', 0):.3f}s")
        print(f"   Context Applied: {', '.join(result.get('context_applied', []))}")
        
        if result.get('query_expansion'):
            print(f"   Query Expanded: {result['query_expansion']}")
        
        # Show recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"   Recommendations: {len(recommendations)} items")
            for rec in recommendations[:2]:  # Show first 2
                print(f"     • {rec}")
        
        # Brief pause
        time.sleep(0.5)
    
    # Trigger evolution
    print(f"\n🧬 TRIGGERING SYSTEM EVOLUTION")
    print("-" * 40)
    
    evolution_result = context_rag.evolve_system()
    
    if evolution_result.get('success'):
        print(f"🎉 Evolution successful!")
        print(f"   Time: {evolution_result['evolution_time']:.2f}s")
        print(f"   Score: {evolution_result['best_score']:.4f}")
    
    # Final status
    print(f"\n📊 FINAL SYSTEM STATUS")
    print("-" * 30)
    
    status = context_rag.get_status()
    print(f"Evolution Status: {status['evolution_status']}")
    print(f"Current Generation: {status['current_generation']}")
    print(f"Total Queries Processed: {status['total_queries']}")
    print(f"Best Score Achieved: {status['best_score']:.4f}")
    print(f"Total Evolution Time: {status['total_evolution_time']:.2f}s")
    
    if status['best_config']:
        config = status['best_config']
        print(f"\nBest Configuration:")
        print(f"  Strategy: {config['chunking_strategy']}")
        print(f"  Chunk Size: {config['base_chunk_size']}")
        print(f"  Overlap: {config['overlap_ratio']}")
        print(f"  Max Context: {config['max_total_ctx_tokens']}")
    
    return context_rag

if __name__ == "__main__":
    simulate_context_rag_session()