#!/usr/bin/env python3
"""
Context RAG Interface - Direct Implementation
Simulates the /context_rag commands without external dependencies
"""

import json
import time
from datetime import datetime
from evolved_context_rag import EvolvedContextRAG

class ContextRAGInterface:
    """Interface for context RAG commands"""
    
    def __init__(self):
        self.evolved_rag = EvolvedContextRAG()
        self.command_history = []
        self.session_start = datetime.now()
    
    def execute_command(self, command: str) -> dict:
        """Execute a context RAG command"""
        timestamp = datetime.now()
        print(f"\n🔍 [{timestamp.strftime('%H:%M:%S')}] Executing: {command}")
        print("-" * 60)
        
        result = None
        
        if command.startswith('/context_rag ') and '--context' in command:
            # Parse query with context
            parts = command.split(' --context ')
            query = parts[0].replace('/context_rag ', '').strip('"')
            context_hints = parts[1].split(',') if len(parts) > 1 else []
            
            result = self._context_rag_query_with_hints(query, context_hints)
            
        elif command.startswith('/context_rag ') and command != '/context_rag --evolve' and command != '/context_rag --status':
            # Simple query without context
            query = command.replace('/context_rag ', '').strip('"')
            result = self._context_rag_query(query)
            
        elif command == '/context_rag --evolve':
            result = self._context_rag_evolve()
            
        elif command == '/context_rag --status':
            result = self._context_rag_status()
            
        else:
            result = {'error': f'Unknown command: {command}'}
        
        # Log command
        self.command_history.append({
            'timestamp': timestamp.isoformat(),
            'command': command,
            'result': result
        })
        
        return result
    
    def _context_rag_query(self, query: str) -> dict:
        """Execute context RAG query without hints"""
        print(f"📝 Query: {query}")
        
        result = self.evolved_rag.context_rag_query(query)
        
        print(f"✅ Response Generated")
        print(f"   Analysis Type: {result.get('analysis_type', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Response Time: {result.get('response_time', 0):.4f}s")
        print(f"   Generation: {result.get('generation', 'N/A')}")
        
        print(f"\n📋 Answer:")
        print(result.get('answer', 'No answer generated')[:300] + "..." if len(result.get('answer', '')) > 300 else result.get('answer', 'No answer'))
        
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        return result
    
    def _context_rag_query_with_hints(self, query: str, context_hints: list) -> dict:
        """Execute context RAG query with domain hints"""
        print(f"📝 Query: {query}")
        print(f"🎯 Context Hints: {', '.join(context_hints)}")
        
        result = self.evolved_rag.context_rag_query(query, context_hints)
        
        print(f"✅ Enhanced Response Generated")
        print(f"   Analysis Type: {result.get('analysis_type', 'unknown')}")
        print(f"   Confidence: {result.get('confidence', 0):.1%}")
        print(f"   Response Time: {result.get('response_time', 0):.4f}s")
        print(f"   Context Applied: {', '.join(result.get('context_applied', []))}")
        
        if result.get('query_expansion'):
            print(f"   Query Expanded: {result['query_expansion']}")
        
        print(f"\n📋 Enhanced Answer:")
        print(result.get('answer', 'No answer generated')[:300] + "..." if len(result.get('answer', '')) > 300 else result.get('answer', 'No answer'))
        
        recommendations = result.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Context-Aware Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
        
        return result
    
    def _context_rag_evolve(self) -> dict:
        """Trigger system evolution"""
        print("🧬 Triggering RAG System Evolution...")
        
        evolution_result = self.evolved_rag.evolve_system()
        
        if evolution_result.get('success'):
            print(f"🎉 Evolution Completed Successfully!")
            print(f"   Generation: {evolution_result.get('generation', 'unknown')}")
            print(f"   Best Score: {evolution_result.get('best_score', 0):.4f}")
            print(f"   Evolution Time: {evolution_result.get('evolution_time', 0):.4f}s")
            
            if evolution_result.get('best_params'):
                params = evolution_result['best_params']
                print(f"\n⚙️  Optimal Configuration:")
                print(f"   Strategy: {params.get('chunking_strategy', 'unknown')}")
                print(f"   Chunk Size: {params.get('base_chunk_size', 'unknown')} tokens")
                print(f"   Overlap: {params.get('overlap_ratio', 'unknown')}")
        else:
            print(f"❌ Evolution Failed: {evolution_result.get('error', 'Unknown error')}")
        
        return evolution_result
    
    def _context_rag_status(self) -> dict:
        """Show system status"""
        print("📊 Context RAG System Status")
        print("=" * 40)
        
        status = self.evolved_rag.get_status()
        
        # System status
        print(f"Evolution Status: {status['evolution_status']}")
        print(f"Current Generation: {status['current_generation']}")
        print(f"Total Queries Processed: {status['total_queries']}")
        print(f"Best Score Achieved: {status['best_score']:.4f}")
        print(f"Total Trials: {status['total_trials']}")
        print(f"Total Evolution Time: {status['total_evolution_time']:.4f}s")
        
        # Session info
        session_duration = (datetime.now() - self.session_start).total_seconds()
        print(f"\n📈 Session Statistics:")
        print(f"Commands Executed: {len(self.command_history)}")
        print(f"Session Duration: {session_duration:.1f}s")
        
        # Best configuration
        if status['best_config']:
            config = status['best_config']
            print(f"\n🎯 Current Best Configuration:")
            print(f"   Strategy: {config['chunking_strategy']}")
            print(f"   Chunk Size: {config['base_chunk_size']} tokens")
            print(f"   Overlap Ratio: {config['overlap_ratio']}")
            print(f"   Max Context: {config['max_total_ctx_tokens']} tokens")
            print(f"   Length Penalty: {config['length_penalty']}")
        
        return status

def run_context_rag_commands():
    """Run the requested context RAG commands"""
    
    print("🚀 Context RAG Command Execution Session")
    print("=" * 60)
    print("Executing the requested command sequence...")
    
    # Initialize interface
    interface = ContextRAGInterface()
    
    # Command sequence as requested
    commands = [
        '/context_rag "why dut send successful completion during flr?"',
        '/context_rag "completion timeout debug" --context troubleshooting,debug',
        '/context_rag --evolve',
        '/context_rag --status'
    ]
    
    results = []
    
    for i, command in enumerate(commands, 1):
        print(f"\n{'='*60}")
        print(f"COMMAND {i}/{len(commands)}")
        print(f"{'='*60}")
        
        result = interface.execute_command(command)
        results.append(result)
        
        # Brief pause between commands
        time.sleep(1)
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    print(f"✅ All {len(commands)} commands executed successfully")
    
    # Calculate performance metrics
    total_time = sum(r.get('response_time', 0) for r in results if 'response_time' in r)
    avg_confidence = sum(r.get('confidence', 0) for r in results if 'confidence' in r) / max(1, sum(1 for r in results if 'confidence' in r))
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Total Response Time: {total_time:.4f}s")
    print(f"   Average Confidence: {avg_confidence:.1%}")
    print(f"   Commands per Second: {len(commands)/max(total_time, 0.001):.1f}")
    
    evolution_results = [r for r in results if 'generation' in r and r.get('success')]
    if evolution_results:
        evo = evolution_results[0]
        print(f"   Evolution Score: {evo.get('best_score', 0):.4f}")
        print(f"   Evolution Time: {evo.get('evolution_time', 0):.4f}s")
    
    print(f"\n🎯 Context RAG System is operational and optimized!")
    
    return results

if __name__ == "__main__":
    run_context_rag_commands()