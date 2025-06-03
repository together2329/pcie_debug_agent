#!/usr/bin/env python3
"""
Deploy Unified RAG System
One-command deployment of the complete unified RAG system
"""

import sys
import importlib.util
from pathlib import Path

def deploy_unified_rag():
    """Deploy the complete unified RAG system"""
    
    print("🚀 Deploying Unified RAG System for PCIe Debug Agent")
    print("=" * 60)
    
    try:
        # Import the integration module
        from unified_rag_integration import integrate_unified_rag
        from automated_rag_test_suite import PCIeRAGTestSuite
        
        print("✅ Unified RAG modules loaded")
        
        # Try to import and initialize the interactive shell
        try:
            from src.cli.interactive import PCIeInteractiveShell
            shell = PCIeInteractiveShell()
            print("✅ PCIe Interactive Shell initialized")
        except Exception as e:
            print(f"⚠️  Using mock shell (interactive not available): {e}")
            
            # Create mock shell for testing
            class MockShell:
                def __init__(self):
                    self.vector_store = None
                    self.model_manager = None
                    self.rag_engine = None
            
            shell = MockShell()
        
        # Integrate unified RAG system
        shell = integrate_unified_rag(shell)
        print("✅ Unified RAG system integrated")
        
        # Run initial quality test
        print("\n🧪 Running initial quality assessment...")
        
        try:
            # Quick test
            test_suite = PCIeRAGTestSuite()
            critical_tests = [tc for tc in test_suite.test_cases if tc.compliance_critical][:2]
            
            if hasattr(shell, 'unified_rag'):
                test_results = []
                for test_case in critical_tests:
                    result = test_suite._run_single_test(test_case, shell.unified_rag)
                    test_results.append(result)
                
                avg_score = sum(r.score for r in test_results) / len(test_results)
                print(f"✅ Initial quality score: {avg_score:.2f}/1.00")
                
                if avg_score >= 0.7:
                    print("🎉 System ready for production use!")
                elif avg_score >= 0.5:
                    print("⚠️  System functional but needs improvement")
                else:
                    print("❌ System needs significant improvement")
            else:
                print("⚠️  Quality test skipped (unified RAG not available)")
        
        except Exception as e:
            print(f"⚠️  Quality test failed: {e}")
        
        # Show usage instructions
        print(f"\n{'='*60}")
        print("🎯 DEPLOYMENT COMPLETE")
        print(f"{'='*60}")
        
        print("\n📚 Available Commands:")
        print("   /rag <question>              - Intelligent PCIe analysis")
        print("   /rag --status                - Show system status")
        print("   /rag --test                  - Run quality test suite")
        print("   /rag --engines               - List available engines")
        print("   /rag --config                - Show configuration")
        
        print("\n🔍 Example Queries:")
        print('   /rag "why dut send successful return during flr?"')
        print('   /rag "PCIe completion timeout debug" --engine production')
        print('   /rag "LTSSM stuck in polling state"')
        
        print("\n📈 Quality Features:")
        print("   • Auto-testing every hour")
        print("   • Multi-engine fallback")
        print("   • Performance monitoring")
        print("   • Compliance checking")
        
        print("\n🎛️  Management Commands:")
        print("   /rag --test      # Run comprehensive quality test")
        print("   /rag --status    # Check system health")
        
        return shell
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check that vector database is built")
        print("3. Verify model configuration")
        return None

def quick_test():
    """Quick test of the unified system"""
    print("🧪 Quick Test of Unified RAG System")
    print("-" * 40)
    
    shell = deploy_unified_rag()
    
    if shell and hasattr(shell, 'unified_rag'):
        # Test with the critical FLR question
        test_question = "why dut send successful return of completion during flr ? I expect crs return"
        print(f"\n🔍 Testing: {test_question}")
        
        result = shell.unified_rag.query(test_question)
        
        print(f"\n📝 Answer (Confidence: {result['confidence']:.1%}):")
        print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])
        
        print(f"\n⚡ Performance:")
        print(f"   Engine: {result['engine']}")
        print(f"   Response Time: {result.get('response_time', 0):.2f}s")
        print(f"   Sources: {len(result.get('sources', []))}")
        
        if result['confidence'] >= 0.7:
            print("\n✅ EXCELLENT - System working well!")
        elif result['confidence'] >= 0.5:
            print("\n✅ GOOD - System functional")
        else:
            print("\n⚠️  NEEDS IMPROVEMENT - Check configuration")
    
    else:
        print("❌ Deployment failed - cannot run test")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        deploy_unified_rag()