#!/usr/bin/env python3
"""
Deploy Enhanced Context-Based RAG System
One-command deployment of the complete self-evolving RAG system
"""

import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy', 'pandas', 'pydantic', 'python-dotenv',
        'langchain', 'sentence-transformers', 'faiss-cpu', 'optuna',
        'tqdm', 'pyyaml'
    ]
    
    package_status = {}
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            package_status[package] = True
        except ImportError:
            package_status[package] = False
    
    return package_status

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing AutoRAG dependencies...")
    
    try:
        # Install from requirements file
        requirements_file = Path("requirements_auto_rag.txt")
        if requirements_file.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
        else:
            # Install core packages manually
            core_packages = [
                "numpy", "pandas", "pydantic", "python-dotenv",
                "langchain>=0.2.0", "sentence-transformers", "faiss-cpu", "optuna",
                "PyMuPDF", "tqdm", "pyyaml"
            ]
            
            for package in core_packages:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package
                ])
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Setup required directories"""
    directories = ["specs", "logs", "data"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directory structure created")

def run_initial_evolution():
    """Run initial system evolution"""
    print("🚀 Running initial RAG system evolution...")
    
    try:
        from auto_rag_system import AutoRAGAgent, EvolutionConfig
        
        # Quick evolution for setup
        config = EvolutionConfig(
            max_trials=10,  # Quick setup
            target_recall=0.80,
            max_time_minutes=5
        )
        
        agent = AutoRAGAgent(config)
        best_params = agent.evolve()
        
        print("✅ Initial evolution completed")
        return True
        
    except Exception as e:
        print(f"⚠️  Initial evolution failed: {e}")
        print("   System will use default configuration")
        return False

def test_system():
    """Test the enhanced RAG system"""
    print("🧪 Testing enhanced RAG system...")
    
    try:
        from enhanced_context_rag import ContextualRAGEngine, ContextualRAGConfig
        
        # Initialize engine
        config = ContextualRAGConfig(
            auto_evolution_enabled=False,  # Disable for testing
            context_window_size=3
        )
        
        engine = ContextualRAGEngine(config)
        
        # Test query
        test_question = "why dut send successful return during flr?"
        result = engine.query(test_question, ["compliance", "reset"])
        
        print(f"Test query: {test_question}")
        print(f"Response confidence: {result['confidence']:.1%}")
        print(f"Response time: {result.get('response_time', 0):.2f}s")
        
        if result['confidence'] > 0.3:
            print("✅ System test passed")
            return True
        else:
            print("⚠️  System test shows low confidence")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def integrate_with_shell():
    """Integrate with existing interactive shell"""
    print("🔗 Integrating with PCIe Debug Agent shell...")
    
    try:
        # Check if interactive shell exists
        shell_path = Path("src/cli/interactive.py")
        
        if shell_path.exists():
            # Create integration script
            integration_code = '''
# Add this to your interactive shell initialization

try:
    from enhanced_context_rag import integrate_contextual_rag
    integrate_contextual_rag(self)
    print("✅ Enhanced contextual RAG integrated")
except Exception as e:
    print(f"⚠️  Contextual RAG integration failed: {e}")
'''
            
            with open("rag_integration_snippet.py", "w") as f:
                f.write(integration_code)
            
            print("✅ Integration snippet created: rag_integration_snippet.py")
            return True
        else:
            print("⚠️  Interactive shell not found, manual integration required")
            return False
            
    except Exception as e:
        print(f"❌ Integration setup failed: {e}")
        return False

def deploy_enhanced_rag():
    """Deploy the complete enhanced RAG system"""
    
    print("🚀 Deploying Enhanced Context-Based RAG System")
    print("=" * 60)
    
    success_count = 0
    total_steps = 6
    
    # Step 1: Check dependencies
    print("\n1. Checking dependencies...")
    package_status = check_dependencies()
    missing_packages = [pkg for pkg, status in package_status.items() if not status]
    
    if missing_packages:
        print(f"   Missing packages: {', '.join(missing_packages)}")
        if install_dependencies():
            success_count += 1
    else:
        print("   ✅ All dependencies satisfied")
        success_count += 1
    
    # Step 2: Setup directories
    print("\n2. Setting up directory structure...")
    setup_directories()
    success_count += 1
    
    # Step 3: Run initial evolution
    print("\n3. Running initial system evolution...")
    if run_initial_evolution():
        success_count += 1
    
    # Step 4: Test system
    print("\n4. Testing enhanced RAG system...")
    if test_system():
        success_count += 1
    
    # Step 5: Integration setup
    print("\n5. Setting up integration...")
    if integrate_with_shell():
        success_count += 1
    
    # Step 6: Final validation
    print("\n6. Final system validation...")
    try:
        from enhanced_context_rag import ContextualRAGEngine
        from auto_rag_system import AutoRAGAgent
        
        print("   ✅ All modules importable")
        success_count += 1
    except Exception as e:
        print(f"   ❌ Module import failed: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 DEPLOYMENT SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nSteps completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        status = "🟢 READY"
    elif success_count >= 4:
        print("✅ DEPLOYMENT MOSTLY SUCCESSFUL")
        status = "🟡 FUNCTIONAL"
    else:
        print("⚠️  DEPLOYMENT PARTIALLY FAILED")
        status = "🟠 LIMITED"
    
    print(f"\nSystem Status: {status}")
    
    print(f"\n📚 Available Commands:")
    print("   /context_rag <question>              - Enhanced contextual query")
    print("   /context_rag <question> --context h1,h2 - Query with context hints")
    print("   /context_rag --evolve                - Trigger system evolution")
    print("   /context_rag --status                - Show system status")
    
    print(f"\n🔍 Example Usage:")
    print('   /context_rag "why dut send successful return during flr?"')
    print('   /context_rag "completion timeout debug" --context troubleshooting,debug')
    print('   /context_rag --evolve')
    
    print(f"\n📈 Enhanced Features:")
    print("   • Self-evolving optimization with Optuna")
    print("   • Contextual query expansion and filtering")
    print("   • PCIe domain-specific intelligence")
    print("   • Adaptive chunking strategies (fixed/heading-aware/sliding)")
    print("   • Real-time performance monitoring")
    print("   • Auto-evolution every 24 hours")
    
    print(f"\n🛠️  Manual Integration (if needed):")
    print("   1. Add to your interactive shell:")
    print("      from enhanced_context_rag import integrate_contextual_rag")
    print("      integrate_contextual_rag(shell)")
    print("   2. Or use the integration snippet in rag_integration_snippet.py")
    
    return success_count >= 4

def quick_demo():
    """Quick demonstration of the enhanced system"""
    print("🎬 Enhanced Context RAG Demo")
    print("-" * 40)
    
    try:
        from enhanced_context_rag import ContextualRAGEngine, ContextualRAGConfig
        
        # Initialize with demo config
        config = ContextualRAGConfig(
            auto_evolution_enabled=False,
            context_window_size=2,
            enable_query_expansion=True
        )
        
        engine = ContextualRAGEngine(config)
        
        # Demo queries
        demo_queries = [
            ("why dut send successful return during flr?", ["compliance", "reset"]),
            ("completion timeout causes", ["debug", "troubleshooting"]),
            ("PCIe TLP header format", ["specification", "format"])
        ]
        
        for question, context_hints in demo_queries:
            print(f"\n🔍 Query: {question}")
            print(f"   Context: {', '.join(context_hints)}")
            
            result = engine.query(question, context_hints)
            
            print(f"   Confidence: {result['confidence']:.1%}")
            print(f"   Time: {result.get('response_time', 0):.2f}s")
            print(f"   Answer: {result['answer'][:100]}...")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        quick_demo()
    else:
        deploy_enhanced_rag()