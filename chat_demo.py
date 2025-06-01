#!/usr/bin/env python
"""
Interactive Chat Demo with Local LLM for PCIe Debugging
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress Metal warnings and set quiet mode
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GGML_METAL_LOG_LEVEL"] = "0"
os.environ["LLAMA_CPP_LOG_LEVEL"] = "ERROR"

import logging
import time
from src.models.local_llm_provider import LocalLLMProvider
from src.models.model_manager import ModelManager
from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery
from src.rag.vector_store import FAISSVectorStore
from src.config.settings import load_settings
from src.models.pcie_prompts import PCIePromptTemplates

# Set up logging (suppress verbose output for demo)
logging.basicConfig(level=logging.WARNING)

class PCIeChatDemo:
    def __init__(self):
        self.engine = None
        self.model_manager = None
        self.setup_complete = False
        
    def setup_system(self):
        """Initialize the PCIe debugging system"""
        print("🔧 Initializing PCIe Debug Agent...")
        print("=" * 50)
        
        try:
            # Load settings
            config_path = Path("configs/settings.yaml")
            settings = load_settings(config_path)
            print("✅ Configuration loaded")
            
            # Initialize model manager
            self.model_manager = ModelManager()
            
            # Load embedding model
            print("📊 Loading embedding model...")
            self.model_manager.load_embedding_model(settings.embedding.model)
            print("✅ Embedding model ready")
            
            # Initialize local LLM
            print("🧠 Initializing local LLM...")
            self.model_manager.initialize_llm(
                provider=settings.llm.provider,
                model_name=settings.llm.model,
                models_dir=settings.local_llm.models_dir,
                n_ctx=settings.local_llm.n_ctx,
                n_gpu_layers=settings.local_llm.n_gpu_layers,
                verbose=False
            )
            print("✅ Local LLM ready")
            
            # Create vector store
            # Use the parent directory for the vector store path
            vector_store_dir = Path(settings.vector_store.index_path).parent
            vector_store = FAISSVectorStore(
                index_path=str(vector_store_dir),
                index_type=settings.vector_store.index_type,
                dimension=settings.embedding.dimension
            )
            print("✅ Vector store ready")
            
            # Add sample PCIe data for demo
            self.add_sample_data(vector_store)
            
            # Create RAG engine
            self.engine = EnhancedRAGEngine(
                vector_store=vector_store,
                model_manager=self.model_manager,
                llm_provider=settings.llm.provider,
                llm_model=settings.llm.model,
                temperature=0.1,
                max_tokens=500  # Reasonable response length
            )
            print("✅ RAG engine ready")
            
            self.setup_complete = True
            print("\n🎉 PCIe Debug Agent is ready for chat!")
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            print("Please check the troubleshooting guide.")
            return False
            
        return True
    
    def add_sample_data(self, vector_store):
        """Add sample PCIe log data for demonstration"""
        print("📋 Loading sample PCIe data...")
        
        sample_logs = [
            "PCIe: 0000:01:00.0 AER: Multiple Uncorrected (Non-Fatal) error received",
            "PCIe: 0000:01:00.0 link training failed, LTSSM stuck in Polling.Compliance", 
            "PCIe: 0000:01:00.0 PCIe Bus Error: severity=Uncorrected, type=Transaction Layer",
            "PCIe: 0000:02:00.0 link up, speed 8.0 GT/s, width x16",
            "PCIe: 0000:03:00.0 device configuration timeout, disabling device",
            "PCIe: 0000:01:00.0 PME# enabled",
            "PCIe: 0000:01:00.0 enabling device (0000 -> 0003)",
            "PCIe: 0000:04:00.0 correctable error detected, type=Physical Layer",
            "PCIe: Root Port link training timeout, falling back to Gen1",
            "PCIe: 0000:01:00.0 Maximum Payload Size set to 256 bytes"
        ]
        
        # Generate embeddings and add to vector store
        embeddings = self.model_manager.generate_embeddings(sample_logs)
        metadata = [
            {"source": "demo_log", "line": i+1, "level": "ERROR" if "error" in log else "INFO"}
            for i, log in enumerate(sample_logs)
        ]
        
        vector_store.add_documents(sample_logs, embeddings, metadata)
        print(f"✅ Added {len(sample_logs)} sample log entries")
    
    def show_help(self):
        """Show available commands and sample queries"""
        print("\n💡 Chat Commands:")
        print("  /help    - Show this help")
        print("  /examples - Show example queries")
        print("  /clear   - Clear chat history")
        print("  /status  - Show system status")
        print("  /quit    - Exit chat")
        
        print("\n🔍 Sample Queries:")
        queries = [
            "What PCIe errors occurred?",
            "Analyze the link training failure",
            "What devices had configuration timeouts?",
            "Check for correctable errors",
            "What is the link speed and width?",
            "Analyze transaction layer errors",
            "What happened to device 0000:01:00.0?",
            "Show me performance issues"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")
        
        print("\n📋 Analysis Types Available:")
        templates = PCIePromptTemplates.get_available_templates()
        for name, desc in templates.items():
            print(f"  • {desc}")
    
    def show_examples(self):
        """Show example queries with expected responses"""
        print("\n📚 Example Queries & What to Expect:")
        print("=" * 50)
        
        examples = [
            {
                "query": "What PCIe errors occurred?",
                "description": "General error analysis - will identify and categorize all errors",
                "type": "General Analysis"
            },
            {
                "query": "Analyze the link training failure in detail",
                "description": "Specialized link training analysis - LTSSM states, speed negotiation",
                "type": "Link Training Analysis"
            },
            {
                "query": "What performance issues are affecting the system?",
                "description": "Performance analysis - throughput, latency, bottlenecks",
                "type": "Performance Analysis"
            },
            {
                "query": "What happened to device 0000:01:00.0?",
                "description": "Device-specific analysis - configuration, errors, status",
                "type": "Device Analysis"
            }
        ]
        
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. {ex['type']}")
            print(f"   Query: \"{ex['query']}\"")
            print(f"   Expects: {ex['description']}")
    
    def show_status(self):
        """Show current system status"""
        print("\n📊 System Status:")
        print("=" * 30)
        
        if not self.setup_complete:
            print("❌ System not initialized")
            return
            
        # Check model status
        if hasattr(self.model_manager.llm_manager._clients.get('local', {}), 'model_path'):
            provider = self.model_manager.llm_manager._clients['local']
            print(f"🧠 LLM Model: {provider.model_name}")
            print(f"📁 Model Path: {provider.model_path}")
            print(f"💾 Model Loaded: {provider.llm is not None}")
        
        # Show metrics if available
        if self.engine:
            metrics = self.engine.get_metrics()
            print(f"📈 Queries Processed: {metrics['queries_processed']}")
            print(f"⏱️  Avg Response Time: {metrics['average_response_time']:.2f}s")
            print(f"🎯 Avg Confidence: {metrics['average_confidence']:.3f}")
            print(f"💨 Cache Hits: {metrics['cache_hits']}")
    
    def process_query(self, query):
        """Process a user query and return response"""
        if not self.setup_complete:
            return "❌ System not initialized. Please restart the demo."
        
        try:
            print(f"\n🔍 Processing: {query}")
            print("⏳ Analyzing...")
            
            start_time = time.time()
            
            # Create RAG query
            rag_query = RAGQuery(
                query=query,
                context_window=3,
                min_similarity=0.3,
                rerank=True
            )
            
            # Get response
            result = self.engine.query(rag_query)
            response_time = time.time() - start_time
            
            print(f"\n🤖 PCIe Debug Agent Response:")
            print("=" * 50)
            print(result.answer)
            
            print(f"\n📊 Analysis Details:")
            print(f"   Response Time: {response_time:.1f}s")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Sources Used: {len(result.sources)}")
            
            if result.sources:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result.sources[:2], 1):
                    content = source.get('content', '')[:80]
                    score = source.get('score', 0)
                    print(f"   {i}. [{score:.3f}] {content}...")
            
            if result.reasoning:
                print(f"\n🧠 Reasoning: {result.reasoning[:100]}...")
                
            return result.answer
            
        except Exception as e:
            error_msg = f"❌ Query failed: {e}"
            print(error_msg)
            return error_msg
    
    def run_chat(self):
        """Run the interactive chat loop"""
        print("\n🎯 Starting PCIe Debug Chat Demo")
        print("=" * 50)
        print("Type your PCIe debugging questions or use /help for commands")
        print("Type /quit to exit")
        
        while True:
            try:
                # Get user input
                print("\n" + "─" * 50)
                user_input = input("🔧 PCIe Debug > ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'quit' or command == 'exit':
                        print("👋 Thanks for using PCIe Debug Agent!")
                        break
                    elif command == 'help':
                        self.show_help()
                    elif command == 'examples':
                        self.show_examples()
                    elif command == 'status':
                        self.show_status()
                    elif command == 'clear':
                        print("\n" * 50)  # Clear screen
                        print("🧹 Chat history cleared")
                    else:
                        print(f"❓ Unknown command: {user_input}")
                        print("Use /help to see available commands")
                else:
                    # Process as PCIe query
                    self.process_query(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main entry point"""
    print("🚀 PCIe Debug Agent - Interactive Chat Demo")
    print("=" * 60)
    
    # Create and setup demo
    demo = PCIeChatDemo()
    
    if demo.setup_system():
        demo.show_help()
        demo.run_chat()
    else:
        print("❌ Failed to initialize system")
        print("Please check your configuration and try again")

if __name__ == "__main__":
    main()