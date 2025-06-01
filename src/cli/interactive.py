"""
Interactive CLI mode for PCIe Debug Agent
Similar to Claude Code's interactive REPL
"""

import os
import sys
import json
import cmd
import readline
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.models.model_selector import get_model_selector
from src.models.embedding_selector import get_embedding_selector
from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.enhanced_rag_engine import EnhancedRAGEngine
from src.config.settings import load_settings
from src.cli.utils.output import print_success, print_error, print_info, print_warning
from src.cli.utils.token_counter import TokenCounter
from src.cli.memory import MemoryManager
from src.cli.session_manager import SessionManager


class PCIeDebugShell(cmd.Cmd):
    """Interactive shell for PCIe debugging"""
    
    intro = """
╔═══════════════════════════════════════════════════════════╗
║                  PCIe Debug Agent v1.0.0                  ║
║           AI-Powered PCIe Log Analysis Tool               ║
╚═══════════════════════════════════════════════════════════╝

Interactive PCIe debugging assistant. Type help or ? for commands.
Type your PCIe questions directly or use slash commands.
    """
    
    prompt = '🔧 > '
    
    def __init__(self, model_id: Optional[str] = None, verbose: bool = False, max_turns: int = 50):
        super().__init__()
        self.verbose = verbose
        self.analysis_verbose = False  # Separate flag for analysis verbosity
        self.max_turns = max_turns
        self.turn_count = 0
        
        # Initialize components
        self.model_selector = get_model_selector()
        if model_id:
            self.model_selector.switch_model(model_id)
        
        self.memory_manager = MemoryManager()
        self.session_manager = SessionManager()
        self.token_counter = TokenCounter()
        self.embedding_selector = get_embedding_selector()
        
        # RAG components
        self.vector_store = None
        self.rag_engine = None
        self.rag_search_mode = 'semantic'  # Default search mode
        self.hybrid_engine = None  # Will be initialized when needed
        
        # Session state
        self.current_session = None
        self.conversation_history = []
        self.session_tokens = {"input": 0, "output": 0}  # Track tokens per session
        
        # Streaming configuration
        self.streaming_enabled = True  # Enable streaming by default
        self.stream_delay = 0.01  # Small delay between chunks for better visual effect
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize RAG system"""
        try:
            if self.verbose:
                print("🔧 Initializing PCIe Debug Agent...")
            
            # Initialize multi-model vector database manager
            from src.vectorstore.multi_model_manager import MultiModelVectorManager
            self.vector_manager = MultiModelVectorManager()
            
            # Try to load vector store for current embedding model
            current_model = self.embedding_selector.get_current_model()
            self.vector_store = None
            self.rag_enabled = False
            
            # Check if we need to migrate legacy database
            legacy_path = Path("data/vectorstore/index.faiss")
            if legacy_path.exists() and not any(self.vector_manager.list_models().values()):
                if self.verbose:
                    print("🔄 Migrating legacy vector database...")
                self.vector_manager.migrate_legacy()
            
            # Try to load vector store for current model
            self.vector_store = self.vector_manager.load(current_model)
            if self.vector_store:
                self.rag_enabled = True
                if self.verbose:
                    print(f"   Vector store ({current_model}): {self.vector_store.index.ntotal} documents")
            else:
                if self.verbose:
                    available_models = [name for name, info in self.vector_manager.list_models().items() if info['exists']]
                    if available_models:
                        print_info(f"ℹ️ No vector database for {current_model}. Available: {', '.join(available_models)}")
                        print_info("   Use '/rag_model <model>' to switch or build database for current model")
                    else:
                        print_info("ℹ️ No vector databases found. RAG features disabled.")
                        print_info("   Run 'pcie-debug vectordb build' to enable semantic search")
            
            # Create model wrapper
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            class ModelWrapper:
                def __init__(self, selector, embedding_selector, rag_enabled=False):
                    self.selector = selector
                    self.embedding_selector = embedding_selector
                    self.rag_enabled = rag_enabled
                
                def generate_completion(self, prompt: str, **kwargs) -> str:
                    # Filter out parameters that LocalLLMProvider doesn't support
                    filtered_kwargs = {k: v for k, v in kwargs.items() 
                                     if k not in ['provider', 'model']}
                    return self.selector.generate_completion(prompt, **filtered_kwargs)
                
                def generate_embeddings(self, texts: List[str]) -> np.ndarray:
                    """Generate embeddings for texts"""
                    if not self.rag_enabled:
                        raise RuntimeError("RAG not enabled - no vector database")
                    provider = self.embedding_selector.get_current_provider()
                    return provider.encode(texts)
            
            # Initialize RAG engine only if vector store is available and compatible
            if self.rag_enabled and self.vector_store:
                # Check dimension compatibility
                embedding_dim = self.embedding_selector.get_current_provider().get_dimension()
                vector_db_dim = self.vector_store.dimension
                
                if embedding_dim == vector_db_dim:
                    self.rag_engine = EnhancedRAGEngine(
                        vector_store=self.vector_store,
                        model_manager=ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=True)
                    )
                else:
                    # Dimension mismatch - disable RAG
                    print_warning(f"⚠️  Dimension mismatch detected!")
                    print_info(f"   Vector DB dimension: {vector_db_dim}")
                    print_info(f"   Embedding model dimension: {embedding_dim}")
                    print_info(f"   Current embedding model: {self.embedding_selector.get_current_model()}")
                    print_info(f"   RAG temporarily disabled to prevent errors")
                    print_info(f"   Run 'pcie-debug vectordb build --force' to rebuild with current embedding model")
                    
                    self.rag_enabled = False
                    self.rag_engine = None
                    self.model_wrapper = ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=False)
            else:
                # Use direct model without RAG
                self.rag_engine = None
                self.model_wrapper = ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=False)
            
            if self.verbose:
                print("✅ System ready!")
                if not self.rag_enabled:
                    print("   (Running in direct mode without RAG)")
                
                # Show current embedding model
                current_embedding = self.embedding_selector.get_current_model()
                embedding_info = self.embedding_selector.get_model_info()
                print(f"   Embedding model: {current_embedding}")
                print(f"   Provider: {embedding_info.get('provider', 'unknown')}")
                print(f"   Dimension: {embedding_info.get('dimension', 'unknown')}")
                if embedding_info.get('cost') != 'free':
                    print(f"   Cost: {embedding_info.get('cost', 'unknown')}")
                
        except Exception as e:
            print_error(f"Failed to initialize system: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def do_help(self, arg):
        """Show help information"""
        if arg:
            super().do_help(arg)
        else:
            print("""
Available Commands:
  help                 Show this help
  /model [model-id]    List or switch AI models
  /rag_model [model]   List or switch RAG embedding models
  /memory              Manage persistent memory
  /session             Manage conversation sessions
  /clear               Clear current conversation
  /status              Show system status
  /tokens              Show detailed token usage
  /review              Request code review
  /search <query>      Search knowledge base
  /analyze <log>       Analyze PCIe log file
  /config              Show configuration
  /verbose [on/off]    Toggle verbose analysis mode
  /rag [on/off]        Toggle RAG (Retrieval-Augmented Generation)
  /rag_mode [mode]     Select RAG search mode (semantic/hybrid/keyword)
  /rag_status          Show detailed RAG and vector DB status
  /knowledge_base      Show RAG knowledge base content and status
  /kb                  Alias for /knowledge_base
  /stream [on/off]     Toggle real-time streaming responses
  /vim                 Enable vim mode
  /exit, /quit         Exit the shell

Direct Usage:
  Just type your PCIe debugging questions directly!
  
Examples:
  > Why is PCIe link training failing?
  > /model llama-3.2-3b
  > /rag_model text-embedding-3-small
  > /analyze logs/pcie_error.log
  > /search completion timeout
  > /verbose on
            """)
    
    def do_model(self, arg):
        """List or switch AI models"""
        from src.cli.commands.model import model
        
        if not arg:
            # List models
            current = self.model_selector.get_current_model()
            print("\n📊 Available Models:")
            print("-" * 50)
            
            models = {
                "gpt-4o-mini": {"desc": "GPT-4o-Mini - Efficient OpenAI model (2.5M tokens/day)", "available": True},
                "mock-llm": {"desc": "Mock LLM - Built-in offline model", "available": True},
                "llama-3.2-3b": {"desc": "Llama 3.2 3B - Fast local analysis", "available": True},
                "deepseek-r1-7b": {"desc": "DeepSeek R1 7B - Detailed reasoning (Ollama) ⚠️ Memory intensive", "available": True},
                "gpt-4o": {"desc": "GPT-4o - Latest OpenAI model (250K tokens/day)", "available": True},
                "gpt-4": {"desc": "GPT-4 - Best quality (250K tokens/day)", "available": True},
                "claude-3-opus": {"desc": "Claude 3 Opus - Best quality (API)", "available": False}
            }
            
            for model_id, info in models.items():
                marker = "✓" if model_id == current else " "
                status = "✅" if info["available"] else "❌"
                print(f"  {marker} {model_id:15} {status} - {info['desc']}")
            
            print(f"\nCurrent: {current}")
            print("Use '/model <name>' to switch")
        else:
            # Check if model is available before switching
            available_models = ["gpt-4o-mini", "mock-llm", "llama-3.2-3b", "deepseek-r1-7b", "gpt-4o", "gpt-4"]
            
            if arg not in available_models:
                if arg in ["claude-3-opus"]:
                    print_error(f"❌ Model '{arg}' is not available")
                    print("   🔑 API model requires API key configuration")
                    print("   💡 Available models: mock-llm, llama-3.2-3b, deepseek-r1-7b, gpt-4o, gpt-4")
                else:
                    print_error(f"❌ Unknown model: {arg}")
                    print("   💡 Use '/model' to see available models")
                return
            
            # Special warning for DeepSeek R1
            if arg == "deepseek-r1-7b":
                print("⚠️  DeepSeek R1 is memory intensive and may timeout on M1 Macs")
                print("   💡 Tip: Use llama-3.2-3b for faster, more reliable responses")
            
            # Switch model
            if self.model_selector.switch_model(arg):
                print_success(f"✅ Switched to {arg}")
                self.prompt = f'🔧 [{arg}] > '
            else:
                print_error(f"❌ Failed to switch to {arg}")
    
    def do_memory(self, arg):
        """Manage persistent memory"""
        if not arg:
            # Show current memory
            memory = self.memory_manager.get_memory()
            if memory:
                print("\n📝 Current Memory:")
                print("-" * 40)
                for key, value in memory.items():
                    print(f"{key}: {value}")
            else:
                print("📝 No memory entries")
        elif arg.startswith("set "):
            # Set memory
            parts = arg[4:].split(" ", 1)
            if len(parts) == 2:
                self.memory_manager.set_memory(parts[0], parts[1])
                print_success(f"✅ Memory set: {parts[0]}")
        elif arg.startswith("del "):
            # Delete memory
            key = arg[4:]
            self.memory_manager.delete_memory(key)
            print_success(f"✅ Memory deleted: {key}")
        else:
            print("Usage: /memory [set <key> <value>] [del <key>]")
    
    def do_session(self, arg):
        """Manage conversation sessions"""
        if not arg:
            # List sessions
            sessions = self.session_manager.list_sessions()
            if sessions:
                print("\n📋 Recent Sessions:")
                for session in sessions[:10]:
                    print(f"  {session['id'][:8]}: {session['title']} ({session['timestamp']})")
            else:
                print("📋 No saved sessions")
        elif arg.startswith("save"):
            # Save current session
            if self.conversation_history:
                session_id = self.session_manager.save_session(
                    self.conversation_history,
                    title=f"PCIe Debug Session {datetime.now().strftime('%H:%M')}"
                )
                print_success(f"✅ Session saved: {session_id}")
        elif arg.startswith("load "):
            # Load session
            session_id = arg[5:]
            session = self.session_manager.load_session(session_id)
            if session:
                self.conversation_history = session['conversation']
                print_success(f"✅ Session loaded: {session_id}")
            else:
                print_error(f"❌ Session not found: {session_id}")
    
    def do_clear(self, arg):
        """Clear conversation history"""
        self.conversation_history.clear()
        self.turn_count = 0
        self.session_tokens = {"input": 0, "output": 0}
        print_success("✅ Conversation cleared")
    
    def do_status(self, arg):
        """Show system status"""
        current_model = self.model_selector.get_current_model()
        doc_count = self.vector_store.index.ntotal if self.vector_store else 0
        
        verbose_status = "ON" if self.analysis_verbose else "OFF"
        rag_status = "ENABLED" if self.rag_enabled else "DISABLED"
        
        print(f"""
🔧 PCIe Debug Agent Status
{'='*60}

🤖 Model Configuration:
   Current Model: {current_model}
   Provider: {self._get_model_provider(current_model)}
   
📚 RAG (Retrieval-Augmented Generation):
   Status: {rag_status}
   Search Mode: {self.rag_search_mode.upper()}
   Vector Database: {'Loaded' if self.vector_store else 'Not Loaded'}
   Documents Indexed: {doc_count:,}
   Embedding Model: {self.embedding_selector.get_current_model()}
   Embedding Dimension: {self.embedding_selector.get_current_provider().get_dimension() if self.embedding_selector.get_current_provider() else 'N/A'}
   Embedding Provider: {self.embedding_selector.get_model_info().get('provider', 'N/A')}
   
💬 Session Information:
   Current Turn: {self.turn_count}/{self.max_turns}
   Session ID: {self.current_session or 'None'}
   Conversation History: {len(self.conversation_history)} messages
   Memory Entries: {len(self.memory_manager.get_memory())}
   
🎯 Token Usage (This Session):
   Input Tokens: {self.token_counter.format_token_count(self.session_tokens['input'])}
   Output Tokens: {self.token_counter.format_token_count(self.session_tokens['output'])}
   Total Tokens: {self.token_counter.format_token_count(self.session_tokens['input'] + self.session_tokens['output'])}
   
⚙️  Settings:
   Verbose Analysis: {verbose_status}
   Streaming Responses: {'ON' if self.streaming_enabled else 'OFF'}
   Stream Delay: {self.stream_delay}s
   Auto-save Sessions: {'Yes' if hasattr(self, 'auto_save') and self.auto_save else 'No'}
   
📊 Performance:
   RAG Pipeline: {'~1-3s per query' if self.rag_enabled else 'N/A'}
   Direct Mode: {'~0.5-2s per query' if not self.rag_enabled else 'N/A'}
   
{'='*60}
        """)
    
    def do_search(self, arg):
        """Search knowledge base"""
        if not arg:
            print("Usage: /search <query>")
            return
        
        if not self.rag_enabled or not self.vector_store:
            print_error("❌ Vector database not loaded. RAG features are disabled.")
            print_info("   Run 'pcie-debug vectordb build' to enable search")
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Generate embedding for search
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            query_embedding = embedding_model.encode(arg)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, k=5)
            
            print(f"\n🔍 Search results for: '{arg}'")
            print("-" * 50)
            
            for i, (content, metadata, score) in enumerate(results, 1):
                source = metadata.get('source', 'Unknown')
                print(f"\n{i}. {source} (Score: {score:.3f})")
                print(f"   {content[:200]}...")
            
        except Exception as e:
            print_error(f"Search failed: {e}")
    
    def do_analyze(self, arg):
        """Analyze PCIe log file"""
        if not arg:
            print("Usage: /analyze <log-file-path>")
            return
        
        log_path = Path(arg)
        if not log_path.exists():
            print_error(f"Log file not found: {arg}")
            return
        
        try:
            with open(log_path, 'r') as f:
                log_content = f.read()
            
            print(f"🔍 Analyzing log file: {log_path.name}")
            
            if self.rag_enabled and self.rag_engine:
                # Use RAG for analysis
                from src.rag.enhanced_rag_engine import RAGQuery
                rag_query = RAGQuery(query=f"Analyze this PCIe log file: {log_content}", context_window=5)
                result = self.rag_engine.query(rag_query)
                
                if result and hasattr(result, 'answer'):
                    response = result.answer
                else:
                    response = None
            else:
                # Direct analysis without RAG
                prompt = f"""You are a PCIe debugging expert. Analyze the following PCIe log file and provide:

1. **Summary**: Overview of the log contents
2. **Issues Found**: Any errors, warnings, or anomalies
3. **Root Cause Analysis**: Likely causes of any issues
4. **Recommendations**: Steps to debug or fix issues

Log file: {log_path.name}
Contents:
{log_content[:5000]}...  # Truncate if too long

Please provide a comprehensive analysis."""
                
                response = self.model_wrapper.generate_completion(prompt)
            
            if response:
                print("\n💡 Analysis Result:")
                print("-" * 60)
                print(response)
                print("-" * 60)
                
                # Add to conversation history
                self._add_to_history("system", f"Analyzed log file: {log_path.name}")
                self._add_to_history("assistant", response)
            else:
                print_error("❌ Analysis failed")
                
        except Exception as e:
            print_error(f"Failed to analyze log: {e}")
    
    def do_config(self, arg):
        """Show configuration"""
        try:
            settings = load_settings()
            print("\n⚙️ Configuration:")
            print("-" * 30)
            print(f"LLM Provider: {getattr(settings.llm, 'provider', 'N/A')}")
            print(f"Embedding Model: {getattr(settings.embedding, 'model', 'N/A')}")
            print(f"Vector Store: {getattr(settings.vector_store, 'index_path', 'N/A')}")
            print(f"Models Dir: {getattr(settings.local_llm, 'models_dir', 'N/A')}")
        except Exception as e:
            print_error(f"Failed to load config: {e}")
    
    def do_rag(self, arg):
        """Toggle RAG (Retrieval-Augmented Generation) on/off"""
        if not arg:
            # Show current RAG status
            status = "ENABLED" if self.rag_enabled else "DISABLED"
            print(f"\n🔧 RAG (Retrieval-Augmented Generation) Status: {status}")
            
            if self.rag_enabled:
                print("\nRAG is currently ENABLED:")
                print("  ✅ Vector database is loaded")
                if self.vector_store:
                    print(f"  ✅ {self.vector_store.index.ntotal} documents indexed")
                print("  ✅ Queries use semantic search for context")
                print("  ✅ Responses include source citations")
            else:
                print("\nRAG is currently DISABLED:")
                print("  ❌ No vector database loaded")
                print("  ❌ Queries sent directly to LLM")
                print("  ❌ No context retrieval")
                
            print("\nUsage: /rag on|off")
            return
            
        if arg.lower() in ["on", "true", "1", "yes", "enable"]:
            if self.rag_enabled:
                print("✅ RAG is already enabled")
                return
                
            # Try to enable RAG
            if not self.vector_store:
                # Try to load vector store
                vector_db_path = Path("data/vectorstore")
                if vector_db_path.exists():
                    try:
                        print("🔄 Loading vector database...")
                        self.vector_store = FAISSVectorStore.load(str(vector_db_path))
                        
                        # Check dimension compatibility
                        embedding_dim = self.embedding_selector.get_current_provider().get_dimension()
                        vector_db_dim = self.vector_store.dimension
                        
                        if embedding_dim != vector_db_dim:
                            print_error(f"❌ Dimension mismatch!")
                            print_info(f"   Vector DB dimension: {vector_db_dim}")
                            print_info(f"   Embedding model dimension: {embedding_dim}")
                            print_info(f"   Current embedding model: {self.embedding_selector.get_current_model()}")
                            print_info(f"   Run 'pcie-debug vectordb build --force' to rebuild with current embedding model")
                            return
                        
                        self.rag_enabled = True
                        
                        # Re-initialize RAG engine
                        from sentence_transformers import SentenceTransformer
                        
                        class ModelWrapper:
                            def __init__(self, selector, embedding_selector):
                                self.selector = selector
                                self.embedding_selector = embedding_selector
                            
                            def generate_completion(self, prompt: str, **kwargs) -> str:
                                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                                 if k not in ['provider', 'model']}
                                return self.selector.generate_completion(prompt, **filtered_kwargs)
                            
                            def generate_embeddings(self, texts: List[str]) -> np.ndarray:
                                provider = self.embedding_selector.get_current_provider()
                                return provider.encode(texts)
                        
                        self.rag_engine = EnhancedRAGEngine(
                            vector_store=self.vector_store,
                            model_manager=ModelWrapper(self.model_selector, self.embedding_selector)
                        )
                        
                        print_success("✅ RAG enabled successfully!")
                        print(f"   Vector store: {self.vector_store.index.ntotal} documents")
                        print(f"   Embedding model: {self.embedding_selector.get_current_model()}")
                        print(f"   Dimensions match: {embedding_dim}")
                    except Exception as e:
                        print_error(f"❌ Failed to enable RAG: {e}")
                        print_info("   Run 'pcie-debug vectordb build' to create vector database")
                else:
                    print_error("❌ No vector database found")
                    print_info("   Run 'pcie-debug vectordb build' to create vector database")
            else:
                self.rag_enabled = True
                print_success("✅ RAG enabled")
                
        elif arg.lower() in ["off", "false", "0", "no", "disable"]:
            if not self.rag_enabled:
                print("✅ RAG is already disabled")
                return
                
            self.rag_enabled = False
            
            # Create model wrapper for direct mode if needed
            if not hasattr(self, 'model_wrapper'):
                from sentence_transformers import SentenceTransformer
                
                class ModelWrapper:
                    def __init__(self, selector, embedding_selector, rag_enabled=False):
                        self.selector = selector
                        self.embedding_selector = embedding_selector
                        self.rag_enabled = rag_enabled
                    
                    def generate_completion(self, prompt: str, **kwargs) -> str:
                        filtered_kwargs = {k: v for k, v in kwargs.items() 
                                         if k not in ['provider', 'model']}
                        return self.selector.generate_completion(prompt, **filtered_kwargs)
                
                self.model_wrapper = ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=False)
            
            print_success("✅ RAG disabled")
            print_info("   Queries will be sent directly to LLM without context retrieval")
        else:
            print_error("❌ Invalid option. Use: /rag on|off")
    
    def do_rag_status(self, arg):
        """Show detailed RAG and vector database status"""
        print(f"\n🔧 RAG Status Analysis")
        print("="*60)
        
        # Show RAG enabled status
        rag_status = "ENABLED" if self.rag_enabled else "DISABLED"
        print(f"\nRAG Mode: {rag_status}")
        
        # Show embedding model info
        current_embedding = self.embedding_selector.get_current_model()
        embedding_info = self.embedding_selector.get_model_info()
        embedding_dim = self.embedding_selector.get_current_provider().get_dimension()
        
        print(f"\nEmbedding Model:")
        print(f"   Current: {current_embedding}")
        print(f"   Provider: {embedding_info.get('provider', 'unknown')}")
        print(f"   Dimension: {embedding_dim}")
        print(f"   Cost: {embedding_info.get('cost', 'unknown')}")
        
        # Show vector database status
        vector_db_path = Path("data/vectorstore")
        if vector_db_path.exists():
            try:
                # Try to load and check compatibility
                temp_store = FAISSVectorStore.load(str(vector_db_path))
                vector_db_dim = temp_store.dimension
                doc_count = temp_store.index.ntotal
                
                print(f"\nVector Database:")
                print(f"   Location: {vector_db_path}")
                print(f"   Documents: {doc_count:,}")
                print(f"   Dimension: {vector_db_dim}")
                
                # Check compatibility
                if embedding_dim == vector_db_dim:
                    print(f"   Compatibility: ✅ COMPATIBLE")
                    if not self.rag_enabled:
                        print(f"   💡 You can enable RAG with '/rag on'")
                else:
                    print(f"   Compatibility: ❌ DIMENSION MISMATCH")
                    print(f"   Issue: Embedding model ({embedding_dim}D) != Vector DB ({vector_db_dim}D)")
                    print(f"   Solution: Rebuild vector database with current embedding model")
                
            except Exception as e:
                print(f"\nVector Database:")
                print(f"   Location: {vector_db_path}")
                print(f"   Status: ❌ Error loading - {e}")
        else:
            print(f"\nVector Database:")
            print(f"   Status: ❌ Not found")
            print(f"   Location: {vector_db_path}")
        
        # Show recommendations
        print(f"\n💡 Recommendations:")
        if not vector_db_path.exists():
            print(f"   • Run 'pcie-debug vectordb build' to create vector database")
        elif not self.rag_enabled:
            if vector_db_path.exists():
                try:
                    temp_store = FAISSVectorStore.load(str(vector_db_path))
                    vector_db_dim = temp_store.dimension
                    if embedding_dim != vector_db_dim:
                        print(f"   • Run 'pcie-debug vectordb build --force' to rebuild for current embedding model")
                    else:
                        print(f"   • Run '/rag on' to enable RAG with existing vector database")
                except:
                    print(f"   • Run 'pcie-debug vectordb build --force' to rebuild vector database")
        else:
            print(f"   • RAG is working properly!")
            print(f"   • Use '/verbose on' to see detailed analysis steps")
        
        print("="*60)
    
    def _generate_streaming_response(self, prompt: str) -> str:
        """Generate streaming response and display in real-time"""
        import sys
        import time
        
        print("\n💡 Response:")
        print("-" * 60)
        
        response_parts = []
        
        try:
            # Generate streaming completion
            for chunk in self.model_selector.generate_completion_stream(prompt):
                if chunk:
                    print(chunk, end='', flush=True)
                    response_parts.append(chunk)
                    # Small delay for better visual effect
                    if self.stream_delay > 0:
                        time.sleep(self.stream_delay)
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
            # Fallback to non-streaming
            return self.model_wrapper.generate_completion(prompt)
        
        print()  # New line after streaming
        print("-" * 60)
        
        return "".join(response_parts)
    
    def _display_source_files(self, sources):
        """Display source files used in RAG retrieval"""
        from pathlib import Path
        
        # Extract unique file paths
        file_paths = set()
        file_chunks = {}
        
        for source in sources:
            metadata = source.get('metadata', {})
            source_path = metadata.get('source', 'Unknown')
            
            if source_path != 'Unknown':
                file_paths.add(source_path)
                filename = Path(source_path).name
                if filename not in file_chunks:
                    file_chunks[filename] = []
                file_chunks[filename].append({
                    'score': source.get('score', 0.0),
                    'content_length': len(source.get('content', ''))
                })
        
        if file_paths:
            print(f"\n📁 Knowledge Base Files Used ({len(file_paths)} files):")
            print("-" * 60)
            
            # Sort files by highest relevance score
            sorted_files = []
            for filename, chunks in file_chunks.items():
                max_score = max(chunk['score'] for chunk in chunks)
                total_content = sum(chunk['content_length'] for chunk in chunks)
                sorted_files.append((filename, len(chunks), max_score, total_content))
            
            sorted_files.sort(key=lambda x: x[2], reverse=True)  # Sort by max score
            
            for filename, chunk_count, max_score, total_content in sorted_files:
                # Get file description
                file_description = self._get_file_description(filename)
                
                print(f"  📄 {filename}")
                print(f"     └─ {file_description}")
                print(f"     └─ {chunk_count} chunk{'s' if chunk_count > 1 else ''} used, "
                      f"max relevance: {max_score:.3f}, {total_content} chars")
            
            print("-" * 60)
    
    def _get_file_description(self, filename):
        """Get description of knowledge base file"""
        descriptions = {
            "aer_error_handling.md": "Advanced Error Reporting configuration and monitoring",
            "power_management_issues.md": "PCIe power states and ASPM troubleshooting", 
            "pcie_error_scenarios.md": "Common PCIe error scenarios and solutions",
            "ltssm_states_guide.md": "Link Training State Machine states and transitions",
            "tlp_error_analysis.md": "Transaction Layer Packet analysis and debugging",
            "signal_integrity_troubleshooting.md": "Signal integrity issues and physical layer debugging"
        }
        
        return descriptions.get(filename, "PCIe technical documentation")
    
    def do_knowledge_base(self, arg):
        """Show current RAG knowledge base content and status"""
        print(f"\n📚 Multi-Model RAG Knowledge Base Status")
        print("="*80)
        
        # Get current embedding model info
        current_embedding = self.embedding_selector.get_current_model()
        embedding_info = self.embedding_selector.get_model_info()
        
        print(f"\n🎯 Current Configuration:")
        print(f"   Active Model: {current_embedding}")
        print(f"   Provider: {embedding_info.get('provider', 'unknown')}")
        print(f"   Dimension: {embedding_info.get('dimension', 'unknown')}D")
        print(f"   RAG Status: {'✅ ENABLED' if self.rag_enabled else '❌ DISABLED'}")
        
        # List all available vector databases
        models_info = self.vector_manager.list_models()
        available_models = {name: info for name, info in models_info.items() if info['exists']}
        
        if available_models:
            print(f"\n🗄️ Available Vector Databases ({len(available_models)} models):")
            print("-" * 80)
            
            for model_name, info in available_models.items():
                # Load stats for this model
                stats = self.vector_manager.get_stats(model_name)
                if stats:
                    is_current = "👉 " if model_name == current_embedding else "   "
                    status = "✅ ACTIVE" if (model_name == current_embedding and self.rag_enabled) else "⚪ READY"
                    
                    print(f"{is_current}📊 {model_name}")
                    print(f"      Status: {status}")
                    print(f"      Vectors: {stats['total_vectors']:,}")
                    print(f"      Dimension: {stats['dimension']}D")
                    print(f"      Size: {stats['size_mb']:.1f}MB")
                    print(f"      Path: {Path(stats['path']).name}")
                    print()
            
            # Show detailed content for current model if available
            if current_embedding in available_models and self.vector_store:
                print(f"\n📖 Current Model Content ({current_embedding}):")
                print("-" * 80)
                
                # Analyze document sources from metadata
                if hasattr(self.vector_store, 'metadata') and self.vector_store.metadata:
                    sources = {}
                    for meta in self.vector_store.metadata:
                        source = meta.get('source', 'unknown')
                        filename = Path(source).name if source != 'unknown' else 'unknown'
                        sources[filename] = sources.get(filename, 0) + 1
                    
                    # Calculate total file size
                    total_size = 0
                    for filename, chunk_count in sorted(sources.items()):
                        if filename != 'unknown':
                            file_path = Path("data/knowledge_base") / filename
                            if file_path.exists():
                                size_kb = file_path.stat().st_size / 1024
                                total_size += size_kb
                                content_focus = self._get_content_focus(filename)
                                print(f"  📄 {filename:<35} {chunk_count} chunks ({size_kb:.1f}KB)")
                                print(f"     {' '*37} └─ {content_focus}")
                    
                    print(f"\n📊 Content Summary:")
                    print(f"   Total files: {len([f for f in sources.keys() if f != 'unknown'])}")
                    print(f"   Total chunks: {len(self.vector_store.metadata)}")
                    print(f"   Total size: {total_size:.1f}KB")
                    
                    coverage_areas = self._get_coverage_areas(sources.keys())
                    print(f"\n🎯 Coverage Areas:")
                    for area in coverage_areas:
                        print(f"   • {area}")
                        
        else:
            print(f"\n❌ No vector databases found")
            print(f"   Run 'pcie-debug vectordb build' to create database for {current_embedding}")
        
        # Show management options
        print(f"\n💡 Management Options:")
        print(f"   📊 View all databases: /knowledge_base")
        print(f"   🔄 Switch models: /rag_model <model>")
        print(f"   🔧 Build database: pcie-debug vectordb build")
        print(f"   📈 Compare performance: python rag_performance_comparison.py")
        
        print("\n" + "="*80)
    
    def do_kb(self, arg):
        """Alias for /knowledge_base command"""
        self.do_knowledge_base(arg)
    
    def _get_content_focus(self, filename):
        """Determine content focus based on filename"""
        focus_map = {
            "aer_error_handling.md": "Advanced Error Reporting, correctable/uncorrectable errors",
            "power_management_issues.md": "PCIe power states (D0-D3, L0-L3), ASPM configuration", 
            "pcie_error_scenarios.md": "Common PCIe error scenarios and troubleshooting",
            "ltssm_states_guide.md": "Link Training State Machine states and transitions",
            "tlp_error_analysis.md": "Transaction Layer Packet analysis and debugging",
            "signal_integrity_troubleshooting.md": "Signal integrity issues and physical layer debugging"
        }
        return focus_map.get(filename, "PCIe technical documentation")
    
    def _get_coverage_areas(self, filenames):
        """Get coverage areas based on available files"""
        areas = []
        if any("aer" in f for f in filenames):
            areas.append("Error handling and AER configuration")
        if any("power" in f for f in filenames):
            areas.append("Power management troubleshooting")
        if any("ltssm" in f for f in filenames):
            areas.append("Link training and LTSSM states")
        if any("tlp" in f for f in filenames):
            areas.append("Transaction layer packet analysis") 
        if any("signal" in f for f in filenames):
            areas.append("Signal integrity issues")
        if any("error" in f for f in filenames):
            areas.append("Common error scenarios and solutions")
        
        if not areas:
            areas.append("PCIe technical documentation")
        
        return areas
    
    def do_rag_model(self, arg):
        """List or switch RAG embedding models"""
        if not arg:
            # List available embedding models
            current = self.embedding_selector.get_current_model()
            models = self.embedding_selector.list_models()
            
            print("\n🧮 Available Embedding Models:")
            print("-" * 70)
            
            # Show models in preferred order
            preferred_order = [
                "text-embedding-3-small",
                "text-embedding-3-large", 
                "text-embedding-ada-002",
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "multi-qa-MiniLM-L6-cos-v1"
            ]
            
            for model_id in preferred_order:
                if model_id in models:
                    config = models[model_id]
                    marker = "✓" if model_id == current else " "
                    available = "✅" if self.embedding_selector.is_available(model_id) else "❌"
                    provider_type = "API" if config["cost"] != "free" else "Local"
                    
                    # Mark preferred default
                    default_marker = " [DEFAULT]" if model_id == "text-embedding-3-small" else ""
                    
                    print(f"  {marker} {model_id:<25} {available} - {config['description']}{default_marker}")
                    print(f"    {' '*27} {provider_type} | {config['speed']} | {config['cost']}")
            
            print(f"\nCurrent: {current}")
            print("Use '/rag_model <name>' to switch")
            print("\nℹ️  text-embedding-3-small is the default when OpenAI API key is available")
            return
        
        # Switch embedding model
        if not self.embedding_selector.is_available(arg):
            if arg in self.embedding_selector.list_models():
                model_config = self.embedding_selector.list_models()[arg]
                if model_config["provider"].__name__ == "OpenAIEmbeddingProvider":
                    print_error(f"❌ Model '{arg}' requires OpenAI API key")
                    print_info("   Set OPENAI_API_KEY environment variable")
                else:
                    print_error(f"❌ Model '{arg}' is not available")
            else:
                print_error(f"❌ Unknown embedding model: {arg}")
                print_info("   Use '/rag_model' to see available models")
            return
        
        if self.embedding_selector.switch_model(arg):
            print_success(f"✅ Switched to embedding model: {arg}")
            
            # Get model info
            info = self.embedding_selector.get_model_info(arg)
            print_info(f"   Dimension: {info['dimension']}")
            print_info(f"   Provider: {info.get('provider', 'unknown')}")
            print_info(f"   Cost: {info['cost']}")
            
            # Try to load vector database for the new model
            self.vector_store = self.vector_manager.load(arg)
            if self.vector_store:
                self.rag_enabled = True
                print_success(f"✅ Loaded existing vector database for {arg}")
                print_info(f"   Vector store: {self.vector_store.index.ntotal} documents")
                
                # Re-initialize RAG engine with new vector store
                from sentence_transformers import SentenceTransformer
                import numpy as np
                
                class ModelWrapper:
                    def __init__(self, selector, embedding_selector, rag_enabled=False):
                        self.selector = selector
                        self.embedding_selector = embedding_selector
                        self.rag_enabled = rag_enabled
                    
                    def generate_completion(self, prompt: str, **kwargs) -> str:
                        filtered_kwargs = {k: v for k, v in kwargs.items() 
                                         if k not in ['provider', 'model']}
                        return self.selector.generate_completion(prompt, **filtered_kwargs)
                    
                    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
                        if not self.rag_enabled:
                            raise RuntimeError("RAG not enabled - no vector database")
                        provider = self.embedding_selector.get_current_provider()
                        return provider.encode(texts)
                
                self.rag_engine = EnhancedRAGEngine(
                    vector_store=self.vector_store,
                    model_manager=ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=True)
                )
            else:
                # No vector database for this model
                self.rag_enabled = False
                self.rag_engine = None
                print_warning(f"⚠️  No vector database found for {arg}")
                print_info(f"   RAG disabled for this model")
                print_info(f"   Run 'pcie-debug vectordb build' to create database for {arg}")
                
                # Show available models
                available_models = [name for name, info in self.vector_manager.list_models().items() if info['exists']]
                if available_models:
                    print_info(f"   Available databases: {', '.join(available_models)}")
        else:
            print_error(f"❌ Failed to switch to {arg}")
    
    def do_rag_mode(self, arg):
        """Select RAG search mode (semantic/hybrid/keyword)"""
        if not arg:
            # Show current mode and available options
            current_mode = getattr(self, 'rag_search_mode', 'semantic')
            print(f"\n🔍 Current RAG Search Mode: {current_mode.upper()}")
            print("\nAvailable Search Modes:")
            print("-" * 50)
            print("  semantic  - Pure vector/semantic search (default)")
            print("             • Uses embeddings to find similar concepts")
            print("             • Best for conceptual queries")
            print("             • Fast and accurate for topic similarity")
            print()
            print("  hybrid    - Combines semantic + keyword search")
            print("             • Uses both embeddings and BM25 scoring")
            print("             • Best overall performance")
            print("             • Balances concept and keyword matching")
            print()
            print("  keyword   - Pure BM25 keyword search")
            print("             • Traditional text matching")
            print("             • Best for exact term searches")
            print("             • Good for specific error codes/names")
            print()
            print("Usage: /rag_mode <semantic|hybrid|keyword>")
            return
        
        mode = arg.lower()
        valid_modes = ['semantic', 'hybrid', 'keyword']
        
        if mode not in valid_modes:
            print_error(f"❌ Invalid mode: {mode}")
            print_info(f"   Valid modes: {', '.join(valid_modes)}")
            return
        
        # Check if RAG is enabled
        if not self.rag_enabled or not self.vector_store:
            print_error("❌ RAG is not enabled")
            print_info("   Enable RAG first with '/rag on'")
            return
        
        # Set the search mode
        self.rag_search_mode = mode
        
        # Initialize hybrid search engine if needed
        if mode in ['hybrid', 'keyword'] and not hasattr(self, 'hybrid_engine'):
            try:
                from src.rag.hybrid_search import HybridSearchEngine
                print("🔄 Initializing hybrid search engine...")
                self.hybrid_engine = HybridSearchEngine(
                    vector_store=self.vector_store,
                    index_path=f"data/vectorstore/bm25_index_{self.embedding_selector.get_current_model()}.pkl"
                )
                print_success("✅ Hybrid search engine initialized")
            except Exception as e:
                print_error(f"❌ Failed to initialize hybrid search: {e}")
                print_info("   Falling back to semantic search")
                self.rag_search_mode = 'semantic'
                return
        
        print_success(f"✅ RAG search mode set to: {mode.upper()}")
        
        # Show mode-specific info
        if mode == 'semantic':
            print_info("   🧠 Using pure semantic search with embeddings")
            print_info("   📊 Best for conceptual and topic-based queries")
        elif mode == 'hybrid':
            print_info("   🔄 Combining semantic + BM25 keyword search")
            print_info("   ⚖️  Balanced approach for best overall performance")
            if hasattr(self, 'hybrid_engine'):
                stats = self.hybrid_engine.get_statistics()
                print_info(f"   📚 BM25 vocabulary: {stats['bm25_vocabulary_size']:,} unique terms")
                print_info(f"   📈 Average doc length: {stats['average_doc_length']:.1f} tokens")
        elif mode == 'keyword':
            print_info("   🔍 Using pure BM25 keyword search")
            print_info("   📝 Best for exact term and error code searches")
    
    def _perform_search_with_mode(self, rag_query, query_embedding=None):
        """Perform search using the selected RAG mode"""
        try:
            if self.rag_search_mode == 'semantic':
                # Use standard RAG engine (semantic search)
                return self.rag_engine.query(rag_query)
            
            elif self.rag_search_mode in ['hybrid', 'keyword']:
                # Use hybrid search engine
                if not hasattr(self, 'hybrid_engine') or self.hybrid_engine is None:
                    # Initialize hybrid engine if not already done
                    from src.rag.hybrid_search import HybridSearchEngine
                    self.hybrid_engine = HybridSearchEngine(
                        vector_store=self.vector_store,
                        index_path=f"data/vectorstore/bm25_index_{self.embedding_selector.get_current_model()}.pkl"
                    )
                
                # Generate embedding if not provided
                if query_embedding is None:
                    current_embedding_provider = self.embedding_selector.get_current_provider()
                    query_embedding = current_embedding_provider.encode([rag_query.query])[0]
                
                # Perform search based on mode
                if self.rag_search_mode == 'hybrid':
                    # Hybrid search - combine semantic + keyword
                    hybrid_results = self.hybrid_engine.hybrid_search(
                        query=rag_query.query,
                        query_embedding=query_embedding,
                        k=rag_query.context_window
                    )
                else:  # keyword mode
                    # Pure keyword search using BM25
                    keyword_results = self.hybrid_engine.keyword_search(
                        query=rag_query.query,
                        k=rag_query.context_window
                    )
                    # Convert to hybrid result format for consistency
                    hybrid_results = []
                    for idx, score in keyword_results:
                        if idx < len(self.vector_store.documents):
                            from src.rag.hybrid_search import HybridSearchResult
                            result = HybridSearchResult(
                                content=self.vector_store.documents[idx],
                                metadata=self.vector_store.metadata[idx] if idx < len(self.vector_store.metadata) else {},
                                semantic_score=0.0,
                                keyword_score=score,
                                combined_score=score,
                                rank=len(hybrid_results) + 1
                            )
                            hybrid_results.append(result)
                
                # Convert hybrid results to RAG engine result format
                return self._convert_hybrid_to_rag_result(hybrid_results, rag_query.query)
            
            else:
                # Fallback to semantic search
                return self.rag_engine.query(rag_query)
        
        except Exception as e:
            if self.analysis_verbose:
                print_error(f"❌ Search mode '{self.rag_search_mode}' failed: {e}")
                print_info("   Falling back to semantic search")
            # Fallback to standard semantic search
            return self.rag_engine.query(rag_query)
    
    def _convert_hybrid_to_rag_result(self, hybrid_results, query):
        """Convert hybrid search results to RAG engine result format"""
        # Create a mock RAG result that matches the expected format
        class MockRAGResult:
            def __init__(self, sources, query):
                self.sources = sources
                self.query = query
                self.confidence = self._calculate_confidence(sources)
                
                # Generate answer using the model with retrieved context
                context_text = "\n\n".join([
                    f"Source {i+1}:\n{result.content[:500]}..."
                    for i, result in enumerate(sources[:3])
                ])
                
                prompt = f"""You are a PCIe debugging expert. Based on the following context, provide a detailed technical analysis for the user's query.

Query: {query}

Context from knowledge base:
{context_text}

Please structure your response with:
1. **Analysis**: Technical explanation based on the context
2. **Root Cause**: Most likely cause of the issue
3. **Impact**: How this affects system operation  
4. **Recommendations**: Specific debugging steps or fixes

Be concise but thorough in your technical analysis."""
                
                self.answer = self._generate_answer_with_model(prompt)
            
            def _calculate_confidence(self, sources):
                """Calculate confidence based on search scores"""
                if not sources:
                    return 0.0
                avg_score = sum(s.combined_score for s in sources) / len(sources)
                return min(avg_score, 1.0)  # Cap at 1.0
            
            def _generate_answer_with_model(self, prompt):
                """Generate answer using the current model"""
                try:
                    # Use the model wrapper to generate response
                    if hasattr(self, 'model_wrapper'):
                        return self.model_wrapper.generate_completion(prompt)
                    else:
                        # Fallback - get the model selector and generate directly
                        from src.models.model_selector import get_model_selector
                        model_selector = get_model_selector()
                        return model_selector.generate_completion(prompt)
                except Exception as e:
                    return f"Error generating response: {e}"
        
        # Convert hybrid results to the format expected by RAG result
        converted_sources = []
        for result in hybrid_results:
            source = {
                'content': result.content,
                'metadata': result.metadata,
                'score': result.combined_score
            }
            converted_sources.append(source)
        
        # Create mock result with proper answer generation
        mock_result = MockRAGResult(converted_sources, query)
        
        # Patch the answer generation to use the instance's model wrapper
        if hasattr(self, 'model_wrapper'):
            context_text = "\n\n".join([
                f"Source {i+1}:\n{result.content[:500]}..."
                for i, result in enumerate(hybrid_results[:3])
            ])
            
            prompt = f"""You are a PCIe debugging expert. Based on the following context, provide a detailed technical analysis for the user's query.

Query: {query}

Context from knowledge base:
{context_text}

Please structure your response with:
1. **Analysis**: Technical explanation based on the context
2. **Root Cause**: Most likely cause of the issue
3. **Impact**: How this affects system operation  
4. **Recommendations**: Specific debugging steps or fixes

Be concise but thorough in your technical analysis."""
            
            mock_result.answer = self.model_wrapper.generate_completion(prompt)
        
        return mock_result
    
    def do_tokens(self, arg):
        """Show detailed token usage information"""
        model_name = self.model_selector.get_current_model()
        limits = self.token_counter.get_model_limits(model_name)
        
        print(f"""
📊 Token Usage Analysis
{'='*60}

🎯 Current Session:
   Input Tokens:  {self.token_counter.format_token_count(self.session_tokens['input'])}
   Output Tokens: {self.token_counter.format_token_count(self.session_tokens['output'])}
   Total Tokens:  {self.token_counter.format_token_count(self.session_tokens['input'] + self.session_tokens['output'])}

📏 Model Limits ({model_name}):
   Context Window: {limits['context']:,} tokens
   Max Output:     {limits['max_output']:,} tokens

📈 Usage Percentage:
   Context Usage: {(self.session_tokens['input'] + self.session_tokens['output']) / limits['context'] * 100:.1f}%
   Output Usage:  {self.session_tokens['output'] / limits['max_output'] * 100:.1f}%

💡 Token Estimation Guide:
   • 1 token ≈ 4 characters (English)
   • 1 token ≈ 0.75 words
   • 100 tokens ≈ 75 words ≈ 3-4 sentences
   • 1,000 tokens ≈ 750 words ≈ 1.5 pages
   • 10,000 tokens ≈ 7,500 words ≈ 15 pages

💰 Cost Estimation (OpenAI pricing):
   • GPT-4o-mini: $0.075 per 1M input, $0.300 per 1M output
   • GPT-4o: $2.50 per 1M input, $10.00 per 1M output
   • GPT-4: $10.00 per 1M input, $30.00 per 1M output

{'='*60}
        """)
    
    def do_verbose(self, arg):
        """Toggle verbose analysis mode"""
        if not arg:
            # Show current status
            status = "ON" if self.analysis_verbose else "OFF"
            print(f"\n🔧 Verbose Analysis Mode: {status}")
            print("\nVerbose mode shows detailed analysis steps:")
            print("  📊 Vector search process")
            print("  🧠 LLM prompt construction")
            print("  ⏱️  Response timing")
            print("  📚 Source document details")
            print("\nUsage: /verbose on|off")
        elif arg.lower() in ["on", "true", "1", "yes"]:
            self.analysis_verbose = True
            print_success("✅ Verbose analysis mode enabled")
            print("   🔍 Will show detailed analysis steps")
        elif arg.lower() in ["off", "false", "0", "no"]:
            self.analysis_verbose = False
            print_success("✅ Verbose analysis mode disabled")
            print("   🔇 Will show concise responses only")
        else:
            print_error("❌ Invalid option. Use: /verbose on|off")
    
    def do_stream(self, arg):
        """Toggle real-time streaming responses"""
        if not arg:
            # Show current status
            status = "ON" if self.streaming_enabled else "OFF"
            print(f"\n🌊 Streaming Mode: {status}")
            print("\nStreaming mode shows LLM responses in real-time:")
            print("  ⚡ Text appears as it's generated")
            print("  👁️  Better user experience and feedback")
            print("  ⏱️  See progress without waiting")
            print(f"  🔧 Stream delay: {self.stream_delay}s between chunks")
            print("\nUsage: /stream on|off")
        elif arg.lower() in ["on", "true", "1", "yes"]:
            self.streaming_enabled = True
            print_success("✅ Real-time streaming enabled")
            print("   🌊 LLM responses will appear in real-time")
        elif arg.lower() in ["off", "false", "0", "no"]:
            self.streaming_enabled = False
            print_success("✅ Real-time streaming disabled")
            print("   📄 LLM responses will appear all at once")
        else:
            print_error("❌ Invalid option. Use: /stream on|off")
    
    def do_vim(self, arg):
        """Enable vim mode for input editing"""
        try:
            readline.parse_and_bind("set editing-mode vi")
            print_success("✅ Vim mode enabled")
        except:
            print_error("❌ Failed to enable vim mode")
    
    def do_exit(self, arg):
        """Exit the shell"""
        return True
    
    def do_quit(self, arg):
        """Exit the shell"""
        return True
    
    def do_EOF(self, arg):
        """Handle Ctrl+D"""
        print("\n👋 Goodbye!")
        return True
    
    def default(self, line):
        """Handle non-command input as PCIe questions"""
        if not line.strip():
            return
        
        # Handle slash commands
        if line.startswith('/'):
            # Convert slash command to method call
            parts = line[1:].split(None, 1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            
            # Map common slash commands to methods
            if hasattr(self, f'do_{cmd}'):
                return getattr(self, f'do_{cmd}')(arg)
            else:
                print(f"❌ Unknown command: /{cmd}")
                print("   Type /help for available commands")
                return
        
        # Check turn limit
        if self.turn_count >= self.max_turns:
            print_warning(f"⚠️ Maximum turns ({self.max_turns}) reached. Use /clear to reset.")
            return
        
        # Process as PCIe debugging query
        self._process_query(line)
        self.turn_count += 1
    
    def _process_query(self, query: str):
        """Process a PCIe debugging query"""
        try:
            model_name = self.model_selector.get_current_model()
            print(f"\n🔍 Analyzing with {model_name}...")
            
            start_time = time.time()
            embedding_time = 0
            search_time = 0
            llm_time = 0
            
            input_tokens = 0
            output_tokens = 0
            
            if self.rag_enabled and self.rag_engine:
                # Use RAG pipeline (streaming not yet supported with RAG)
                original_streaming = self.streaming_enabled
                self.streaming_enabled = False  # Temporarily disable streaming for RAG
                if self.analysis_verbose:
                    print(f"\n📝 Query: '{query}'")
                    print(f"   Length: {len(query)} characters")
                    if original_streaming:
                        print("   ℹ️  Streaming temporarily disabled for RAG pipeline")
                    print("\n🔧 Analysis Pipeline (RAG ENABLED):")
                    print("\n  1️⃣ Generating embeddings for query...")
                    embedding_start = time.time()
                
                # Use RAG with selected search mode
                from src.rag.enhanced_rag_engine import RAGQuery
                rag_query = RAGQuery(query=query, context_window=5)
                
                if self.analysis_verbose:
                    # Show embedding details
                    current_embedding_provider = self.embedding_selector.get_current_provider()
                    query_embedding = current_embedding_provider.encode([query])[0]
                    embedding_time = time.time() - embedding_start
                    
                    current_embedding_model = self.embedding_selector.get_current_model()
                    print(f"     ✓ Embedding model: {current_embedding_model}")
                    print(f"     ✓ Embedding dimension: {len(query_embedding)}")
                    print(f"     ✓ Embedding time: {embedding_time:.3f}s")
                    print(f"     ✓ Embedding norm: {np.linalg.norm(query_embedding):.4f}")
                    
                    print("\n  2️⃣ Searching vector database...")
                    search_start = time.time()
                    print(f"     → Vector DB size: {self.vector_store.index.ntotal} documents")
                    print(f"     → Search method: {self.rag_search_mode.upper()}")
                    print(f"     → Top-k retrieval: {rag_query.context_window} documents")
                
                # Use custom search based on selected mode
                result = self._perform_search_with_mode(rag_query, query_embedding if self.analysis_verbose else None)
                
                if self.analysis_verbose:
                    search_time = time.time() - search_start - embedding_time
                    print(f"     ✓ Search completed in {search_time:.3f}s")
                    
                    print(f"\n  3️⃣ Retrieved {len(getattr(result, 'sources', []))} source documents")
                    print(f"\n  4️⃣ Generating LLM response with {model_name}...")
                    llm_start = time.time()
                
                end_time = time.time()
                
                if self.analysis_verbose:
                    llm_time = end_time - llm_start
                
                if result and hasattr(result, 'answer'):
                    response = result.answer
                    sources = getattr(result, 'sources', [])
                    confidence = getattr(result, 'confidence', None)
                    
                    # Show source files used
                    if sources:
                        self._display_source_files(sources)
                    
                    # Estimate tokens for RAG (prompt includes context)
                    context_text = "\n".join([s.get('content', '')[:500] for s in sources[:5]])
                    full_prompt = f"{query}\n\nContext:\n{context_text}"
                    input_tokens = self.token_counter.count_tokens(full_prompt, model_name)
                    output_tokens = self.token_counter.count_tokens(response, model_name)
                else:
                    response = "No response generated"
                    sources = []
                    confidence = None
            else:
                # Direct LLM mode without RAG
                if self.analysis_verbose:
                    print(f"\n📝 Query: '{query}'")
                    print(f"   Length: {len(query)} characters")
                    print("\n🔧 Analysis Pipeline (RAG DISABLED):")
                    print("\n  ⚠️  No vector database loaded")
                    print("  🤖 Sending query directly to LLM...")
                    print(f"     → Model: {model_name}")
                    print(f"     → Mode: Direct (no context retrieval)")
                
                # Create a PCIe-focused prompt
                prompt = f"""You are a PCIe debugging expert. Please provide a detailed technical analysis for the following query:

Query: {query}

Please structure your response with:
1. **Analysis**: Technical explanation
2. **Root Cause**: Most likely cause of the issue
3. **Impact**: How this affects system operation
4. **Recommendations**: Specific debugging steps or fixes

Be concise but thorough in your technical analysis."""
                
                # Count input tokens
                input_tokens = self.token_counter.count_tokens(prompt, model_name)
                
                if self.analysis_verbose:
                    llm_start = time.time()
                
                if self.streaming_enabled:
                    response = self._generate_streaming_response(prompt)
                else:
                    response = self.model_wrapper.generate_completion(prompt)
                sources = []
                confidence = None
                
                end_time = time.time()
                
                if self.analysis_verbose:
                    llm_time = end_time - llm_start
                
                # Count output tokens
                output_tokens = self.token_counter.count_tokens(response, model_name)
                
                # Restore original streaming setting if it was defined
                if 'original_streaming' in locals():
                    self.streaming_enabled = original_streaming
            
            if response and response != "No response generated":
                if not self.streaming_enabled:
                    # Non-streaming mode: print all at once
                    print("\n💡 Response:")
                    print("-" * 60)
                    print(response)
                    print("-" * 60)
                
                if self.analysis_verbose:
                    print(f"\n📊 Analysis Details:")
                    print(f"  ⏱️  Total response time: {end_time - start_time:.2f}s")
                    
                    if self.rag_enabled and 'result' in locals() and hasattr(result, 'sources'):
                        # RAG mode - show detailed timing breakdown
                        print(f"\n  ⏲️  Time Breakdown:")
                        print(f"     • Embedding generation: {embedding_time:.3f}s")
                        print(f"     • Vector search: {search_time:.3f}s")
                        print(f"     • LLM processing: {llm_time:.3f}s")
                        print(f"     • Other operations: {end_time - start_time - embedding_time - search_time - llm_time:.3f}s")
                        
                        print(f"\n  📚 RAG Pipeline Results:")
                        print(f"     • Sources retrieved: {len(result.sources)}")
                        print(f"     • Context window: {rag_query.context_window} documents")
                        if hasattr(result, 'confidence'):
                            print(f"     • Confidence score: {result.confidence:.1%}")
                        
                        # Show detailed source information
                        if result.sources and len(result.sources) > 0:
                            from pathlib import Path
                            print(f"\n📖 Source Documents (Top {min(5, len(result.sources))}):")
                            print("=" * 80)
                            for i, source in enumerate(result.sources[:5], 1):  # Show top 5 sources
                                source_info = source.get('metadata', {})
                                content = source.get('content', '')
                                score = source.get('score', 0.0)
                                
                                source_path = source_info.get('source', 'Unknown')
                                filename = Path(source_path).name if source_path != 'Unknown' else 'Unknown'
                                file_description = self._get_file_description(filename)
                                
                                print(f"\n  [{i}] Relevance Score: {score:.4f}")
                                print(f"      📄 File: {filename}")
                                print(f"      📝 Description: {file_description}")
                                if source_path != 'Unknown':
                                    print(f"      🗂️  Full Path: {source_path}")
                                if 'chunk_id' in source_info:
                                    print(f"      🔗 Chunk ID: {source_info['chunk_id']}")
                                if 'page' in source_info:
                                    print(f"      📄 Page: {source_info['page']}")
                                print(f"      📊 Content ({len(content)} chars):")
                                # Show more content in verbose mode
                                preview_length = 300
                                if len(content) > preview_length:
                                    print(f"      {content[:preview_length]}...")
                                else:
                                    print(f"      {content}")
                            print("=" * 80)
                            
                        # Show embedding statistics
                        embedding_info = self.embedding_selector.get_model_info()
                        print(f"\n🧮 Embedding Statistics:")
                        print(f"   • Model: {embedding_info.get('model', 'N/A')}")
                        print(f"   • Provider: {embedding_info.get('provider', 'N/A')}")
                        print(f"   • Dimension: {embedding_info.get('dimension', 'N/A')}")
                        print(f"   • Cost: {embedding_info.get('cost', 'N/A')}")
                        print(f"   • Similarity metric: Cosine")
                        print(f"   • Index type: FAISS IndexFlatIP")
                    else:
                        # Direct mode - no sources
                        print("\n  🚀 Direct LLM Mode (RAG DISABLED)")
                        print(f"     • No vector database search performed")
                        print(f"     • Query sent directly to {model_name}")
                        print(f"     • Processing time: {end_time - start_time:.3f}s")
                    
                    # Always show token counts in verbose mode
                    print(f"\n📊 Token Usage:")
                    print(f"   • Input tokens: {self.token_counter.format_token_count(input_tokens)}")
                    print(f"   • Output tokens: {self.token_counter.format_token_count(output_tokens)}")
                    print(f"   • Total tokens: {self.token_counter.format_token_count(input_tokens + output_tokens)}")
                    
                    # Check token limits
                    token_usage = self.token_counter.check_token_usage(input_tokens, output_tokens, model_name)
                    if token_usage['within_limits']:
                        print(f"   • Context usage: {token_usage['context_usage_pct']:.1f}% of {token_usage['context_limit']:,} limit")
                        print(f"   • Output usage: {token_usage['output_usage_pct']:.1f}% of {token_usage['output_limit']:,} limit")
                    else:
                        print(f"   ⚠️  Token limits exceeded!")
                        if token_usage['total_tokens'] > token_usage['context_limit']:
                            print(f"      Context limit exceeded: {token_usage['total_tokens']:,} > {token_usage['context_limit']:,}")
                        if token_usage['output_tokens'] > token_usage['output_limit']:
                            print(f"      Output limit exceeded: {token_usage['output_tokens']:,} > {token_usage['output_limit']:,}")
                elif self.verbose:
                    print(f"\n⏱️ Response time: {end_time - start_time:.1f}s")
                    if self.rag_enabled and 'result' in locals() and hasattr(result, 'sources'):
                        print(f"📚 Sources: {len(result.sources)}")
                        if hasattr(result, 'confidence'):
                            print(f"📊 Confidence: {result.confidence:.1%}")
                
                # Add to conversation history
                self._add_to_history("user", query)
                self._add_to_history("assistant", response)
                
                # Update session token counts
                self.session_tokens["input"] += input_tokens
                self.session_tokens["output"] += output_tokens
                
            else:
                print_error("❌ No response generated")
                
        except Exception as e:
            print_error(f"❌ Error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def _add_to_history(self, role: str, content: str):
        """Add entry to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def _get_model_provider(self, model_name: str) -> str:
        """Get provider type for a model"""
        provider_map = {
            "gpt-4o-mini": "OpenAI API",
            "gpt-4o": "OpenAI API",
            "gpt-4": "OpenAI API",
            "claude-3-opus": "Anthropic API",
            "llama-3.2-3b": "Local GGUF",
            "deepseek-r1-7b": "Ollama",
            "mock-llm": "Built-in Mock"
        }
        return provider_map.get(model_name, "Unknown")
    
    def cmdloop(self, intro=None):
        """Override cmdloop to handle errors gracefully"""
        try:
            super().cmdloop(intro)
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Type /exit to quit.")
            self.cmdloop()
        except Exception as e:
            print_error(f"\n❌ Unexpected error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()


def start_interactive_mode(model_id: Optional[str] = None, verbose: bool = False, 
                          max_turns: int = 50, continue_session: bool = False,
                          analysis_verbose: bool = False):
    """Start interactive PCIe debugging shell"""
    
    # Suppress warnings unless verbose
    if not verbose:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["GGML_METAL_LOG_LEVEL"] = "0"
    
    shell = PCIeDebugShell(model_id=model_id, verbose=verbose, max_turns=max_turns)
    shell.analysis_verbose = analysis_verbose
    
    # Load most recent session if continue requested
    if continue_session:
        sessions = shell.session_manager.list_sessions()
        if sessions:
            latest = sessions[0]
            session = shell.session_manager.load_session(latest['id'])
            if session:
                shell.conversation_history = session['conversation']
                print_info(f"📋 Continued session: {latest['title']}")
    
    shell.cmdloop()