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
        
        # Session state
        self.current_session = None
        self.conversation_history = []
        self.session_tokens = {"input": 0, "output": 0}  # Track tokens per session
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize RAG system"""
        try:
            if self.verbose:
                print("🔧 Initializing PCIe Debug Agent...")
            
            # Try to load vector store (optional)
            vector_db_path = Path("data/vectorstore")
            self.vector_store = None
            self.rag_enabled = False
            
            if vector_db_path.exists():
                try:
                    # Use the load class method to load existing store
                    self.vector_store = FAISSVectorStore.load(str(vector_db_path))
                    self.rag_enabled = True
                    
                    if self.verbose:
                        print(f"   Vector store: {self.vector_store.index.ntotal} documents")
                except Exception as e:
                    print_warning(f"⚠️ Vector database found but couldn't load: {e}")
                    print_info("   Continuing without RAG support")
            else:
                print_info("ℹ️ No vector database found. RAG features disabled.")
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
            
            # Initialize RAG engine only if vector store is available
            if self.rag_enabled and self.vector_store:
                self.rag_engine = EnhancedRAGEngine(
                    vector_store=self.vector_store,
                    model_manager=ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=True)
                )
            else:
                # Use direct model without RAG
                self.rag_engine = None
                self.model_wrapper = ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=False)
            
            if self.verbose:
                print("✅ System ready!")
                if not self.rag_enabled:
                    print("   (Running in direct mode without RAG)")
                
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
  /vim                 Enable vim mode
  /exit, /quit         Exit the shell

Direct Usage:
  Just type your PCIe debugging questions directly!
  
Examples:
  > Why is PCIe link training failing?
  > /model llama-3.2-3b
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
                    def __init__(self, selector, rag_enabled=False):
                        self.selector = selector
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
    
    def do_rag_model(self, arg):
        """List or switch RAG embedding models"""
        if not arg:
            # List available embedding models
            current = self.embedding_selector.get_current_model()
            models = self.embedding_selector.list_models()
            
            print("\n🧮 Available Embedding Models:")
            print("-" * 70)
            
            for model_id, config in models.items():
                marker = "✓" if model_id == current else " "
                available = "✅" if self.embedding_selector.is_available(model_id) else "❌"
                provider_type = "API" if config["cost"] != "free" else "Local"
                
                print(f"  {marker} {model_id:<25} {available} - {config['description']}")
                print(f"    {' '*27} {provider_type} | {config['speed']} | {config['cost']}")
            
            print(f"\nCurrent: {current}")
            print("Use '/rag_model <name>' to switch")
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
            
            # If RAG is enabled, warn about rebuilding vector store
            if self.rag_enabled:
                print_warning("⚠️  Embedding model changed!")
                print_info("   Vector database may need rebuilding for optimal performance")
                print_info("   Run 'pcie-debug vectordb build --force' to rebuild with new embeddings")
                
                # Disable RAG temporarily to avoid dimension mismatch
                self.rag_enabled = False
                print_info("   RAG temporarily disabled to prevent errors")
                print_info("   Re-enable with '/rag on' after rebuilding vector database")
        else:
            print_error(f"❌ Failed to switch to {arg}")
    
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
                # Use RAG pipeline
                if self.analysis_verbose:
                    print(f"\n📝 Query: '{query}'")
                    print(f"   Length: {len(query)} characters")
                    print("\n🔧 Analysis Pipeline (RAG ENABLED):")
                    print("\n  1️⃣ Generating embeddings for query...")
                    embedding_start = time.time()
                
                # Use RAG engine for analysis
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
                    print(f"     → Search method: FAISS (cosine similarity)")
                    print(f"     → Top-k retrieval: {rag_query.context_window} documents")
                
                result = self.rag_engine.query(rag_query)
                
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
                
                response = self.model_wrapper.generate_completion(prompt)
                sources = []
                confidence = None
                
                end_time = time.time()
                
                if self.analysis_verbose:
                    llm_time = end_time - llm_start
                
                # Count output tokens
                output_tokens = self.token_counter.count_tokens(response, model_name)
            
            if response and response != "No response generated":
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
                            print(f"\n📖 Source Documents (Top {min(5, len(result.sources))}):")
                            print("=" * 80)
                            for i, source in enumerate(result.sources[:5], 1):  # Show top 5 sources
                                source_info = source.get('metadata', {})
                                content = source.get('content', '')
                                score = source.get('score', 0.0)
                                
                                print(f"\n  [{i}] Relevance Score: {score:.4f}")
                                print(f"      Source: {source_info.get('source', 'Unknown')}")
                                if 'chunk_id' in source_info:
                                    print(f"      Chunk ID: {source_info['chunk_id']}")
                                if 'page' in source_info:
                                    print(f"      Page: {source_info['page']}")
                                print(f"      Content ({len(content)} chars):")
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