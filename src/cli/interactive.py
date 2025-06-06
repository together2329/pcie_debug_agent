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
from typing import Dict, List, Optional, Any, Union

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
        self.rag_search_mode = 'semantic'  # Default search mode (most reliable)
        self.hybrid_engine = None  # Will be initialized when needed
        
        # Session state
        self.current_session = None
        self.conversation_history = []
        self.session_tokens = {"input": 0, "output": 0}  # Track tokens per session
        self.session_start_time = time.time()  # Track session duration
        
        # Streaming configuration
        self.streaming_enabled = True  # Enable streaming by default
        self.stream_delay = 0.01  # Small delay between chunks for better visual effect
        
        # Initialize model wrapper/manager
        self.model_wrapper = None
        self.model_manager = None  # Will be set during initialization
        
        # Initialize system
        self._initialize_system()
        
        # Setup tab completion for slash commands
        self._setup_tab_completion()
        
        # Configure logging to suppress empty error messages
        self._configure_logging()
    
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
                
                def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
                    """Embed text(s) - alias for generate_embeddings for compatibility"""
                    if isinstance(texts, str):
                        texts = [texts]
                    return self.generate_embeddings(texts)
            
            # Initialize RAG engine only if vector store is available and compatible
            if self.rag_enabled and self.vector_store:
                # Check dimension compatibility
                embedding_dim = self.embedding_selector.get_current_provider().get_dimension()
                vector_db_dim = self.vector_store.dimension
                
                if embedding_dim == vector_db_dim:
                    self.model_wrapper = ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=True)
                    self.model_manager = self.model_wrapper  # Set model_manager for unified RAG
                    self.rag_engine = EnhancedRAGEngine(
                        vector_store=self.vector_store,
                        model_manager=self.model_wrapper
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
                    self.model_manager = self.model_wrapper  # Set model_manager even without RAG
            else:
                # Use direct model without RAG
                self.rag_engine = None
                self.model_wrapper = ModelWrapper(self.model_selector, self.embedding_selector, rag_enabled=False)
                self.model_manager = self.model_wrapper  # Set model_manager even without RAG
            
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
    
    def _setup_tab_completion(self):
        """Setup tab completion for slash commands"""
        try:
            # Set up readline tab completion
            readline.set_completer_delims(' \t\n;')
            readline.parse_and_bind("tab: complete")
            
            # Enable history
            readline.set_history_length(1000)
            
            if self.verbose:
                print("✅ Tab completion enabled")
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Tab completion setup failed: {e}")
    
    def _configure_logging(self):
        """Configure logging to suppress empty error messages"""
        import logging
        
        # Custom filter to suppress empty error messages
        class EmptyErrorFilter(logging.Filter):
            def filter(self, record):
                # Skip empty error messages from vector store
                if record.levelname == 'ERROR' and 'Error searching vector store:' in record.getMessage():
                    # Check if the error message is effectively empty
                    msg = record.getMessage()
                    error_part = msg.split(':', 1)[-1].strip()
                    if not error_part:
                        return False  # Suppress this log record
                return True
        
        # Add filter to the faiss_store logger
        faiss_logger = logging.getLogger('src.vectorstore.faiss_store')
        faiss_logger.addFilter(EmptyErrorFilter())
        
        # Set appropriate logging level
        if not self.verbose:
            # Suppress info and debug messages in non-verbose mode
            logging.getLogger('src').setLevel(logging.WARNING)
    
    def completenames(self, text, *ignored):
        """Override cmd.Cmd completenames to handle slash commands"""
        if text.startswith('/'):
            # Handle slash command completion
            text_without_slash = text[1:]  # Remove the leading '/'
            commands = self._get_available_commands()
            matches = []
            
            for cmd_name, _ in commands:
                if cmd_name.startswith(text_without_slash):
                    matches.append('/' + cmd_name)
            
            return matches
        else:
            # Default behavior for non-slash commands
            return super().completenames(text, *ignored)
    
    def complete(self, text, state):
        """Enhanced completion that handles both slash commands and arguments"""
        # Get the current line
        line = readline.get_line_buffer()
        
        # Check if we're completing a slash command
        if line.strip().startswith('/'):
            # Split the line to see if we're completing the command or its arguments
            parts = line.strip().split()
            
            if len(parts) == 1 and not line.endswith(' '):
                # We're still completing the command name
                matches = self.completenames(text, None)
            else:
                # We're completing arguments for a specific command
                if len(parts) >= 1:
                    cmd_with_slash = parts[0]
                    cmd_name = cmd_with_slash[1:]  # Remove the '/'
                    matches = self._complete_command_args(cmd_name, text, line)
                else:
                    matches = []
        else:
            # For non-slash commands, use default completion
            matches = []
        
        # Return the match for the current state
        try:
            return matches[state]
        except IndexError:
            return None
    
    def _complete_command_args(self, cmd_name, text, line):
        """Complete arguments for specific commands"""
        matches = []
        
        # Command-specific argument completion
        if cmd_name == 'model':
            models = ["gpt-4o-mini", "mock-llm", "llama-3.2-3b", "deepseek-r1-7b", "gpt-4o", "gpt-4"]
            matches = [model for model in models if model.startswith(text)]
        
        elif cmd_name == 'rag_model':
            embedding_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002", 
                              "all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"]
            matches = [model for model in embedding_models if model.startswith(text)]
        
        elif cmd_name == 'rag_mode':
            modes = ["semantic", "hybrid", "keyword", "unified"]
            matches = [mode for mode in modes if mode.startswith(text)]
        
        elif cmd_name in ['rag_files', 'rag_check']:
            # No arguments needed for these commands
            matches = []
        
        elif cmd_name == 'rag':
            options = ["on", "off"]
            matches = [opt for opt in options if opt.startswith(text)]
        
        elif cmd_name == 'verbose':
            options = ["on", "off"]
            matches = [opt for opt in options if opt.startswith(text)]
        
        elif cmd_name == 'stream':
            options = ["on", "off"]
            matches = [opt for opt in options if opt.startswith(text)]
        
        elif cmd_name == 'analyze':
            # Complete file paths for log files
            import glob
            try:
                # Look for log files
                if text:
                    pattern = text + '*'
                else:
                    pattern = '*'
                
                # Search in common log directories
                log_patterns = [
                    pattern,
                    f"logs/{pattern}",
                    f"sample_logs/{pattern}",
                    f"*.log"
                ]
                
                for log_pattern in log_patterns:
                    matches.extend(glob.glob(log_pattern))
                
                # Filter matches that start with text
                matches = [match for match in matches if match.startswith(text)]
            except:
                matches = []
        
        elif cmd_name in ['session']:
            # For session commands, offer subcommands
            if 'save' not in line and 'load' not in line:
                subcommands = ["save", "load"]
                matches = [sub for sub in subcommands if sub.startswith(text)]
        
        elif cmd_name in ['memory']:
            # For memory commands, offer subcommands
            if 'set' not in line and 'del' not in line:
                subcommands = ["set", "del"]
                matches = [sub for sub in subcommands if sub.startswith(text)]
        
        return matches
    
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
  /rag_mode [mode]     Select RAG search mode (semantic/hybrid/keyword/unified/pcie)
  /rag_status          Show detailed RAG and vector DB status
  /knowledge_base      Show RAG knowledge base content and status
  /kb                  Alias for /knowledge_base
  /rag_files           Show which files are indexed in each RAG database
  /rag_check           Quick check of current RAG database readiness
  /stream [on/off]     Toggle real-time streaming responses
  /vim                 Enable vim mode
  /exit, /quit         Exit the shell

Direct Usage:
  Just type your PCIe debugging questions directly!
  
💡 Tab Completion:
  • Press TAB to auto-complete slash commands
  • Press TAB again to cycle through options
  • Works for commands: /ra[TAB] → /rag, /rag_mode, etc.
  • Works for arguments: /rag_mode [TAB] → semantic, hybrid, keyword
  
🔧 PCIe Mode (Recommended):
  /rag_mode pcie       Switch to PCIe-optimized adaptive chunking
  pcie-debug pcie build --force    Build PCIe knowledge base
  pcie-debug pcie query "LTSSM states"    Query with filters

Examples:
  > Why is PCIe link training failing?
  > /rag_mode pcie
  > What are LTSSM timeout conditions?
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
        """Clear conversation history and free up context"""
        self.conversation_history.clear()
        self.turn_count = 0
        self.session_tokens = {"input": 0, "output": 0}
        self.session_start_time = time.time()  # Reset session timer
        print_success("✅ Conversation cleared and session timer reset")
    
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
    
    
    def do_msearch(self, arg):
        """Metadata-enhanced search with filters
        Usage: /msearch <query> [--version VERSION] [--type TYPE] [--component COMP]
        
        Examples:
            /msearch "link training" --version 4.0
            /msearch "error handling" --type error_log --component endpoint
            /msearch "LTSSM timeout" --severity error
        """
        if not arg:
            print("Usage: /msearch <query> [--version VERSION] [--type TYPE] [--component COMP]")
            return
        
        if not self.rag_enabled or not self.vector_store:
            print_error("❌ Vector database not loaded. RAG features are disabled.")
            return
        
        try:
            # Parse arguments
            parts = arg.split()
            query_parts = []
            filters = {}
            
            i = 0
            while i < len(parts):
                if parts[i].startswith('--'):
                    if i + 1 < len(parts):
                        key = parts[i][2:]
                        value = parts[i + 1]
                        filters[key] = value
                        i += 2
                    else:
                        i += 1
                else:
                    query_parts.append(parts[i])
                    i += 1
            
            query = ' '.join(query_parts)
            
            # Show search info
            print(f"\n🔍 Metadata Search: '{query}'")
            if filters:
                print("📋 Filters:")
                for key, value in filters.items():
                    print(f"   {key}: {value}")
            print("-" * 50)
            
            # Create metadata query
            from src.rag.metadata_enhanced_rag import MetadataRAGQuery, MetadataEnhancedRAGEngine
            from src.rag.metadata_extractor import PCIeDocumentType, PCIeVersion, ErrorSeverity
            
            # Initialize metadata engine if not exists
            if not hasattr(self, 'metadata_engine'):
                from src.rag.metadata_extractor import MetadataExtractor
                metadata_extractor = MetadataExtractor(self.model_manager, self.llm_model)
                self.metadata_engine = MetadataEnhancedRAGEngine(
                    self.vector_store,
                    self.model_manager,
                    metadata_extractor
                )
            
            # Build metadata query
            metadata_query = MetadataRAGQuery(
                query=query,
                context_window=5,
                pcie_versions=[filters.get('version')] if 'version' in filters else None,
                document_types=[PCIeDocumentType(filters.get('type'))] if 'type' in filters else None,
                error_severity=ErrorSeverity(filters.get('severity')) if 'severity' in filters else None,
                components=[filters.get('component')] if 'component' in filters else None
            )
            
            # Execute query
            response = self.metadata_engine.query_with_metadata(metadata_query)
            
            # Display results
            print(f"\n💡 Answer (Confidence: {response.confidence:.1%}):")
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)} documents):")
                for i, source in enumerate(response.sources, 1):
                    print(f"\n{i}. {source['title']}")
                    metadata = source['metadata']
                    if metadata.get('pcie_version'):
                        print(f"   PCIe: {', '.join(metadata['pcie_version'])}")
                    if metadata.get('document_type'):
                        print(f"   Type: {metadata['document_type']}")
                    print(f"   Score: {source['relevance_score']:.3f}")
            
            # Show performance info in verbose mode
            if self.analysis_verbose and response.metadata:
                print(f"\n⚡ Performance:")
                print(f"   Query time: {response.metadata.get('query_time', 0):.2f}s")
                if response.metadata.get('metadata_filters_applied'):
                    print(f"   Filtered: {response.metadata.get('initial_results', 0)} → {response.metadata.get('filtered_results', 0)} results")
                    
        except Exception as e:
            print_error(f"❌ Metadata search failed: {str(e)}")
            if self.analysis_verbose:
                import traceback
                traceback.print_exc()
    
    def do_mextract(self, arg):
        """Extract metadata from a document
        Usage: /mextract <file_path> [--quick]
        """
        if not arg:
            print("Usage: /mextract <file_path> [--quick]")
            return
            
        parts = arg.split()
        file_path = parts[0]
        quick = '--quick' in parts
        
        try:
            from pathlib import Path
            path = Path(file_path)
            
            if not path.exists():
                print_error(f"❌ File not found: {file_path}")
                return
            
            print_info(f"📄 Extracting metadata from: {path.name}")
            
            # Read file content
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Initialize extractor if needed
            if not hasattr(self, 'metadata_extractor'):
                from src.rag.metadata_extractor import MetadataExtractor
                self.metadata_extractor = MetadataExtractor(self.model_manager, self.llm_model)
            
            if quick:
                print_info("Using quick extraction (regex-based)...")
                metadata = self.metadata_extractor.extract_quick_metadata(content)
                
                print_success("\n✅ Quick metadata extracted:")
                for key, value in metadata.items():
                    if value and value != []:
                        print(f"  {key}: {value}")
            else:
                print_info("Using LLM extraction (this may take a moment)...")
                
                # Run async extraction
                import asyncio
                metadata = asyncio.run(
                    self.metadata_extractor.extract_metadata(content, str(path))
                )
                
                print_success("\n✅ Rich metadata extracted:")
                print(f"  Type: {metadata.document_type.value}")
                print(f"  Title: {metadata.title}")
                print(f"  Summary: {metadata.summary}")
                
                if metadata.pcie_version:
                    print(f"  PCIe versions: {[v.value for v in metadata.pcie_version]}")
                if metadata.topics:
                    print(f"  Topics: {', '.join(metadata.topics)}")
                if metadata.error_codes:
                    print(f"  Error codes: {', '.join(metadata.error_codes[:5])}")
                if metadata.components:
                    print(f"  Components: {', '.join(metadata.components)}")
                
                print(f"\n  Confidence: {metadata.confidence_score:.1%}")
                
        except Exception as e:
            print_error(f"❌ Metadata extraction failed: {str(e)}")
    
    def do_mstats(self, arg):
        """Show metadata statistics for indexed documents"""
        if not self.vector_store:
            print_error("❌ Vector database not loaded")
            return
            
        try:
            print_info("📊 Metadata Statistics")
            print("=" * 60)
            
            total_docs = len(self.vector_store.documents)
            has_metadata = sum(1 for m in self.vector_store.metadata if m and len(m) > 1)
            
            print(f"\nTotal documents: {total_docs}")
            print(f"With rich metadata: {has_metadata} ({has_metadata/total_docs*100:.1f}%)")
            
            # Analyze metadata
            doc_types = {}
            pcie_versions = {}
            topics_count = {}
            
            for metadata in self.vector_store.metadata:
                if metadata:
                    # Document types
                    doc_type = metadata.get('document_type', 'unknown')
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    # PCIe versions
                    for version in metadata.get('pcie_version', []):
                        pcie_versions[version] = pcie_versions.get(version, 0) + 1
                    
                    # Topics
                    for topic in metadata.get('topics', []):
                        topics_count[topic] = topics_count.get(topic, 0) + 1
            
            if doc_types:
                print("\nDocument types:")
                for dtype, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {dtype}: {count}")
            
            if pcie_versions:
                print("\nPCIe versions coverage:")
                for version, count in sorted(pcie_versions.items()):
                    print(f"  PCIe {version}: {count} documents")
            
            if topics_count:
                print("\nTop topics:")
                for topic, count in sorted(topics_count.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {topic}: {count}")
                    
        except Exception as e:
            print_error(f"❌ Failed to get statistics: {str(e)}")
    

    def do_urag(self, arg):
        """Unified RAG query using intelligent method combination
        Usage: /urag <query> [--strategy STRATEGY] [--priority PRIORITY] [--time TIME]
        
        Strategies: fast, balanced, comprehensive, adaptive (default)
        Priority: speed, accuracy, balance (default)
        Time: max response time in seconds
        
        Examples:
            /urag "PCIe 4.0 link training issues"
            /urag "debug LTSSM timeout" --strategy comprehensive --priority accuracy
            /urag "Gen3 speed" --strategy fast --time 1.0
        """
        if not arg:
            print("Usage: /urag <query> [--strategy STRATEGY] [--priority PRIORITY] [--time TIME]")
            return
        
        if not self.rag_enabled or not self.vector_store:
            print_error("❌ Vector database not loaded. RAG features are disabled.")
            return
        
        try:
            # Parse arguments
            parts = arg.split()
            query_parts = []
            options = {}
            
            i = 0
            while i < len(parts):
                if parts[i].startswith('--'):
                    if i + 1 < len(parts):
                        key = parts[i][2:]
                        value = parts[i + 1]
                        options[key] = value
                        i += 2
                    else:
                        i += 1
                else:
                    query_parts.append(parts[i])
                    i += 1
            
            query = ' '.join(query_parts)
            
            # Show query info
            print(f"\n🔄 Unified RAG Query: '{query}'")
            if options:
                print("⚙️  Options:")
                for key, value in options.items():
                    print(f"   {key}: {value}")
            print("-" * 60)
            
            # Initialize unified RAG engine if not exists
            if not hasattr(self, 'unified_rag'):
                from src.rag.unified_rag_engine import UnifiedRAGEngine
                self.unified_rag = UnifiedRAGEngine(
                    self.vector_store,
                    self.model_manager
                )
            
            # Create unified query
            from src.rag.unified_rag_engine import UnifiedRAGQuery, ProcessingStrategy
            
            # Map strategy strings to enums
            strategy_map = {
                "fast": ProcessingStrategy.FAST_ONLY,
                "balanced": ProcessingStrategy.BALANCED,
                "comprehensive": ProcessingStrategy.COMPREHENSIVE,
                "adaptive": ProcessingStrategy.ADAPTIVE,
                "cascading": ProcessingStrategy.CASCADING
            }
            
            strategy = strategy_map.get(options.get('strategy', 'adaptive'), ProcessingStrategy.ADAPTIVE)
            priority = options.get('priority', 'balance')
            max_time = float(options.get('time', 0)) if options.get('time') else None
            
            unified_query = UnifiedRAGQuery(
                query=query,
                strategy=strategy,
                priority=priority,
                max_response_time=max_time,
                user_expertise=getattr(self, 'user_expertise', 'intermediate')
            )
            
            # Execute unified query
            import asyncio
            response = asyncio.run(self.unified_rag.query(unified_query))
            
            # Display results
            print(f"\n🎯 Answer (Confidence: {response.confidence:.1%}):")
            print(response.answer)
            
            if response.sources:
                print(f"\n📚 Sources ({len(response.sources)} documents):")
                for i, source in enumerate(response.sources[:5], 1):
                    print(f"\n{i}. Document {source['document_id']}")
                    if 'metadata' in source and source['metadata'].get('title'):
                        print(f"   Title: {source['metadata']['title']}")
                    print(f"   Score: {source['final_score']:.3f}")
                    
                    # Show method contributions
                    contributions = source.get('method_contributions', {})
                    if contributions:
                        contrib_str = ", ".join([f"{method}: {score:.2f}" for method, score in contributions.items()])
                        print(f"   Methods: {contrib_str}")
                    
                    consensus = source.get('consensus_score', 0)
                    if consensus > 0:
                        print(f"   Consensus: {consensus:.1%}")
            
            # Show performance breakdown
            print(f"\n⚡ Performance:")
            print(f"   Total time: {response.total_processing_time:.2f}s")
            print(f"   Methods used: {', '.join(response.methods_used)}")
            
            if self.analysis_verbose:
                print(f"   Method breakdown:")
                for method, time_taken in response.processing_breakdown.items():
                    contribution = response.method_contributions.get(method, 0)
                    print(f"     {method}: {time_taken:.2f}s (weight: {contribution:.1%})")
                
                print(f"   Result diversity: {response.result_diversity:.1%}")
                print(f"   Consensus score: {response.consensus_score:.1%}")
                
                if response.metadata:
                    print(f"   Query type: {response.metadata.get('query_type', 'unknown')}")
                    print(f"   Total documents: {response.metadata.get('total_documents_considered', 0)}")
                    
        except Exception as e:
            print_error(f"❌ Unified RAG failed: {str(e)}")
            if self.analysis_verbose:
                import traceback
                traceback.print_exc()
    
    def do_urag_status(self, arg):
        """Show unified RAG system status and metrics"""
        try:
            if not hasattr(self, 'unified_rag'):
                print_info("🔄 Unified RAG engine not initialized")
                print("   Run a /urag query first to initialize")
                return
            
            metrics = self.unified_rag.metrics
            
            print("🔄 Unified RAG System Status")
            print("=" * 60)
            
            print(f"\n📊 Overall Statistics:")
            print(f"   Queries processed: {metrics['queries_processed']}")
            print(f"   Average processing time: {metrics['avg_processing_time']:.2f}s")
            
            if metrics['method_usage_stats']:
                print(f"\n🎯 Method Usage Statistics:")
                for method, stats in metrics['method_usage_stats'].items():
                    print(f"   {method}:")
                    print(f"     Used: {stats['count']} times")
                    print(f"     Avg time: {stats['avg_time']:.2f}s")
            
            if metrics['strategy_effectiveness']:
                print(f"\n⚙️  Strategy Effectiveness:")
                for strategy, stats in metrics['strategy_effectiveness'].items():
                    print(f"   {strategy}:")
                    print(f"     Used: {stats['count']} times")
                    print(f"     Avg time: {stats['avg_time']:.2f}s")
                    print(f"     Avg methods: {stats['avg_methods']:.1f}")
            
            # Performance recommendations
            print(f"\n💡 Performance Insights:")
            if metrics['queries_processed'] > 5:
                fastest_method = min(metrics['method_usage_stats'].items(), 
                                   key=lambda x: x[1]['avg_time'])
                print(f"   Fastest method: {fastest_method[0]} ({fastest_method[1]['avg_time']:.2f}s)")
                
                most_used = max(metrics['method_usage_stats'].items(),
                              key=lambda x: x[1]['count'])
                print(f"   Most used method: {most_used[0]} ({most_used[1]['count']} times)")
            else:
                print("   Run more queries for detailed insights")
                
        except Exception as e:
            print_error(f"❌ Failed to get unified RAG status: {str(e)}")
    
    def do_urag_config(self, arg):
        """Configure unified RAG system settings
        Usage: /urag_config [show|set <key> <value>]
        
        Available settings:
            expertise: beginner, intermediate, expert
            default_strategy: fast, balanced, comprehensive, adaptive
            default_priority: speed, accuracy, balance
            timeout: default timeout in seconds
        """
        if not arg:
            arg = "show"
        
        try:
            parts = arg.split()
            command = parts[0]
            
            if command == "show":
                print("🔄 Unified RAG Configuration")
                print("=" * 50)
                
                config = getattr(self, 'urag_config', {
                    'expertise': 'intermediate',
                    'default_strategy': 'adaptive',
                    'default_priority': 'balance',
                    'timeout': None
                })
                
                for key, value in config.items():
                    print(f"   {key}: {value}")
                
            elif command == "set" and len(parts) >= 3:
                key = parts[1]
                value = parts[2]
                
                if not hasattr(self, 'urag_config'):
                    self.urag_config = {}
                
                # Validate settings
                valid_settings = {
                    'expertise': ['beginner', 'intermediate', 'expert'],
                    'default_strategy': ['fast', 'balanced', 'comprehensive', 'adaptive'],
                    'default_priority': ['speed', 'accuracy', 'balance'],
                    'timeout': 'numeric'
                }
                
                if key in valid_settings:
                    if valid_settings[key] == 'numeric':
                        try:
                            value = float(value)
                        except ValueError:
                            print_error(f"❌ {key} must be a number")
                            return
                    elif value not in valid_settings[key]:
                        print_error(f"❌ Invalid value for {key}. Valid: {valid_settings[key]}")
                        return
                    
                    self.urag_config[key] = value
                    print_success(f"✅ Set {key} = {value}")
                else:
                    print_error(f"❌ Unknown setting: {key}")
            else:
                print("Usage: /urag_config [show|set <key> <value>]")
                
        except Exception as e:
            print_error(f"❌ Configuration failed: {str(e)}")
    
    def do_urag_test(self, arg):
        """Test unified RAG with different strategies
        Usage: /urag_test <query>
        
        Runs the same query with all strategies and compares results
        """
        if not arg:
            print("Usage: /urag_test <query>")
            return
        
        if not self.rag_enabled or not self.vector_store:
            print_error("❌ Vector database not loaded")
            return
        
        try:
            print(f"🧪 Testing Unified RAG Strategies")
            print(f"Query: '{arg}'")
            print("=" * 60)
            
            # Initialize unified RAG if needed
            if not hasattr(self, 'unified_rag'):
                from src.rag.unified_rag_engine import UnifiedRAGEngine
                self.unified_rag = UnifiedRAGEngine(
                    self.vector_store,
                    self.model_manager
                )
            
            from src.rag.unified_rag_engine import UnifiedRAGQuery, ProcessingStrategy
            
            strategies = [
                ("Fast Only", ProcessingStrategy.FAST_ONLY),
                ("Balanced", ProcessingStrategy.BALANCED),
                ("Comprehensive", ProcessingStrategy.COMPREHENSIVE),
                ("Adaptive", ProcessingStrategy.ADAPTIVE)
            ]
            
            results = {}
            
            for strategy_name, strategy in strategies:
                print(f"\n🔄 Testing {strategy_name}...")
                
                query = UnifiedRAGQuery(
                    query=arg,
                    strategy=strategy,
                    priority="balance"
                )
                
                import asyncio
                response = asyncio.run(self.unified_rag.query(query))
                
                results[strategy_name] = response
                
                print(f"   Time: {response.total_processing_time:.2f}s")
                print(f"   Confidence: {response.confidence:.1%}")
                print(f"   Methods: {', '.join(response.methods_used)}")
                print(f"   Sources: {len(response.sources)}")
            
            # Comparison summary
            print(f"\n📊 Strategy Comparison:")
            print(f"{'Strategy':<15} {'Time':<8} {'Confidence':<12} {'Methods':<10} {'Sources':<8}")
            print("-" * 55)
            
            for strategy_name, response in results.items():
                print(f"{strategy_name:<15} {response.total_processing_time:.2f}s{'':<3} "
                      f"{response.confidence:.1%}{'':<6} "
                      f"{len(response.methods_used):<10} "
                      f"{len(response.sources):<8}")
            
            # Best strategy recommendation
            best_balanced = max(results.items(), 
                              key=lambda x: x[1].confidence - x[1].total_processing_time * 0.1)
            print(f"\n🏆 Recommended: {best_balanced[0]}")
            
        except Exception as e:
            print_error(f"❌ Testing failed: {str(e)}")
            if self.analysis_verbose:
                import traceback
                traceback.print_exc()
    
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
                        
                        # Re-initialize RAG engine using the existing ModelWrapper class
                        # Update the existing model wrapper with RAG enabled
                        if hasattr(self.model_wrapper, 'rag_enabled'):
                            self.model_wrapper.rag_enabled = True
                        
                        self.rag_engine = EnhancedRAGEngine(
                            vector_store=self.vector_store,
                            model_manager=self.model_wrapper
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
    
    def do_rag_files(self, arg):
        """Show which files are indexed in each RAG database"""
        print(f"\n📂 RAG Database Files Overview")
        print("=" * 80)
        
        # Get all available vector databases
        models_info = self.vector_manager.list_models()
        available_dbs = [(name, info) for name, info in models_info.items() if info['exists']]
        
        if not available_dbs:
            print("❌ No RAG databases found")
            print("   Run 'pcie-debug vectordb build' to create a database")
            return
        
        # Check knowledge base source directory
        kb_path = Path("data/knowledge_base")
        source_files = set()
        if kb_path.exists():
            source_files = {f.name for f in kb_path.glob("*.md")}
            print(f"\n📁 Source Knowledge Base Directory: {kb_path}")
            print(f"   Total source files available: {len(source_files)}")
            if source_files:
                print("   Files:")
                for f in sorted(source_files):
                    print(f"     • {f}")
        
        print(f"\n🗄️ Vector Databases Status ({len(available_dbs)} databases):")
        print("-" * 80)
        
        # For each database, show indexed files
        for db_name, db_info in available_dbs:
            print(f"\n📊 Database: {db_name}")
            print(f"   Path: {Path(db_info['path']).name}")
            
            # Get statistics for this database
            stats = self.vector_manager.get_stats(db_name)
            if stats:
                print(f"   Vectors: {stats['total_vectors']:,}")
                print(f"   Dimension: {stats['dimension']}D")
            
            # Load this database temporarily to check files
            try:
                temp_store = self.vector_manager.load(db_name)
                if temp_store and hasattr(temp_store, 'metadata'):
                    # Extract unique files from metadata
                    indexed_files = {}
                    for meta in temp_store.metadata:
                        source = meta.get('source', 'unknown')
                        filename = Path(source).name if source != 'unknown' else 'unknown'
                        if filename != 'unknown':
                            indexed_files[filename] = indexed_files.get(filename, 0) + 1
                    
                    print(f"   Indexed files: {len(indexed_files)}")
                    if indexed_files:
                        for filename, chunks in sorted(indexed_files.items()):
                            # Check if file still exists in source
                            status = "✅" if filename in source_files else "⚠️"
                            print(f"     {status} {filename} ({chunks} chunks)")
                    
                    # Check for missing files
                    missing_in_db = source_files - set(indexed_files.keys())
                    if missing_in_db:
                        print(f"   ⚠️  Not indexed from source: {', '.join(sorted(missing_in_db))}")
                else:
                    print("   ⚠️  Unable to read file information")
            except Exception as e:
                print(f"   ❌ Error reading database: {e}")
        
        # Summary and recommendations
        print(f"\n💡 Summary:")
        current_model = self.embedding_selector.get_current_model()
        print(f"   Current embedding model: {current_model}")
        
        current_db_exists = current_model in [name for name, _ in available_dbs]
        if current_db_exists:
            print(f"   ✅ Database exists for current model")
        else:
            print(f"   ❌ No database for current model")
            print(f"   💡 Run 'pcie-debug vectordb build' to create it")
        
        print(f"\n📝 Legend:")
        print(f"   ✅ = File exists in both source and database")
        print(f"   ⚠️  = File in database but not in source (may be outdated)")
        
        print("\n" + "=" * 80)
    
    def do_rag_check(self, arg):
        """Quick check of current RAG database readiness"""
        current_model = self.embedding_selector.get_current_model()
        
        print(f"\n🔍 RAG Quick Check")
        print("=" * 50)
        print(f"Current Model: {current_model}")
        
        # Check if database exists
        if not self.vector_manager.exists(current_model):
            print(f"\n❌ No database for {current_model}")
            print("   Run 'pcie-debug vectordb build' to create it")
            return
        
        # Get stats
        stats = self.vector_manager.get_stats(current_model)
        if not stats:
            print(f"\n⚠️  Database exists but cannot read stats")
            return
        
        print(f"\n✅ Database Ready!")
        print(f"   Vectors: {stats['total_vectors']:,}")
        print(f"   Dimension: {stats['dimension']}D")
        print(f"   Size: {stats['size_mb']:.1f}MB")
        
        # Check if loaded
        if self.rag_enabled and self.vector_store:
            print(f"\n✅ RAG Status: ACTIVE")
            print(f"   Search mode: {self.rag_search_mode.upper()}")
        else:
            print(f"\n⚠️  RAG Status: NOT LOADED")
            print(f"   Run '/rag on' to enable")
        
        # Quick file check
        kb_path = Path("data/knowledge_base")
        if kb_path.exists():
            source_count = len(list(kb_path.glob("*.md")))
            print(f"\n📁 Source files: {source_count}")
            
            # Check indexed files
            try:
                store = self.vector_manager.load(current_model)
                if store and hasattr(store, 'metadata'):
                    indexed_files = set()
                    for meta in store.metadata:
                        source = meta.get('source', 'unknown')
                        filename = Path(source).name if source != 'unknown' else 'unknown'
                        if filename != 'unknown':
                            indexed_files.add(filename)
                    
                    print(f"📂 Indexed files: {len(indexed_files)}")
                    
                    if len(indexed_files) < source_count:
                        print(f"   ⚠️  {source_count - len(indexed_files)} files not indexed")
                        print(f"   Run 'pcie-debug vectordb build --force' to rebuild")
                    else:
                        print(f"   ✅ All source files indexed")
            except:
                pass
        
        print("=" * 50)
    
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
            print("  unified   - Adaptive multi-strategy search (BEST)")
            print("             • Intelligently combines all methods")
            print("             • Adaptive query routing")
            print("             • Best overall performance")
            print()
            print("  pcie      - PCIe-optimized adaptive chunking")
            print("             • 1000-word chunks with semantic boundaries")
            print("             • PCIe concept extraction and boosting")
            print("             • Technical level filtering")
            print("             • Optimized for PCIe specifications")
            print()
            print("Usage: /rag_mode <semantic|hybrid|keyword|unified|pcie>")
            return
        
        mode = arg.lower()
        valid_modes = ['semantic', 'hybrid', 'keyword', 'unified', 'pcie']
        
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
        
        # Initialize PCIe mode if selected
        if mode == 'pcie':
            self._initialize_pcie_mode()
        
        # Initialize hybrid search engine if needed
        if mode in ['hybrid', 'keyword'] and (not hasattr(self, 'hybrid_engine') or self.hybrid_engine is None):
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
            if hasattr(self, 'hybrid_engine') and self.hybrid_engine is not None:
                try:
                    stats = self.hybrid_engine.get_statistics()
                    print_info(f"   📚 BM25 vocabulary: {stats['bm25_vocabulary_size']:,} unique terms")
                    print_info(f"   📈 Average doc length: {stats['average_doc_length']:.1f} tokens")
                except Exception as e:
                    print_info(f"   ℹ️  Statistics will be available after first search")
        elif mode == 'keyword':
            print_info("   🔍 Using pure BM25 keyword search")
            print_info("   📝 Best for exact term and error code searches")
        elif mode == 'unified':
            print_info("   🚀 Using Unified Adaptive RAG (BEST)")
            print_info("   🧠 Intelligently combines semantic, keyword, and metadata search")
            print_info("   🎯 Adaptive query routing for optimal results")
            print_info("   ⚡ Best overall performance and accuracy")
        elif mode == 'pcie':
            print_info("   🔧 Using PCIe-optimized adaptive chunking")
            print_info("   📏 1000-word chunks with semantic boundaries")
            print_info("   🧮 PCIe concept extraction and boosting")
            print_info("   🎯 Technical level filtering for precision")
            print_info("   ⚡ Optimized for PCIe specification queries")
    
    def do_json_query(self, arg):
        """Execute a structured JSON query in current RAG mode"""
        if not arg.strip():
            print("Usage: /json_query <your question>")
            print("Example: /json_query What are PCIe LTSSM states?")
            return
        
        if not self.rag_enabled:
            print_error("❌ RAG not enabled - no vector database available")
            print_info("   Use '/rag_model <model>' to enable RAG")
            return
            
        try:
            if self.rag_search_mode == 'pcie' and hasattr(self, 'pcie_rag_engine') and self.pcie_rag_engine:
                # Use PCIe structured output
                results = self.pcie_rag_engine.query(
                    query=arg.strip(),
                    return_structured=True
                )
                
                # Output JSON
                import json
                output = results.to_json(indent=2)
                print(output)
                
            else:
                print_error("❌ Structured output only available in PCIe mode")
                print_info("   Switch to PCIe mode: /rag_mode pcie")
                
        except Exception as e:
            print_error(f"❌ Error executing structured query: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def _initialize_pcie_mode(self):
        """Initialize PCIe RAG mode"""
        try:
            if not hasattr(self, 'pcie_rag_engine') or self.pcie_rag_engine is None:
                from src.rag.pcie_rag_engine import PCIeRAGEngine
                
                current_model = self.embedding_selector.get_current_model()
                print_info("🔄 Initializing PCIe adaptive RAG engine...")
                
                self.pcie_rag_engine = PCIeRAGEngine(
                    embedding_model=current_model,
                    chunk_config={
                        'target_size': 1000,
                        'max_size': 1500,
                        'min_size': 200,
                        'overlap_size': 200
                    }
                )
                
                # Check if knowledge base exists
                if not self.pcie_rag_engine._is_knowledge_base_current():
                    print_info("📚 Building PCIe knowledge base (this may take a moment)...")
                    kb_success = self.pcie_rag_engine.build_knowledge_base(
                        knowledge_base_path="data/knowledge_base",
                        force_rebuild=False
                    )
                    
                    if not kb_success:
                        print_error("❌ Failed to build PCIe knowledge base")
                        print_info("   Run 'pcie-debug pcie build --force' to rebuild")
                        return False
                
                # Get stats
                stats = self.pcie_rag_engine.get_pcie_mode_stats()
                print_success("✅ PCIe adaptive RAG engine ready!")
                total_vectors = stats.get('total_vectors', 'unknown')
                if isinstance(total_vectors, int):
                    print_info(f"   📊 Vectors: {total_vectors:,}")
                else:
                    print_info(f"   📊 Vectors: {total_vectors}")
                print_info(f"   🧮 Chunking: {stats.get('chunking_strategy', 'adaptive')}")
                print_info(f"   🎯 Concept boosting: {'enabled' if stats.get('concept_boosting_enabled') else 'disabled'}")
                
                return True
        except Exception as e:
            print_error(f"❌ Failed to initialize PCIe mode: {str(e)}")
            return False
    
    def _is_pcie_query(self, query: str) -> bool:
        """Detect if query is PCIe-related"""
        pcie_keywords = [
            'pcie', 'ltssm', 'tlp', 'flr', 'aer', 'aspm', 'completion timeout',
            'malformed tlp', 'link training', 'power management', 'hot reset',
            'function level reset', 'advanced error reporting', 'data link layer',
            'transaction layer', 'physical layer', 'system architecture',
            'pci express', 'endpoint', 'root complex', 'switch', 'bridge',
            'config space', 'bar', 'capability', 'extended capability',
            'ecrc', 'lcrc', 'dllp', 'ordered set', 'equalization',
            'signal integrity', 'eye diagram', 'compliance pattern'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in pcie_keywords)
    
    def _suggest_pcie_mode(self, query: str):
        """Suggest switching to PCIe mode for PCIe queries"""
        print_info("🔧 Detected PCIe-related query!")
        print_info("   💡 Consider switching to PCIe mode for better results:")
        print_info("   👉 Type '/rag_mode pcie' for optimized PCIe answers")
        print("")
    
    def _perform_search_with_mode(self, rag_query, query_embedding=None):
        """Perform search using the selected RAG mode"""
        try:
            if self.rag_search_mode == 'semantic':
                # Use standard RAG engine (semantic search)
                return self.rag_engine.query(rag_query)
            
            elif self.rag_search_mode == 'unified':
                # Use unified RAG engine (adaptive multi-strategy)
                if not hasattr(self, 'unified_rag') or self.unified_rag is None:
                    # Initialize unified RAG engine if not already done
                    from src.rag.unified_rag_engine import UnifiedRAGEngine
                    self.unified_rag = UnifiedRAGEngine(
                        vector_store=self.vector_store,
                        model_manager=self.model_manager
                    )
                
                # Convert to UnifiedRAGQuery
                from src.rag.unified_rag_engine import UnifiedRAGQuery
                unified_query = UnifiedRAGQuery(
                    query=rag_query.query,
                    strategy="adaptive",  # Let it choose the best strategy
                    priority="balance"    # Balance speed and quality
                )
                
                # Execute unified query
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    unified_response = loop.run_until_complete(self.unified_rag.query(unified_query))
                finally:
                    loop.close()
                
                # Convert unified response to standard RAG result format
                from src.rag.enhanced_rag_engine import RAGResponse
                rag_result = RAGResponse(
                    answer=unified_response.answer,
                    sources=unified_response.sources[:rag_query.context_window],
                    confidence=unified_response.confidence,
                    metadata=unified_response.metadata
                )
                
                return rag_result
            
            elif self.rag_search_mode == 'pcie':
                # Use PCIe RAG engine
                if not hasattr(self, 'pcie_rag_engine') or self.pcie_rag_engine is None:
                    # Initialize PCIe RAG engine
                    from src.rag.pcie_rag_engine import PCIeRAGEngine
                    
                    current_model = self.embedding_selector.get_current_model()
                    print("🔄 Initializing PCIe adaptive RAG engine...")
                    
                    self.pcie_rag_engine = PCIeRAGEngine(
                        embedding_model=current_model,
                        chunk_config={
                            'target_size': 1000,
                            'max_size': 1500,
                            'min_size': 200,
                            'overlap_size': 200
                        }
                    )
                    
                    # Build knowledge base if needed
                    kb_success = self.pcie_rag_engine.build_knowledge_base(
                        knowledge_base_path="data/knowledge_base",
                        force_rebuild=False
                    )
                    
                    if not kb_success:
                        print_error("❌ Failed to build PCIe knowledge base")
                        print_info("   Falling back to semantic search")
                        self.rag_search_mode = 'semantic'
                        return self.rag_engine.query(rag_query)
                    
                    print_success("✅ PCIe adaptive RAG engine initialized")
                
                # Execute PCIe query
                pcie_results = self.pcie_rag_engine.query(
                    query=rag_query.query,
                    top_k=rag_query.context_window
                )
                
                # Convert PCIe results to standard RAG result format
                from src.rag.enhanced_rag_engine import RAGResponse
                
                if pcie_results:
                    # Convert first result to main answer
                    best_result = pcie_results[0]
                    
                    # Build sources from all results
                    sources = []
                    for result in pcie_results:
                        source = {
                            'content': result.content,
                            'metadata': result.metadata,
                            'score': result.score,
                            'pcie_layer': result.metadata.get('pcie_layer', 'general'),
                            'technical_level': result.technical_level,
                            'semantic_type': result.semantic_type,
                            'section': result.source_section
                        }
                        sources.append(source)
                    
                    rag_result = RAGResponse(
                        answer=best_result.content,  # Use best match as answer
                        sources=sources,
                        confidence=best_result.score,
                        metadata={
                            'mode': 'pcie_adaptive',
                            'chunking_strategy': 'adaptive',
                            'total_results': len(pcie_results)
                        }
                    )
                    
                    return rag_result
                else:
                    # No results found
                    return RAGResponse(
                        answer="No relevant PCIe information found.",
                        sources=[],
                        confidence=0.0,
                        metadata={'mode': 'pcie_adaptive', 'error': 'no_results'}
                    )
            
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
                    f"Source {i+1}:\n{(result['content'] if isinstance(result, dict) else result.content)[:500]}..."
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
                # Sources are dictionaries with 'score' key, not objects with combined_score attribute
                avg_score = sum(s['score'] if isinstance(s, dict) else s.combined_score for s in sources) / len(sources)
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
    
    def _show_command_suggestions(self):
        """Show available commands in a nice formatted display"""
        commands = self._get_available_commands()
        
        print("\n" + "╭" + "─" * 168 + "╮")
        print("│ > /                                                                                                                                                                        │")
        print("╰" + "─" * 168 + "╯")
        
        for cmd_name, description in commands:
            print(f"  /{cmd_name:<15} {description}")
        
        print()
    
    def _get_available_commands(self):
        """Get list of available commands with descriptions"""
        commands = []
        
        # Get all methods that start with 'do_'
        for attr_name in dir(self):
            if attr_name.startswith('do_') and not attr_name.startswith('do_EOF'):
                cmd_name = attr_name[3:]  # Remove 'do_' prefix
                method = getattr(self, attr_name)
                
                # Get docstring as description
                if hasattr(method, '__doc__') and method.__doc__:
                    description = method.__doc__.strip()
                else:
                    description = f"Execute {cmd_name} command"
                
                commands.append((cmd_name, description))
        
        # Sort commands alphabetically
        commands.sort(key=lambda x: x[0])
        
        return commands
    
    def _find_matching_commands(self, partial_cmd):
        """Find commands that start with the partial command"""
        available_commands = self._get_available_commands()
        matches = []
        
        for cmd_name, _ in available_commands:
            if cmd_name.startswith(partial_cmd.lower()):
                matches.append(cmd_name)
        
        return matches
    
    def do_cost(self, arg):
        """Show the total cost and duration of the current session"""
        session_duration = time.time() - self.session_start_time
        hours = int(session_duration // 3600)
        minutes = int((session_duration % 3600) // 60)
        seconds = int(session_duration % 60)
        
        # Format duration
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        
        current_model = self.model_selector.get_current_model()
        input_tokens = self.session_tokens["input"]
        output_tokens = self.session_tokens["output"]
        total_tokens = input_tokens + output_tokens
        
        # Cost calculation (OpenAI pricing as example)
        cost_info = self._calculate_session_cost(current_model, input_tokens, output_tokens)
        
        print(f"""
💰 Session Cost & Duration Analysis
{'='*60}

⏱️  Session Duration: {duration_str}
🕐 Started: {datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S')}

🤖 Model: {current_model}
📊 Token Usage:
   Input:  {self.token_counter.format_token_count(input_tokens)}
   Output: {self.token_counter.format_token_count(output_tokens)}
   Total:  {self.token_counter.format_token_count(total_tokens)}

💵 Estimated Cost:
   Input:  ${cost_info['input_cost']:.6f}
   Output: ${cost_info['output_cost']:.6f}
   Total:  ${cost_info['total_cost']:.6f}

📈 Session Stats:
   Conversations: {len(self.conversation_history) // 2} interactions
   Turns Used: {self.turn_count}/{self.max_turns}
   RAG Mode: {self.rag_search_mode.upper()}
   Streaming: {'ON' if self.streaming_enabled else 'OFF'}

💡 Cost Breakdown:
   • {cost_info['provider']} pricing
   • Input: ${cost_info['input_rate']:.3f}/1K tokens
   • Output: ${cost_info['output_rate']:.3f}/1K tokens
   
{'='*60}
        """)
    
    def _calculate_session_cost(self, model_name, input_tokens, output_tokens):
        """Calculate estimated cost for the session"""
        # Cost per 1K tokens (USD)
        pricing = {
            "gpt-4o-mini": {
                "provider": "OpenAI",
                "input_rate": 0.000075,  # $0.075/1M tokens
                "output_rate": 0.0003   # $0.300/1M tokens
            },
            "gpt-4o": {
                "provider": "OpenAI", 
                "input_rate": 0.0025,   # $2.50/1M tokens
                "output_rate": 0.01     # $10.00/1M tokens
            },
            "gpt-4": {
                "provider": "OpenAI",
                "input_rate": 0.01,     # $10.00/1M tokens
                "output_rate": 0.03     # $30.00/1M tokens
            },
            "claude-3-opus": {
                "provider": "Anthropic",
                "input_rate": 0.015,    # $15.00/1M tokens
                "output_rate": 0.075    # $75.00/1M tokens
            }
        }
        
        # Default for local/free models
        default_pricing = {
            "provider": "Local/Free",
            "input_rate": 0.0,
            "output_rate": 0.0
        }
        
        model_pricing = pricing.get(model_name, default_pricing)
        
        # Calculate costs (rates are per 1K tokens)
        input_cost = (input_tokens / 1000) * model_pricing["input_rate"]
        output_cost = (output_tokens / 1000) * model_pricing["output_rate"]
        total_cost = input_cost + output_cost
        
        return {
            "provider": model_pricing["provider"],
            "input_rate": model_pricing["input_rate"],
            "output_rate": model_pricing["output_rate"],
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def do_doctor(self, arg):
        """Checks the health of your PCIe Debug Agent installation"""
        print("\n🏥 PCIe Debug Agent Health Check")
        print("=" * 60)
        
        health_status = {"total": 0, "passed": 0, "warnings": 0, "errors": 0}
        
        # Check Python environment
        print("\n📦 Python Environment:")
        self._check_python_version(health_status)
        self._check_dependencies(health_status)
        
        # Check core components
        print("\n🔧 Core Components:")
        self._check_model_providers(health_status)
        self._check_embedding_providers(health_status)
        
        # Check RAG system
        print("\n📚 RAG System:")
        self._check_vector_database(health_status)
        self._check_knowledge_base(health_status)
        
        # Check file system
        print("\n📁 File System:")
        self._check_directories(health_status)
        self._check_permissions(health_status)
        
        # Check memory and performance
        print("\n⚡ Performance:")
        self._check_memory_usage(health_status)
        self._check_disk_space(health_status)
        
        # Summary
        print(f"\n📊 Health Check Summary:")
        print(f"   Total Checks: {health_status['total']}")
        print(f"   ✅ Passed: {health_status['passed']}")
        print(f"   ⚠️  Warnings: {health_status['warnings']}")
        print(f"   ❌ Errors: {health_status['errors']}")
        
        overall_health = "HEALTHY" if health_status['errors'] == 0 else "NEEDS ATTENTION"
        status_emoji = "✅" if health_status['errors'] == 0 else "⚠️"
        
        print(f"\n{status_emoji} Overall Status: {overall_health}")
        print("=" * 60)
    
    def _check_python_version(self, health_status):
        """Check Python version compatibility"""
        health_status['total'] += 1
        try:
            import sys
            version = sys.version_info
            if version >= (3, 8):
                print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (supported)")
                health_status['passed'] += 1
            else:
                print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
                health_status['errors'] += 1
        except Exception as e:
            print(f"   ❌ Could not check Python version: {e}")
            health_status['errors'] += 1
    
    def _check_dependencies(self, health_status):
        """Check critical dependencies"""
        critical_deps = [
            ('numpy', 'numpy'),
            ('torch', 'torch'),
            ('sentence_transformers', 'sentence-transformers'),
            ('faiss', 'faiss-cpu'),
            ('rank_bm25', 'rank-bm25')
        ]
        
        for module_name, package_name in critical_deps:
            health_status['total'] += 1
            try:
                __import__(module_name)
                print(f"   ✅ {package_name}")
                health_status['passed'] += 1
            except ImportError:
                print(f"   ❌ {package_name} (not installed)")
                health_status['errors'] += 1
    
    def _check_model_providers(self, health_status):
        """Check model provider availability"""
        health_status['total'] += 1
        try:
            current_model = self.model_selector.get_current_model()
            print(f"   ✅ Current model: {current_model}")
            health_status['passed'] += 1
        except Exception as e:
            print(f"   ❌ Model selector error: {e}")
            health_status['errors'] += 1
    
    def _check_embedding_providers(self, health_status):
        """Check embedding provider availability"""
        health_status['total'] += 1
        try:
            current_embedding = self.embedding_selector.get_current_model()
            embedding_info = self.embedding_selector.get_model_info()
            print(f"   ✅ Embedding model: {current_embedding} ({embedding_info.get('provider', 'unknown')})")
            health_status['passed'] += 1
        except Exception as e:
            print(f"   ❌ Embedding selector error: {e}")
            health_status['errors'] += 1
    
    def _check_vector_database(self, health_status):
        """Check vector database status"""
        health_status['total'] += 1
        if self.vector_store:
            doc_count = self.vector_store.index.ntotal
            print(f"   ✅ Vector database loaded ({doc_count:,} documents)")
            health_status['passed'] += 1
        else:
            print(f"   ⚠️  Vector database not loaded (RAG disabled)")
            health_status['warnings'] += 1
    
    def _check_knowledge_base(self, health_status):
        """Check knowledge base files"""
        health_status['total'] += 1
        try:
            from pathlib import Path
            kb_path = Path("data/knowledge_base")
            if kb_path.exists():
                files = list(kb_path.glob("*.md"))
                print(f"   ✅ Knowledge base ({len(files)} files)")
                health_status['passed'] += 1
            else:
                print(f"   ⚠️  Knowledge base directory not found")
                health_status['warnings'] += 1
        except Exception as e:
            print(f"   ❌ Knowledge base check failed: {e}")
            health_status['errors'] += 1
    
    def _check_directories(self, health_status):
        """Check required directories"""
        required_dirs = ["data", "logs", "models"]
        
        for dir_name in required_dirs:
            health_status['total'] += 1
            try:
                from pathlib import Path
                dir_path = Path(dir_name)
                if dir_path.exists():
                    print(f"   ✅ {dir_name}/ directory")
                    health_status['passed'] += 1
                else:
                    print(f"   ⚠️  {dir_name}/ directory missing")
                    health_status['warnings'] += 1
            except Exception as e:
                print(f"   ❌ Could not check {dir_name}/ directory: {e}")
                health_status['errors'] += 1
    
    def _check_permissions(self, health_status):
        """Check file permissions"""
        health_status['total'] += 1
        try:
            from pathlib import Path
            import tempfile
            
            # Test write permissions
            test_file = Path("data") / ".health_check_test"
            test_file.touch()
            test_file.unlink()
            print(f"   ✅ File permissions (read/write)")
            health_status['passed'] += 1
        except Exception as e:
            print(f"   ❌ File permission error: {e}")
            health_status['errors'] += 1
    
    def _check_memory_usage(self, health_status):
        """Check memory usage"""
        health_status['total'] += 1
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024**3)
            total_gb = memory.total / (1024**3)
            percent = memory.percent
            
            if percent < 80:
                print(f"   ✅ Memory usage: {used_gb:.1f}GB/{total_gb:.1f}GB ({percent:.1f}%)")
                health_status['passed'] += 1
            else:
                print(f"   ⚠️  High memory usage: {used_gb:.1f}GB/{total_gb:.1f}GB ({percent:.1f}%)")
                health_status['warnings'] += 1
        except ImportError:
            print(f"   ⚠️  Memory check unavailable (psutil not installed)")
            health_status['warnings'] += 1
        except Exception as e:
            print(f"   ❌ Memory check failed: {e}")
            health_status['errors'] += 1
    
    def _check_disk_space(self, health_status):
        """Check available disk space"""
        health_status['total'] += 1
        try:
            import shutil
            from pathlib import Path
            
            total, used, free = shutil.disk_usage(Path.cwd())
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            percent_free = (free / total) * 100
            
            if percent_free > 10:  # More than 10% free
                print(f"   ✅ Disk space: {free_gb:.1f}GB free ({percent_free:.1f}%)")
                health_status['passed'] += 1
            else:
                print(f"   ⚠️  Low disk space: {free_gb:.1f}GB free ({percent_free:.1f}%)")
                health_status['warnings'] += 1
        except Exception as e:
            print(f"   ❌ Disk space check failed: {e}")
            health_status['errors'] += 1
    
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
        """Process default queries using Unified RAG"""
        if line.strip().startswith('/'):
            # Handle slash commands
            super().default(line)
            return
        
        # Check if this looks like a PCIe query and suggest PCIe mode
        if self._is_pcie_query(line) and getattr(self, 'rag_search_mode', 'semantic') != 'pcie':
            self._suggest_pcie_mode(line)
        
        # Use Unified RAG for regular queries
        if hasattr(self, 'unified_rag') and self.unified_rag:
            self._process_unified_rag_query(line)
        else:
            self._process_regular_query(line)
    
    def _process_unified_rag_query(self, query):
        """Process query using Unified RAG"""
        try:
            from src.rag.unified_rag_engine import UnifiedRAGQuery
            
            # Create unified query
            unified_query = UnifiedRAGQuery(
                query=query,
                strategy=self.rag_search_mode if hasattr(self, 'rag_search_mode') else "adaptive",
                priority="balance",
                user_expertise=getattr(self, 'user_expertise', 'intermediate')
            )
            
            # Execute query
            import asyncio
            response = asyncio.run(self.unified_rag.query(unified_query))
            
            # Display result
            print(f"\n💡 Answer (Confidence: {response.confidence:.1%}):")
            print(response.answer)
            
            if response.sources and self.analysis_verbose:
                print(f"\n📚 Sources ({len(response.sources)}):")
                for i, source in enumerate(response.sources[:3], 1):
                    print(f"  {i}. Score: {source.get('final_score', 0):.3f}")
                    if 'metadata' in source and source['metadata'].get('filename'):
                        print(f"     File: {source['metadata']['filename']}")
                
                print(f"\n⚡ Performance:")
                print(f"   Methods: {', '.join(response.methods_used)}")
                print(f"   Time: {response.total_processing_time:.2f}s")
            
        except Exception as e:
            print_error(f"❌ Unified RAG query failed: {e}")
            self._process_regular_query(query)
    
    def _process_regular_query(self, query):
        """Fallback to regular query processing"""
        """Handle non-command input as PCIe questions"""
        if not query.strip():
            return
        
        # Handle slash commands
        if query.startswith('/'):
            # Show command suggestions if just "/" is typed
            if query.strip() == '/':
                self._show_command_suggestions()
                return
            
            # Convert slash command to method call
            parts = query[1:].split(None, 1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            
            # Show partial command matches if command not found
            if not hasattr(self, f'do_{cmd}'):
                matching_commands = self._find_matching_commands(cmd)
                if matching_commands:
                    print(f"❌ Unknown command: /{cmd}")
                    print("📝 Did you mean:")
                    for match in matching_commands[:5]:  # Show top 5 matches
                        print(f"   /{match}")
                else:
                    print(f"❌ Unknown command: /{cmd}")
                    print("   Type / for available commands or /help for detailed help")
                return
            
            # Execute the command
            return getattr(self, f'do_{cmd}')(arg)
        
        # Check turn limit
        if self.turn_count >= self.max_turns:
            print_warning(f"⚠️ Maximum turns ({self.max_turns}) reached. Use /clear to reset.")
            return
        
        # Process as PCIe debugging query
        self._process_query(query)
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
                    
                    # Note: Embedding model is cached after first use for performance
                
                # Use RAG with selected search mode
                from src.rag.enhanced_rag_engine import RAGQuery
                rag_query = RAGQuery(query=query, context_window=5)
                
                if self.analysis_verbose:
                    # Show embedding details
                    current_embedding_provider = self.embedding_selector.get_current_provider()
                    
                    # Check if provider is cached
                    provider_info = current_embedding_provider.get_info()
                    model_name = provider_info.get('model', 'unknown')
                    
                    # Generate embedding
                    query_embedding = current_embedding_provider.encode([query])[0]
                    embedding_time = time.time() - embedding_start
                    
                    current_embedding_model = self.embedding_selector.get_current_model()
                    print(f"     ✓ Embedding model: {current_embedding_model}")
                    print(f"     ✓ Embedding dimension: {len(query_embedding)}")
                    print(f"     ✓ Embedding time: {embedding_time:.3f}s")
                    if embedding_time < 0.1:
                        print(f"     ✓ Model cached: Fast response")
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
    
    def onecmd(self, line):
        """Override to handle slash commands properly"""
        line = line.strip()
        if not line:
            return False
        
        # Handle slash commands
        if line.startswith('/'):
            # Show command suggestions if just "/" is typed
            if line == '/':
                self._show_command_suggestions()
                return False
            
            # Convert slash command to method call
            parts = line[1:].split(None, 1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else ''
            
            # Check if command exists
            if hasattr(self, f'do_{cmd}'):
                # Execute the command
                try:
                    return getattr(self, f'do_{cmd}')(arg)
                except Exception as e:
                    print_error(f"❌ Error executing /{cmd}: {e}")
                    if self.verbose:
                        import traceback
                        traceback.print_exc()
                    return False
            else:
                # Show partial command matches if command not found
                matching_commands = self._find_matching_commands(cmd)
                if matching_commands:
                    print(f"❌ Unknown command: /{cmd}")
                    print("📝 Did you mean:")
                    for match in matching_commands[:5]:  # Show top 5 matches
                        print(f"   /{match}")
                else:
                    print(f"❌ Unknown command: /{cmd}")
                    print("   Type / for available commands or /help for detailed help")
                return False
        else:
            # Process as regular query
            self.default(line)
            return False
    
    def cmdloop(self, intro=None):
        """Override cmdloop to set up completion and handle errors gracefully"""
        # Set the completer function
        readline.set_completer(self.complete)
        
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