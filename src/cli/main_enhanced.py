"""
Enhanced PCIe Debug Agent CLI - Integrated like Claude Code
"""

import click
import sys
import os
import time
from pathlib import Path
from typing import Optional, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cli.commands import analyze, index, search, report, config, test, model
from src.cli.interactive import start_interactive_mode
from src.cli.memory import MemoryManager
from src.cli.session_manager import SessionManager
from src.cli.utils.output import console, print_banner, print_success, print_error, print_info
from src.config.settings import Settings, load_settings
from src.models.model_selector import get_model_selector


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="PCIe Debug Agent")
@click.option(
    "--print", "-p",
    "one_shot",
    help="Run one-off query and exit"
)
@click.option(
    "--model", "-m",
    help="Set AI model for session"
)
@click.option(
    "--continue", "-c",
    "continue_session",
    is_flag=True,
    help="Continue most recent conversation"
)
@click.option(
    "--resume", "-r",
    help="Resume specific session by ID"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--quiet", "-q",
    is_flag=True,
    help="Suppress non-essential output"
)
@click.option(
    "--max-turns",
    default=50,
    help="Maximum number of conversation turns"
)
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.pass_context
def cli(ctx: click.Context, one_shot: Optional[str], model: Optional[str], 
        continue_session: bool, resume: Optional[str], verbose: bool, quiet: bool,
        max_turns: int, config_path: Optional[str]):
    """PCIe Debug Agent - AI-powered PCIe log analysis tool
    
    Interactive mode (default):
        pcie-debug                    # Start interactive session
        pcie-debug -c                 # Continue last session
        pcie-debug -r <session-id>    # Resume specific session
    
    One-shot mode:
        pcie-debug -p "Why is link training failing?"
        pcie-debug --print "Analyze this error log: ..."
    
    Examples:
        pcie-debug                           # Interactive mode
        pcie-debug -p "PCIe completion timeout causes"
        pcie-debug -m gpt-4 -p "Complex PCIe analysis"
        pcie-debug -c                        # Continue last session
        pcie-debug model list                # List available models
        pcie-debug analyze logs/error.log    # Analyze specific log
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        if config_path:
            settings = load_settings(Path(config_path))
        else:
            # Try default locations
            default_config = Path("configs/settings.yaml")
            if default_config.exists():
                settings = load_settings(default_config)
            else:
                settings = load_settings(None)  # Use defaults with env vars
        
        settings.validate()
        ctx.obj["settings"] = settings
        
    except Exception as e:
        if not quiet:
            console.print(f"[red]Error loading configuration: {e}[/red]")
        # Continue with defaults
        ctx.obj["settings"] = None
    
    # Set verbosity
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    
    # Initialize logging
    import logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Suppress warnings unless verbose
    if not verbose:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["GGML_METAL_LOG_LEVEL"] = "0"
    
    # If no subcommand provided, handle special modes
    if ctx.invoked_subcommand is None:
        if one_shot:
            # One-shot mode
            handle_one_shot(one_shot, model, verbose, quiet)
        elif resume:
            # Resume specific session
            handle_resume_session(resume, model, verbose, max_turns)
        else:
            # Interactive mode (default)
            if not quiet:
                print_banner()
            start_interactive_mode(
                model_id=model,
                verbose=verbose,
                max_turns=max_turns,
                continue_session=continue_session
            )


def handle_one_shot(query: str, model_id: Optional[str], verbose: bool, quiet: bool):
    """Handle one-shot query mode"""
    try:
        if not quiet:
            print(f"🔍 Analyzing: {query[:50]}...")
        
        # Initialize components
        from src.rag.vector_store import FAISSVectorStore
        from src.rag.enhanced_rag_engine import EnhancedRAGEngine
        
        model_selector = get_model_selector()
        if model_id:
            if not model_selector.switch_model(model_id):
                print_error(f"Failed to switch to model: {model_id}")
                sys.exit(1)
        
        # Create model wrapper with embedding support
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        class ModelWrapper:
            def __init__(self, selector):
                self.selector = selector
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            def generate_completion(self, prompt: str, **kwargs) -> str:
                return self.selector.generate_completion(prompt, **kwargs)
            
            def generate_embeddings(self, texts: List[str]) -> np.ndarray:
                """Generate embeddings for texts"""
                embeddings = self.embedding_model.encode(texts)
                return np.array(embeddings)
        
        # Initialize RAG system
        vector_store = FAISSVectorStore(
            index_path="data/vectorstore",
            dimension=384
        )
        
        rag_engine = EnhancedRAGEngine(
            vector_store=vector_store,
            model_manager=ModelWrapper(model_selector)
        )
        
        if verbose:
            print(f"Using model: {model_selector.get_current_model()}")
            print(f"Vector store: {vector_store.index.ntotal} documents")
        
        # Process query
        start_time = time.time()
        from src.rag.enhanced_rag_engine import RAGQuery
        rag_query = RAGQuery(query=query, context_window=5)
        result = rag_engine.query(rag_query)
        end_time = time.time()
        
        if result and hasattr(result, 'answer'):
            print("\n" + "="*60)
            print(result.answer)
            print("="*60)
            
            if verbose:
                print(f"\nResponse time: {end_time - start_time:.1f}s")
                print(f"Confidence: {result.confidence:.1%}")
                print(f"Sources used: {len(result.sources)}")
        else:
            print_error("No analysis generated")
            sys.exit(1)
            
    except Exception as e:
        print_error(f"One-shot analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def handle_resume_session(session_id: str, model_id: Optional[str], verbose: bool, max_turns: int):
    """Handle session resumption"""
    try:
        session_manager = SessionManager()
        session = session_manager.load_session(session_id)
        
        if not session:
            print_error(f"Session not found: {session_id}")
            # Show available sessions
            sessions = session_manager.list_sessions(10)
            if sessions:
                print("\nAvailable sessions:")
                for s in sessions:
                    print(f"  {s['id'][:8]}: {s['title']} ({s['created']})")
            sys.exit(1)
        
        print_success(f"📋 Resuming session: {session['title']}")
        
        # Start interactive mode with loaded session
        shell = start_interactive_mode(
            model_id=model_id or session.get('last_model'),
            verbose=verbose,
            max_turns=max_turns,
            continue_session=False
        )
        
        # Load conversation history would be handled in the shell
        
    except Exception as e:
        print_error(f"Failed to resume session: {e}")
        sys.exit(1)


# Register traditional commands
cli.add_command(analyze.analyze)
cli.add_command(index.index)
cli.add_command(search.search)
cli.add_command(report.report)
cli.add_command(config.config)
cli.add_command(test.test)
cli.add_command(model.model)

# Add utility commands
@cli.command()
@click.option("--limit", "-l", default=10, help="Number of sessions to show")
@click.option("--tag", help="Filter by tag")
def sessions(limit: int, tag: Optional[str]):
    """List and manage conversation sessions"""
    try:
        session_manager = SessionManager()
        sessions = session_manager.list_sessions(limit=limit, tag=tag)
        
        if not sessions:
            print("📋 No sessions found")
            return
        
        print(f"\n📋 Recent Sessions ({len(sessions)}):")
        print("-" * 80)
        
        for session in sessions:
            session_id = session['id'][:8]
            title = session['title'][:50]
            created = session['created'][:16]
            turns = session.get('turn_count', 0)
            model = session.get('last_model', 'unknown')
            tags = ', '.join(session.get('tags', []))
            
            print(f"{session_id} | {title:50} | {created} | {turns:2}💬 | {model:12} | {tags}")
        
        print("-" * 80)
        print("Use 'pcie-debug -r <session-id>' to resume a session")
        
    except Exception as e:
        print_error(f"Failed to list sessions: {e}")


@cli.command()
@click.argument("key", required=False)
@click.argument("value", required=False)
@click.option("--delete", "-d", help="Delete memory entry")
@click.option("--clear", is_flag=True, help="Clear all memory")
@click.option("--list", "-l", is_flag=True, help="List all memory entries")
def memory(key: Optional[str], value: Optional[str], delete: Optional[str], 
          clear: bool, list: bool):
    """Manage persistent memory across sessions"""
    try:
        memory_manager = MemoryManager()
        
        if clear:
            memory_manager.clear_memory()
            print_success("✅ Memory cleared")
        elif delete:
            if memory_manager.delete_memory(delete):
                print_success(f"✅ Deleted memory: {delete}")
            else:
                print_error(f"❌ Memory not found: {delete}")
        elif list or (not key and not value):
            # List memory
            memory = memory_manager.list_memory()
            if memory:
                print("\n📝 Persistent Memory:")
                print("-" * 50)
                for k, entry in memory.items():
                    val = entry["value"]
                    updated = entry["updated"]
                    if len(str(val)) > 60:
                        val = str(val)[:60] + "..."
                    print(f"{k:20} | {val:60} | {updated}")
                print("-" * 50)
            else:
                print("📝 No memory entries")
        elif key and value:
            # Set memory
            memory_manager.set_memory(key, value)
            print_success(f"✅ Memory set: {key}")
        elif key:
            # Get memory
            val = memory_manager.get_memory(key)
            if val is not None:
                print(f"{key}: {val}")
            else:
                print_error(f"❌ Memory not found: {key}")
        else:
            print("Usage: pcie-debug memory [key] [value] [--list] [--delete key] [--clear]")
            
    except Exception as e:
        print_error(f"Memory operation failed: {e}")


@cli.command()
def status():
    """Show system status and statistics"""
    try:
        # Model info
        model_selector = get_model_selector()
        current_model = model_selector.get_current_model()
        
        # Vector store info
        try:
            from src.rag.vector_store import FAISSVectorStore
            vector_store = FAISSVectorStore(index_path="data/vectorstore", dimension=384)
            doc_count = vector_store.index.ntotal
        except:
            doc_count = "Unknown"
        
        # Memory info
        memory_manager = MemoryManager()
        memory_count = len(memory_manager.get_memory())
        
        # Session info
        session_manager = SessionManager()
        session_stats = session_manager.get_session_stats()
        
        print(f"""
🔧 PCIe Debug Agent Status
{'='*50}
Current Model: {current_model}
Documents in Vector Store: {doc_count:,}
Memory Entries: {memory_count}
Total Sessions: {session_stats['total_sessions']}
Recent Sessions (7 days): {session_stats['recent_sessions']}
Total Conversation Turns: {session_stats['total_turns']}
Avg Turns per Session: {session_stats['avg_turns_per_session']:.1f}

Model Usage:""")
        
        for model, count in session_stats['model_usage'].items():
            print(f"  {model}: {count} sessions")
        
        print("="*50)
        
    except Exception as e:
        print_error(f"Failed to get status: {e}")


@cli.command()
def update():
    """Check for updates (placeholder)"""
    print("🔄 Checking for updates...")
    print("💡 Update functionality not implemented yet.")
    print("   For now, update manually with 'git pull'")


def main():
    """Main entry point"""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()