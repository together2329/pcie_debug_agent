"""Analyze command with Hybrid LLM support for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
from typing import Optional

from src.cli.utils.output import (
    console, print_error, print_success, print_info, print_warning,
    print_analysis_result, create_progress_bar
)
from src.rag.enhanced_rag_engine_hybrid import EnhancedRAGEngineHybrid, RAGQuery
from src.vectorstore.faiss_store import FAISSVectorStore
from src.models.model_manager import ModelManager


@click.command()
@click.argument(
    "log_path",
    type=click.Path(exists=True),
    required=False
)
@click.option(
    "--query", "-q",
    help="Analysis query",
    prompt="Enter your analysis query"
)
@click.option(
    "--analysis-type", "-t",
    type=click.Choice(["quick", "detailed", "auto"]),
    default="auto",
    help="Analysis type: quick (Llama), detailed (DeepSeek), or auto"
)
@click.option(
    "--model", "-m",
    help="Specific model to use (overrides analysis type)"
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json", "markdown"]),
    default="text",
    help="Output format"
)
@click.option(
    "--confidence", "-c",
    type=float,
    default=0.7,
    help="Minimum confidence threshold"
)
@click.option(
    "--context-window", "-w",
    type=int,
    default=5,
    help="Number of context documents to use"
)
@click.option(
    "--max-time", "-x",
    type=float,
    default=30.0,
    help="Maximum response time in seconds"
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable query caching"
)
@click.pass_context
def analyze_hybrid(
    ctx: click.Context,
    log_path: Optional[str],
    query: str,
    analysis_type: str,
    model: Optional[str],
    output: str,
    confidence: float,
    context_window: int,
    max_time: float,
    no_cache: bool
):
    """Analyze PCIe logs with Hybrid AI-powered insights
    
    Uses intelligent model selection:
    - Quick analysis: Llama 3.2 3B for fast interactive debugging
    - Detailed analysis: DeepSeek Q4_1 for comprehensive investigation
    - Auto: Automatically selects based on query complexity
    
    Examples:
        pcie-debug analyze --query "What errors occurred?" --analysis-type quick
        
        pcie-debug analyze /path/to/logs --query "Find timeout issues" -t auto
        
        pcie-debug analyze -q "Provide detailed root cause analysis" -t detailed
        
        pcie-debug analyze -q "Link training failed" --max-time 15
    """
    settings = ctx.obj["settings"]
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # If log path provided, ensure it's indexed first
        if log_path:
            print_info(f"Analyzing logs from: {log_path}")
            # TODO: Trigger indexing if needed
        
        # Initialize components
        with create_progress_bar("Initializing Hybrid RAG engine") as progress:
            task = progress.add_task("Loading models...", total=4)
            
            # Vector store
            vector_store = FAISSVectorStore(
                index_path=str(settings.vector_store.index_path),
                index_type=settings.vector_store.index_type,
                dimension=settings.embedding.dimension
            )
            progress.update(task, advance=1, description="Vector store loaded")
            
            # Model manager
            model_manager = ModelManager()
            
            # Load embedding model
            if settings.embedding.provider == "local":
                model_manager.load_embedding_model(settings.embedding.model)
            progress.update(task, advance=1, description="Embedding model loaded")
            
            # Initialize Hybrid RAG engine
            rag_engine = EnhancedRAGEngineHybrid(
                vector_store=vector_store,
                model_manager=model_manager,
                models_dir=str(settings.local_llm.models_dir),
                enable_cache=not no_cache
            )
            progress.update(task, advance=1, description="Hybrid LLM initialized")
            
            # Check model status
            status = rag_engine.hybrid_provider.get_model_status()
            if verbose:
                print_info(f"Model status - Llama: {status['llama']['available']}, "
                          f"DeepSeek: {status['deepseek']['available']}")
            
            progress.update(task, advance=1, description="Ready for analysis")
        
        # Create RAG query
        rag_query = RAGQuery(
            query=query,
            context_window=context_window,
            min_similarity=confidence,
            analysis_type=analysis_type,
            max_response_time=max_time
        )
        
        # Show analysis type info
        if analysis_type == "quick":
            print_info("🚀 Using quick analysis mode (Llama 3.2 3B)")
        elif analysis_type == "detailed":
            print_info("🔬 Using detailed analysis mode (DeepSeek Q4_1)")
        else:
            print_info("🤖 Using auto mode (intelligent model selection)")
        
        # Execute query
        with create_progress_bar("Analyzing PCIe errors") as progress:
            task = progress.add_task("Processing query...", total=None)
            
            response = rag_engine.query(rag_query)
            
            progress.update(task, completed=True, description="Analysis complete")
        
        # Display results based on output format
        if output == "json":
            import json
            result = {
                "query": query,
                "answer": response.answer,
                "confidence": response.confidence,
                "model_used": response.model_used,
                "analysis_type": response.analysis_type,
                "response_time": response.response_time,
                "sources": response.sources,
                "metadata": response.metadata
            }
            console.print_json(data=result)
            
        elif output == "markdown":
            # Markdown format
            markdown = f"""# PCIe Error Analysis

## Query
{query}

## Analysis Type
- **Type**: {response.analysis_type}
- **Model**: {response.model_used}
- **Response Time**: {response.response_time:.2f}s
- **Confidence**: {response.confidence:.2%}

## Answer
{response.answer}

## Sources
"""
            for i, source in enumerate(response.sources, 1):
                markdown += f"\n### Source {i}\n"
                markdown += f"- **File**: {source['metadata'].get('source', 'Unknown')}\n"
                markdown += f"- **Similarity**: {source.get('similarity', 0):.2%}\n"
                markdown += f"- **Content**: {source['content'][:200]}...\n"
            
            console.print(markdown)
            
        else:
            # Text format
            print_analysis_result(
                answer=response.answer,
                confidence=response.confidence,
                sources=response.sources
            )
            
            # Show model info
            print_info(f"\n📊 Analysis Details:")
            print_info(f"  Model Used: {response.model_used}")
            print_info(f"  Analysis Type: {response.analysis_type}")
            print_info(f"  Response Time: {response.response_time:.2f}s")
            
            if response.metadata.get("fallback_used"):
                print_warning("  ⚠️  Fallback model was used")
            
            if response.metadata.get("error"):
                print_warning(f"  ⚠️  Error: {response.metadata['error']}")
        
        # Show performance metrics if verbose
        if verbose:
            metrics = rag_engine.get_performance_metrics()
            print_info(f"\n📈 Performance Metrics:")
            print_info(f"  Total Queries: {metrics['queries_processed']}")
            print_info(f"  Avg Response Time: {metrics['average_response_time']:.2f}s")
            print_info(f"  Avg Confidence: {metrics['average_confidence']:.2%}")
            print_info(f"  Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
            print_info(f"  Model Usage: Llama={metrics['model_usage']['llama']}, "
                      f"DeepSeek={metrics['model_usage']['deepseek']}")
        
        print_success("\n✅ Analysis complete!")
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)