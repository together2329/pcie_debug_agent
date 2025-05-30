"""Analyze command for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
from typing import Optional

from src.cli.utils.output import (
    console, print_error, print_success, print_info,
    print_analysis_result, create_progress_bar
)
from src.rag.enhanced_rag_engine import EnhancedRAGEngine
from src.rag.vector_store import FAISSVectorStore
from src.rag.model_manager import ModelManager


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
    "--model", "-m",
    help="LLM model to use (overrides config)"
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
    default=3,
    help="Number of context documents to use"
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable query caching"
)
@click.pass_context
def analyze(
    ctx: click.Context,
    log_path: Optional[str],
    query: str,
    model: Optional[str],
    output: str,
    confidence: float,
    context_window: int,
    no_cache: bool
):
    """Analyze PCIe logs with AI-powered insights
    
    Examples:
        pcie-debug analyze --query "What errors occurred?"
        
        pcie-debug analyze /path/to/logs --query "Find timeout issues"
        
        pcie-debug analyze -q "Explain the PCIe link training failure" --model gpt-4
    """
    settings = ctx.obj["settings"]
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # If log path provided, ensure it's indexed first
        if log_path:
            print_info(f"Analyzing logs from: {log_path}")
            # TODO: Trigger indexing if needed
        
        # Initialize components
        with create_progress_bar("Initializing RAG engine") as progress:
            task = progress.add_task("Loading models...", total=3)
            
            # Vector store
            vector_store = FAISSVectorStore(
                index_path=settings.vector_store_path,
                index_type=settings.index_type,
                dimension=settings.embedding_dimension
            )
            progress.update(task, advance=1)
            
            # Model manager
            model_manager = ModelManager(
                embedding_model=settings.embedding_model,
                embedding_provider=settings.embedding_provider,
                embedding_api_key=settings.embedding_config.api_key,
                embedding_api_base_url=settings.embedding_config.api_base_url,
                llm_provider=settings.llm_provider,
                llm_model=model or settings.llm_model,
                llm_api_key=settings.llm_config.api_key,
                llm_api_base_url=settings.llm_config.api_base_url
            )
            progress.update(task, advance=1)
            
            # RAG engine
            engine = EnhancedRAGEngine(
                vector_store=vector_store,
                model_manager=model_manager,
                llm_provider=settings.llm_provider,
                llm_model=model or settings.llm_model,
                temperature=0.1,
                max_tokens=2000,
                context_window=context_window,
                enable_cache=not no_cache
            )
            progress.update(task, advance=1)
        
        # Perform analysis
        print_info(f"Analyzing with query: '{query}'")
        
        with create_progress_bar("Performing analysis") as progress:
            task = progress.add_task("Searching and analyzing...", total=None)
            
            result = engine.query(
                query=query,
                min_confidence=confidence
            )
            
            progress.update(task, completed=True)
        
        # Check confidence
        if result.confidence < confidence:
            print_warning(
                f"Result confidence ({result.confidence:.1%}) is below "
                f"threshold ({confidence:.1%})"
            )
        
        # Output results
        if output == "json":
            import json
            console.print(json.dumps(result.to_dict(), indent=2))
        elif output == "markdown":
            # Format as markdown
            md_output = f"# Analysis Result\n\n"
            md_output += f"**Query**: {query}\n\n"
            md_output += f"## Answer\n\n{result.answer}\n\n"
            md_output += f"**Confidence**: {result.confidence:.1%}\n\n"
            
            if result.sources:
                md_output += "## Sources\n\n"
                for source in result.sources:
                    md_output += f"- {source['file']} (relevance: {source['relevance']:.2f})\n"
            
            console.print(md_output)
        else:
            # Default text output
            print_analysis_result(result.to_dict())
        
        # Show cache statistics if verbose
        if verbose and not no_cache:
            print_info(
                f"Cache stats - Hits: {engine.cache_hits}, "
                f"Misses: {engine.cache_misses}, "
                f"Hit rate: {engine.cache_hit_rate:.1%}"
            )
        
        print_success("Analysis complete!")
        
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        print_error(f"Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Analysis failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)