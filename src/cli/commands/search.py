"""Search command for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
from typing import Optional, List
import json

from src.cli.utils.output import (
    console, print_error, print_info, print_warning,
    print_search_results, print_json, create_progress_bar
)
from src.rag.vector_store import FAISSVectorStore
from src.rag.model_manager import ModelManager


@click.command()
@click.argument("query")
@click.option(
    "--limit", "-l",
    type=int,
    default=10,
    help="Maximum number of results"
)
@click.option(
    "--similarity", "-s",
    type=float,
    default=0.7,
    help="Minimum similarity threshold (0-1)"
)
@click.option(
    "--filter", "-f",
    multiple=True,
    help="Filter results by metadata (key=value)"
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json", "detailed"]),
    default="text",
    help="Output format"
)
@click.option(
    "--show-embeddings",
    is_flag=True,
    help="Include embeddings in output (json only)"
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    limit: int,
    similarity: float,
    filter: List[str],
    output: str,
    show_embeddings: bool
):
    """Search indexed documents using semantic similarity
    
    Examples:
        pcie-debug search "PCIe timeout error"
        
        pcie-debug search "link training failed" --limit 5
        
        pcie-debug search "error" --filter severity=ERROR --filter source=pcie.log
        
        pcie-debug search "device enumeration" --output json > results.json
    """
    settings = ctx.obj["settings"]
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # Check if index exists
        if not Path(settings.vector_store.index_path + ".faiss").exists():
            print_error("No index found. Run 'pcie-debug index build' first.")
            sys.exit(1)
        
        # Parse filters
        metadata_filter = {}
        for f in filter:
            if '=' not in f:
                print_warning(f"Invalid filter format: {f} (expected key=value)")
                continue
            key, value = f.split('=', 1)
            metadata_filter[key.strip()] = value.strip()
        
        # Initialize components
        with create_progress_bar("Initializing search") as progress:
            task = progress.add_task("Loading components...", total=2)
            
            # Vector store
            vector_store = FAISSVectorStore(
                index_path=settings.vector_store.index_path,
                index_type=settings.vector_store.index_type,
                dimension=settings.vector_store.dimension
            )
            vector_store.load_index()
            progress.update(task, advance=1)
            
            # Model manager for embeddings
            model_manager = ModelManager(
                embedding_model=settings.embedding.model,
                embedding_provider=settings.embedding.provider,
                embedding_api_key=settings.embedding.api_key,
                embedding_api_base_url=settings.embedding.api_base_url
            )
            progress.update(task, advance=1)
        
        print_info(f"Searching {vector_store.index.ntotal} documents...")
        
        # Generate query embedding
        with create_progress_bar("Processing query") as progress:
            task = progress.add_task("Generating embedding...", total=None)
            query_embedding = model_manager.embed_text(query)
            progress.update(task, completed=True)
        
        # Perform search
        with create_progress_bar("Searching") as progress:
            task = progress.add_task("Finding similar documents...", total=None)
            
            results = vector_store.similarity_search(
                query_embedding=query_embedding,
                k=limit,
                score_threshold=similarity,
                filter=metadata_filter if metadata_filter else None
            )
            
            progress.update(task, completed=True)
        
        # Output results
        if not results:
            print_warning("No results found matching your query")
            return
        
        print_info(f"Found {len(results)} results")
        
        if output == "json":
            # Prepare JSON output
            json_results = []
            for result in results:
                json_result = {
                    "id": result.get("id"),
                    "content": result.get("content"),
                    "score": float(result.get("score", 0)),
                    "metadata": result.get("metadata", {})
                }
                if show_embeddings:
                    json_result["embedding"] = result.get("embedding", []).tolist()
                json_results.append(json_result)
            
            console.print(json.dumps(json_results, indent=2))
            
        elif output == "detailed":
            # Detailed text output
            for i, result in enumerate(results, 1):
                console.print(f"\n[bold cyan]━━━ Result {i} ━━━[/bold cyan]")
                console.print(f"[green]Score:[/green] {result.get('score', 0):.4f}")
                console.print(f"[green]ID:[/green] {result.get('id', 'N/A')}")
                
                # Metadata
                if metadata := result.get('metadata', {}):
                    console.print("[green]Metadata:[/green]")
                    for key, value in metadata.items():
                        console.print(f"  {key}: {value}")
                
                # Content
                console.print(f"\n[green]Content:[/green]")
                console.print(result.get('content', ''))
                
        else:
            # Default text output
            print_search_results(results)
        
        # Show search statistics if verbose
        if verbose:
            console.print(f"\n[dim]Query: {query}[/dim]")
            console.print(f"[dim]Filters: {metadata_filter if metadata_filter else 'None'}[/dim]")
            console.print(f"[dim]Min similarity: {similarity}[/dim]")
        
    except FileNotFoundError as e:
        print_error(f"Index files not found: {e}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Search failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)