"""Index command for PCIe Debug Agent CLI"""

import click
import sys
from pathlib import Path
from typing import List, Optional
import time

from src.cli.utils.output import (
    console, print_error, print_success, print_info, print_warning,
    print_table, print_stats, create_progress_bar, confirm
)
from src.collectors.log_collector import LogCollector
from src.processors.document_chunker import DocumentChunker
from src.processors.embedder import Embedder
from src.rag.vector_store import FAISSVectorStore
from src.rag.model_manager import ModelManager


@click.group()
def index():
    """Manage vector store indexing"""
    pass


@index.command()
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True),
    required=True
)
@click.option(
    "--pattern", "-p",
    multiple=True,
    default=["*.log", "*.txt", "*.v", "*.sv"],
    help="File patterns to index"
)
@click.option(
    "--recursive", "-r",
    is_flag=True,
    help="Recursively search directories"
)
@click.option(
    "--batch-size", "-b",
    type=int,
    default=100,
    help="Batch size for processing"
)
@click.option(
    "--workers", "-w",
    type=int,
    default=4,
    help="Number of parallel workers"
)
@click.option(
    "--chunk-size",
    type=int,
    help="Override default chunk size"
)
@click.option(
    "--chunk-overlap",
    type=int,
    help="Override default chunk overlap"
)
@click.option(
    "--force",
    is_flag=True,
    help="Force rebuild existing index"
)
@click.pass_context
def build(
    ctx: click.Context,
    paths: List[str],
    pattern: List[str],
    recursive: bool,
    batch_size: int,
    workers: int,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    force: bool
):
    """Build or rebuild vector index from logs
    
    Examples:
        pcie-debug index build /path/to/logs
        
        pcie-debug index build . --recursive --pattern "*.log"
        
        pcie-debug index build /logs /more/logs --force
    """
    settings = ctx.obj["settings"]
    verbose = ctx.obj.get("verbose", False)
    
    try:
        # Check if index exists
        index_path = Path(settings.vector_store.index_path)
        if index_path.exists() and not force:
            if not confirm("Index already exists. Rebuild?"):
                print_info("Operation cancelled")
                return
        
        # Initialize components
        print_info("Initializing indexing components...")
        
        # Vector store
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            index_type=settings.vector_store.index_type,
            dimension=settings.vector_store.dimension
        )
        
        # Model manager
        model_manager = ModelManager(
            embedding_model=settings.embedding.model,
            embedding_provider=settings.embedding.provider,
            embedding_api_key=settings.embedding.api_key,
            embedding_api_base_url=settings.embedding.api_base_url
        )
        
        # Collectors and processors
        collector = LogCollector(
            file_patterns=list(pattern),
            recursive=recursive
        )
        
        chunker = DocumentChunker(
            chunk_size=chunk_size or settings.rag.chunk_size,
            chunk_overlap=chunk_overlap or settings.rag.chunk_overlap
        )
        
        embedder = Embedder(
            model_manager=model_manager,
            batch_size=batch_size
        )
        
        # Collect files
        print_info("Scanning for files...")
        all_files = []
        for path in paths:
            files = collector.collect_files(Path(path))
            all_files.extend(files)
        
        if not all_files:
            print_warning("No files found matching the specified patterns")
            return
        
        print_success(f"Found {len(all_files)} files to index")
        
        # Process files
        total_chunks = 0
        start_time = time.time()
        
        with create_progress_bar("Indexing files") as progress:
            task = progress.add_task(
                "Processing...",
                total=len(all_files)
            )
            
            for i, file_path in enumerate(all_files):
                try:
                    # Read file
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    
                    # Chunk document
                    chunks = chunker.chunk_document(
                        content=content,
                        metadata={
                            "source": str(file_path),
                            "file_type": file_path.suffix
                        }
                    )
                    
                    # Generate embeddings
                    embeddings = embedder.embed_batch(
                        [chunk["content"] for chunk in chunks]
                    )
                    
                    # Add to vector store
                    documents = []
                    for chunk, embedding in zip(chunks, embeddings):
                        documents.append({
                            "id": f"{file_path.name}_{chunk['metadata']['chunk_index']}",
                            "content": chunk["content"],
                            "embedding": embedding,
                            "metadata": chunk["metadata"]
                        })
                    
                    vector_store.add_documents(documents)
                    total_chunks += len(documents)
                    
                    if verbose:
                        print_info(f"Indexed {file_path.name}: {len(chunks)} chunks")
                    
                except Exception as e:
                    print_error(f"Failed to index {file_path}: {e}")
                
                progress.update(task, advance=1)
        
        # Save index
        print_info("Saving index...")
        vector_store.save_index()
        
        # Print statistics
        elapsed_time = time.time() - start_time
        stats = {
            "Files Indexed": len(all_files),
            "Total Chunks": total_chunks,
            "Index Size": vector_store.index.ntotal,
            "Processing Time": f"{elapsed_time:.2f}s",
            "Chunks/Second": f"{total_chunks/elapsed_time:.2f}"
        }
        
        print_stats(stats, title="Indexing Statistics")
        print_success("Indexing complete!")
        
    except Exception as e:
        print_error(f"Indexing failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@index.command()
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(exists=True),
    required=True
)
@click.option(
    "--pattern", "-p",
    multiple=True,
    default=["*.log", "*.txt"],
    help="File patterns to update"
)
@click.pass_context
def update(ctx: click.Context, paths: List[str], pattern: List[str]):
    """Update existing index with new files
    
    Examples:
        pcie-debug index update /new/logs
        
        pcie-debug index update . --pattern "*.log"
    """
    settings = ctx.obj["settings"]
    
    try:
        # Load existing index
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            index_type=settings.vector_store.index_type,
            dimension=settings.vector_store.dimension
        )
        
        if not Path(settings.vector_store.index_path + ".faiss").exists():
            print_error("No existing index found. Use 'index build' first.")
            sys.exit(1)
        
        vector_store.load_index()
        print_info(f"Loaded existing index with {vector_store.index.ntotal} vectors")
        
        # TODO: Implement incremental update logic
        print_warning("Update functionality not yet implemented")
        
    except Exception as e:
        print_error(f"Update failed: {e}")
        sys.exit(1)


@index.command()
@click.pass_context
def stats(ctx: click.Context):
    """Show index statistics
    
    Example:
        pcie-debug index stats
    """
    settings = ctx.obj["settings"]
    
    try:
        # Load index
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            index_type=settings.vector_store.index_type,
            dimension=settings.vector_store.dimension
        )
        
        if not Path(settings.vector_store.index_path + ".faiss").exists():
            print_error("No index found")
            sys.exit(1)
        
        vector_store.load_index()
        
        # Get statistics
        stats = vector_store.get_statistics()
        
        # Display statistics
        print_stats(stats, title="Vector Store Statistics")
        
        # Show metadata distribution
        if "metadata_stats" in stats:
            print_info("\nMetadata Distribution:")
            for key, values in stats["metadata_stats"].items():
                console.print(f"\n  {key}:")
                for value, count in values.items():
                    console.print(f"    - {value}: {count}")
        
    except Exception as e:
        print_error(f"Failed to get statistics: {e}")
        sys.exit(1)


@index.command()
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
@click.pass_context
def optimize(ctx: click.Context, confirm: bool):
    """Optimize index for better performance
    
    Example:
        pcie-debug index optimize
    """
    settings = ctx.obj["settings"]
    
    try:
        if not confirm:
            if not confirm("This will optimize the index. Continue?"):
                print_info("Operation cancelled")
                return
        
        # Load index
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            index_type=settings.vector_store.index_type,
            dimension=settings.vector_store.dimension
        )
        
        vector_store.load_index()
        
        print_info("Optimizing index...")
        initial_memory = vector_store.get_memory_usage()
        
        # Optimize
        vector_store.optimize_index()
        
        final_memory = vector_store.get_memory_usage()
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        print_success(f"Optimization complete! Memory reduced by {reduction:.1f}%")
        
        # Save optimized index
        vector_store.save_index()
        
    except Exception as e:
        print_error(f"Optimization failed: {e}")
        sys.exit(1)