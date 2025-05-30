"""Test command for PCIe Debug Agent CLI"""

import click
import sys
import time
from pathlib import Path
from typing import List, Optional
import subprocess

from src.cli.utils.output import (
    console, print_error, print_success, print_info, print_warning,
    print_table, create_progress_bar
)


@click.group()
def test():
    """Run tests and diagnostics"""
    pass


@test.command()
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed test output"
)
@click.pass_context
def connectivity(ctx: click.Context, verbose: bool):
    """Test API connectivity
    
    Example:
        pcie-debug test connectivity
    """
    settings = ctx.obj["settings"]
    results = []
    
    try:
        print_info("Testing API connectivity...\n")
        
        # Test LLM API
        with create_progress_bar("Testing LLM API") as progress:
            task = progress.add_task("Connecting...", total=None)
            
            try:
                if settings.llm.provider == "openai":
                    import openai
                    openai.api_key = settings.llm.api_key
                    if settings.llm.api_base_url:
                        openai.api_base = settings.llm.api_base_url
                    
                    # Test with a simple request
                    response = openai.ChatCompletion.create(
                        model=settings.llm.model,
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=5
                    )
                    results.append(["LLM API", "OpenAI", "✅ Connected", "OK"])
                    
                elif settings.llm.provider == "anthropic":
                    import anthropic
                    client = anthropic.Anthropic(api_key=settings.llm.api_key)
                    response = client.completions.create(
                        model=settings.llm.model,
                        prompt="test",
                        max_tokens_to_sample=5
                    )
                    results.append(["LLM API", "Anthropic", "✅ Connected", "OK"])
                    
                else:
                    results.append(["LLM API", settings.llm.provider, "⚠️  Untested", "Skip"])
                    
            except Exception as e:
                error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                results.append(["LLM API", settings.llm.provider, "❌ Failed", error_msg])
            
            progress.update(task, completed=True)
        
        # Test Embedding API
        with create_progress_bar("Testing Embedding API") as progress:
            task = progress.add_task("Connecting...", total=None)
            
            try:
                if settings.embedding.provider == "openai":
                    import openai
                    openai.api_key = settings.embedding.api_key
                    response = openai.Embedding.create(
                        model=settings.embedding.model,
                        input="test"
                    )
                    results.append(["Embedding API", "OpenAI", "✅ Connected", "OK"])
                    
                elif settings.embedding.provider == "local":
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(settings.embedding.model)
                    embedding = model.encode("test")
                    results.append(["Embedding API", "Local", "✅ Loaded", f"Dim: {len(embedding)}"])
                    
                else:
                    results.append(["Embedding API", settings.embedding.provider, "⚠️  Untested", "Skip"])
                    
            except Exception as e:
                error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
                results.append(["Embedding API", settings.embedding.provider, "❌ Failed", error_msg])
            
            progress.update(task, completed=True)
        
        # Test Vector Store
        with create_progress_bar("Testing Vector Store") as progress:
            task = progress.add_task("Checking...", total=None)
            
            try:
                from src.rag.vector_store import FAISSVectorStore
                
                if Path(settings.vector_store.index_path + ".faiss").exists():
                    store = FAISSVectorStore(
                        index_path=settings.vector_store.index_path,
                        dimension=settings.vector_store.dimension
                    )
                    store.load_index()
                    results.append([
                        "Vector Store", 
                        "FAISS", 
                        "✅ Loaded", 
                        f"{store.index.ntotal} vectors"
                    ])
                else:
                    results.append(["Vector Store", "FAISS", "⚠️  Not found", "No index"])
                    
            except Exception as e:
                results.append(["Vector Store", "FAISS", "❌ Failed", str(e)[:50]])
            
            progress.update(task, completed=True)
        
        # Display results
        print_table(
            headers=["Component", "Provider", "Status", "Details"],
            rows=results,
            title="Connectivity Test Results"
        )
        
        # Summary
        failed = sum(1 for r in results if "❌" in r[2])
        if failed == 0:
            print_success("\nAll connectivity tests passed!")
        else:
            print_error(f"\n{failed} connectivity test(s) failed")
            sys.exit(1)
            
    except Exception as e:
        print_error(f"Connectivity test failed: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@test.command()
@click.option(
    "--sample-size", "-s",
    type=int,
    default=100,
    help="Number of test queries"
)
@click.pass_context
def performance(ctx: click.Context, sample_size: int):
    """Run performance tests
    
    Example:
        pcie-debug test performance --sample-size 50
    """
    settings = ctx.obj["settings"]
    
    try:
        print_info("Running performance tests...\n")
        
        # Import required components
        from src.rag.vector_store import FAISSVectorStore
        from src.rag.model_manager import ModelManager
        import numpy as np
        
        # Initialize components
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            dimension=settings.vector_store.dimension
        )
        
        if not Path(settings.vector_store.index_path + ".faiss").exists():
            print_error("No vector store index found. Run 'pcie-debug index build' first.")
            sys.exit(1)
        
        vector_store.load_index()
        
        model_manager = ModelManager(
            embedding_model=settings.embedding.model,
            embedding_provider=settings.embedding.provider,
            embedding_api_key=settings.embedding.api_key
        )
        
        # Test 1: Embedding generation speed
        print_info("Testing embedding generation speed...")
        test_texts = [f"Test query {i}" for i in range(sample_size)]
        
        start_time = time.time()
        embeddings = model_manager.embed_texts(test_texts)
        embedding_time = time.time() - start_time
        
        embedding_speed = sample_size / embedding_time
        print_success(f"Embedding speed: {embedding_speed:.2f} texts/second")
        
        # Test 2: Vector search speed
        print_info("\nTesting vector search speed...")
        search_times = []
        
        with create_progress_bar("Running searches") as progress:
            task = progress.add_task("Searching...", total=sample_size)
            
            for embedding in embeddings[:sample_size]:
                start_time = time.time()
                results = vector_store.similarity_search(
                    query_embedding=embedding,
                    k=10
                )
                search_times.append(time.time() - start_time)
                progress.update(task, advance=1)
        
        avg_search_time = sum(search_times) / len(search_times)
        p95_search_time = sorted(search_times)[int(len(search_times) * 0.95)]
        
        # Test 3: Memory usage
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Display results
        print_info("\n[bold]Performance Test Results:[/bold]")
        
        metrics = [
            ["Metric", "Value"],
            ["Index Size", f"{vector_store.index.ntotal:,} vectors"],
            ["Embedding Speed", f"{embedding_speed:.2f} texts/sec"],
            ["Avg Search Time", f"{avg_search_time*1000:.2f} ms"],
            ["P95 Search Time", f"{p95_search_time*1000:.2f} ms"],
            ["Search Throughput", f"{1/avg_search_time:.2f} queries/sec"],
            ["Memory Usage", f"{memory_usage:.2f} MB"]
        ]
        
        print_table(
            headers=metrics[0],
            rows=metrics[1:],
            title="Performance Metrics"
        )
        
        # Performance assessment
        print_info("\n[bold]Performance Assessment:[/bold]")
        
        if avg_search_time < 0.1:
            print_success("✅ Search performance: Excellent")
        elif avg_search_time < 0.5:
            print_info("ℹ️  Search performance: Good")
        else:
            print_warning("⚠️  Search performance: Could be improved")
        
        if embedding_speed > 100:
            print_success("✅ Embedding speed: Excellent")
        elif embedding_speed > 50:
            print_info("ℹ️  Embedding speed: Good")
        else:
            print_warning("⚠️  Embedding speed: Could be improved")
            
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        console.print_exception()
        sys.exit(1)


@test.command()
@click.option(
    "--unit", "-u",
    is_flag=True,
    help="Run unit tests"
)
@click.option(
    "--integration", "-i", 
    is_flag=True,
    help="Run integration tests"
)
@click.option(
    "--coverage", "-c",
    is_flag=True,
    help="Generate coverage report"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose test output"
)
@click.pass_context
def suite(
    ctx: click.Context,
    unit: bool,
    integration: bool,
    coverage: bool,
    verbose: bool
):
    """Run test suite
    
    Examples:
        pcie-debug test suite --unit
        
        pcie-debug test suite --unit --integration --coverage
    """
    # Default to all tests if none specified
    if not unit and not integration:
        unit = integration = True
    
    try:
        print_info("Running test suite...\n")
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        if unit:
            cmd.append("tests/unit")
        if integration:
            cmd.append("tests/integration")
        
        if coverage:
            cmd.extend(["--cov=src", "--cov-report=term-missing"])
        
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Run tests
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Display output
        console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr, style="red")
        
        if result.returncode == 0:
            print_success("\nAll tests passed!")
        else:
            print_error("\nTests failed!")
            sys.exit(1)
            
    except FileNotFoundError:
        print_error("pytest not found. Install with: pip install pytest pytest-cov")
        sys.exit(1)
    except Exception as e:
        print_error(f"Test suite failed: {e}")
        sys.exit(1)