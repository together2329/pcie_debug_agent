"""
PCIe RAG CLI Commands

Command-line interface for PCIe adaptive RAG functionality
"""

import logging
from pathlib import Path
from typing import Optional
import click

from src.rag.pcie_rag_engine import PCIeRAGEngine, create_pcie_rag_engine

logger = logging.getLogger(__name__)

@click.group()
def pcie():
    """PCIe-specific RAG commands with adaptive chunking"""
    pass

@pcie.command()
@click.option('--model', '-m', default='text-embedding-3-small', 
              help='Embedding model to use')
@click.option('--force', '-f', is_flag=True, 
              help='Force rebuild even if database exists')
@click.option('--target-size', default=1000, 
              help='Target chunk size in words')
@click.option('--max-size', default=1500, 
              help='Maximum chunk size in words')
@click.option('--overlap', default=200, 
              help='Overlap size in words')
def build(model: str, force: bool, target_size: int, max_size: int, overlap: int):
    """Build PCIe knowledge base with adaptive chunking"""
    
    click.echo(f"🔧 Building PCIe knowledge base with adaptive chunking...")
    click.echo(f"   Model: {model}")
    click.echo(f"   Target chunk size: {target_size} words")
    click.echo(f"   Max chunk size: {max_size} words")
    click.echo(f"   Overlap: {overlap} words")
    
    # Create PCIe RAG engine with custom config
    chunk_config = {
        'target_size': target_size,
        'max_size': max_size,
        'min_size': 200,
        'overlap_size': overlap
    }
    
    try:
        engine = create_pcie_rag_engine(
            embedding_model=model,
            chunk_config=chunk_config
        )
        
        # Build knowledge base
        success = engine.build_knowledge_base(
            knowledge_base_path="data/knowledge_base",
            force_rebuild=force
        )
        
        if success:
            # Get statistics
            stats = engine.get_pcie_mode_stats()
            
            click.echo(f"\n✅ PCIe knowledge base built successfully!")
            click.echo(f"   Total vectors: {stats.get('total_vectors', 'unknown'):,}")
            click.echo(f"   Vector dimension: {stats.get('dimension', 'unknown')}D")
            click.echo(f"   Chunking strategy: {stats.get('chunking_strategy', 'adaptive')}")
            click.echo(f"   Database size: {stats.get('size_mb', 0):.1f}MB")
        else:
            click.echo("❌ Failed to build PCIe knowledge base")
            exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error building PCIe knowledge base: {str(e)}")
        exit(1)

@pcie.command()
@click.argument('query')
@click.option('--model', '-m', default='text-embedding-3-small',
              help='Embedding model to use')
@click.option('--top-k', '-k', default=5,
              help='Number of results to return')
@click.option('--technical-level', '-t', type=click.Choice(['1', '2', '3']),
              help='Filter by technical level (1=basic, 2=intermediate, 3=advanced)')
@click.option('--layer', '-l', 
              type=click.Choice(['physical', 'transaction', 'data_link', 'power_management', 'system_architecture', 'software_interface']),
              help='Filter by PCIe layer')
@click.option('--semantic-type', '-s',
              type=click.Choice(['header_section', 'procedure', 'specification', 'example', 'content']),
              help='Filter by semantic type')
@click.option('--json', '-j', is_flag=True,
              help='Output results as JSON')
@click.option('--pretty', is_flag=True,
              help='Pretty print JSON output')
def query(query: str, model: str, top_k: int, technical_level: Optional[str], 
          layer: Optional[str], semantic_type: Optional[str], json: bool, pretty: bool):
    """Query PCIe knowledge base with adaptive chunking"""
    
    if not json:
        click.echo(f"🔍 PCIe Query: {query}")
        
        if technical_level:
            click.echo(f"   Technical level filter: {technical_level}")
        if layer:
            click.echo(f"   PCIe layer filter: {layer}")
        if semantic_type:
            click.echo(f"   Semantic type filter: {semantic_type}")
    
    try:
        # Create PCIe RAG engine
        engine = create_pcie_rag_engine(embedding_model=model)
        
        # Execute query with filters - request structured output if JSON
        results = engine.query(
            query=query,
            top_k=top_k,
            technical_level_filter=int(technical_level) if technical_level else None,
            pcie_layer_filter=layer,
            semantic_type_filter=semantic_type,
            return_structured=json
        )
        
        if json:
            # Output structured JSON
            if hasattr(results, 'to_json'):
                indent = 2 if pretty else None
                output = results.to_json(indent=indent)
                click.echo(output)
            else:
                # Fallback for legacy results
                import json as json_module
                indent = 2 if pretty else None
                click.echo(json_module.dumps([r.__dict__ for r in results], indent=indent))
            return
        
        # Legacy text output
        if not results:
            click.echo("❌ No results found")
            return
        
        # Display results
        click.echo(f"\n📚 Found {len(results)} results:")
        click.echo("=" * 80)
        
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. {result.source_section}")
            click.echo(f"   Score: {result.score:.3f}")
            click.echo(f"   Technical Level: {result.technical_level}")
            click.echo(f"   Semantic Type: {result.semantic_type}")
            click.echo(f"   PCIe Layer: {result.metadata.get('pcie_layer', 'unknown')}")
            
            if result.pcie_concepts:
                concepts = [c for c in result.pcie_concepts if c]  # Remove empty
                if concepts:
                    click.echo(f"   PCIe Concepts: {', '.join(concepts[:3])}...")  # Show first 3
            
            # Show content preview
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            click.echo(f"   Content: {content_preview}")
            
    except Exception as e:
        click.echo(f"❌ Error querying PCIe knowledge base: {str(e)}")
        exit(1)

@pcie.command()
@click.option('--model', '-m', default='text-embedding-3-small',
              help='Embedding model to use')
def stats(model: str):
    """Show PCIe RAG statistics"""
    
    try:
        engine = create_pcie_rag_engine(embedding_model=model)
        stats = engine.get_pcie_mode_stats()
        
        click.echo(f"📊 PCIe RAG Statistics")
        click.echo("=" * 50)
        click.echo(f"Mode: {stats.get('mode', 'unknown')}")
        click.echo(f"Chunking Strategy: {stats.get('chunking_strategy', 'unknown')}")
        total_vectors = stats.get('total_vectors', 'unknown')
        if isinstance(total_vectors, int):
            click.echo(f"Total Vectors: {total_vectors:,}")
        else:
            click.echo(f"Total Vectors: {total_vectors}")
        click.echo(f"Vector Dimension: {stats.get('dimension', 'unknown')}D")
        click.echo(f"Database Size: {stats.get('size_mb', 0):.1f}MB")
        
        # Show chunk configuration
        chunk_config = stats.get('chunk_config', {})
        if chunk_config:
            click.echo(f"\nChunk Configuration:")
            click.echo(f"  Target Size: {chunk_config.get('target_size', 'unknown')} words")
            click.echo(f"  Max Size: {chunk_config.get('max_size', 'unknown')} words")
            click.echo(f"  Min Size: {chunk_config.get('min_size', 'unknown')} words")
            click.echo(f"  Overlap: {chunk_config.get('overlap_size', 'unknown')} words")
        
        # Show feature flags
        click.echo(f"\nFeatures:")
        click.echo(f"  Concept Boosting: {'✅' if stats.get('concept_boosting_enabled') else '❌'}")
        click.echo(f"  Technical Level Filtering: {'✅' if stats.get('technical_level_filtering_enabled') else '❌'}")
        
    except Exception as e:
        click.echo(f"❌ Error getting PCIe RAG statistics: {str(e)}")

@pcie.command()
@click.option('--model', '-m', default='text-embedding-3-small',
              help='Embedding model to use') 
def rebuild(model: str):
    """Force rebuild PCIe knowledge base"""
    
    click.echo(f"🔄 Force rebuilding PCIe knowledge base...")
    
    try:
        from src.rag.pcie_rag_engine import build_pcie_knowledge_base
        
        success = build_pcie_knowledge_base(
            knowledge_base_path="data/knowledge_base",
            embedding_model=model,
            force_rebuild=True
        )
        
        if success:
            click.echo("✅ PCIe knowledge base rebuilt successfully!")
        else:
            click.echo("❌ Failed to rebuild PCIe knowledge base")
            exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error rebuilding PCIe knowledge base: {str(e)}")
        exit(1)

if __name__ == '__main__':
    pcie()