"""
CLI commands for hybrid search configuration
"""

import click
from pathlib import Path
from src.cli.utils.output import print_success, print_error, print_info, print_warning


@click.command(name="hybrid")
@click.option('--enable/--disable', default=True, help='Enable or disable hybrid search')
@click.option('--alpha', '-a', type=float, default=0.7, help='Weight for semantic search (0-1)')
@click.option('--method', '-m', type=click.Choice(['weighted', 'rrf']), default='weighted', 
              help='Fusion method: weighted or reciprocal rank fusion')
@click.option('--rebuild-index', is_flag=True, help='Rebuild BM25 index')
@click.option('--status', is_flag=True, help='Show hybrid search status')
def hybrid_search_config(enable, alpha, method, rebuild_index, status):
    """Configure hybrid search settings"""
    
    if status:
        # Show current status
        show_hybrid_status()
        return
    
    if rebuild_index:
        # Rebuild BM25 index
        rebuild_bm25_index()
        return
    
    # Update configuration
    update_hybrid_config(enable, alpha, method)


def show_hybrid_status():
    """Show current hybrid search status"""
    from src.vectorstore.faiss_store import FAISSVectorStore
    from src.rag.hybrid_search import HybridSearchEngine
    
    print("\n🔍 Hybrid Search Status")
    print("=" * 60)
    
    # Check if vector store exists
    vector_store_path = Path("data/vectorstore")
    if not vector_store_path.exists():
        print_error("❌ Vector store not found")
        print_info("   Run 'pcie-debug vectordb build' first")
        return
    
    # Check BM25 index
    bm25_index_path = vector_store_path / "bm25_index.pkl"
    if bm25_index_path.exists():
        print_success("✅ BM25 index exists")
        
        # Load and show statistics
        try:
            vector_store = FAISSVectorStore.load(str(vector_store_path))
            hybrid_search = HybridSearchEngine(
                vector_store=vector_store,
                index_path=str(bm25_index_path)
            )
            
            stats = hybrid_search.get_statistics()
            print_info(f"   Documents indexed: {stats['total_documents']}")
            print_info(f"   Vocabulary size: {stats['bm25_vocabulary_size']}")
            print_info(f"   Avg document length: {stats['average_doc_length']:.1f} tokens")
            print_info(f"   Current alpha: {stats['alpha_weight']}")
        except Exception as e:
            print_error(f"   Error loading index: {e}")
    else:
        print_warning("⚠️  BM25 index not found")
        print_info("   Run with --rebuild-index to create")
    
    # Show configuration
    try:
        config_path = Path.home() / ".pcie_debug" / "hybrid_search_config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            print("\n⚙️  Configuration:")
            print_info(f"   Enabled: {config.get('enabled', True)}")
            print_info(f"   Alpha: {config.get('alpha', 0.7)}")
            print_info(f"   Method: {config.get('method', 'weighted')}")
    except:
        print_info("\n⚙️  Using default configuration")


def rebuild_bm25_index():
    """Rebuild BM25 index for hybrid search"""
    from src.vectorstore.faiss_store import FAISSVectorStore
    from src.rag.hybrid_search import HybridSearchEngine
    
    print("\n🔧 Rebuilding BM25 Index")
    print("=" * 60)
    
    # Check vector store
    vector_store_path = Path("data/vectorstore")
    if not vector_store_path.exists():
        print_error("❌ Vector store not found")
        return
    
    try:
        # Load vector store
        print_info("Loading vector store...")
        vector_store = FAISSVectorStore.load(str(vector_store_path))
        print_success(f"✅ Loaded {vector_store.index.ntotal} vectors")
        
        # Create hybrid search engine
        print_info("Building BM25 index...")
        bm25_index_path = vector_store_path / "bm25_index.pkl"
        
        hybrid_search = HybridSearchEngine(
            vector_store=vector_store,
            index_path=str(bm25_index_path)
        )
        
        # Force rebuild
        hybrid_search.build_bm25_index()
        hybrid_search.save_bm25_index()
        
        stats = hybrid_search.get_statistics()
        print_success("✅ BM25 index rebuilt successfully")
        print_info(f"   Documents: {stats['total_documents']}")
        print_info(f"   Vocabulary: {stats['bm25_vocabulary_size']} unique tokens")
        print_info(f"   Index saved to: {bm25_index_path}")
        
    except Exception as e:
        print_error(f"❌ Failed to rebuild index: {e}")


def update_hybrid_config(enable: bool, alpha: float, method: str):
    """Update hybrid search configuration"""
    import json
    
    print("\n⚙️  Updating Hybrid Search Configuration")
    print("=" * 60)
    
    # Validate alpha
    if not 0.0 <= alpha <= 1.0:
        print_error("❌ Alpha must be between 0.0 and 1.0")
        return
    
    # Create config
    config = {
        "enabled": enable,
        "alpha": alpha,
        "method": method
    }
    
    # Save configuration
    config_path = Path.home() / ".pcie_debug" / "hybrid_search_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print_success("✅ Configuration updated")
    print_info(f"   Hybrid search: {'Enabled' if enable else 'Disabled'}")
    print_info(f"   Alpha (semantic weight): {alpha}")
    print_info(f"   Alpha (keyword weight): {1-alpha:.1f}")
    print_info(f"   Fusion method: {method}")
    
    if enable and alpha == 0.0:
        print_warning("⚠️  Alpha=0.0 means keyword search only")
    elif enable and alpha == 1.0:
        print_warning("⚠️  Alpha=1.0 means semantic search only")
    
    # Check if BM25 index exists
    bm25_index_path = Path("data/vectorstore/bm25_index.pkl")
    if enable and not bm25_index_path.exists():
        print_warning("\n⚠️  BM25 index not found")
        print_info("   Run 'pcie-debug hybrid --rebuild-index' to create it")


# Register command
if __name__ == "__main__":
    hybrid_search_config()