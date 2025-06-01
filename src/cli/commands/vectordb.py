"""
Vector Database management commands for PCIe Debug Agent
"""

import click
import os
import time
from pathlib import Path
from typing import Optional, List
import shutil

from src.processors.document_chunker import DocumentChunker
from src.vectorstore.faiss_store import FAISSVectorStore
from src.config.settings import load_settings
from src.models.embedding_selector import get_embedding_selector
from src.cli.utils.output import print_success, print_error, print_info, print_warning


@click.group()
def vectordb():
    """Manage vector database for PCIe knowledge base"""
    pass


@vectordb.command(name="build")
@click.option('--input-dir', '-i', default='data/knowledge_base', help='Input directory with documents')
@click.option('--output-dir', '-o', default='data/vectorstore', help='Output directory for vector database')
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if database exists')
@click.option('--chunk-size', default=1000, help='Chunk size for text splitting')
@click.option('--chunk-overlap', default=200, help='Overlap between chunks')
def build_vectordb(input_dir: str, output_dir: str, force: bool, chunk_size: int, chunk_overlap: int):
    """Build vector database from knowledge base documents"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if vector database already exists
    if output_path.exists() and not force:
        print_warning(f"Vector database already exists at {output_path}")
        print_info("Use --force to rebuild")
        
        # Check current stats
        try:
            store = FAISSVectorStore(index_path=str(output_path))
            print_info(f"Current database has {store.index.ntotal} vectors")
        except:
            pass
        return
    
    # Check if input directory exists
    if not input_path.exists():
        print_error(f"Input directory not found: {input_path}")
        return
    
    print_info("🔧 Building Vector Database...")
    print_info(f"Input: {input_path}")
    print_info(f"Output: {output_path}")
    
    start_time = time.time()
    
    try:
        # Initialize components
        settings = load_settings()
        
        # Create chunker
        chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Process documents
        print_info("\n📄 Processing documents...")
        documents = []
        
        # Find all text files
        text_files = list(input_path.glob('**/*.txt')) + \
                    list(input_path.glob('**/*.md')) + \
                    list(input_path.glob('**/*.log'))
        
        if not text_files:
            print_error("No documents found in input directory")
            return
        
        print_info(f"Found {len(text_files)} documents")
        
        # Process each file
        for i, file_path in enumerate(text_files, 1):
            try:
                print(f"  [{i}/{len(text_files)}] Processing {file_path.name}...", end='', flush=True)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Chunk the document
                chunks = chunker.chunk_document(content, str(file_path))
                documents.extend(chunks)
                
                print(f" ✓ ({len(chunks)} chunks)")
                
            except Exception as e:
                print(f" ✗ Error: {e}")
                continue
        
        print_info(f"\n📊 Total chunks created: {len(documents)}")
        
        # Create vector store
        print_info("\n🧮 Generating embeddings...")
        print_info("This may take a few minutes...")
        
        # Initialize vector store (will create if doesn't exist)
        if output_path.exists() and force:
            shutil.rmtree(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get current embedding model
        embedding_selector = get_embedding_selector()
        embedding_provider = embedding_selector.get_current_provider()
        embedding_info = embedding_selector.get_model_info()
        
        print_info(f"Using embedding model: {embedding_info['model']}")
        print_info(f"Provider: {embedding_info['provider']}")
        print_info(f"Dimension: {embedding_info['dimension']}")
        
        # Create vector store with correct dimension
        store = FAISSVectorStore(
            dimension=embedding_info['dimension'],
            index_path=str(output_path)
        )
        
        # Add documents in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            print(f"  Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}...", end='', flush=True)
            
            texts = [doc['content'] for doc in batch]
            metadatas = [{'source': doc.get('source', 'unknown')} for doc in batch]
            
            # Generate embeddings using selected model
            embeddings = embedding_provider.encode(texts)
            
            store.add_documents(embeddings, texts, metadatas)
            print(" ✓")
        
        # Save the index
        store.save(str(output_path))
        
        end_time = time.time()
        duration = end_time - start_time
        
        print_success(f"\n✅ Vector database built successfully!")
        print_info(f"   Total vectors: {store.index.ntotal}")
        print_info(f"   Time taken: {duration:.1f} seconds")
        print_info(f"   Location: {output_path}")
        
    except Exception as e:
        print_error(f"Failed to build vector database: {e}")
        import traceback
        traceback.print_exc()


@vectordb.command(name="status")
@click.option('--path', '-p', default='data/vectorstore', help='Path to vector database')
def status(path: str):
    """Check vector database status"""
    
    db_path = Path(path)
    
    if not db_path.exists():
        print_error(f"Vector database not found at {db_path}")
        print_info("Run 'pcie-debug vectordb build' to create it")
        return
    
    try:
        # Load existing vector store
        store = FAISSVectorStore.load(str(db_path))
        
        print_info("📊 Vector Database Status")
        print("=" * 50)
        print(f"Location: {db_path}")
        print(f"Total vectors: {store.index.ntotal:,}")
        print(f"Dimension: {store.dimension}")
        print(f"Index type: FAISS")
        
        # Check file sizes
        index_file = db_path / "index.faiss"
        metadata_file = db_path / "metadata.pkl"
        
        if index_file.exists():
            size_mb = index_file.stat().st_size / (1024 * 1024)
            print(f"Index size: {size_mb:.1f} MB")
        
        if metadata_file.exists():
            size_mb = metadata_file.stat().st_size / (1024 * 1024)
            print(f"Metadata size: {size_mb:.1f} MB")
        
        print("=" * 50)
        print_success("✅ Vector database is ready")
        
    except Exception as e:
        print_error(f"Error reading vector database: {e}")


@vectordb.command(name="clear")
@click.option('--path', '-p', default='data/vectorstore', help='Path to vector database')
@click.confirmation_option(prompt='Are you sure you want to delete the vector database?')
def clear(path: str):
    """Clear/delete vector database"""
    
    db_path = Path(path)
    
    if not db_path.exists():
        print_info("Vector database does not exist")
        return
    
    try:
        shutil.rmtree(db_path)
        print_success("✅ Vector database cleared successfully")
    except Exception as e:
        print_error(f"Failed to clear vector database: {e}")


@vectordb.command(name="search")
@click.argument('query')
@click.option('--path', '-p', default='data/vectorstore', help='Path to vector database')
@click.option('--top-k', '-k', default=5, help='Number of results to return')
def search(query: str, path: str, top_k: int):
    """Search vector database"""
    
    db_path = Path(path)
    
    if not db_path.exists():
        print_error(f"Vector database not found at {db_path}")
        return
    
    try:
        # Load existing vector store
        store = FAISSVectorStore.load(str(db_path))
        
        print_info(f"🔍 Searching for: '{query}'")
        print("-" * 50)
        
        # Generate embedding for query using current model
        embedding_selector = get_embedding_selector()
        embedding_provider = embedding_selector.get_current_provider()
        query_embedding = embedding_provider.encode([query])[0]
        
        # Search
        results = store.search(query_embedding, k=top_k)
        
        if not results:
            print_info("No results found")
            return
        
        for i, (content, metadata, score) in enumerate(results, 1):
            source = metadata.get('source', 'Unknown')
            print(f"\n{i}. Source: {source}")
            print(f"   Score: {score:.3f}")
            print(f"   Content: {content[:200]}...")
        
    except Exception as e:
        print_error(f"Search failed: {e}")


if __name__ == "__main__":
    vectordb()