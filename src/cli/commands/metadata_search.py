"""
Metadata-enhanced search commands for the CLI
"""

import click
import asyncio
from typing import Optional, List
from pathlib import Path

from src.rag.metadata_enhanced_rag import (
    MetadataEnhancedRAGEngine, MetadataRAGQuery
)
from src.rag.metadata_extractor import (
    PCIeDocumentType, PCIeVersion, ErrorSeverity
)
from src.cli.utils.output import print_info, print_success, print_error, print_warning
from src.config.settings import load_settings


@click.group()
def metadata():
    """Metadata-enhanced search commands"""
    pass


@metadata.command()
@click.argument('query')
@click.option('--pcie-version', multiple=True, help='Filter by PCIe version (e.g., 3.0, 4.0)')
@click.option('--doc-type', multiple=True, 
              type=click.Choice(['specification', 'error_log', 'debug_log', 'troubleshooting', 
                               'tutorial', 'code', 'configuration']),
              help='Filter by document type')
@click.option('--severity', 
              type=click.Choice(['critical', 'error', 'warning', 'info', 'debug']),
              help='Minimum error severity')
@click.option('--component', multiple=True, 
              help='Filter by PCIe component (e.g., root_complex, endpoint)')
@click.option('--topic', multiple=True,
              help='Filter by topic (e.g., link_training, error_handling)')
@click.option('--no-boost', is_flag=True, help='Disable metadata relevance boosting')
@click.option('--limit', default=5, help='Number of results to return')
def search(query: str, 
          pcie_version: tuple,
          doc_type: tuple,
          severity: Optional[str],
          component: tuple,
          topic: tuple,
          no_boost: bool,
          limit: int):
    """Search with metadata filters"""
    
    print_info(f"🔍 Searching: {query}")
    
    # Show active filters
    if any([pcie_version, doc_type, severity, component, topic]):
        print_info("Active filters:")
        if pcie_version:
            print_info(f"  PCIe versions: {', '.join(pcie_version)}")
        if doc_type:
            print_info(f"  Document types: {', '.join(doc_type)}")
        if severity:
            print_info(f"  Min severity: {severity}")
        if component:
            print_info(f"  Components: {', '.join(component)}")
        if topic:
            print_info(f"  Topics: {', '.join(topic)}")
    
    try:
        # Load settings and initialize engine
        settings = load_settings()
        engine = _get_metadata_engine(settings)
        
        # Create metadata query
        metadata_query = MetadataRAGQuery(
            query=query,
            context_window=limit,
            pcie_versions=list(pcie_version) if pcie_version else None,
            document_types=[PCIeDocumentType(dt) for dt in doc_type] if doc_type else None,
            error_severity=ErrorSeverity(severity) if severity else None,
            components=list(component) if component else None,
            topics=list(topic) if topic else None,
            use_metadata_boost=not no_boost
        )
        
        # Execute query
        response = engine.query_with_metadata(metadata_query)
        
        # Display results
        print_success(f"\n📊 Found {len(response.sources)} relevant documents")
        print_info(f"Confidence: {response.confidence:.2%}\n")
        
        # Show answer
        print_info("💡 Answer:")
        print(response.answer)
        
        # Show sources with metadata
        if response.sources:
            print_info("\n📚 Sources:")
            for i, source in enumerate(response.sources, 1):
                print(f"\n{i}. {source['title']}")
                metadata = source['metadata']
                
                # Show relevant metadata
                if metadata.get('pcie_version'):
                    print(f"   PCIe: {', '.join(metadata['pcie_version'])}")
                if metadata.get('document_type'):
                    print(f"   Type: {metadata['document_type']}")
                if metadata.get('topics'):
                    print(f"   Topics: {', '.join(metadata['topics'][:3])}")
                if metadata.get('error_codes'):
                    print(f"   Errors: {', '.join(metadata['error_codes'][:3])}")
                print(f"   Score: {source['relevance_score']:.3f}")
        
        # Show query metadata
        if response.metadata:
            print_info(f"\n⏱️  Query time: {response.metadata['query_time']:.2f}s")
            if response.metadata.get('metadata_filters_applied'):
                print_info(f"Filtered: {response.metadata['initial_results']} → {response.metadata['filtered_results']} results")
        
    except Exception as e:
        print_error(f"Search failed: {str(e)}")


@metadata.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--quick', is_flag=True, help='Use quick extraction (no LLM)')
def extract(file_path: str, quick: bool):
    """Extract metadata from a document"""
    
    file_path = Path(file_path)
    print_info(f"📄 Extracting metadata from: {file_path.name}")
    
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get extractor
        settings = load_settings()
        from src.models.model_manager import ModelManager
        from src.rag.metadata_extractor import MetadataExtractor
        
        model_manager = ModelManager()
        extractor = MetadataExtractor(model_manager)
        
        if quick:
            # Quick extraction
            print_info("Using quick extraction (regex-based)...")
            metadata = extractor.extract_quick_metadata(content)
            
            print_success("\n✅ Metadata extracted:")
            for key, value in metadata.items():
                if value:
                    print(f"  {key}: {value}")
        else:
            # LLM extraction
            print_info("Using LLM extraction...")
            
            # Run async extraction
            async def extract_async():
                return await extractor.extract_metadata(content, str(file_path))
            
            metadata = asyncio.run(extract_async())
            
            print_success("\n✅ Metadata extracted:")
            print(f"  Type: {metadata.document_type.value}")
            print(f"  Title: {metadata.title}")
            print(f"  Summary: {metadata.summary}")
            
            if metadata.pcie_version:
                print(f"  PCIe versions: {[v.value for v in metadata.pcie_version]}")
            if metadata.topics:
                print(f"  Topics: {metadata.topics}")
            if metadata.error_codes:
                print(f"  Error codes: {metadata.error_codes}")
            if metadata.components:
                print(f"  Components: {metadata.components}")
            if metadata.keywords:
                print(f"  Keywords: {metadata.keywords[:5]}")
            
            print(f"\n  Confidence: {metadata.confidence_score:.2%}")
            
    except Exception as e:
        print_error(f"Extraction failed: {str(e)}")


@metadata.command()
@click.argument('directory', type=click.Path(exists=True))
@click.option('--pattern', default='*.md', help='File pattern to process')
@click.option('--quick', is_flag=True, help='Use quick extraction')
@click.option('--force', is_flag=True, help='Force re-extraction')
def index(directory: str, pattern: str, quick: bool, force: bool):
    """Index directory with metadata extraction"""
    
    directory = Path(directory)
    print_info(f"📁 Indexing directory: {directory}")
    print_info(f"Pattern: {pattern}")
    
    try:
        # Find files
        files = list(directory.rglob(pattern))
        print_info(f"Found {len(files)} files to process")
        
        if not files:
            print_warning("No files found matching pattern")
            return
        
        # Initialize engine
        settings = load_settings()
        engine = _get_metadata_engine(settings)
        
        # Process files
        async def process_all():
            return await engine.batch_process_documents(
                [str(f) for f in files],
                quick_extract=quick
            )
        
        print_info("Processing files...")
        results = asyncio.run(process_all())
        
        # Show results
        print_success(f"\n✅ Indexing complete:")
        print(f"  Processed: {results['processed']}")
        print(f"  Failed: {results['failed']}")
        
        # Show sample metadata
        if results['documents']:
            print_info("\nSample metadata:")
            for doc in results['documents'][:3]:
                metadata = doc['metadata']
                print(f"\n  {doc['document_id']}")
                print(f"    Type: {metadata.get('document_type', 'unknown')}")
                print(f"    Title: {metadata.get('title', 'Unknown')}")
                
    except Exception as e:
        print_error(f"Indexing failed: {str(e)}")


@metadata.command()
def stats():
    """Show metadata statistics"""
    
    print_info("📊 Metadata Statistics")
    
    try:
        settings = load_settings()
        engine = _get_metadata_engine(settings)
        
        # Get all documents
        total_docs = len(engine.vector_store.documents)
        
        # Analyze metadata
        doc_types = {}
        pcie_versions = {}
        components = {}
        topics = {}
        has_metadata = 0
        
        for metadata in engine.vector_store.metadata:
            if metadata:
                has_metadata += 1
                
                # Count document types
                doc_type = metadata.get('document_type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Count PCIe versions
                for version in metadata.get('pcie_version', []):
                    pcie_versions[version] = pcie_versions.get(version, 0) + 1
                
                # Count components
                for component in metadata.get('components', []):
                    components[component] = components.get(component, 0) + 1
                
                # Count topics
                for topic in metadata.get('topics', []):
                    topics[topic] = topics.get(topic, 0) + 1
        
        # Display statistics
        print(f"\nTotal documents: {total_docs}")
        print(f"With metadata: {has_metadata} ({has_metadata/total_docs*100:.1f}%)")
        
        print("\nDocument types:")
        for doc_type, count in sorted(doc_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {doc_type}: {count}")
        
        print("\nPCIe versions:")
        for version, count in sorted(pcie_versions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {version}: {count}")
        
        print("\nTop components:")
        for component, count in sorted(components.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {component}: {count}")
        
        print("\nTop topics:")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {topic}: {count}")
            
    except Exception as e:
        print_error(f"Failed to get statistics: {str(e)}")


def _get_metadata_engine(settings):
    """Initialize metadata-enhanced RAG engine"""
    from src.vectorstore.faiss_store import FAISSVectorStore
    from src.models.model_manager import ModelManager
    
    # Initialize components
    vector_store = FAISSVectorStore(
        dimension=settings.embedding_dimension,
        index_path=settings.vector_store_path
    )
    
    model_manager = ModelManager()
    
    # Create engine
    engine = MetadataEnhancedRAGEngine(
        vector_store=vector_store,
        model_manager=model_manager,
        llm_model=settings.default_model
    )
    
    return engine