import streamlit as st
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import asyncio
import threading
import queue
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Import your existing modules
from src.config.settings import Settings
from src.collectors.log_collector import LogCollector, UVMError
from src.processors.document_chunker import DocumentChunker
from src.processors.code_chunker import SystemVerilogChunker
from src.processors.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.retriever import Retriever
from src.rag.analyzer import Analyzer
from src.reports.report_generator import ReportGenerator
from src.models.model_manager import ModelManager

# Page configuration
st.set_page_config(
    page_title="UVM Debug Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    .success-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-message {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .vector-db-stats {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .file-preview {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        max-height: 200px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'errors' not in st.session_state:
    st.session_state.errors = []
if 'settings' not in st.session_state:
    st.session_state.settings = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()
if 'embedding_progress' not in st.session_state:
    st.session_state.embedding_progress = 0
if 'embedded_chunks' not in st.session_state:
    st.session_state.embedded_chunks = []
if 'vector_store_stats' not in st.session_state:
    st.session_state.vector_store_stats = {}

def load_config(config_path: str) -> Settings:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return Settings(**config_data)

def save_config(config_data: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, allow_unicode=True, sort_keys=False)

def create_sidebar():
    """Create sidebar with configuration options"""
    st.sidebar.title("🔧 Configuration")
    
    # Model Selection
    st.sidebar.header("Model Selection")
    
    # Get model manager
    model_manager = st.session_state.model_manager
    
    # Embedding Model Provider
    embedding_providers = ["Local", "OpenAI", "Cohere", "Voyage", "Custom"]
    embedding_provider = st.sidebar.selectbox(
        "Embedding Provider",
        options=embedding_providers,
        help="Choose embedding model provider"
    )
    
    # Embedding Model based on provider
    if embedding_provider == "Local":
        embedding_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            "sentence-transformers/all-roberta-large-v1"
        ]
    elif embedding_provider == "OpenAI":
        embedding_models = [
            "openai/text-embedding-ada-002",
            "openai/text-embedding-3-small",
            "openai/text-embedding-3-large"
        ]
    elif embedding_provider == "Cohere":
        embedding_models = ["cohere/embed-english-v3.0"]
    elif embedding_provider == "Voyage":
        embedding_models = ["voyage/voyage-01"]
    else:  # Custom
        embedding_models = ["custom/api"]
    
    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        options=embedding_models,
        help="Choose the embedding model"
    )
    
    # Show model info
    if selected_embedding:
        model_info = model_manager.embedding_manager.get_model_info(selected_embedding)
        if model_info:
            st.sidebar.info(f"{model_info.description}")
    
    # Embedding API Configuration
    if embedding_provider != "Local":
        st.sidebar.subheader("Embedding API Configuration")
        
        if embedding_provider in ["OpenAI", "Cohere", "Voyage"]:
            embedding_api_key = st.sidebar.text_input(
                f"{embedding_provider} API Key",
                type="password",
                help=f"Your {embedding_provider} API key"
            )
            embedding_base_url = None
            embedding_headers = None
            embedding_dimension = model_info.dimension if model_info else 768
        else:  # Custom
            embedding_api_key = st.sidebar.text_input(
                "API Key (Optional)",
                type="password",
                help="API key if required"
            )
            embedding_base_url = st.sidebar.text_input(
                "Embedding API Base URL",
                placeholder="https://api.example.com/v1/embeddings",
                help="Base URL for custom embedding API"
            )
            
            embedding_headers = st.sidebar.text_area(
                "Additional Headers (JSON)",
                placeholder='{"X-Custom-Header": "value"}',
                help="Additional headers for API requests"
            )
            
            embedding_dimension = st.sidebar.number_input(
                "Embedding Dimension",
                min_value=1,
                max_value=4096,
                value=768,
                help="Dimension of the embedding vectors"
            )
    else:
        embedding_api_key = None
        embedding_base_url = None
        embedding_headers = None
        embedding_dimension = model_manager.embedding_manager.get_model_info(selected_embedding).dimension
    
    # Test Embedding API Connection
    if embedding_provider != "Local":
        if st.sidebar.button("🔌 Test Embedding API", key="test_embedding_api"):
            with st.spinner("Testing embedding API connection..."):
                try:
                    # Configure API
                    if embedding_provider == "Custom":
                        headers = json.loads(embedding_headers) if embedding_headers else {}
                        model_manager.configure_embedding_api(
                            selected_embedding,
                            api_key=embedding_api_key,
                            base_url=embedding_base_url,
                            headers=headers
                        )
                    else:
                        model_manager.configure_embedding_api(
                            selected_embedding,
                            api_key=embedding_api_key,
                            base_url=embedding_base_url if embedding_base_url else None
                        )
                    
                    # Test connection with a sample text
                    test_text = "This is a test sentence for embedding API connection."
                    result = model_manager.test_embedding_api(
                        text=test_text,
                        model_name=selected_embedding
                    )
                    
                    if result['success']:
                        st.sidebar.success("✅ " + result['message'])
                        st.session_state.api_test_results['embedding'] = result
                        
                        # Show embedding dimension if successful
                        if 'dimension' in result:
                            st.sidebar.info(f"Embedding dimension: {result['dimension']}")
                    else:
                        st.sidebar.error("❌ " + result['message'])
                except Exception as e:
                    st.sidebar.error(f"❌ Error: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # LLM Provider
    llm_providers = ["openai", "anthropic", "ollama", "custom"]
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=llm_providers,
        format_func=lambda x: x.capitalize(),
        help="Choose LLM provider"
    )
    
    # LLM Model
    llm_models = model_manager.llm_manager.SUPPORTED_MODELS.get(llm_provider, {})
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        options=list(llm_models.keys()),
        format_func=lambda x: llm_models[x].name if x in llm_models else x
    )
    
    if llm_model and llm_model in llm_models:
        model_info = llm_models[llm_model]
        st.sidebar.info(f"{model_info.description}")
    
    # LLM API Configuration
    st.sidebar.subheader("LLM API Configuration")
    
    if llm_provider == "openai":
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key"
        )
        openai_base_url = st.sidebar.text_input(
            "OpenAI Base URL (Optional)",
            placeholder="https://api.openai.com",
            help="Custom OpenAI-compatible API endpoint"
        )
        custom_headers = None
    elif llm_provider == "anthropic":
        anthropic_api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            help="Your Anthropic API key"
        )
        anthropic_base_url = st.sidebar.text_input(
            "Anthropic Base URL (Optional)",
            placeholder="https://api.anthropic.com",
            help="Custom Anthropic API endpoint"
        )
        custom_headers = None
    elif llm_provider == "ollama":
        ollama_base_url = st.sidebar.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="Ollama API endpoint"
        )
        custom_headers = None
    elif llm_provider == "custom":
        custom_api_key = st.sidebar.text_input(
            "API Key (Optional)",
            type="password",
            help="API key if required"
        )
        custom_base_url = st.sidebar.text_input(
            "LLM API Base URL",
            placeholder="https://api.example.com/v1/chat",
            help="Base URL for custom LLM API"
        )
        custom_headers = st.sidebar.text_area(
            "Additional Headers (JSON)",
            placeholder='{"X-Custom-Header": "value"}',
            help="Additional headers for API requests"
        )
    
    # Test LLM API Connection
    if st.sidebar.button("🔌 Test LLM API", key="test_llm_api"):
        with st.spinner("Testing LLM API connection..."):
            try:
                # Configure and initialize API
                if llm_provider == "openai":
                    model_manager.configure_llm_api(
                        llm_provider,
                        api_key=openai_api_key,
                        base_url=openai_base_url if openai_base_url else None
                    )
                    model_manager.initialize_llm(
                        llm_provider,
                        api_key=openai_api_key,
                        base_url=openai_base_url if openai_base_url else None
                    )
                elif llm_provider == "anthropic":
                    model_manager.configure_llm_api(
                        llm_provider,
                        api_key=anthropic_api_key,
                        base_url=anthropic_base_url if anthropic_base_url else None
                    )
                    model_manager.initialize_llm(
                        llm_provider,
                        api_key=anthropic_api_key,
                        base_url=anthropic_base_url if anthropic_base_url else None
                    )
                elif llm_provider == "ollama":
                    model_manager.initialize_llm(
                        llm_provider,
                        base_url=ollama_base_url
                    )
                elif llm_provider == "custom":
                    headers = json.loads(custom_headers) if custom_headers else {}
                    model_manager.configure_llm_api(
                        llm_provider,
                        api_key=custom_api_key,
                        base_url=custom_base_url,
                        headers=headers
                    )
                    model_manager.initialize_llm(
                        llm_provider,
                        api_key=custom_api_key,
                        base_url=custom_base_url,
                        headers=headers
                    )
                
                # Test connection
                result = model_manager.test_api_connection("llm", llm_provider, llm_model)
                
                if result['success']:
                    st.sidebar.success("✅ " + result['message'])
                    st.session_state.api_test_results['llm'] = result
                else:
                    st.sidebar.error("❌ " + result['message'])
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # Processing Settings
    st.sidebar.header("Processing Settings")
    
    chunk_size = st.sidebar.slider(
        "Chunk Size",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
        help="Size of text chunks for processing"
    )
    
    chunk_overlap = st.sidebar.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Overlap between consecutive chunks"
    )
    
    retrieval_top_k = st.sidebar.slider(
        "Retrieval Top K",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of documents to retrieve for each query"
    )
    
    temperature = st.sidebar.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Controls randomness in LLM responses (0=deterministic, 1=creative)"
    )
    
    # Log Collection Settings
    st.sidebar.header("Log Collection")
    
    log_dirs = st.sidebar.text_area(
        "Log Directories",
        value="logs/simulation\nlogs/verification",
        help="One directory per line"
    ).split('\n')
    
    # Advanced Settings
    with st.sidebar.expander("Advanced Settings"):
        batch_size = st.sidebar.number_input(
            "Batch Size",
            min_value=1,
            max_value=100,
            value=32,
            help="Batch size for embedding generation"
        )
        
        max_tokens = st.sidebar.number_input(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=2000,
            help="Maximum tokens in LLM response"
        )
        
        parallel_workers = st.sidebar.number_input(
            "Parallel Workers",
            min_value=1,
            max_value=16,
            value=4,
            help="Number of parallel workers for processing"
        )
    
    # Create settings object
    settings_dict = {
        "embedding": {
            "model": selected_embedding,
            "batch_size": batch_size,
            "dimension": embedding_dimension,
            "api_key": embedding_api_key,
            "base_url": embedding_base_url,
            "headers": json.loads(embedding_headers) if embedding_headers else {}
        },
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_key": openai_api_key if llm_provider == "openai" else None,
            "openai_base_url": openai_base_url if llm_provider == "openai" else None,
            "anthropic_api_key": anthropic_api_key if llm_provider == "anthropic" else None,
            "anthropic_base_url": anthropic_base_url if llm_provider == "anthropic" else None,
            "ollama_base_url": ollama_base_url if llm_provider == "ollama" else None,
            "custom_api_key": custom_api_key if llm_provider == "custom" else None,
            "custom_base_url": custom_base_url if llm_provider == "custom" else None,
            "custom_headers": json.loads(custom_headers) if llm_provider == "custom" and custom_headers else {}
        },
        "rag": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "retrieval_top_k": retrieval_top_k
        },
        "log_directories": log_dirs,
        "batch": {
            "batch_size": batch_size,
            "parallel_workers": parallel_workers
        }
    }
    
    return settings_dict

def display_metrics(errors: List[UVMError], analysis_results: List[Dict[str, Any]]):
    """Display analysis metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Errors",
            value=len(errors),
            delta=None
        )
    
    with col2:
        fatal_errors = len([e for e in errors if e.severity == "FATAL"])
        st.metric(
            label="Fatal Errors",
            value=fatal_errors,
            delta=None
        )
    
    with col3:
        error_errors = len([e for e in errors if e.severity == "ERROR"])
        st.metric(
            label="Error Level",
            value=error_errors,
            delta=None
        )
    
    with col4:
        warnings = len([e for e in errors if e.severity == "WARNING"])
        st.metric(
            label="Warnings",
            value=warnings,
            delta=None
        )

def create_error_distribution_charts(errors: List[UVMError]):
    """Create error distribution visualizations"""
    if not errors:
        st.warning("No errors to visualize")
        return
    
    # Convert errors to DataFrame
    error_data = []
    for error in errors:
        error_data.append({
            'timestamp': error.timestamp,
            'severity': error.severity,
            'component': error.component,
            'message': error.message[:50] + '...' if len(error.message) > 50 else error.message,
            'file': error.file_path
        })
    
    df = pd.DataFrame(error_data)
    
    # Severity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Error Distribution by Severity")
        severity_counts = df['severity'].value_counts()
        fig_severity = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            color_discrete_map={
                'FATAL': '#c0392b',
                'ERROR': '#e74c3c',
                'WARNING': '#f39c12',
                'INFO': '#3498db'
            }
        )
        st.plotly_chart(fig_severity, use_container_width=True)
    
    with col2:
        st.subheader("Error Distribution by Component")
        component_counts = df['component'].value_counts().head(10)
        fig_component = px.bar(
            x=component_counts.values,
            y=component_counts.index,
            orientation='h',
            labels={'x': 'Count', 'y': 'Component'}
        )
        st.plotly_chart(fig_component, use_container_width=True)
    
    # Timeline view
    if 'timestamp' in df.columns:
        st.subheader("Error Timeline")
        # Convert timestamp to datetime if needed
        timeline_df = df.copy()
        timeline_df['time'] = pd.to_datetime(timeline_df['timestamp'], errors='coerce')
        timeline_df = timeline_df.dropna(subset=['time'])
        
        if not timeline_df.empty:
            fig_timeline = px.scatter(
                timeline_df,
                x='time',
                y='component',
                color='severity',
                hover_data=['message'],
                color_discrete_map={
                    'FATAL': '#c0392b',
                    'ERROR': '#e74c3c',
                    'WARNING': '#f39c12',
                    'INFO': '#3498db'
                }
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

def display_analysis_results(analysis_results: List[Dict[str, Any]]):
    """Display detailed analysis results"""
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    st.header("📊 Detailed Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Individual Errors", "Root Cause Summary", "Recommendations"])
    
    with tab1:
        # Individual error analysis
        for idx, result in enumerate(analysis_results):
            with st.expander(f"Error #{idx + 1} - {result.get('metadata', {}).get('error_id', 'Unknown')}", expanded=idx < 3):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Root Cause Analysis**")
                    st.write(result.get('root_cause', 'Not available'))
                    
                    st.markdown("**Component Analysis**")
                    st.write(result.get('component_analysis', 'Not available'))
                
                with col2:
                    st.markdown("**Metadata**")
                    metadata = result.get('metadata', {})
                    st.json(metadata)
                
                st.markdown("**Suggested Fixes**")
                fixes = result.get('suggested_fixes', 'Not available')
                if isinstance(fixes, list):
                    for fix in fixes:
                        st.write(f"• {fix}")
                else:
                    st.write(fixes)
                
                st.markdown("**Prevention Guidelines**")
                prevention = result.get('prevention', 'Not available')
                st.write(prevention)
    
    with tab2:
        # Root cause summary
        st.subheader("Root Cause Distribution")
        
        # Extract and count root causes
        root_causes = {}
        for result in analysis_results:
            cause = str(result.get('root_cause', 'Unknown'))[:100]
            root_causes[cause] = root_causes.get(cause, 0) + 1
        
        # Create bar chart
        if root_causes:
            fig_causes = px.bar(
                x=list(root_causes.values()),
                y=list(root_causes.keys()),
                orientation='h',
                labels={'x': 'Count', 'y': 'Root Cause'}
            )
            st.plotly_chart(fig_causes, use_container_width=True)
    
    with tab3:
        # Recommendations summary
        st.subheader("Common Recommendations")
        
        all_fixes = []
        all_preventions = []
        
        for result in analysis_results:
            fixes = result.get('suggested_fixes', [])
            if isinstance(fixes, list):
                all_fixes.extend(fixes)
            
            prevention = result.get('prevention', [])
            if isinstance(prevention, list):
                all_preventions.extend(prevention)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Most Common Fixes**")
            fix_counts = pd.Series(all_fixes).value_counts().head(10)
            for fix, count in fix_counts.items():
                st.write(f"• {fix} ({count} occurrences)")
        
        with col2:
            st.markdown("**Prevention Guidelines**")
            prevention_counts = pd.Series(all_preventions).value_counts().head(10)
            for guideline, count in prevention_counts.items():
                st.write(f"• {guideline} ({count} occurrences)")

def process_documents_for_embedding(settings_dict: Dict[str, Any], doc_paths: List[Path], code_paths: List[Path]):
    """Process documents and code files for embedding"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize processors
        doc_chunker = DocumentChunker(
            chunk_size=settings_dict['rag']['chunk_size'],
            chunk_overlap=settings_dict['rag']['chunk_overlap']
        )
        
        code_chunker = SystemVerilogChunker(
            max_chunk_size=settings_dict['rag']['chunk_size']
        )
        
        # Get model manager
        model_manager = st.session_state.model_manager
        
        chunks = []
        total_files = len(doc_paths) + len(code_paths)
        processed_files = 0
        
        # Process documents
        status_text.text("📄 Processing specification documents...")
        for doc_path in doc_paths:
            try:
                doc_chunks = doc_chunker.chunk_documents(doc_path)
                chunks.extend(doc_chunks)
                processed_files += 1
                progress_bar.progress(processed_files / total_files * 0.5)
                status_text.text(f"📄 Processing: {doc_path.name}")
            except Exception as e:
                st.warning(f"Error processing {doc_path}: {str(e)}")
        
        # Process code files
        status_text.text("💻 Processing SystemVerilog code...")
        for code_path in code_paths:
            try:
                code_chunks = code_chunker.chunk_sv_file(code_path)
                chunks.extend(code_chunks)
                processed_files += 1
                progress_bar.progress(0.5 + (processed_files - len(doc_paths)) / total_files * 0.3)
                status_text.text(f"💻 Processing: {code_path.name}")
            except Exception as e:
                st.warning(f"Error processing {code_path}: {str(e)}")
        
        # Generate embeddings
        status_text.text("🧮 Generating embeddings...")
        progress_bar.progress(0.8)
        
        if chunks:
            # Generate embeddings using model manager
            embeddings = model_manager.generate_embeddings(
                texts=[chunk.content for chunk in chunks],
                model_name=settings_dict['embedding']['model']
            )
            
            # Create or update vector store
            if st.session_state.vector_store is None:
                st.session_state.vector_store = FAISSVectorStore(
                    dimension=settings_dict['embedding']['dimension'],
                    index_type="IndexFlatIP"
                )
            
            # Add to vector store
            st.session_state.vector_store.add_documents(
                embeddings=embeddings,
                documents=[chunk.content for chunk in chunks],
                metadata=[chunk.metadata for chunk in chunks]
            )
            
            # Store chunks for visualization
            st.session_state.embedded_chunks = chunks
            
            # Update stats
            st.session_state.vector_store_stats = st.session_state.vector_store.get_stats()
            
            progress_bar.progress(1.0)
            status_text.text(f"✅ Successfully processed {len(chunks)} chunks from {total_files} files")
            
            return True, len(chunks)
        else:
            status_text.text("⚠️ No chunks generated from the selected files")
            return False, 0
            
    except Exception as e:
        st.error(f"Error during embedding: {str(e)}")
        return False, 0

def visualize_embeddings(embeddings: np.ndarray, metadata: List[Dict[str, Any]], method: str = "PCA"):
    """Visualize embeddings using dimensionality reduction"""
    if len(embeddings) == 0:
        st.warning("No embeddings to visualize")
        return
    
    # Reduce dimensions
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "UMAP":
        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
        except:
            st.error("UMAP not installed. Please install with: pip install umap-learn")
            return
    
    # Fit and transform
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create visualization dataframe
    viz_data = []
    for i, (embedding, meta) in enumerate(zip(reduced_embeddings, metadata)):
        viz_data.append({
            'x': embedding[0],
            'y': embedding[1],
            'file_type': meta.get('file_type', 'unknown'),
            'file_name': meta.get('file_name', 'unknown'),
            'chunk_idx': i,
            'content_preview': str(meta.get('content', ''))[:100] + '...'
        })
    
    df = pd.DataFrame(viz_data)
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='file_type',
        hover_data=['file_name', 'chunk_idx', 'content_preview'],
        title=f'Document Embeddings Visualization ({method})',
        labels={'x': f'{method} Component 1', 'y': f'{method} Component 2'}
    )
    
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)

def display_vector_store_stats():
    """Display vector store statistics"""
    if st.session_state.vector_store is None:
        st.info("No vector store created yet. Please process documents first.")
        return
    
    stats = st.session_state.vector_store.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", stats.get('num_documents', 0))
    
    with col2:
        st.metric("Embedding Dimension", stats.get('dimension', 0))
    
    with col3:
        st.metric("Index Type", stats.get('index_type', 'N/A'))
    
    with col4:
        st.metric("Total Vectors", stats.get('total_size', 0))
    
    # Document type distribution
    if st.session_state.embedded_chunks:
        st.subheader("Document Type Distribution")
        
        type_counts = {}
        for chunk in st.session_state.embedded_chunks:
            file_type = chunk.metadata.get('file_type', 'unknown')
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Chunks by File Type"
        )
        st.plotly_chart(fig, use_container_width=True)

def run_analysis(settings_dict: Dict[str, Any], start_time: Optional[str] = None):
    """Run the analysis with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Check if vector store exists
        if st.session_state.vector_store is None:
            st.error("Please setup embeddings first in the 'Embedding Setup' tab")
            return False, "No vector store available"
        
        # Step 1: Collect errors
        status_text.text("📂 Collecting error logs...")
        progress_bar.progress(10)
        
        collector = LogCollector(settings_dict)
        errors = []
        
        for log_dir in settings_dict['log_directories']:
            log_path = Path(log_dir)
            if log_path.exists():
                log_files = collector.collect_logs(start_time=start_time)
                for log_file in log_files:
                    file_errors = collector.extract_errors(log_file)
                    errors.extend(file_errors)
        
        st.session_state.errors = errors
        status_text.text(f"✅ Collected {len(errors)} errors")
        progress_bar.progress(40)
        
        # Step 2: Analyze errors using existing vector store
        status_text.text("🤖 Analyzing errors with AI...")
        progress_bar.progress(50)
        
        retriever = Retriever(
            vector_store=st.session_state.vector_store,
            embedder=st.session_state.model_manager
        )
        
        analyzer = Analyzer(
            llm_provider=settings_dict['llm']['provider'],
            model=settings_dict['llm']['model'],
            temperature=settings_dict['llm']['temperature'],
            max_tokens=settings_dict['llm']['max_tokens'],
            api_key=settings_dict['llm'].get('openai_api_key') or settings_dict['llm'].get('anthropic_api_key')
        )
        
        analysis_results = []
        for i, error in enumerate(errors):
            # Update progress
            progress = 50 + int((i / len(errors)) * 40)
            progress_bar.progress(progress)
            status_text.text(f"🤖 Analyzing error {i+1}/{len(errors)}...")
            
            # Retrieve context
            context = retriever.retrieve(
                query=error.message,
                k=settings_dict['rag']['retrieval_top_k']
            )
            
            # Analyze error
            analysis = analyzer.analyze_error(error, context)
            analysis_results.append(analysis)
        
        st.session_state.analysis_results = analysis_results
        
        # Step 3: Generate reports
        status_text.text("📊 Generating reports...")
        progress_bar.progress(95)
        
        report_generator = ReportGenerator()
        
        # Generate reports
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        html_report = report_generator.generate_report(
            errors=errors,
            analysis_results=analysis_results,
            output_path=report_dir / "report.html",
            format="html"
        )
        
        md_report = report_generator.generate_report(
            errors=errors,
            analysis_results=analysis_results,
            output_path=report_dir / "report.md",
            format="markdown"
        )
        
        summary_report = report_generator.generate_summary(
            errors=errors,
            analysis_results=analysis_results,
            output_path=report_dir / "summary.yaml"
        )
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        return True, "Analysis completed successfully"
        
    except Exception as e:
        return False, f"Error during analysis: {str(e)}"

def generate_html_report(errors_df: pd.DataFrame) -> str:
    """Generate HTML report from error analysis"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>UVM Error Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .summary { margin-bottom: 20px; }
            .error-table { width: 100%; border-collapse: collapse; }
            .error-table th, .error-table td { 
                border: 1px solid #ddd; 
                padding: 8px; 
                text-align: left; 
            }
            .error-table th { background-color: #f5f5f5; }
            .fatal { color: #c0392b; }
            .error { color: #e74c3c; }
            .warning { color: #f1c40f; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>UVM Error Analysis Report</h1>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Errors: {total_errors}</p>
            <p>Fatal Errors: {fatal_count}</p>
            <p>Errors: {error_count}</p>
            <p>Warnings: {warning_count}</p>
        </div>
        
        <h2>Error Details</h2>
        <table class="error-table">
            <tr>
                <th>Timestamp</th>
                <th>Severity</th>
                <th>Component</th>
                <th>Message</th>
            </tr>
            {error_rows}
        </table>
    </body>
    </html>
    """
    
    # Generate error rows
    error_rows = ""
    for _, row in errors_df.iterrows():
        severity_class = row['severity'].lower()
        error_rows += f"""
        <tr>
            <td>{row['timestamp']}</td>
            <td class="{severity_class}">{row['severity']}</td>
            <td>{row['component']}</td>
            <td>{row['message']}</td>
        </tr>
        """
    
    # Fill template
    report_html = html_template.format(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_errors=len(errors_df),
        fatal_count=len(errors_df[errors_df['severity'] == 'FATAL']),
        error_count=len(errors_df[errors_df['severity'] == 'ERROR']),
        warning_count=len(errors_df[errors_df['severity'] == 'WARNING']),
        error_rows=error_rows
    )
    
    return report_html

def main():
    st.title("🔍 UVM Debug Agent")
    st.markdown("AI-powered Universal Verification Methodology error analysis and debugging tool")
    
    # Create sidebar
    settings_dict = create_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📚 Embedding Setup", "🔍 Analysis", "📊 Results", "📄 Reports"])
    
    with tab1:
        st.header("Dashboard")
        
        # Vector Store Stats
        st.subheader("Vector Store Status")
        display_vector_store_stats()
        
        # Quick stats
        if st.session_state.errors:
            st.subheader("Error Analysis Status")
            display_metrics(st.session_state.errors, st.session_state.analysis_results or [])
            
            # Error distribution charts
            st.header("Error Distribution")
            create_error_distribution_charts(st.session_state.errors)
        else:
            st.info("No analysis data available. First setup embeddings, then run an analysis to see results.")
    
    with tab2:
        st.header("📚 Document Embedding Setup")
        st.markdown("Configure and process documents for the knowledge base")
        
        # Document selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Specification Documents")
            doc_dir = st.text_input(
                "Specification Directory",
                value="data/specs",
                help="Directory containing UVM specification documents"
            )
            
            doc_extensions = st.multiselect(
                "Document File Types",
                options=['.pdf', '.md', '.txt', '.docx'],
                default=['.pdf', '.md', '.txt'],
                help="Select file types to process"
            )
            
            # Preview documents
            if st.button("🔍 Preview Documents", key="preview_docs"):
                doc_path = Path(doc_dir)
                if doc_path.exists():
                    doc_files = []
                    for ext in doc_extensions:
                        doc_files.extend(doc_path.glob(f"**/*{ext}"))
                    
                    if doc_files:
                        st.success(f"Found {len(doc_files)} document files")
                        with st.expander("Document Files", expanded=True):
                            for file in doc_files[:10]:  # Show first 10
                                st.text(f"📄 {file.relative_to(doc_path)}")
                    else:
                        st.warning("No documents found with selected extensions")
                else:
                    st.error(f"Directory not found: {doc_dir}")
        
        with col2:
            st.subheader("💻 SystemVerilog Code")
            code_dir = st.text_input(
                "Testbench Directory",
                value="data/testbench",
                help="Directory containing SystemVerilog testbench files"
            )
            
            code_extensions = st.multiselect(
                "Code File Types",
                options=['.sv', '.svh', '.v', '.vh'],
                default=['.sv', '.svh'],
                help="Select code file types to process"
            )
            
            # Preview code files
            if st.button("🔍 Preview Code Files", key="preview_code"):
                code_path = Path(code_dir)
                if code_path.exists():
                    code_files = []
                    for ext in code_extensions:
                        code_files.extend(code_path.glob(f"**/*{ext}"))
                    
                    if code_files:
                        st.success(f"Found {len(code_files)} code files")
                        with st.expander("Code Files", expanded=True):
                            for file in code_files[:10]:  # Show first 10
                                st.text(f"💻 {file.relative_to(code_path)}")
                    else:
                        st.warning("No code files found with selected extensions")
                else:
                        st.error(f"Directory not found: {code_dir}")
        
        st.markdown("---")
        
        # Embedding configuration
        st.subheader("🔧 Embedding Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chunk_method = st.selectbox(
                "Chunking Method",
                options=["Fixed Size", "Sentence-based", "Semantic"],
                help="Method for splitting documents into chunks"
            )
        
        with col2:
            min_chunk_size = st.number_input(
                "Min Chunk Size",
                min_value=50,
                max_value=500,
                value=100,
                help="Minimum size for a chunk"
            )
        
        with col3:
            embedding_batch = st.number_input(
                "Embedding Batch Size",
                min_value=1,
                max_value=100,
                value=32,
                help="Number of chunks to process at once"
            )
        
        # Process button
        if st.button("🚀 Process Documents and Generate Embeddings", type="primary", use_container_width=True):
            # Collect files
            doc_path = Path(doc_dir)
            code_path = Path(code_dir)
            
            doc_files = []
            code_files = []
            
            if doc_path.exists():
                for ext in doc_extensions:
                    doc_files.extend(doc_path.glob(f"**/*{ext}"))
            
            if code_path.exists():
                for ext in code_extensions:
                    code_files.extend(code_path.glob(f"**/*{ext}"))
            
            if doc_files or code_files:
                success, num_chunks = process_documents_for_embedding(
                    settings_dict,
                    doc_files,
                    code_files
                )
                
                if success:
                    st.success(f"✅ Successfully processed {num_chunks} chunks")
                    st.balloons()
                else:
                    st.error("Failed to process documents")
            else:
                st.error("No files found to process")
        
        # Vector Store Management
        st.markdown("---")
        st.subheader("💾 Vector Store Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Vector Store"):
                if st.session_state.vector_store:
                    try:
                        save_path = Path("data/vectorstore")
                        st.session_state.vector_store.save(str(save_path))
                        st.success(f"Vector store saved to {save_path}")
                    except Exception as e:
                        st.error(f"Error saving vector store: {str(e)}")
                else:
                    st.warning("No vector store to save")
        
        with col2:
            if st.button("📂 Load Vector Store"):
                try:
                    load_path = Path("data/vectorstore")
                    if load_path.exists():
                        st.session_state.vector_store = FAISSVectorStore.load(str(load_path))
                        st.session_state.vector_store_stats = st.session_state.vector_store.get_stats()
                        st.success("Vector store loaded successfully")
                    else:
                        st.error(f"Vector store not found at {load_path}")
                except Exception as e:
                    st.error(f"Error loading vector store: {str(e)}")
        
        with col3:
            if st.button("🗑️ Clear Vector Store"):
                if st.session_state.vector_store:
                    st.session_state.vector_store.clear()
                    st.session_state.embedded_chunks = []
                    st.session_state.vector_store_stats = {}
                    st.success("Vector store cleared")
                else:
                    st.info("No vector store to clear")
        
        # Embedding Visualization
        if st.session_state.vector_store and st.session_state.embedded_chunks:
            st.markdown("---")
            st.subheader("📊 Embedding Visualization")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                viz_method = st.selectbox(
                    "Visualization Method",
                    options=["PCA", "t-SNE", "UMAP"],
                    help="Method for reducing dimensions"
                )
                
                if st.button("🎨 Visualize Embeddings"):
                    with col2:
                        with st.spinner(f"Generating {viz_method} visualization..."):
                            # Get embeddings from chunks
                            embeddings = []
                            metadata = []
                            
                            # Generate embeddings for visualization
                            model_manager = st.session_state.model_manager
                            chunk_texts = [chunk.content for chunk in st.session_state.embedded_chunks[:1000]]  # Limit to 1000
                            
                            embeddings = model_manager.generate_embeddings(
                                texts=chunk_texts,
                                model_name=settings_dict['embedding']['model']
                            )
                            
                            metadata = [chunk.metadata for chunk in st.session_state.embedded_chunks[:1000]]
                            
                            visualize_embeddings(embeddings, metadata, viz_method)
            
            # Search in Vector Store
            st.markdown("---")
            st.subheader("🔍 Test Vector Store Search")
            
            test_query = st.text_input(
                "Test Query",
                placeholder="Enter a test query to search the vector store...",
                help="Test the retrieval capability of your vector store"
            )
            
            if test_query:
                if st.button("Search"):
                    with st.spinner("Searching..."):
                        retriever = Retriever(
                            vector_store=st.session_state.vector_store,
                            embedder=st.session_state.model_manager
                        )
                        
                        results = retriever.retrieve(
                            query=test_query,
                            k=5
                        )
                        
                        if results:
                            st.success(f"Found {len(results)} relevant documents")
                            
                            for i, result in enumerate(results):
                                with st.expander(f"Result {i+1} (Score: {result['score']:.3f})"):
                                    st.markdown("**Metadata:**")
                                    st.json(result['metadata'])
                                    st.markdown("**Content Preview:**")
                                    st.text(result['content'][:500] + "...")
                        else:
                            st.warning("No results found")
    
    with tab3:
        st.header("Run Analysis")
        
        # Check if embeddings are ready
        if st.session_state.vector_store is None:
            st.warning("⚠️ Please setup document embeddings first in the 'Embedding Setup' tab before running analysis.")
            st.stop()
        
        # Analysis Configuration
        st.subheader("🔧 Analysis Configuration")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Time range selection
            use_time_filter = st.checkbox("Filter by time range")
            
            if use_time_filter:
                col_date, col_time = st.columns(2)
                with col_date:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=7)
                    )
                with col_time:
                    start_time = st.time_input("Start Time", value=datetime.now().time())
                
                start_datetime = datetime.combine(start_date, start_time)
            else:
                start_datetime = None
        
        with col2:
            st.markdown("### Quick Actions")
            if st.button("🔄 Clear Cache", help="Clear all cached data"):
                st.session_state.clear()
                st.success("Cache cleared!")
        
        # Show current vector store status
        if st.session_state.vector_store:
            stats = st.session_state.vector_store_stats
            st.info(f"📚 Vector Store Ready: {stats.get('num_documents', 0)} documents indexed")
        
        # Analysis Settings
        st.subheader("⚙️ Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_depth = st.select_slider(
                "Analysis Depth",
                options=["Basic", "Standard", "Detailed", "Comprehensive"],
                value="Standard",
                help="Level of detail in error analysis"
            )
            
            include_suggestions = st.checkbox(
                "Include Fix Suggestions",
                value=True,
                help="Generate suggested fixes for each error"
            )
        
        with col2:
            include_prevention = st.checkbox(
                "Include Prevention Guidelines",
                value=True,
                help="Generate prevention guidelines for each error type"
            )
            
            generate_reports = st.checkbox(
                "Generate Reports",
                value=True,
                help="Generate HTML and Markdown reports"
            )
        
        # Display current configuration
        with st.expander("Current Configuration", expanded=False):
            st.json(settings_dict)
        
        # Analysis Execution
        st.markdown("---")
        st.subheader("🚀 Execute Analysis")
        
        if st.button("Start Analysis", type="primary", use_container_width=True):
            # Validate settings
            if settings_dict['llm']['provider'] == 'openai' and not settings_dict['llm']['openai_api_key']:
                st.error("Please provide OpenAI API key in the sidebar")
            elif settings_dict['llm']['provider'] == 'anthropic' and not settings_dict['llm']['anthropic_api_key']:
                st.error("Please provide Anthropic API key in the sidebar")
            else:
                # Run analysis
                success, message = run_analysis(
                    settings_dict,
                    start_time=start_datetime.strftime("%Y-%m-%d %H:%M:%S") if start_datetime else None
                )
                
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.error(message)
    
    with tab4:
        st.header("Analysis Results")
        
        if st.session_state.analysis_results:
            display_analysis_results(st.session_state.analysis_results)
        else:
            st.info("No analysis results available. Run an analysis first.")
    
    with tab5:
        st.header("Generated Reports")
        
        report_dir = Path("reports")
        if report_dir.exists():
            report_files = list(report_dir.glob("*"))
            
            if report_files:
                for report_file in report_files:
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.text(f"📄 {report_file.name}")
                    
                    with col2:
                        st.text(f"Size: {report_file.stat().st_size // 1024} KB")
                    
                    with col3:
                        with open(report_file, 'rb') as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=report_file.name,
                                mime="text/html" if report_file.suffix == ".html" else "text/plain"
                            )
                
                # Preview HTML report
                if (report_dir / "report.html").exists():
                    st.subheader("HTML Report Preview")
                    with open(report_dir / "report.html", 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600, scrolling=True)
            else:
                st.info("No reports generated yet.")
        else:
            st.info("No reports directory found.")

if __name__ == "__main__":
    main() 