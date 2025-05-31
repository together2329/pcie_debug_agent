"""
Streamlit application with Hybrid LLM support for PCIe Debug Agent
"""

import streamlit as st
import yaml
from pathlib import Path
import sys
import os
import time

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config.settings import Settings, load_settings as load_config
from src.rag.enhanced_rag_engine_hybrid import EnhancedRAGEngineHybrid, RAGQuery
from src.vectorstore.faiss_store import FAISSVectorStore
from src.models.model_manager import ModelManager
from src.models.hybrid_llm_provider import HybridLLMProvider

def load_settings() -> Settings:
    """Load configuration"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
    
    try:
        if config_path.exists():
            return load_config(config_path)
        else:
            st.warning("Configuration file not found. Using default settings.")
            return load_config(None)
    except Exception as e:
        st.error(f"Failed to load settings: {str(e)}")
        st.stop()

def initialize_components(settings: Settings):
    """Initialize all components"""
    try:
        # Vector store
        vector_store = FAISSVectorStore(
            index_path=str(settings.vector_store.index_path),
            index_type=settings.vector_store.index_type,
            dimension=settings.embedding.dimension
        )
        
        # Model manager
        model_manager = ModelManager()
        
        # Load embedding model if local
        if settings.embedding.provider == "local":
            model_manager.load_embedding_model(settings.embedding.model)
        
        # Hybrid RAG engine
        rag_engine = EnhancedRAGEngineHybrid(
            vector_store=vector_store,
            model_manager=model_manager,
            models_dir=str(settings.local_llm.models_dir),
            enable_cache=True
        )
        
        return rag_engine
        
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        st.stop()

def main():
    """Main application with Hybrid LLM support"""
    # Page configuration
    st.set_page_config(
        page_title="PCIe Debug Agent - Hybrid LLM",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        .analysis-card {
            padding: 1rem;
            border-radius: 0.5rem;
            background: #f0f2f6;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load settings
    settings = load_settings()
    
    # Initialize RAG engine
    if 'rag_engine' not in st.session_state:
        with st.spinner("Initializing Hybrid LLM system..."):
            st.session_state.rag_engine = initialize_components(settings)
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 PCIe Debug Agent")
        st.caption("Hybrid LLM Edition v1.1")
        
        # Model status
        status = st.session_state.rag_engine.hybrid_provider.get_model_status()
        
        st.markdown("### 🎯 Model Status")
        col1, col2 = st.columns(2)
        with col1:
            if status['llama']['available']:
                st.success("🦙 Llama Ready")
            else:
                st.error("🦙 Llama Error")
        with col2:
            if status['deepseek']['available']:
                st.success("🤖 DeepSeek Ready")
            else:
                st.warning("🤖 DeepSeek N/A")
        
        st.divider()
        
        # Analysis mode selector
        st.markdown("### ⚡ Analysis Mode")
        analysis_mode = st.radio(
            "Select analysis type:",
            ["🚀 Quick (Llama)", "🔬 Detailed (DeepSeek)", "🤖 Auto"],
            index=2,
            help="Quick: Fast interactive analysis\nDetailed: Comprehensive investigation\nAuto: Let AI decide"
        )
        
        # Map display names to values
        mode_map = {
            "🚀 Quick (Llama)": "quick",
            "🔬 Detailed (DeepSeek)": "detailed",
            "🤖 Auto": "auto"
        }
        selected_mode = mode_map[analysis_mode]
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            max_response_time = st.slider(
                "Max Response Time (seconds)",
                min_value=10,
                max_value=180,
                value=30,
                step=10
            )
            
            context_window = st.slider(
                "Context Documents",
                min_value=1,
                max_value=10,
                value=5
            )
            
            min_similarity = st.slider(
                "Min Similarity Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        
        st.divider()
        
        # Performance metrics
        metrics = st.session_state.rag_engine.get_performance_metrics()
        st.markdown("### 📊 Performance")
        st.metric("Total Queries", metrics['queries_processed'])
        st.metric("Avg Response", f"{metrics['average_response_time']:.1f}s")
        st.metric("Cache Hits", f"{metrics['cache_hit_rate']:.0%}")
        
        # Model usage
        st.markdown("### 🤖 Model Usage")
        llama_usage = metrics['model_usage']['llama']
        deepseek_usage = metrics['model_usage']['deepseek']
        total_usage = llama_usage + deepseek_usage
        
        if total_usage > 0:
            st.progress(llama_usage / total_usage, text=f"Llama: {llama_usage}")
            st.progress(deepseek_usage / total_usage, text=f"DeepSeek: {deepseek_usage}")
    
    # Main content
    st.title("🔍 PCIe Error Analysis with Hybrid AI")
    st.markdown("Intelligent model selection for optimal debugging experience")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Analysis", "🔎 Search", "📈 Analytics"])
    
    with tab1:
        # Chat interface
        st.markdown("### Ask about PCIe errors")
        
        # Error log input
        with st.expander("📋 Paste Error Log (optional)", expanded=False):
            error_log = st.text_area(
                "Error log content:",
                height=150,
                placeholder="[10:15:30] PCIe: Link training failed...\n[10:15:31] PCIe: Device timeout..."
            )
        
        # Query input
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Your question:",
                placeholder="What caused the link training failure? How to fix it?"
            )
        with col2:
            analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
        
        # Process query
        if analyze_button and query:
            # Create query with error log if provided
            full_query = query
            if error_log:
                full_query = f"{query}\n\nError Log:\n{error_log}"
            
            # Create RAG query
            rag_query = RAGQuery(
                query=full_query,
                analysis_type=selected_mode,
                max_response_time=max_response_time,
                context_window=context_window,
                min_similarity=min_similarity
            )
            
            # Show progress
            with st.spinner(f"Analyzing with {selected_mode} mode..."):
                start_time = time.time()
                response = st.session_state.rag_engine.query(rag_query)
                elapsed_time = time.time() - start_time
            
            # Display results
            st.markdown("### 📊 Analysis Results")
            
            # Model info card
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Used", response.model_used.split('-')[0])
            with col2:
                st.metric("Response Time", f"{response.response_time:.1f}s")
            with col3:
                st.metric("Confidence", f"{response.confidence:.0%}")
            with col4:
                st.metric("Analysis Type", response.analysis_type)
            
            # Main response
            st.markdown("### 💡 Answer")
            st.markdown(f'<div class="analysis-card">{response.answer}</div>', unsafe_allow_html=True)
            
            # Sources
            if response.sources:
                with st.expander(f"📚 Sources ({len(response.sources)} documents)"):
                    for i, source in enumerate(response.sources, 1):
                        st.markdown(f"**Source {i}** - Similarity: {source.get('similarity', 0):.2%}")
                        st.text(source['content'][:300] + "...")
                        st.divider()
            
            # Metadata
            if response.metadata.get("fallback_used"):
                st.warning("⚠️ Fallback model was used due to primary model unavailability")
            
            if response.metadata.get("error"):
                st.error(f"Error during analysis: {response.metadata['error']}")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        example_cols = st.columns(3)
        
        examples = [
            ("🚀 Quick", "What is a PCIe TLP error?"),
            ("🔬 Detailed", "Analyze correlation between thermal and link errors"),
            ("🤖 Auto", "Debug intermittent device disconnection")
        ]
        
        for col, (mode, example) in zip(example_cols, examples):
            with col:
                if st.button(f"{mode}: {example[:20]}...", use_container_width=True):
                    st.session_state.example_query = example
                    st.rerun()
    
    with tab2:
        # Search interface
        st.markdown("### 🔎 Search PCIe Knowledge Base")
        
        search_query = st.text_input(
            "Search for:",
            placeholder="LTSSM recovery process"
        )
        
        search_cols = st.columns([3, 1, 1])
        with search_cols[1]:
            num_results = st.number_input("Results", min_value=1, max_value=20, value=5)
        with search_cols[2]:
            search_button = st.button("Search", type="primary", use_container_width=True)
        
        if search_button and search_query:
            with st.spinner("Searching..."):
                # Use retriever directly
                retriever = st.session_state.rag_engine.retriever
                results = retriever.retrieve(
                    query=search_query,
                    k=num_results,
                    min_similarity=min_similarity
                )
            
            st.markdown(f"### Found {len(results)} results")
            
            for i, doc in enumerate(results, 1):
                with st.expander(f"Result {i} - Similarity: {doc.get('similarity', 0):.2%}"):
                    st.markdown(f"**Source:** {doc.get('metadata', {}).get('source', 'Unknown')}")
                    st.markdown(doc['content'])
    
    with tab3:
        # Analytics
        st.markdown("### 📈 System Analytics")
        
        metrics = st.session_state.rag_engine.get_performance_metrics()
        
        # Analysis type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Analysis Types Used")
            for atype, count in metrics['analysis_types'].items():
                st.metric(atype.capitalize(), count)
        
        with col2:
            st.markdown("#### Model Performance")
            st.metric("🦙 Llama Queries", metrics['model_usage']['llama'])
            st.metric("🤖 DeepSeek Queries", metrics['model_usage']['deepseek'])
        
        # Performance over time (placeholder)
        st.markdown("#### Response Time Trend")
        st.line_chart({"Average Response Time": [metrics['average_response_time']]})
        
        # Clear cache button
        if st.button("🗑️ Clear Cache"):
            st.session_state.rag_engine.clear_cache()
            st.success("Cache cleared!")
            st.rerun()
    
    # Handle example query
    if 'example_query' in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
        st.rerun()

if __name__ == "__main__":
    main()