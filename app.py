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
    
    # Embedding Model
    embedding_models = model_manager.embedding_manager.SUPPORTED_MODELS
    selected_embedding = st.sidebar.selectbox(
        "Embedding Model",
        options=list(embedding_models.keys()),
        format_func=lambda x: embedding_models[x].name,
        help="Choose the embedding model for document vectorization"
    )
    
    if selected_embedding:
        model_info = embedding_models[selected_embedding]
        st.sidebar.info(f"{model_info.description}\nMemory: {model_info.memory_usage}")
    
    # LLM Provider
    llm_providers = model_manager.llm_manager.SUPPORTED_MODELS
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        options=list(llm_providers.keys()),
        format_func=lambda x: x.capitalize()
    )
    
    # LLM Model
    llm_models = llm_providers[llm_provider]
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        options=list(llm_models.keys()),
        format_func=lambda x: llm_models[x].name
    )
    
    if llm_model:
        model_info = llm_models[llm_model]
        st.sidebar.info(f"{model_info.description}\nContext: {model_info.context_window:,} tokens")
    
    # API Keys
    st.sidebar.header("API Configuration")
    
    if llm_provider == "openai":
        openai_api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key"
        )
    elif llm_provider == "anthropic":
        anthropic_api_key = st.sidebar.text_input(
            "Anthropic API Key",
            type="password",
            help="Your Anthropic API key"
        )
    else:
        openai_api_key = None
        anthropic_api_key = None
    
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
            "dimension": embedding_models[selected_embedding].dimension
        },
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "openai_api_key": openai_api_key if llm_provider == "openai" else None,
            "anthropic_api_key": anthropic_api_key if llm_provider == "anthropic" else None
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

def run_analysis(settings_dict: Dict[str, Any], start_time: Optional[str] = None):
    """Run the analysis with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
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
        progress_bar.progress(30)
        
        # Step 2: Process documents
        status_text.text("📄 Processing documents and code...")
        progress_bar.progress(40)
        
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
        
        # Process documents
        chunks = []
        data_dir = Path("data")
        
        # Process specs
        for spec_file in data_dir.glob("specs/**/*"):
            if spec_file.suffix in ['.pdf', '.md', '.txt']:
                doc_chunks = doc_chunker.chunk_documents(spec_file)
                chunks.extend(doc_chunks)
        
        # Process code
        for code_file in data_dir.glob("testbench/**/*.sv"):
            code_chunks = code_chunker.chunk_sv_file(code_file)
            chunks.extend(code_chunks)
        
        status_text.text(f"✅ Processed {len(chunks)} document chunks")
        progress_bar.progress(50)
        
        # Step 3: Generate embeddings
        status_text.text("🧮 Generating embeddings...")
        progress_bar.progress(60)
        
        # Generate embeddings using model manager
        embeddings = model_manager.generate_embeddings(
            texts=[chunk.content for chunk in chunks],
            model_name=settings_dict['embedding']['model']
        )
        
        # Step 4: Create vector store
        status_text.text("💾 Creating vector store...")
        progress_bar.progress(70)
        
        vector_store = FAISSVectorStore(
            dimension=settings_dict['embedding']['dimension'],
            index_type="IndexFlatIP"
        )
        
        vector_store.add_documents(
            embeddings=embeddings,
            documents=[chunk.content for chunk in chunks],
            metadata=[chunk.metadata for chunk in chunks]
        )
        
        st.session_state.vector_store = vector_store
        
        # Step 5: Analyze errors
        status_text.text("🤖 Analyzing errors with AI...")
        progress_bar.progress(80)
        
        retriever = Retriever(
            vector_store=vector_store,
            embedder=model_manager
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
            progress = 80 + int((i / len(errors)) * 15)
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
        
        # Step 6: Generate reports
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

def main():
    st.title("🔍 UVM Debug Agent")
    st.markdown("AI-powered Universal Verification Methodology error analysis and debugging tool")
    
    # Create sidebar
    settings_dict = create_sidebar()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🔍 Analysis", "📊 Results", "📄 Reports"])
    
    with tab1:
        st.header("Dashboard")
        
        # Quick stats
        if st.session_state.errors:
            display_metrics(st.session_state.errors, st.session_state.analysis_results or [])
            
            # Error distribution charts
            st.header("Error Distribution")
            create_error_distribution_charts(st.session_state.errors)
        else:
            st.info("No analysis data available. Run an analysis to see results.")
    
    with tab2:
        st.header("Run Analysis")
        
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
        
        # Run analysis button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
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
        
        # Display current configuration
        with st.expander("Current Configuration", expanded=False):
            st.json(settings_dict)
    
    with tab3:
        st.header("Analysis Results")
        
        if st.session_state.analysis_results:
            display_analysis_results(st.session_state.analysis_results)
        else:
            st.info("No analysis results available. Run an analysis first.")
    
    with tab4:
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