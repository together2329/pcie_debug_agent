"""
Enhanced Streamlit application for the PCIe Debug Agent
"""

import streamlit as st
import yaml
from pathlib import Path
import sys
import os
from datetime import datetime
import json

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import Settings, load_settings
from src.rag.enhanced_rag_engine import EnhancedRAGEngine
from src.ui.interactive_chat import InteractiveChatInterface
from src.ui.semantic_search import SemanticSearchInterface

# Page config must be first Streamlit command
st.set_page_config(
    page_title="PCIe Debug Agent",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/pcie_debug_agent',
        'Report a bug': "https://github.com/yourusername/pcie_debug_agent/issues",
        'About': "# PCIe Debug Agent\nAn intelligent tool for PCIe debugging and analysis"
    }
)

# Custom CSS for better UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Custom theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --background-color: #0e1117;
        --text-color: #fafafa;
        --border-color: #262730;
    }
    
    /* Improved sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
        padding: 2rem 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    /* Card-like containers */
    .stat-card {
        background-color: #262730;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }
    
    /* Improved metric styling */
    [data-testid="metric-container"] {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        background-color: transparent;
        border-radius: 0.5rem;
        color: #888;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #262730;
        border-radius: 0.5rem;
        padding: 0.75rem;
        border: 1px solid var(--border-color);
    }
    
    /* Success/Error/Warning messages */
    .stAlert {
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a4a4a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5a5a5a;
    }
    </style>
""", unsafe_allow_html=True)

class PCIeDebugAgentApp:
    """Main application class for PCIe Debug Agent"""
    
    def __init__(self):
        self.initialize_session_state()
        self.settings = self.load_settings()
        self.setup_rag_engine()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.current_tab = "Chat"
            st.session_state.theme = "dark"
            st.session_state.notifications = []
            st.session_state.last_activity = datetime.now()
    
    def load_settings(self) -> Settings:
        """Load configuration settings"""
        config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
        
        # Use the new load_settings function which handles env vars properly
        try:
            if config_path.exists():
                return load_settings(config_path)
            else:
                # Create default settings with env vars
                st.warning("Configuration file not found. Using default settings with environment variables.")
                return load_settings(None)
        except Exception as e:
            self.show_error("Failed to load settings", str(e))
            st.stop()
    
    def setup_rag_engine(self):
        """Initialize RAG engine with error handling"""
        if 'rag_engine' not in st.session_state:
            try:
                with st.spinner("🚀 Initializing PCIe Debug Agent..."):
                    st.session_state.rag_engine = EnhancedRAGEngine(self.settings)
                self.show_success("System initialized successfully!")
            except Exception as e:
                self.show_error("Failed to initialize RAG engine", str(e))
                st.stop()
    
    def show_notification(self, message: str, type: str = "info"):
        """Show a notification with automatic dismissal"""
        notification = {
            'message': message,
            'type': type,
            'timestamp': datetime.now()
        }
        st.session_state.notifications.append(notification)
        
        # Keep only recent notifications
        st.session_state.notifications = [
            n for n in st.session_state.notifications
            if (datetime.now() - n['timestamp']).seconds < 10
        ]
    
    def show_success(self, message: str, details: str = None):
        """Show success message"""
        if details:
            st.success(f"✅ {message}\n\n{details}")
        else:
            st.success(f"✅ {message}")
    
    def show_error(self, message: str, details: str = None):
        """Show error message"""
        if details:
            st.error(f"❌ {message}\n\n```\n{details}\n```")
        else:
            st.error(f"❌ {message}")
    
    def show_warning(self, message: str, details: str = None):
        """Show warning message"""
        if details:
            st.warning(f"⚠️ {message}\n\n{details}")
        else:
            st.warning(f"⚠️ {message}")
    
    def render_sidebar(self):
        """Render enhanced sidebar"""
        with st.sidebar:
            # Logo and title
            st.markdown("""
                <div style='text-align: center; padding: 1rem 0;'>
                    <h1 style='color: #1f77b4; margin: 0;'>🔧 PCIe Debug Agent</h1>
                    <p style='color: #888; font-size: 0.9rem; margin: 0.5rem 0;'>
                        Intelligent PCIe Debugging Assistant
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Navigation
            st.subheader("📍 Navigation")
            tabs = ["💬 Chat", "🔍 Search", "📊 Analytics", "⚙️ Settings", "📚 Help"]
            
            selected_tab = st.radio(
                "Select Mode",
                tabs,
                label_visibility="collapsed",
                key="navigation_radio"
            )
            
            st.session_state.current_tab = selected_tab.split()[1]  # Remove emoji
            
            st.divider()
            
            # System Status
            self.render_system_status()
            
            st.divider()
            
            # Quick Actions
            self.render_quick_actions()
    
    def render_system_status(self):
        """Render system status in sidebar"""
        st.subheader("📈 System Status")
        
        # Create status card
        status_html = f"""
        <div class='stat-card'>
            <h4 style='margin: 0 0 1rem 0; color: #1f77b4;'>🔧 Configuration</h4>
            <p style='margin: 0.5rem 0;'><b>Embedding:</b> {self.settings.embedding_model}</p>
            <p style='margin: 0.5rem 0;'><b>LLM:</b> {self.settings.llm_provider}</p>
            <p style='margin: 0.5rem 0;'><b>Chunk Size:</b> {self.settings.chunk_size}</p>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)
        
        # Performance metrics
        if hasattr(st.session_state, 'rag_engine'):
            metrics = st.session_state.rag_engine.get_metrics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Queries",
                    metrics['queries_processed'],
                    delta=f"+{metrics.get('queries_today', 0)} today"
                )
            with col2:
                st.metric(
                    "Avg Time",
                    f"{metrics['average_response_time']:.1f}s",
                    delta=f"{metrics.get('time_change', 0):.1f}s"
                )
            
            # Cache efficiency
            cache_rate = (metrics['cache_hits'] / max(metrics['queries_processed'], 1)) * 100
            st.metric(
                "Cache Efficiency",
                f"{cache_rate:.1f}%",
                delta=f"{metrics.get('cache_change', 0):.1f}%"
            )
    
    def render_quick_actions(self):
        """Render quick action buttons"""
        st.subheader("⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
            
            if st.button("📥 Import", use_container_width=True):
                self.show_import_dialog()
        
        with col2:
            if st.button("📤 Export", use_container_width=True):
                self.show_export_dialog()
            
            if st.button("🗑️ Clear", use_container_width=True):
                self.clear_session_data()
    
    def render_main_content(self):
        """Render main content area based on selected tab"""
        # Notification area
        if st.session_state.notifications:
            for notification in st.session_state.notifications:
                if notification['type'] == 'success':
                    st.success(notification['message'])
                elif notification['type'] == 'error':
                    st.error(notification['message'])
                else:
                    st.info(notification['message'])
        
        # Tab content
        if st.session_state.current_tab == "Chat":
            self.render_chat_interface()
        elif st.session_state.current_tab == "Search":
            self.render_search_interface()
        elif st.session_state.current_tab == "Analytics":
            self.render_analytics()
        elif st.session_state.current_tab == "Settings":
            self.render_settings()
        elif st.session_state.current_tab == "Help":
            self.render_help()
    
    def render_chat_interface(self):
        """Render enhanced chat interface"""
        chat_interface = InteractiveChatInterface()
        
        # Add header with context
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title("💬 PCIe Debug Assistant")
        with col2:
            active_context = st.selectbox(
                "Context",
                ["General", "Error Analysis", "Performance", "Configuration"],
                label_visibility="collapsed"
            )
        with col3:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Render chat
        chat_interface.render()
    
    def render_search_interface(self):
        """Render enhanced search interface"""
        search_interface = SemanticSearchInterface()
        
        # Add header with filters
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.title("🔍 Semantic Search")
        with col2:
            file_filter = st.multiselect(
                "File Types",
                [".log", ".txt", ".v", ".sv", ".list"],
                default=[".log", ".txt"],
                label_visibility="collapsed"
            )
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=datetime.now(),
                label_visibility="collapsed"
            )
        
        st.markdown("---")
        
        # Render search
        search_interface.render()
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.title("📊 Analytics Dashboard")
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Documents",
                "1,234",
                delta="+12 today",
                help="Total documents in the knowledge base"
            )
        
        with col2:
            st.metric(
                "Total Queries",
                "567",
                delta="+45 this week",
                help="Total queries processed"
            )
        
        with col3:
            st.metric(
                "Avg Confidence",
                "87.3%",
                delta="+2.1%",
                help="Average confidence score"
            )
        
        with col4:
            st.metric(
                "Response Time",
                "1.2s",
                delta="-0.3s",
                help="Average response time"
            )
        
        st.markdown("---")
        
        # Charts
        tab1, tab2, tab3 = st.tabs(["📈 Usage Trends", "🎯 Performance", "📝 Logs"])
        
        with tab1:
            st.subheader("Query Volume Over Time")
            # Add placeholder chart
            st.info("Query volume chart will be displayed here")
        
        with tab2:
            st.subheader("Performance Metrics")
            # Add placeholder chart
            st.info("Performance metrics will be displayed here")
        
        with tab3:
            st.subheader("Recent Activity Logs")
            # Add placeholder logs
            st.info("Activity logs will be displayed here")
    
    def render_settings(self):
        """Render enhanced settings interface"""
        st.title("⚙️ Settings")
        
        # Settings tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model", "📊 RAG", "🎨 Interface", "🔐 Advanced"])
        
        with tab1:
            self.render_model_settings()
        
        with tab2:
            self.render_rag_settings()
        
        with tab3:
            self.render_interface_settings()
        
        with tab4:
            self.render_advanced_settings()
    
    def render_model_settings(self):
        """Render model configuration settings"""
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Embedding Model")
            embedding_model = st.selectbox(
                "Model",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=0,
                help="Choose the embedding model for document vectorization"
            )
            
            embedding_dim = st.number_input(
                "Embedding Dimension",
                min_value=128,
                max_value=4096,
                value=1536,
                step=128,
                help="Dimension of the embedding vectors"
            )
        
        with col2:
            st.markdown("#### Language Model")
            llm_provider = st.selectbox(
                "Provider",
                ["openai", "anthropic", "ollama", "custom"],
                index=0,
                help="Choose the LLM provider"
            )
            
            llm_model = st.selectbox(
                "Model",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
                index=0,
                help="Choose the language model"
            )
        
        if st.button("💾 Save Model Settings", type="primary"):
            self.save_settings({
                'embedding_model': embedding_model,
                'embedding_dim': embedding_dim,
                'llm_provider': llm_provider,
                'llm_model': llm_model
            })
    
    def render_rag_settings(self):
        """Render RAG configuration settings"""
        st.subheader("RAG Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Chunking")
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=self.settings.chunk_size,
                step=50,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=self.settings.chunk_overlap,
                step=25,
                help="Overlap between consecutive chunks"
            )
        
        with col2:
            st.markdown("#### Retrieval")
            top_k = st.slider(
                "Top K Results",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of documents to retrieve"
            )
            
            min_similarity = st.slider(
                "Min Similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum similarity threshold"
            )
        
        if st.button("💾 Save RAG Settings", type="primary"):
            self.save_settings({
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap,
                'top_k': top_k,
                'min_similarity': min_similarity
            })
    
    def render_interface_settings(self):
        """Render interface settings"""
        st.subheader("Interface Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Appearance")
            theme = st.selectbox(
                "Theme",
                ["Dark", "Light", "Auto"],
                index=0,
                help="Choose the interface theme"
            )
            
            show_timestamps = st.checkbox(
                "Show Timestamps",
                value=True,
                help="Display timestamps in chat and search results"
            )
            
            show_confidence = st.checkbox(
                "Show Confidence Scores",
                value=True,
                help="Display confidence scores for responses"
            )
        
        with col2:
            st.markdown("#### Behavior")
            auto_scroll = st.checkbox(
                "Auto-scroll Chat",
                value=True,
                help="Automatically scroll to latest message"
            )
            
            enable_shortcuts = st.checkbox(
                "Keyboard Shortcuts",
                value=True,
                help="Enable keyboard shortcuts"
            )
            
            notification_sound = st.checkbox(
                "Notification Sound",
                value=False,
                help="Play sound for notifications"
            )
        
        if st.button("💾 Save Interface Settings", type="primary"):
            self.save_interface_settings({
                'theme': theme,
                'show_timestamps': show_timestamps,
                'show_confidence': show_confidence,
                'auto_scroll': auto_scroll,
                'enable_shortcuts': enable_shortcuts,
                'notification_sound': notification_sound
            })
    
    def render_advanced_settings(self):
        """Render advanced settings"""
        st.subheader("Advanced Settings")
        
        st.warning("⚠️ These settings are for advanced users. Incorrect configuration may affect system performance.")
        
        with st.expander("🔧 System Configuration", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                cache_size = st.number_input(
                    "Cache Size (MB)",
                    min_value=100,
                    max_value=10000,
                    value=1000,
                    step=100,
                    help="Maximum cache size in megabytes"
                )
                
                max_workers = st.number_input(
                    "Max Workers",
                    min_value=1,
                    max_value=16,
                    value=4,
                    help="Maximum number of worker threads"
                )
            
            with col2:
                timeout = st.number_input(
                    "Request Timeout (s)",
                    min_value=10,
                    max_value=300,
                    value=60,
                    help="Request timeout in seconds"
                )
                
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=100,
                    value=10,
                    help="Batch size for processing"
                )
        
        with st.expander("🔑 API Configuration", expanded=False):
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="Enter your API key",
                help="API key for the selected LLM provider"
            )
            
            api_endpoint = st.text_input(
                "API Endpoint",
                placeholder="https://api.example.com",
                help="Custom API endpoint (optional)"
            )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Advanced Settings", type="primary", use_container_width=True):
                self.save_advanced_settings({
                    'cache_size': cache_size,
                    'max_workers': max_workers,
                    'timeout': timeout,
                    'batch_size': batch_size,
                    'api_key': api_key,
                    'api_endpoint': api_endpoint
                })
        
        with col2:
            if st.button("🔄 Reset to Defaults", type="secondary", use_container_width=True):
                self.reset_settings()
    
    def render_help(self):
        """Render help and documentation"""
        st.title("📚 Help & Documentation")
        
        # Quick start guide
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started with PCIe Debug Agent
            
            1. **Chat Mode** - Ask questions about PCIe debugging in natural language
            2. **Search Mode** - Search through logs and documentation
            3. **Analytics** - View usage statistics and performance metrics
            4. **Settings** - Configure the system to your needs
            
            #### Common Use Cases:
            - 🐛 Debug PCIe errors and issues
            - 📊 Analyze performance bottlenecks
            - 🔍 Search through large log files
            - 💡 Get suggestions for fixes
            """)
        
        # FAQ
        with st.expander("❓ Frequently Asked Questions"):
            st.markdown("""
            **Q: How do I upload new log files?**
            A: Use the import feature in the sidebar or drag and drop files directly.
            
            **Q: What file formats are supported?**
            A: We support .log, .txt, .v, .sv, and .list files.
            
            **Q: How accurate are the responses?**
            A: The confidence score indicates the reliability of each response.
            
            **Q: Can I export my analysis?**
            A: Yes, use the export feature to save your sessions and results.
            """)
        
        # Keyboard shortcuts
        with st.expander("⌨️ Keyboard Shortcuts"):
            shortcuts = {
                "Ctrl/Cmd + K": "Quick search",
                "Ctrl/Cmd + /": "Focus chat input",
                "Ctrl/Cmd + S": "Save settings",
                "Ctrl/Cmd + E": "Export data",
                "Ctrl/Cmd + R": "Refresh page",
                "Esc": "Close dialogs"
            }
            
            for shortcut, description in shortcuts.items():
                st.markdown(f"**{shortcut}** - {description}")
        
        # Contact support
        st.markdown("---")
        st.markdown("""
        ### 🆘 Need More Help?
        
        - 📧 Email: support@pcie-debug.com
        - 💬 Discord: [Join our community](https://discord.gg/pcie-debug)
        - 📖 Documentation: [Full docs](https://docs.pcie-debug.com)
        - 🐛 Report issues: [GitHub](https://github.com/yourusername/pcie_debug_agent/issues)
        """)
    
    def save_settings(self, settings_dict: dict):
        """Save settings to configuration file"""
        try:
            # Update settings object
            for key, value in settings_dict.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            
            # Save to file
            config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(self.settings.to_dict(), f, default_flow_style=False)
            
            # Reinitialize RAG engine
            st.session_state.rag_engine = EnhancedRAGEngine(self.settings)
            
            self.show_success("Settings saved successfully!")
            st.rerun()
        
        except Exception as e:
            self.show_error("Failed to save settings", str(e))
    
    def save_interface_settings(self, settings_dict: dict):
        """Save interface settings"""
        for key, value in settings_dict.items():
            st.session_state[f"ui_{key}"] = value
        self.show_success("Interface settings saved!")
    
    def save_advanced_settings(self, settings_dict: dict):
        """Save advanced settings with validation"""
        # Validate API key if provided
        if settings_dict.get('api_key'):
            # Here you would validate the API key
            pass
        
        self.save_settings(settings_dict)
    
    def reset_settings(self):
        """Reset settings to defaults"""
        if st.confirm("Are you sure you want to reset all settings to defaults?"):
            # Reset logic here
            self.show_success("Settings reset to defaults!")
            st.rerun()
    
    def show_import_dialog(self):
        """Show import dialog"""
        st.session_state.show_import = True
    
    def show_export_dialog(self):
        """Show export dialog"""
        # Export current session data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'chat_history': st.session_state.get('chat_history', []),
            'search_history': st.session_state.get('search_history', []),
            'settings': self.settings.to_dict()
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
        
        st.download_button(
            label="📥 Download Export",
            data=json_str,
            file_name=f"pcie_debug_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def clear_session_data(self):
        """Clear session data with confirmation"""
        if st.confirm("Are you sure you want to clear all session data?"):
            st.session_state.chat_history = []
            st.session_state.search_history = []
            self.show_success("Session data cleared!")
            st.rerun()
    
    def run(self):
        """Run the application"""
        # Render sidebar
        self.render_sidebar()
        
        # Render main content
        self.render_main_content()
        
        # Update last activity
        st.session_state.last_activity = datetime.now()

def main():
    """Entry point"""
    app = PCIeDebugAgentApp()
    app.run()

if __name__ == "__main__":
    main()