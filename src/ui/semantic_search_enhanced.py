"""
Enhanced Semantic Search Interface with improved UX and features
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import defaultdict
import re

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery, RAGResponse

class EnhancedSearchInterface:
    """Enhanced semantic search interface with advanced features"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_search_presets()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if 'saved_searches' not in st.session_state:
            st.session_state.saved_searches = []
        
        if 'search_filters' not in st.session_state:
            st.session_state.search_filters = {
                'file_types': [],
                'date_range': None,
                'min_score': 0.5,
                'categories': [],
                'tags': []
            }
        
        if 'search_settings' not in st.session_state:
            st.session_state.search_settings = {
                'top_k': 10,
                'min_similarity': 0.5,
                'rerank': True,
                'group_by_file': True,
                'show_metadata': True,
                'highlight_matches': True,
                'show_preview': True,
                'preview_length': 300
            }
        
        if 'search_view_mode' not in st.session_state:
            st.session_state.search_view_mode = 'cards'  # cards, list, compact
    
    def setup_search_presets(self):
        """Setup predefined search presets"""
        self.search_presets = {
            "Error Patterns": {
                "icon": "🐛",
                "queries": [
                    "PCIe error timeout",
                    "link training failed",
                    "completion timeout",
                    "ECRC error",
                    "fatal error"
                ],
                "filters": {
                    "file_types": [".log", ".err"],
                    "min_score": 0.7
                }
            },
            "Configuration": {
                "icon": "⚙️",
                "queries": [
                    "PCIe configuration space",
                    "BAR setup",
                    "capability registers",
                    "power management",
                    "link width negotiation"
                ],
                "filters": {
                    "file_types": [".cfg", ".conf", ".yaml"],
                    "min_score": 0.6
                }
            },
            "Performance": {
                "icon": "📊",
                "queries": [
                    "throughput optimization",
                    "latency measurement",
                    "bandwidth utilization",
                    "credit flow",
                    "buffer management"
                ],
                "filters": {
                    "file_types": [".log", ".perf", ".csv"],
                    "min_score": 0.6
                }
            },
            "Debug Traces": {
                "icon": "🔍",
                "queries": [
                    "TLP dump",
                    "DLLP trace",
                    "physical layer",
                    "training sequence",
                    "link state"
                ],
                "filters": {
                    "file_types": [".trc", ".log", ".dump"],
                    "min_score": 0.5
                }
            }
        }
    
    def render(self):
        """Render the enhanced search interface"""
        # Check RAG engine
        if st.session_state.rag_engine is None:
            self.render_setup_prompt()
            return
        
        # Main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_search_area()
        
        with col2:
            self.render_search_sidebar()
        
        # Results area
        if st.session_state.search_history:
            self.render_search_results()
    
    def render_setup_prompt(self):
        """Render setup prompt when system not initialized"""
        st.warning("⚠️ System not initialized")
        st.markdown("""
        ### 🔍 Semantic Search Setup
        
        To use the semantic search:
        1. Initialize the embedding system
        2. Index your PCIe logs and documentation
        3. Start searching!
        """)
    
    def render_search_area(self):
        """Render main search area"""
        # Search header
        st.markdown("## 🔍 Semantic Search")
        st.markdown("Search through PCIe logs, documentation, and debug traces with AI-powered understanding")
        
        # Search presets
        self.render_search_presets()
        
        # Search input
        self.render_search_input()
        
        # Active filters
        self.render_active_filters()
    
    def render_search_presets(self):
        """Render search preset buttons"""
        st.markdown("### 🎯 Quick Searches")
        
        cols = st.columns(len(self.search_presets))
        
        for i, (preset_name, preset_data) in enumerate(self.search_presets.items()):
            with cols[i]:
                if st.button(
                    f"{preset_data['icon']} {preset_name}",
                    key=f"preset_{preset_name}",
                    use_container_width=True,
                    help=f"Search for {preset_name.lower()} related content"
                ):
                    self.apply_search_preset(preset_name)
    
    def render_search_input(self):
        """Render enhanced search input"""
        # Search form
        with st.form(key="search_form", clear_on_submit=False):
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search Query",
                    placeholder="Enter keywords, error codes, or natural language questions...",
                    label_visibility="collapsed",
                    help="Use natural language or specific terms"
                )
            
            with col2:
                search_mode = st.selectbox(
                    "Mode",
                    ["Smart", "Exact", "Fuzzy"],
                    label_visibility="collapsed",
                    help="Search mode affects matching behavior"
                )
            
            with col3:
                submit_button = st.form_submit_button(
                    "🔍 Search",
                    use_container_width=True,
                    type="primary"
                )
        
        # Advanced search options
        with st.expander("🔧 Advanced Search", expanded=False):
            self.render_advanced_search_options()
        
        # Handle search submission
        if submit_button and search_query:
            self.process_search(search_query, search_mode)
            st.rerun()
    
    def render_advanced_search_options(self):
        """Render advanced search options"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # File type filters
            available_types = [".log", ".txt", ".cfg", ".yaml", ".v", ".sv", ".list"]
            st.session_state.search_filters['file_types'] = st.multiselect(
                "File Types",
                available_types,
                default=st.session_state.search_filters['file_types'],
                help="Filter by file extensions"
            )
            
            # Category filters
            categories = ["Errors", "Warnings", "Info", "Debug", "Configuration"]
            st.session_state.search_filters['categories'] = st.multiselect(
                "Categories",
                categories,
                default=st.session_state.search_filters['categories'],
                help="Filter by content category"
            )
        
        with col2:
            # Date range
            date_option = st.selectbox(
                "Date Range",
                ["All Time", "Last 24 Hours", "Last Week", "Last Month", "Custom"],
                help="Filter by document date"
            )
            
            if date_option == "Custom":
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
                st.session_state.search_filters['date_range'] = (start_date, end_date)
            elif date_option != "All Time":
                # Calculate date range
                end_date = datetime.now()
                if date_option == "Last 24 Hours":
                    start_date = end_date - timedelta(days=1)
                elif date_option == "Last Week":
                    start_date = end_date - timedelta(weeks=1)
                else:  # Last Month
                    start_date = end_date - timedelta(days=30)
                st.session_state.search_filters['date_range'] = (start_date, end_date)
            else:
                st.session_state.search_filters['date_range'] = None
        
        with col3:
            # Score threshold
            st.session_state.search_filters['min_score'] = st.slider(
                "Minimum Relevance",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.search_filters['min_score'],
                step=0.05,
                help="Minimum relevance score for results"
            )
            
            # Result limit
            st.session_state.search_settings['top_k'] = st.number_input(
                "Max Results",
                min_value=5,
                max_value=100,
                value=st.session_state.search_settings['top_k'],
                step=5,
                help="Maximum number of results to return"
            )
    
    def render_active_filters(self):
        """Render active filters as chips"""
        active_filters = []
        
        if st.session_state.search_filters['file_types']:
            active_filters.extend([f"📄 {ft}" for ft in st.session_state.search_filters['file_types']])
        
        if st.session_state.search_filters['categories']:
            active_filters.extend([f"🏷️ {cat}" for cat in st.session_state.search_filters['categories']])
        
        if st.session_state.search_filters['date_range']:
            active_filters.append(f"📅 Date range")
        
        if st.session_state.search_filters['min_score'] > 0.5:
            active_filters.append(f"⭐ Score ≥ {st.session_state.search_filters['min_score']:.1f}")
        
        if active_filters:
            st.markdown("**Active Filters:**")
            cols = st.columns(len(active_filters) + 1)
            
            for i, filter_text in enumerate(active_filters):
                with cols[i]:
                    st.info(filter_text)
            
            with cols[-1]:
                if st.button("❌ Clear All", key="clear_filters"):
                    self.clear_filters()
                    st.rerun()
    
    def render_search_sidebar(self):
        """Render search sidebar with settings and history"""
        st.markdown("### 🎛️ Search Settings")
        
        # View mode
        st.session_state.search_view_mode = st.radio(
            "View Mode",
            ["cards", "list", "compact"],
            format_func=lambda x: {"cards": "🃏 Cards", "list": "📋 List", "compact": "📄 Compact"}[x],
            horizontal=True
        )
        
        # Display settings
        st.session_state.search_settings['show_preview'] = st.checkbox(
            "Show Preview",
            value=st.session_state.search_settings['show_preview']
        )
        
        st.session_state.search_settings['highlight_matches'] = st.checkbox(
            "Highlight Matches",
            value=st.session_state.search_settings['highlight_matches']
        )
        
        st.session_state.search_settings['group_by_file'] = st.checkbox(
            "Group by File",
            value=st.session_state.search_settings['group_by_file']
        )
        
        st.divider()
        
        # Search history
        self.render_search_history()
        
        st.divider()
        
        # Saved searches
        self.render_saved_searches()
    
    def render_search_history(self):
        """Render search history"""
        st.markdown("### 📜 Recent Searches")
        
        recent_searches = st.session_state.search_history[-5:]
        
        if recent_searches:
            for search in reversed(recent_searches):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    if st.button(
                        f"🔍 {search['query'][:30]}...",
                        key=f"history_{search['timestamp']}",
                        use_container_width=True
                    ):
                        self.rerun_search(search)
                
                with col2:
                    st.caption(self.format_time_ago(search['timestamp']))
        else:
            st.info("No recent searches")
    
    def render_saved_searches(self):
        """Render saved searches"""
        st.markdown("### 💾 Saved Searches")
        
        if st.session_state.saved_searches:
            for saved in st.session_state.saved_searches:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(
                        f"⭐ {saved['name']}",
                        key=f"saved_{saved['id']}",
                        use_container_width=True,
                        help=saved.get('description', '')
                    ):
                        self.load_saved_search(saved)
                
                with col2:
                    if st.button("🗑️", key=f"delete_{saved['id']}"):
                        self.delete_saved_search(saved['id'])
                        st.rerun()
        else:
            st.info("No saved searches")
        
        # Save current search
        if st.session_state.search_history:
            if st.button("💾 Save Current Search", use_container_width=True):
                self.show_save_search_dialog()
    
    def render_search_results(self):
        """Render search results with selected view mode"""
        latest_search = st.session_state.search_history[-1]
        
        # Results header
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"### 🔍 Results for: *{latest_search['query']}*")
            st.caption(f"Found {len(latest_search['results'])} results in {latest_search['metadata'].get('query_time', 0):.2f}s")
        
        with col2:
            # Sort options
            sort_by = st.selectbox(
                "Sort by",
                ["Relevance", "Date", "File Name"],
                label_visibility="collapsed"
            )
        
        with col3:
            # Export results
            if st.button("📤 Export", use_container_width=True):
                self.export_results(latest_search)
        
        # Apply sorting
        sorted_results = self.sort_results(latest_search['results'], sort_by)
        
        # Render based on view mode
        if st.session_state.search_view_mode == 'cards':
            self.render_results_cards(sorted_results)
        elif st.session_state.search_view_mode == 'list':
            self.render_results_list(sorted_results)
        else:  # compact
            self.render_results_compact(sorted_results)
        
        # Results analytics
        self.render_results_analytics(latest_search)
    
    def render_results_cards(self, results: List[Dict[str, Any]]):
        """Render results as cards"""
        if st.session_state.search_settings['group_by_file']:
            grouped = self.group_results_by_file(results)
            
            for file_name, file_results in grouped.items():
                with st.expander(f"📄 {file_name} ({len(file_results)} matches)", expanded=True):
                    for i, result in enumerate(file_results):
                        self.render_result_card(result, i)
        else:
            for i, result in enumerate(results):
                self.render_result_card(result, i)
    
    def render_result_card(self, result: Dict[str, Any], index: int):
        """Render individual result as a card"""
        with st.container():
            # Card styling
            st.markdown("""
                <style>
                .result-card {
                    background-color: #262730;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    border: 1px solid #464646;
                    margin-bottom: 1rem;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Card content
            col1, col2, col3 = st.columns([6, 2, 2])
            
            with col1:
                # Title and metadata
                metadata = result.get('metadata', {})
                title_parts = []
                
                if 'section' in metadata:
                    title_parts.append(f"📑 {metadata['section']}")
                if 'line_number' in metadata:
                    title_parts.append(f"Line {metadata['line_number']}")
                
                if title_parts:
                    st.markdown(f"**{' | '.join(title_parts)}**")
                
                # Content preview
                if st.session_state.search_settings['show_preview']:
                    content = result['content']
                    preview_length = st.session_state.search_settings['preview_length']
                    
                    if st.session_state.search_settings['highlight_matches']:
                        # Highlight search terms
                        highlighted = self.highlight_matches(content, st.session_state.search_history[-1]['query'])
                        st.markdown(highlighted[:preview_length] + "..." if len(highlighted) > preview_length else highlighted)
                    else:
                        st.text(content[:preview_length] + "..." if len(content) > preview_length else content)
            
            with col2:
                # Relevance score with visual indicator
                score = result.get('score', 0)
                score_color = self.get_score_color(score)
                
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 2rem; color: {score_color};'>
                            {self.get_score_emoji(score)}
                        </div>
                        <div style='font-size: 0.9rem; color: {score_color};'>
                            {score:.1%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                # Actions
                if st.button("👁️ View", key=f"view_{index}", use_container_width=True):
                    self.show_result_detail(result)
                
                if st.button("📋 Copy", key=f"copy_{index}", use_container_width=True):
                    st.clipboard(result['content'])
                    st.toast("Copied to clipboard!")
            
            # Tags and categories
            if 'tags' in metadata or 'category' in metadata:
                tag_cols = st.columns(8)
                tag_index = 0
                
                if 'category' in metadata:
                    with tag_cols[tag_index]:
                        st.info(f"🏷️ {metadata['category']}")
                    tag_index += 1
                
                for tag in metadata.get('tags', [])[:7]:
                    if tag_index < 8:
                        with tag_cols[tag_index]:
                            st.success(f"#{tag}")
                        tag_index += 1
    
    def render_results_list(self, results: List[Dict[str, Any]]):
        """Render results as a list"""
        for i, result in enumerate(results):
            with st.container():
                col1, col2, col3, col4 = st.columns([5, 2, 2, 1])
                
                with col1:
                    metadata = result.get('metadata', {})
                    file_name = metadata.get('file_name', 'Unknown')
                    st.markdown(f"**{i+1}. {file_name}**")
                    
                    if 'section' in metadata:
                        st.caption(f"Section: {metadata['section']}")
                
                with col2:
                    if 'line_number' in metadata:
                        st.caption(f"Line: {metadata['line_number']}")
                
                with col3:
                    score = result.get('score', 0)
                    st.metric("Score", f"{score:.1%}", label_visibility="collapsed")
                
                with col4:
                    if st.button("→", key=f"open_{i}"):
                        self.show_result_detail(result)
                
                if st.session_state.search_settings['show_preview']:
                    st.text(result['content'][:200] + "...")
                
                if i < len(results) - 1:
                    st.divider()
    
    def render_results_compact(self, results: List[Dict[str, Any]]):
        """Render results in compact view"""
        # Create DataFrame for compact display
        data = []
        for result in results:
            metadata = result.get('metadata', {})
            data.append({
                'File': metadata.get('file_name', 'Unknown'),
                'Section': metadata.get('section', '-'),
                'Line': metadata.get('line_number', '-'),
                'Score': f"{result.get('score', 0):.1%}",
                'Preview': result['content'][:100] + "..."
            })
        
        df = pd.DataFrame(data)
        
        # Display as interactive table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.TextColumn(
                    "Score",
                    help="Relevance score",
                    width="small"
                ),
                "Preview": st.column_config.TextColumn(
                    "Preview",
                    help="Content preview",
                    width="large"
                )
            }
        )
    
    def render_results_analytics(self, search: Dict[str, Any]):
        """Render search results analytics"""
        with st.expander("📊 Results Analytics", expanded=False):
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Results",
                    len(search['results'])
                )
            
            with col2:
                avg_score = sum(r.get('score', 0) for r in search['results']) / max(len(search['results']), 1)
                st.metric(
                    "Avg Relevance",
                    f"{avg_score:.1%}"
                )
            
            with col3:
                unique_files = len(set(r.get('metadata', {}).get('file_name', '') for r in search['results']))
                st.metric(
                    "Unique Files",
                    unique_files
                )
            
            with col4:
                st.metric(
                    "Query Time",
                    f"{search['metadata'].get('query_time', 0):.2f}s"
                )
            
            # Visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Score Distribution", "📁 File Distribution", "🏷️ Category Distribution"])
            
            with tab1:
                self.render_score_distribution(search['results'])
            
            with tab2:
                self.render_file_distribution(search['results'])
            
            with tab3:
                self.render_category_distribution(search['results'])
    
    def render_score_distribution(self, results: List[Dict[str, Any]]):
        """Render score distribution chart"""
        scores = [r.get('score', 0) for r in results]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=20,
            name='Score Distribution',
            marker_color='#1f77b4',
            opacity=0.75
        ))
        
        fig.update_layout(
            title="Relevance Score Distribution",
            xaxis_title="Score",
            yaxis_title="Count",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_file_distribution(self, results: List[Dict[str, Any]]):
        """Render file distribution chart"""
        file_counts = defaultdict(int)
        for result in results:
            file_name = result.get('metadata', {}).get('file_name', 'Unknown')
            file_counts[file_name] += 1
        
        # Top 10 files
        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f[1] for f in sorted_files],
            y=[f[0] for f in sorted_files],
            orientation='h',
            marker_color='#ff7f0e',
            text=[f[1] for f in sorted_files],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Top Files by Result Count",
            xaxis_title="Number of Results",
            yaxis_title="File Name",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_category_distribution(self, results: List[Dict[str, Any]]):
        """Render category distribution chart"""
        category_counts = defaultdict(int)
        for result in results:
            category = result.get('metadata', {}).get('category', 'Uncategorized')
            category_counts[category] += 1
        
        if category_counts:
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=list(category_counts.keys()),
                values=list(category_counts.values()),
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3
            ))
            
            fig.update_layout(
                title="Results by Category",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category information available")
    
    def process_search(self, query: str, mode: str = "Smart"):
        """Process search query with filters"""
        # Apply search mode transformations
        if mode == "Exact":
            processed_query = f'"{query}"'
        elif mode == "Fuzzy":
            processed_query = self.make_fuzzy_query(query)
        else:
            processed_query = query
        
        # Create RAG query with filters
        rag_query = RAGQuery(
            query=processed_query,
            context_window=st.session_state.search_settings['top_k'],
            rerank=st.session_state.search_settings['rerank'],
            min_similarity=st.session_state.search_filters['min_score'],
            include_metadata=True,
            filters=self.build_search_filters()
        )
        
        # Execute search
        with st.spinner(f"🔍 Searching for '{query}'..."):
            response = st.session_state.rag_engine.query(rag_query)
        
        # Save to history
        search_record = {
            'query': query,
            'mode': mode,
            'timestamp': datetime.now(),
            'results': response.sources,
            'metadata': response.metadata,
            'filters': st.session_state.search_filters.copy(),
            'settings': st.session_state.search_settings.copy()
        }
        
        st.session_state.search_history.append(search_record)
    
    def build_search_filters(self) -> Dict[str, Any]:
        """Build filter dictionary for search"""
        filters = {}
        
        if st.session_state.search_filters['file_types']:
            filters['file_types'] = st.session_state.search_filters['file_types']
        
        if st.session_state.search_filters['date_range']:
            filters['date_range'] = st.session_state.search_filters['date_range']
        
        if st.session_state.search_filters['categories']:
            filters['categories'] = st.session_state.search_filters['categories']
        
        if st.session_state.search_filters['tags']:
            filters['tags'] = st.session_state.search_filters['tags']
        
        return filters
    
    def make_fuzzy_query(self, query: str) -> str:
        """Convert query to fuzzy search format"""
        # Simple fuzzy query - in production use proper fuzzy matching
        words = query.split()
        fuzzy_words = [f"{word}~" for word in words if len(word) > 3]
        return " ".join(fuzzy_words)
    
    def highlight_matches(self, content: str, query: str) -> str:
        """Highlight query matches in content"""
        # Simple highlighting - in production use proper text highlighting
        words = query.lower().split()
        highlighted = content
        
        for word in words:
            if len(word) > 2:  # Only highlight words longer than 2 chars
                pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
                highlighted = pattern.sub(r'**\1**', highlighted)
        
        return highlighted
    
    def group_results_by_file(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group results by file name"""
        grouped = defaultdict(list)
        for result in results:
            file_name = result.get('metadata', {}).get('file_name', 'Unknown')
            grouped[file_name].append(result)
        return dict(grouped)
    
    def sort_results(self, results: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
        """Sort results based on criteria"""
        if sort_by == "Relevance":
            return sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        elif sort_by == "Date":
            return sorted(results, key=lambda x: x.get('metadata', {}).get('date', ''), reverse=True)
        else:  # File Name
            return sorted(results, key=lambda x: x.get('metadata', {}).get('file_name', ''))
    
    def get_score_color(self, score: float) -> str:
        """Get color based on score"""
        if score >= 0.8:
            return "#28a745"
        elif score >= 0.6:
            return "#ffc107"
        else:
            return "#dc3545"
    
    def get_score_emoji(self, score: float) -> str:
        """Get emoji based on score"""
        if score >= 0.8:
            return "🎯"
        elif score >= 0.6:
            return "✓"
        else:
            return "○"
    
    def format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as relative time"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        delta = datetime.now() - timestamp
        
        if delta.seconds < 60:
            return "now"
        elif delta.seconds < 3600:
            return f"{delta.seconds // 60}m"
        elif delta.days < 1:
            return f"{delta.seconds // 3600}h"
        else:
            return f"{delta.days}d"
    
    def apply_search_preset(self, preset_name: str):
        """Apply a search preset"""
        preset = self.search_presets[preset_name]
        
        # Apply filters
        for key, value in preset.get('filters', {}).items():
            if key in st.session_state.search_filters:
                st.session_state.search_filters[key] = value
        
        # Run first query
        if preset['queries']:
            self.process_search(preset['queries'][0])
    
    def clear_filters(self):
        """Clear all search filters"""
        st.session_state.search_filters = {
            'file_types': [],
            'date_range': None,
            'min_score': 0.5,
            'categories': [],
            'tags': []
        }
    
    def rerun_search(self, search: Dict[str, Any]):
        """Rerun a previous search"""
        # Restore filters and settings
        st.session_state.search_filters = search.get('filters', {})
        st.session_state.search_settings = search.get('settings', {})
        
        # Run search
        self.process_search(search['query'], search.get('mode', 'Smart'))
    
    def show_result_detail(self, result: Dict[str, Any]):
        """Show detailed view of a result"""
        st.session_state.show_result_detail = result
    
    def export_results(self, search: Dict[str, Any]):
        """Export search results"""
        export_data = {
            'query': search['query'],
            'timestamp': search['timestamp'].isoformat(),
            'filters': search['filters'],
            'results': search['results'],
            'metadata': search['metadata']
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            label="📥 Download Results",
            data=json_str,
            file_name=f"search_results_{search['timestamp'].strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    def show_save_search_dialog(self):
        """Show dialog to save current search"""
        with st.form("save_search_form"):
            st.subheader("💾 Save Search")
            
            name = st.text_input("Search Name", placeholder="My Important Search")
            description = st.text_area("Description", placeholder="Optional description")
            
            if st.form_submit_button("Save"):
                if name:
                    self.save_current_search(name, description)
                    st.success("Search saved!")
                    st.rerun()
                else:
                    st.error("Please enter a name")
    
    def save_current_search(self, name: str, description: str = ""):
        """Save current search configuration"""
        if st.session_state.search_history:
            latest = st.session_state.search_history[-1]
            
            saved_search = {
                'id': f"saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'name': name,
                'description': description,
                'query': latest['query'],
                'mode': latest.get('mode', 'Smart'),
                'filters': latest['filters'],
                'settings': latest['settings'],
                'created_at': datetime.now()
            }
            
            st.session_state.saved_searches.append(saved_search)
    
    def load_saved_search(self, saved: Dict[str, Any]):
        """Load a saved search"""
        # Restore filters and settings
        st.session_state.search_filters = saved['filters']
        st.session_state.search_settings = saved['settings']
        
        # Run search
        self.process_search(saved['query'], saved['mode'])
    
    def delete_saved_search(self, search_id: str):
        """Delete a saved search"""
        st.session_state.saved_searches = [
            s for s in st.session_state.saved_searches
            if s['id'] != search_id
        ]