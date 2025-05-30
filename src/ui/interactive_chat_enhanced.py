"""
Enhanced Interactive Chat Interface for RAG-based conversations
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re
from collections import defaultdict

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery, RAGResponse

class EnhancedChatInterface:
    """Enhanced interactive chat interface with improved UX"""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_chat_templates()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = []
        
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = self.create_new_session()
        
        if 'chat_settings' not in st.session_state:
            st.session_state.chat_settings = {
                'context_window': 5,
                'min_similarity': 0.5,
                'rerank': True,
                'show_sources': True,
                'show_confidence': True,
                'auto_suggest': True,
                'stream_response': True,
                'context_mode': 'adaptive'
            }
        
        if 'pinned_messages' not in st.session_state:
            st.session_state.pinned_messages = []
        
        if 'message_feedback' not in st.session_state:
            st.session_state.message_feedback = {}
    
    def setup_chat_templates(self):
        """Setup predefined chat templates"""
        self.chat_templates = {
            "Debug Error": {
                "icon": "🐛",
                "prompts": [
                    "I'm seeing this PCIe error: {error_message}. What could be causing it?",
                    "How do I debug {error_type} errors in PCIe?",
                    "What are common causes of PCIe link training failures?"
                ]
            },
            "Performance Analysis": {
                "icon": "📊",
                "prompts": [
                    "How can I improve PCIe throughput?",
                    "What factors affect PCIe latency?",
                    "Analyze the performance bottlenecks in my PCIe setup"
                ]
            },
            "Configuration": {
                "icon": "⚙️",
                "prompts": [
                    "What's the optimal PCIe configuration for {use_case}?",
                    "How do I configure PCIe Gen {generation} settings?",
                    "Explain PCIe power management states"
                ]
            },
            "Learning": {
                "icon": "📚",
                "prompts": [
                    "Explain how PCIe {concept} works",
                    "What's the difference between {term1} and {term2}?",
                    "Give me a beginner's guide to PCIe debugging"
                ]
            }
        }
    
    def create_new_session(self) -> str:
        """Create a new chat session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = {
            'id': session_id,
            'name': f"Session {len(st.session_state.chat_sessions) + 1}",
            'created_at': datetime.now(),
            'messages': [],
            'context': []
        }
        st.session_state.chat_sessions.append(session)
        return session_id
    
    def render(self):
        """Render the enhanced chat interface"""
        # Check RAG engine
        if st.session_state.rag_engine is None:
            self.render_setup_prompt()
            return
        
        # Main layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            self.render_chat_area()
        
        with col2:
            self.render_chat_sidebar()
    
    def render_setup_prompt(self):
        """Render setup prompt when system not initialized"""
        st.warning("⚠️ System not initialized")
        st.markdown("""
        ### 🚀 Getting Started
        
        To start using the chat interface:
        1. Initialize the embedding system
        2. Load your PCIe logs and documentation
        3. Start asking questions!
        
        **Quick Setup:**
        """)
        
        if st.button("🔧 Initialize System", type="primary"):
            # Initialize system
            st.rerun()
    
    def render_chat_area(self):
        """Render main chat area"""
        # Chat header
        self.render_chat_header()
        
        # Messages container
        chat_container = st.container(height=500)
        
        with chat_container:
            self.render_messages()
        
        # Input area
        self.render_input_area()
    
    def render_chat_header(self):
        """Render chat header with session info"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            current_session = self.get_current_session()
            st.markdown(f"### 💬 {current_session['name']}")
        
        with col2:
            if st.button("📌 Pin Important", use_container_width=True):
                self.show_pin_dialog()
        
        with col3:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                self.clear_current_chat()
    
    def render_messages(self):
        """Render chat messages with enhanced styling"""
        messages = self.get_current_messages()
        
        for i, message in enumerate(messages):
            is_pinned = message.get('id') in st.session_state.pinned_messages
            
            if message['role'] == 'user':
                self.render_user_message(message, i, is_pinned)
            else:
                self.render_assistant_message(message, i, is_pinned)
    
    def render_user_message(self, message: dict, index: int, is_pinned: bool):
        """Render user message with actions"""
        with st.chat_message("user", avatar="👤"):
            col1, col2 = st.columns([10, 1])
            
            with col1:
                st.markdown(message['content'])
                if is_pinned:
                    st.caption("📌 Pinned")
            
            with col2:
                if st.button("📋", key=f"copy_user_{index}", help="Copy"):
                    st.clipboard(message['content'])
                    st.toast("Copied to clipboard!")
    
    def render_assistant_message(self, message: dict, index: int, is_pinned: bool):
        """Render assistant message with enhanced features"""
        with st.chat_message("assistant", avatar="🤖"):
            # Message content
            if st.session_state.chat_settings.get('stream_response') and index == len(self.get_current_messages()) - 1:
                # Stream the latest response
                self.stream_message(message['content'])
            else:
                st.markdown(message['content'])
            
            # Message metadata
            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
            
            with col1:
                if st.session_state.chat_settings['show_confidence'] and 'confidence' in message:
                    confidence = message['confidence']
                    color = self.get_confidence_color(confidence)
                    st.markdown(
                        f"<span style='color: {color}; font-size: 0.9em;'>🎯 {confidence:.0%} confident</span>",
                        unsafe_allow_html=True
                    )
            
            with col2:
                if 'timestamp' in message:
                    time_ago = self.format_time_ago(message['timestamp'])
                    st.caption(f"⏰ {time_ago}")
            
            with col3:
                feedback = st.session_state.message_feedback.get(message.get('id'), None)
                col3_1, col3_2 = st.columns(2)
                
                with col3_1:
                    if st.button("👍", key=f"like_{index}", help="Helpful"):
                        self.record_feedback(message.get('id'), 'positive')
                
                with col3_2:
                    if st.button("👎", key=f"dislike_{index}", help="Not helpful"):
                        self.record_feedback(message.get('id'), 'negative')
            
            with col4:
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("📋", key=f"copy_{index}", help="Copy"):
                        st.clipboard(message['content'])
                        st.toast("Copied!")
                
                with action_col2:
                    if st.button("📌", key=f"pin_{index}", help="Pin"):
                        self.toggle_pin(message.get('id'))
                
                with action_col3:
                    if st.button("🔄", key=f"regen_{index}", help="Regenerate"):
                        self.regenerate_response(index)
            
            # Sources
            if st.session_state.chat_settings['show_sources'] and 'sources' in message:
                self.render_sources_enhanced(message['sources'], f"sources_{index}")
            
            # Follow-up suggestions
            if 'suggestions' in message and st.session_state.chat_settings['auto_suggest']:
                self.render_follow_up_suggestions(message['suggestions'], index)
    
    def render_sources_enhanced(self, sources: List[Dict[str, Any]], key_prefix: str):
        """Render sources with enhanced visualization"""
        if not sources:
            return
        
        with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
            # Source summary
            source_types = defaultdict(int)
            for source in sources:
                file_type = Path(source.get('metadata', {}).get('file_name', '')).suffix
                source_types[file_type] += 1
            
            # Display summary
            summary_cols = st.columns(len(source_types))
            for i, (file_type, count) in enumerate(source_types.items()):
                with summary_cols[i]:
                    st.metric(file_type or "Other", count)
            
            st.divider()
            
            # Detailed sources
            for i, source in enumerate(sources):
                self.render_source_card(source, i)
    
    def render_source_card(self, source: dict, index: int):
        """Render individual source as a card"""
        metadata = source.get('metadata', {})
        
        # Create expandable card
        with st.container():
            col1, col2, col3 = st.columns([6, 2, 2])
            
            with col1:
                file_info = []
                if 'file_name' in metadata:
                    file_info.append(f"📄 **{metadata['file_name']}**")
                if 'page' in metadata:
                    file_info.append(f"Page {metadata['page']}")
                if 'line_number' in metadata:
                    file_info.append(f"Line {metadata['line_number']}")
                
                st.markdown(" | ".join(file_info))
            
            with col2:
                score = source.get('score', 0)
                st.metric("Relevance", f"{score:.2f}", label_visibility="collapsed")
            
            with col3:
                if st.button("View", key=f"view_source_{index}"):
                    self.show_source_detail(source)
            
            # Content preview with syntax highlighting
            content = source['content']
            if self.is_code_content(content):
                st.code(content[:300] + "..." if len(content) > 300 else content)
            else:
                st.text(content[:300] + "..." if len(content) > 300 else content)
            
            if index < len(source) - 1:
                st.divider()
    
    def render_follow_up_suggestions(self, suggestions: List[str], message_index: int):
        """Render follow-up question suggestions"""
        st.markdown("**💡 Follow-up questions:**")
        
        cols = st.columns(min(len(suggestions), 3))
        for i, suggestion in enumerate(suggestions[:3]):
            with cols[i]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{message_index}_{i}",
                    use_container_width=True
                ):
                    self.process_query(suggestion)
    
    def render_input_area(self):
        """Render enhanced input area"""
        # Template selector
        col1, col2 = st.columns([1, 4])
        
        with col1:
            template_name = st.selectbox(
                "Template",
                ["None"] + list(self.chat_templates.keys()),
                label_visibility="collapsed"
            )
        
        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([10, 1])
            
            with col1:
                # Show template prompts if selected
                if template_name != "None":
                    template = self.chat_templates[template_name]
                    prompt_options = template['prompts']
                    selected_prompt = st.selectbox(
                        "Select a prompt template:",
                        prompt_options,
                        label_visibility="collapsed"
                    )
                    
                    # Allow customization
                    user_input = st.text_area(
                        "Customize your question:",
                        value=selected_prompt,
                        height=100,
                        label_visibility="collapsed"
                    )
                else:
                    user_input = st.text_area(
                        "Ask anything about PCIe debugging...",
                        height=100,
                        placeholder="Type your question here or use / for commands",
                        label_visibility="collapsed"
                    )
            
            with col2:
                submit_button = st.form_submit_button(
                    "Send",
                    use_container_width=True,
                    type="primary"
                )
        
        # Handle submission
        if submit_button and user_input:
            # Check for commands
            if user_input.startswith("/"):
                self.handle_command(user_input)
            else:
                self.process_query(user_input)
            st.rerun()
    
    def render_chat_sidebar(self):
        """Render chat sidebar with sessions and settings"""
        st.markdown("### 🗂️ Chat Sessions")
        
        # Session list
        for session in st.session_state.chat_sessions[-5:]:  # Show last 5 sessions
            is_current = session['id'] == st.session_state.current_session_id
            
            if is_current:
                st.info(f"📍 {session['name']}")
            else:
                if st.button(
                    f"📄 {session['name']}",
                    key=f"session_{session['id']}",
                    use_container_width=True
                ):
                    self.switch_session(session['id'])
        
        if st.button("➕ New Session", use_container_width=True):
            self.create_new_session()
            st.rerun()
        
        st.divider()
        
        # Quick settings
        st.markdown("### ⚡ Quick Settings")
        
        st.session_state.chat_settings['show_sources'] = st.checkbox(
            "Show Sources",
            value=st.session_state.chat_settings['show_sources']
        )
        
        st.session_state.chat_settings['show_confidence'] = st.checkbox(
            "Show Confidence",
            value=st.session_state.chat_settings['show_confidence']
        )
        
        st.session_state.chat_settings['auto_suggest'] = st.checkbox(
            "Auto-suggest Questions",
            value=st.session_state.chat_settings['auto_suggest']
        )
        
        st.session_state.chat_settings['stream_response'] = st.checkbox(
            "Stream Responses",
            value=st.session_state.chat_settings['stream_response']
        )
        
        # Context mode
        st.session_state.chat_settings['context_mode'] = st.radio(
            "Context Mode",
            ["adaptive", "full", "minimal"],
            index=0,
            help="How much context to include in queries"
        )
        
        st.divider()
        
        # Pinned messages
        if st.session_state.pinned_messages:
            st.markdown("### 📌 Pinned Messages")
            if st.button("View Pinned", use_container_width=True):
                self.show_pinned_messages()
        
        # Export/Import
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export", use_container_width=True):
                self.export_chat()
        
        with col2:
            if st.button("📥 Import", use_container_width=True):
                self.import_chat()
    
    def process_query(self, query: str):
        """Process user query with enhanced context"""
        # Add to history
        message_id = self.generate_message_id()
        user_message = {
            'id': message_id,
            'role': 'user',
            'content': query,
            'timestamp': datetime.now()
        }
        
        self.add_message_to_current_session(user_message)
        
        # Build context based on mode
        context = self.build_query_context(query)
        
        # Create RAG query
        rag_query = RAGQuery(
            query=query,
            context=context,
            context_window=st.session_state.chat_settings['context_window'],
            rerank=st.session_state.chat_settings['rerank'],
            min_similarity=st.session_state.chat_settings['min_similarity'],
            include_metadata=True
        )
        
        # Process with loading animation
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.rag_engine.query(rag_query)
        
        # Generate follow-up suggestions
        suggestions = self.generate_follow_up_suggestions(query, response)
        
        # Add response to history
        response_id = self.generate_message_id()
        assistant_message = {
            'id': response_id,
            'role': 'assistant',
            'content': response.answer,
            'sources': response.sources,
            'confidence': response.confidence,
            'reasoning': response.reasoning,
            'metadata': response.metadata,
            'suggestions': suggestions,
            'timestamp': datetime.now()
        }
        
        self.add_message_to_current_session(assistant_message)
    
    def build_query_context(self, query: str) -> List[Dict[str, str]]:
        """Build context for query based on settings"""
        mode = st.session_state.chat_settings['context_mode']
        messages = self.get_current_messages()
        
        if mode == 'minimal':
            # Only include last exchange
            context = messages[-2:] if len(messages) >= 2 else messages
        elif mode == 'full':
            # Include all messages
            context = messages
        else:  # adaptive
            # Include relevant messages based on similarity
            context = self.get_relevant_context(query, messages)
        
        return [
            {'role': msg['role'], 'content': msg['content']}
            for msg in context
        ]
    
    def get_relevant_context(self, query: str, messages: List[dict], max_context: int = 5) -> List[dict]:
        """Get relevant context messages using similarity"""
        # Simple keyword-based relevance for now
        # In production, use embeddings for similarity
        query_words = set(query.lower().split())
        
        scored_messages = []
        for msg in messages[-10:]:  # Look at last 10 messages
            msg_words = set(msg['content'].lower().split())
            score = len(query_words.intersection(msg_words)) / len(query_words)
            scored_messages.append((score, msg))
        
        # Sort by relevance and take top messages
        scored_messages.sort(key=lambda x: x[0], reverse=True)
        return [msg for _, msg in scored_messages[:max_context]]
    
    def generate_follow_up_suggestions(self, query: str, response: RAGResponse) -> List[str]:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        
        # Extract key concepts from response
        concepts = self.extract_concepts(response.answer)
        
        # Generate suggestions based on concepts
        if "error" in query.lower():
            suggestions.extend([
                f"What are the root causes of this error?",
                f"How can I prevent this error in the future?",
                f"Show me similar error patterns"
            ])
        elif "performance" in query.lower():
            suggestions.extend([
                f"What metrics should I monitor?",
                f"How can I optimize this further?",
                f"Compare with best practices"
            ])
        else:
            # Generic suggestions based on concepts
            for concept in concepts[:2]:
                suggestions.append(f"Tell me more about {concept}")
            suggestions.append("What are the next steps?")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple implementation - in production use NLP
        # Look for capitalized terms, technical terms, etc.
        words = text.split()
        concepts = []
        
        for word in words:
            if word.isupper() and len(word) > 2:  # Acronyms
                concepts.append(word)
            elif word.startswith("PCIe"):  # PCIe specific terms
                concepts.append(word)
        
        return list(set(concepts))
    
    def handle_command(self, command: str):
        """Handle chat commands"""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        
        if cmd == "/help":
            self.show_help()
        elif cmd == "/clear":
            self.clear_current_chat()
        elif cmd == "/export":
            self.export_chat()
        elif cmd == "/stats":
            self.show_chat_stats()
        elif cmd == "/debug":
            self.toggle_debug_mode()
        else:
            st.error(f"Unknown command: {cmd}")
    
    def stream_message(self, content: str):
        """Stream message content for better UX"""
        placeholder = st.empty()
        
        # Simulate streaming
        for i in range(0, len(content), 50):
            placeholder.markdown(content[:i+50])
            # In production, this would be actual streaming from LLM
    
    def get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence score"""
        if confidence >= 0.8:
            return "#28a745"  # Green
        elif confidence >= 0.6:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red
    
    def format_time_ago(self, timestamp: datetime) -> str:
        """Format timestamp as relative time"""
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        delta = datetime.now() - timestamp
        
        if delta.seconds < 60:
            return "just now"
        elif delta.seconds < 3600:
            minutes = delta.seconds // 60
            return f"{minutes}m ago"
        elif delta.days < 1:
            hours = delta.seconds // 3600
            return f"{hours}h ago"
        else:
            return timestamp.strftime("%b %d")
    
    def is_code_content(self, content: str) -> bool:
        """Check if content appears to be code"""
        code_indicators = ['{', '}', '()', ';', '//', '/*', 'def ', 'class ', 'function']
        return any(indicator in content for indicator in code_indicators)
    
    # Helper methods
    def get_current_session(self) -> dict:
        """Get current chat session"""
        for session in st.session_state.chat_sessions:
            if session['id'] == st.session_state.current_session_id:
                return session
        return None
    
    def get_current_messages(self) -> List[dict]:
        """Get messages for current session"""
        session = self.get_current_session()
        return session['messages'] if session else []
    
    def add_message_to_current_session(self, message: dict):
        """Add message to current session"""
        session = self.get_current_session()
        if session:
            session['messages'].append(message)
    
    def generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def switch_session(self, session_id: str):
        """Switch to different chat session"""
        st.session_state.current_session_id = session_id
    
    def clear_current_chat(self):
        """Clear current chat session"""
        session = self.get_current_session()
        if session:
            session['messages'] = []
    
    def toggle_pin(self, message_id: str):
        """Toggle message pin status"""
        if message_id in st.session_state.pinned_messages:
            st.session_state.pinned_messages.remove(message_id)
        else:
            st.session_state.pinned_messages.append(message_id)
    
    def record_feedback(self, message_id: str, feedback_type: str):
        """Record user feedback for message"""
        st.session_state.message_feedback[message_id] = feedback_type
        st.toast(f"Thanks for your feedback! {'👍' if feedback_type == 'positive' else '👎'}")
    
    def regenerate_response(self, message_index: int):
        """Regenerate response for a specific message"""
        messages = self.get_current_messages()
        if message_index > 0 and messages[message_index-1]['role'] == 'user':
            query = messages[message_index-1]['content']
            # Remove current response
            messages.pop(message_index)
            # Regenerate
            self.process_query(query)
    
    def show_source_detail(self, source: dict):
        """Show detailed view of a source"""
        st.session_state.show_source_detail = source
    
    def show_help(self):
        """Show help dialog"""
        st.info("""
        ### Chat Commands
        - `/help` - Show this help
        - `/clear` - Clear chat history
        - `/export` - Export chat session
        - `/stats` - Show chat statistics
        - `/debug` - Toggle debug mode
        """)
    
    def show_chat_stats(self):
        """Show chat statistics"""
        messages = self.get_current_messages()
        total_messages = len(messages)
        user_messages = len([m for m in messages if m['role'] == 'user'])
        assistant_messages = total_messages - user_messages
        
        avg_confidence = sum(m.get('confidence', 0) for m in messages if m['role'] == 'assistant') / max(assistant_messages, 1)
        
        st.info(f"""
        ### Chat Statistics
        - Total messages: {total_messages}
        - Your messages: {user_messages}
        - Assistant messages: {assistant_messages}
        - Average confidence: {avg_confidence:.0%}
        """)
    
    def toggle_debug_mode(self):
        """Toggle debug mode"""
        st.session_state.debug_mode = not st.session_state.get('debug_mode', False)
        st.toast(f"Debug mode: {'ON' if st.session_state.debug_mode else 'OFF'}")
    
    def export_chat(self):
        """Export current chat session"""
        session = self.get_current_session()
        if not session:
            return
        
        export_data = {
            'session': session,
            'settings': st.session_state.chat_settings,
            'exported_at': datetime.now().isoformat()
        }
        
        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
        
        st.download_button(
            label="📥 Download Chat Export",
            data=json_str,
            file_name=f"chat_export_{session['id']}.json",
            mime="application/json"
        )
    
    def import_chat(self):
        """Import chat session"""
        uploaded_file = st.file_uploader(
            "Upload chat export",
            type=['json'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                # Import session
                st.session_state.chat_sessions.append(data['session'])
                st.session_state.current_session_id = data['session']['id']
                st.success("Chat imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to import: {str(e)}")
    
    def show_pinned_messages(self):
        """Show dialog with pinned messages"""
        pinned = []
        for session in st.session_state.chat_sessions:
            for msg in session['messages']:
                if msg.get('id') in st.session_state.pinned_messages:
                    pinned.append(msg)
        
        if pinned:
            for msg in pinned:
                st.info(f"📌 {msg['content'][:100]}...")
        else:
            st.info("No pinned messages yet")