"""
Interactive Chat Interface for RAG-based conversations
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery, RAGResponse

class InteractiveChatInterface:
    """대화형 채팅 인터페이스"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = None
            
        if 'chat_settings' not in st.session_state:
            st.session_state.chat_settings = {
                'context_window': 5,
                'min_similarity': 0.5,
                'rerank': True,
                'show_sources': True,
                'show_confidence': True
            }
    
    def render(self):
        """채팅 인터페이스 렌더링"""
        st.header("🤖 AI Assistant Chat")
        
        # RAG 엔진 확인
        if st.session_state.rag_engine is None:
            st.warning("⚠️ Please setup embeddings and initialize the system first.")
            return
        
        # 채팅 설정 사이드바
        self._render_chat_settings()
        
        # 채팅 히스토리 표시
        self._render_chat_history()
        
        # 입력 인터페이스
        self._render_input_interface()
        
        # 성능 메트릭 표시
        self._render_metrics()
    
    def _render_chat_settings(self):
        """채팅 설정 렌더링"""
        with st.sidebar.expander("💬 Chat Settings", expanded=True):
            st.session_state.chat_settings['context_window'] = st.slider(
                "Context Window",
                min_value=1,
                max_value=20,
                value=st.session_state.chat_settings['context_window'],
                help="Number of sources to retrieve"
            )
            
            st.session_state.chat_settings['min_similarity'] = st.slider(
                "Minimum Similarity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.chat_settings['min_similarity'],
                step=0.1,
                help="Minimum similarity score for retrieved documents"
            )
            
            st.session_state.chat_settings['rerank'] = st.checkbox(
                "Enable Reranking",
                value=st.session_state.chat_settings['rerank'],
                help="Rerank retrieved documents for better relevance"
            )
            
            st.session_state.chat_settings['show_sources'] = st.checkbox(
                "Show Sources",
                value=st.session_state.chat_settings['show_sources'],
                help="Display source documents with answers"
            )
            
            st.session_state.chat_settings['show_confidence'] = st.checkbox(
                "Show Confidence Score",
                value=st.session_state.chat_settings['show_confidence'],
                help="Display confidence score for answers"
            )
            
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
    
    def _render_chat_history(self):
        """채팅 히스토리 렌더링"""
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(message['content'])
                else:
                    with st.chat_message("assistant"):
                        # 답변 표시
                        st.write(message['content'])
                        
                        # 신뢰도 표시
                        if st.session_state.chat_settings['show_confidence'] and 'confidence' in message:
                            confidence = message['confidence']
                            color = self._get_confidence_color(confidence)
                            st.markdown(
                                f"<small style='color: {color}'>Confidence: {confidence:.2%}</small>",
                                unsafe_allow_html=True
                            )
                        
                        # 소스 표시
                        if st.session_state.chat_settings['show_sources'] and 'sources' in message:
                            self._render_sources(message['sources'], f"sources_{i}")
    
    def _render_input_interface(self):
        """입력 인터페이스 렌더링"""
        # 입력 폼
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask a question...",
                    placeholder="Type your question here",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send", use_container_width=True)
        
        # 질문 제안
        st.markdown("**Suggested questions:**")
        col1, col2, col3 = st.columns(3)
        
        suggestions = [
            "What is UVM?",
            "How to debug UVM errors?",
            "Explain sequence items",
            "What are virtual interfaces?",
            "How to use scoreboard?",
            "Explain factory pattern"
        ]
        
        for i, suggestion in enumerate(suggestions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    self._process_query(suggestion)
                    st.rerun()
        
        # 폼 제출 처리
        if submit_button and user_input:
            self._process_query(user_input)
            st.rerun()
    
    def _process_query(self, query: str):
        """쿼리 처리"""
        # 사용자 메시지 추가
        st.session_state.chat_history.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })
        
        # RAG 쿼리 생성
        rag_query = RAGQuery(
            query=query,
            context_window=st.session_state.chat_settings['context_window'],
            rerank=st.session_state.chat_settings['rerank'],
            min_similarity=st.session_state.chat_settings['min_similarity'],
            include_metadata=True
        )
        
        # RAG 엔진 호출
        with st.spinner("Thinking..."):
            response = st.session_state.rag_engine.query(rag_query)
        
        # 응답 메시지 추가
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response.answer,
            'sources': response.sources,
            'confidence': response.confidence,
            'reasoning': response.reasoning,
            'metadata': response.metadata,
            'timestamp': datetime.now().isoformat()
        })
    
    def _render_sources(self, sources: List[Dict[str, Any]], key_prefix: str):
        """소스 문서 렌더링"""
        if not sources:
            return
        
        with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
            for i, source in enumerate(sources):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # 메타데이터 표시
                    metadata = source.get('metadata', {})
                    source_info = []
                    
                    if 'file_name' in metadata:
                        source_info.append(f"📄 {metadata['file_name']}")
                    if 'page' in metadata:
                        source_info.append(f"Page {metadata['page']}")
                    if 'section' in metadata:
                        source_info.append(f"Section: {metadata['section']}")
                    
                    if source_info:
                        st.markdown(f"**{' | '.join(source_info)}**")
                    
                    # 내용 미리보기
                    content_preview = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
                    st.text(content_preview)
                
                with col2:
                    # 관련성 점수
                    score = source.get('score', 0)
                    st.metric("Score", f"{score:.3f}")
                
                if i < len(sources) - 1:
                    st.divider()
    
    def _render_metrics(self):
        """성능 메트릭 렌더링"""
        if st.session_state.rag_engine is None:
            return
        
        metrics = st.session_state.rag_engine.get_metrics()
        
        with st.expander("📊 Performance Metrics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Queries",
                    metrics['queries_processed']
                )
            
            with col2:
                st.metric(
                    "Avg Response Time",
                    f"{metrics['average_response_time']:.2f}s"
                )
            
            with col3:
                st.metric(
                    "Avg Confidence",
                    f"{metrics['average_confidence']:.2%}"
                )
            
            with col4:
                st.metric(
                    "Cache Hits",
                    metrics['cache_hits']
                )
            
            # 쿼리 히스토리 차트
            if st.session_state.chat_history:
                self._render_query_history_chart()
    
    def _render_query_history_chart(self):
        """쿼리 히스토리 차트 렌더링"""
        # 데이터 준비
        data = []
        for msg in st.session_state.chat_history:
            if msg['role'] == 'assistant' and 'confidence' in msg:
                data.append({
                    'timestamp': pd.to_datetime(msg['timestamp']),
                    'confidence': msg['confidence'],
                    'response_time': msg.get('metadata', {}).get('query_time', 0)
                })
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # 신뢰도 차트
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Query Confidence Over Time",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            yaxis_range=[0, 1],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _get_confidence_color(self, confidence: float) -> str:
        """신뢰도에 따른 색상 반환"""
        if confidence >= 0.8:
            return "#27ae60"  # Green
        elif confidence >= 0.6:
            return "#f39c12"  # Orange
        else:
            return "#e74c3c"  # Red
    
    def export_chat_history(self) -> str:
        """채팅 히스토리 내보내기"""
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'settings': st.session_state.chat_settings,
            'history': st.session_state.chat_history
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def import_chat_history(self, data: str):
        """채팅 히스토리 가져오기"""
        try:
            imported_data = json.loads(data)
            st.session_state.chat_history = imported_data.get('history', [])
            st.session_state.chat_settings.update(imported_data.get('settings', {}))
            return True
        except Exception as e:
            st.error(f"Failed to import chat history: {str(e)}")
            return False 