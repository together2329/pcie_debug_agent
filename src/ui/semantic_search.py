"""
Semantic Search Interface for RAG-based document search
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGQuery, RAGResponse

class SemanticSearchInterface:
    """시맨틱 검색 인터페이스"""
    
    def __init__(self):
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
            
        if 'search_settings' not in st.session_state:
            st.session_state.search_settings = {
                'top_k': 5,
                'min_similarity': 0.5,
                'rerank': True,
                'group_by_file': True,
                'show_metadata': True
            }
    
    def render(self):
        """검색 인터페이스 렌더링"""
        st.header("🔍 Semantic Search")
        
        # RAG 엔진 확인
        if st.session_state.rag_engine is None:
            st.warning("⚠️ Please setup embeddings and initialize the system first.")
            return
        
        # 검색 설정 사이드바
        self._render_search_settings()
        
        # 검색 인터페이스
        self._render_search_interface()
        
        # 검색 결과
        if st.session_state.search_history:
            self._render_search_results()
    
    def _render_search_settings(self):
        """검색 설정 렌더링"""
        with st.sidebar.expander("🔧 Search Settings", expanded=True):
            st.session_state.search_settings['top_k'] = st.slider(
                "Number of Results",
                min_value=1,
                max_value=20,
                value=st.session_state.search_settings['top_k'],
                help="Maximum number of results to return"
            )
            
            st.session_state.search_settings['min_similarity'] = st.slider(
                "Minimum Similarity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.search_settings['min_similarity'],
                step=0.1,
                help="Minimum similarity score for results"
            )
            
            st.session_state.search_settings['rerank'] = st.checkbox(
                "Enable Reranking",
                value=st.session_state.search_settings['rerank'],
                help="Rerank results for better relevance"
            )
            
            st.session_state.search_settings['group_by_file'] = st.checkbox(
                "Group by File",
                value=st.session_state.search_settings['group_by_file'],
                help="Group results by source file"
            )
            
            st.session_state.search_settings['show_metadata'] = st.checkbox(
                "Show Metadata",
                value=st.session_state.search_settings['show_metadata'],
                help="Display document metadata"
            )
            
            if st.button("Clear Search History", type="secondary"):
                st.session_state.search_history = []
                st.rerun()
    
    def _render_search_interface(self):
        """검색 인터페이스 렌더링"""
        # 검색 폼
        with st.form(key="search_form", clear_on_submit=True):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                search_query = st.text_input(
                    "Enter your search query...",
                    placeholder="Type your search query here",
                    label_visibility="collapsed"
                )
            
            with col2:
                submit_button = st.form_submit_button("Search", use_container_width=True)
        
        # 검색 제안
        st.markdown("**Suggested searches:**")
        col1, col2, col3 = st.columns(3)
        
        suggestions = [
            "UVM sequence",
            "Virtual interface",
            "Scoreboard implementation",
            "Factory pattern",
            "Coverage collection",
            "Assertion syntax"
        ]
        
        for i, suggestion in enumerate(suggestions):
            col = [col1, col2, col3][i % 3]
            with col:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    self._process_search(suggestion)
                    st.rerun()
        
        # 폼 제출 처리
        if submit_button and search_query:
            self._process_search(search_query)
            st.rerun()
    
    def _process_search(self, query: str):
        """검색 처리"""
        # RAG 쿼리 생성
        rag_query = RAGQuery(
            query=query,
            context_window=st.session_state.search_settings['top_k'],
            rerank=st.session_state.search_settings['rerank'],
            min_similarity=st.session_state.search_settings['min_similarity'],
            include_metadata=True
        )
        
        # RAG 엔진 호출
        with st.spinner("Searching..."):
            response = st.session_state.rag_engine.query(rag_query)
        
        # 검색 결과 저장
        st.session_state.search_history.append({
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'results': response.sources,
            'metadata': response.metadata
        })
    
    def _render_search_results(self):
        """검색 결과 렌더링"""
        latest_search = st.session_state.search_history[-1]
        
        # 검색 정보
        st.markdown(f"### Search Results for: *{latest_search['query']}*")
        st.markdown(f"*Found {len(latest_search['results'])} results*")
        
        # 결과 그룹화
        if st.session_state.search_settings['group_by_file']:
            self._render_grouped_results(latest_search['results'])
        else:
            self._render_flat_results(latest_search['results'])
        
        # 검색 메트릭
        self._render_search_metrics(latest_search)
    
    def _render_grouped_results(self, results: List[Dict[str, Any]]):
        """그룹화된 결과 렌더링"""
        # 파일별로 그룹화
        grouped_results = {}
        for result in results:
            file_name = result.get('metadata', {}).get('file_name', 'Unknown')
            if file_name not in grouped_results:
                grouped_results[file_name] = []
            grouped_results[file_name].append(result)
        
        # 각 파일별로 결과 표시
        for file_name, file_results in grouped_results.items():
            with st.expander(f"📄 {file_name} ({len(file_results)} matches)", expanded=True):
                for result in file_results:
                    self._render_result_item(result)
    
    def _render_flat_results(self, results: List[Dict[str, Any]]):
        """평면화된 결과 렌더링"""
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1}", expanded=i==0):
                self._render_result_item(result)
    
    def _render_result_item(self, result: Dict[str, Any]):
        """결과 항목 렌더링"""
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # 메타데이터 표시
            if st.session_state.search_settings['show_metadata']:
                metadata = result.get('metadata', {})
                metadata_info = []
                
                if 'page' in metadata:
                    metadata_info.append(f"Page {metadata['page']}")
                if 'section' in metadata:
                    metadata_info.append(f"Section: {metadata['section']}")
                if 'line_number' in metadata:
                    metadata_info.append(f"Line: {metadata['line_number']}")
                
                if metadata_info:
                    st.markdown(f"**{' | '.join(metadata_info)}**")
            
            # 내용 표시
            st.text(result['content'])
        
        with col2:
            # 관련성 점수
            score = result.get('score', 0)
            st.metric("Score", f"{score:.3f}")
    
    def _render_search_metrics(self, search: Dict[str, Any]):
        """검색 메트릭 렌더링"""
        with st.expander("📊 Search Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Query Time",
                    f"{search['metadata'].get('query_time', 0):.2f}s"
                )
            
            with col2:
                st.metric(
                    "Average Score",
                    f"{sum(r.get('score', 0) for r in search['results']) / len(search['results']):.3f}"
                )
            
            with col3:
                st.metric(
                    "Results",
                    len(search['results'])
                )
            
            # 점수 분포 차트
            self._render_score_distribution(search['results'])
    
    def _render_score_distribution(self, results: List[Dict[str, Any]]):
        """점수 분포 차트 렌더링"""
        scores = [r.get('score', 0) for r in results]
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=10,
            name='Score Distribution',
            marker_color='blue'
        ))
        
        fig.update_layout(
            title="Score Distribution",
            xaxis_title="Similarity Score",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def export_search_history(self) -> str:
        """검색 히스토리 내보내기"""
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'settings': st.session_state.search_settings,
            'history': st.session_state.search_history
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def import_search_history(self, data: str):
        """검색 히스토리 가져오기"""
        try:
            imported_data = json.loads(data)
            st.session_state.search_history = imported_data.get('history', [])
            st.session_state.search_settings.update(imported_data.get('settings', {}))
            return True
        except Exception as e:
            st.error(f"Failed to import search history: {str(e)}")
            return False 