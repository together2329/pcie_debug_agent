"""
Main Streamlit application for the UVM Debug Agent
"""

import streamlit as st
import yaml
from pathlib import Path
import sys
import os

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import Settings
from src.rag.enhanced_rag_engine import EnhancedRAGEngine
from src.ui.interactive_chat import InteractiveChatInterface
from src.ui.semantic_search import SemanticSearchInterface

def load_settings() -> Settings:
    """설정 파일 로드"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
    
    try:
        if config_path.exists():
            return Settings.load_settings(str(config_path))
        else:
            # Create default settings with env vars
            st.warning("Configuration file not found. Using default settings with environment variables.")
            return Settings.load_settings(None)
    except Exception as e:
        st.error(f"Failed to load settings: {str(e)}")
        st.stop()

def initialize_rag_engine(settings: Settings) -> EnhancedRAGEngine:
    """RAG 엔진 초기화"""
    try:
        return EnhancedRAGEngine(settings)
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        st.stop()

def main():
    """메인 애플리케이션"""
    # 페이지 설정
    st.set_page_config(
        page_title="UVM Debug Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 커스텀 CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 설정 로드
    settings = load_settings()
    
    # RAG 엔진 초기화
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = initialize_rag_engine(settings)
    
    # 사이드바
    with st.sidebar:
        st.title("🤖 UVM Debug Agent")
        
        # 탭 선택
        tab = st.radio(
            "Select Mode",
            ["Chat", "Search", "Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # 상태 표시
        st.markdown("### System Status")
        st.markdown(f"**Embedding Model:** {settings.embedding_model}")
        st.markdown(f"**LLM Provider:** {settings.llm_provider}")
        st.markdown(f"**Chunk Size:** {settings.chunk_size}")
        
        # 메트릭
        metrics = st.session_state.rag_engine.get_metrics()
        st.markdown("### Performance")
        st.metric("Total Queries", metrics['queries_processed'])
        st.metric("Avg Response Time", f"{metrics['average_response_time']:.2f}s")
        st.metric("Cache Hits", metrics['cache_hits'])
    
    # 메인 컨텐츠
    if tab == "Chat":
        chat_interface = InteractiveChatInterface()
        chat_interface.render()
    
    elif tab == "Search":
        search_interface = SemanticSearchInterface()
        search_interface.render()
    
    else:  # Settings
        st.header("⚙️ Settings")
        
        # 설정 수정 폼
        with st.form("settings_form"):
            st.subheader("Embedding Settings")
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=0
            )
            
            st.subheader("LLM Settings")
            llm_provider = st.selectbox(
                "LLM Provider",
                ["openai", "anthropic", "ollama", "custom"],
                index=0
            )
            
            llm_model = st.selectbox(
                "LLM Model",
                ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"],
                index=0
            )
            
            st.subheader("RAG Settings")
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=1000,
                value=settings.chunk_size,
                step=50
            )
            
            overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=200,
                value=settings.chunk_overlap,
                step=10
            )
            
            if st.form_submit_button("Save Settings"):
                # 설정 업데이트
                settings.embedding_model = embedding_model
                settings.llm_provider = llm_provider
                settings.llm_model = llm_model
                settings.chunk_size = chunk_size
                settings.chunk_overlap = overlap
                
                # 설정 저장
                config_path = Path(__file__).parent.parent.parent / "configs" / "settings.yaml"
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(settings.to_dict(), f, default_flow_style=False)
                
                # RAG 엔진 재초기화
                st.session_state.rag_engine = initialize_rag_engine(settings)
                st.success("Settings saved successfully!")
                st.rerun()

if __name__ == "__main__":
    main() 