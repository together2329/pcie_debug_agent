"""
UVM Debug Agent 메인 애플리케이션
"""

import argparse
import logging
from pathlib import Path
import sys
import os
import yaml
import streamlit as st

from config.settings import Settings, load_settings
from rag.enhanced_rag_engine import EnhancedRAGEngine
from ui.interactive_chat import InteractiveChatInterface
from ui.semantic_search import SemanticSearchInterface
from rag.vector_store import FAISSVectorStore
from rag.model_manager import ModelManager

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log')
        ]
    )

def load_app_settings() -> Settings:
    """설정 파일 로드"""
    config_path = Path("configs/settings.yaml")
    
    if config_path.exists():
        settings = load_settings(config_path)
    else:
        # Create default settings with env vars
        print("Configuration file not found. Using default settings with environment variables.")
        settings = load_settings(None)
    
    settings.validate()
    return settings

def initialize_rag_engine(settings: Settings) -> EnhancedRAGEngine:
    """RAG 엔진 초기화"""
    try:
        # 벡터 스토어 초기화
        vector_store = FAISSVectorStore(
            index_path=settings.vector_store.index_path,
            index_type=settings.vector_store.index_type,
            dimension=settings.vector_store.dimension
        )
        
        # 모델 매니저 초기화
        model_manager = ModelManager(
            embedding_model=settings.embedding.model,
            embedding_provider=settings.embedding.provider,
            embedding_api_key=settings.embedding.api_key,
            embedding_api_base_url=settings.embedding.api_base_url,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            llm_api_key=settings.llm.api_key,
            llm_api_base_url=settings.llm.api_base_url
        )
        
        # RAG 엔진 초기화
        engine = EnhancedRAGEngine(
            vector_store=vector_store,
            model_manager=model_manager,
            llm_provider=settings.llm.provider,
            llm_model=settings.llm.model,
            temperature=0.1,
            max_tokens=2000
        )
        return engine
    except Exception as e:
        logging.error(f"RAG 엔진 초기화 실패: {str(e)}")
        raise

def run_web_interface(settings: Settings):
    """웹 인터페이스 실행"""
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🔍",
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
    
    # RAG 엔진 초기화
    rag_engine = initialize_rag_engine(settings)
    
    # 사이드바
    with st.sidebar:
        st.title("🔍 UVM Debug Agent")
        
        # 모드 선택
        mode = st.radio(
            "모드 선택",
            ["Chat", "Search", "Settings"],
            index=0
        )
        
        # 시스템 상태
        st.markdown("---")
        st.subheader("시스템 상태")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("임베딩 모델", settings.embedding.model)
            st.metric("LLM 제공자", settings.llm.provider)
        with col2:
            st.metric("청크 크기", settings.rag.chunk_size)
            st.metric("총 쿼리 수", rag_engine.total_queries)
        
        # 성능 메트릭
        st.markdown("---")
        st.subheader("성능 메트릭")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("평균 응답 시간", f"{rag_engine.avg_response_time:.2f}s")
            st.metric("캐시 히트", f"{rag_engine.cache_hits}")
        with col2:
            st.metric("평균 신뢰도", f"{rag_engine.avg_confidence:.2%}")
            st.metric("캐시 미스", f"{rag_engine.cache_misses}")
    
    # 메인 컨텐츠
    if mode == "Chat":
        chat_interface = InteractiveChatInterface(rag_engine, settings)
        chat_interface.render()
    
    elif mode == "Search":
        search_interface = SemanticSearchInterface(rag_engine, settings)
        search_interface.render()
    
    else:  # Settings
        st.title("설정")
        
        with st.form("settings_form"):
            st.subheader("임베딩 설정")
            embedding_provider = st.selectbox(
                "임베딩 제공자",
                ["Local", "OpenAI", "Cohere", "Voyage", "Custom"],
                index=["Local", "OpenAI", "Cohere", "Voyage", "Custom"].index(settings.embedding.provider)
            )
            
            if embedding_provider == "Custom":
                embedding_model = st.text_input("임베딩 모델", settings.embedding.model)
                embedding_api_key = st.text_input("API 키", settings.embedding.api_key or "", type="password")
                embedding_base_url = st.text_input("Base URL", settings.embedding.api_base_url or "")
                embedding_headers = st.text_area("추가 헤더 (JSON)", "{}")
            else:
                if embedding_provider == "OpenAI":
                    embedding_model = st.selectbox(
                        "임베딩 모델",
                        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                        index=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"].index(settings.embedding.model)
                    )
                elif embedding_provider == "Cohere":
                    embedding_model = st.selectbox(
                        "임베딩 모델",
                        ["embed-english-v3.0", "embed-multilingual-v3.0"],
                        index=["embed-english-v3.0", "embed-multilingual-v3.0"].index(settings.embedding.model)
                    )
                elif embedding_provider == "Voyage":
                    embedding_model = st.selectbox(
                        "임베딩 모델",
                        ["voyage-01", "voyage-02"],
                        index=["voyage-01", "voyage-02"].index(settings.embedding.model)
                    )
                else:
                    embedding_model = st.text_input("임베딩 모델", settings.embedding.model)
                
                embedding_api_key = st.text_input("API 키", settings.embedding.api_key or "", type="password")
                embedding_base_url = st.text_input("Base URL (선택사항)", settings.embedding.api_base_url or "")
                embedding_headers = "{}"
            
            st.subheader("LLM 설정")
            llm_provider = st.selectbox(
                "LLM 제공자",
                ["openai", "anthropic", "ollama", "custom"],
                index=["openai", "anthropic", "ollama", "custom"].index(settings.llm.provider)
            )
            
            if llm_provider == "custom":
                llm_model = st.text_input("LLM 모델", settings.llm.model)
                llm_api_key = st.text_input("API 키", settings.llm.api_key or "", type="password")
                llm_base_url = st.text_input("Base URL", settings.llm.api_base_url or "")
                llm_headers = st.text_area("추가 헤더 (JSON)", "{}")
            else:
                if llm_provider == "openai":
                    llm_model = st.selectbox(
                        "LLM 모델",
                        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                        index=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"].index(settings.llm.model)
                    )
                elif llm_provider == "anthropic":
                    llm_model = st.selectbox(
                        "LLM 모델",
                        ["claude-3-opus", "claude-3-sonnet", "claude-2"],
                        index=["claude-3-opus", "claude-3-sonnet", "claude-2"].index(settings.llm.model)
                    )
                elif llm_provider == "ollama":
                    llm_model = st.selectbox(
                        "LLM 모델",
                        ["llama2", "mistral", "codellama"],
                        index=["llama2", "mistral", "codellama"].index(settings.llm.model)
                    )
                else:
                    llm_model = st.text_input("LLM 모델", settings.llm.model)
                
                llm_api_key = st.text_input("API 키", settings.llm.api_key or "", type="password")
                llm_base_url = st.text_input("Base URL (선택사항)", settings.llm.api_base_url or "")
                llm_headers = "{}"
            
            st.subheader("RAG 설정")
            chunk_size = st.number_input("청크 크기", min_value=100, max_value=1000, value=settings.rag.chunk_size)
            chunk_overlap = st.number_input("청크 오버랩", min_value=0, max_value=100, value=settings.rag.chunk_overlap)
            context_window = st.number_input("컨텍스트 윈도우", min_value=1, max_value=10, value=settings.rag.context_window)
            min_similarity = st.slider("최소 유사도", min_value=0.0, max_value=1.0, value=settings.rag.min_similarity, step=0.05)
            rerank = st.checkbox("재순위화", value=settings.rag.rerank)
            
            st.subheader("UI 설정")
            theme = st.selectbox("테마", ["light", "dark"], index=["light", "dark"].index(settings.ui.theme))
            max_width = st.number_input("최대 너비", min_value=800, max_value=2000, value=settings.ui.max_width)
            show_confidence = st.checkbox("신뢰도 표시", value=settings.ui.show_confidence)
            show_sources = st.checkbox("소스 표시", value=settings.ui.show_sources)
            group_by_file = st.checkbox("파일별 그룹화", value=settings.ui.group_by_file)
            
            if st.form_submit_button("설정 저장"):
                # 설정 업데이트
                settings.embedding.provider = embedding_provider.lower()
                settings.embedding.model = embedding_model
                settings.embedding.api_key = embedding_api_key
                settings.embedding.api_base_url = embedding_base_url
                
                settings.llm.provider = llm_provider
                settings.llm.model = llm_model
                settings.llm.api_key = llm_api_key
                settings.llm.api_base_url = llm_base_url
                
                settings.rag.chunk_size = chunk_size
                settings.rag.chunk_overlap = chunk_overlap
                settings.rag.context_window = context_window
                settings.rag.min_similarity = min_similarity
                settings.rag.rerank = rerank
                
                settings.ui.theme = theme
                settings.ui.max_width = max_width
                settings.ui.show_confidence = show_confidence
                settings.ui.show_sources = show_sources
                settings.ui.group_by_file = group_by_file
                
                # 설정 저장
                with open("configs/settings.yaml", 'w') as f:
                    yaml.dump(settings.to_dict(), f, default_flow_style=False)
                
                st.success("설정이 저장되었습니다.")
                
                # RAG 엔진 재초기화
                st.info("RAG 엔진을 재초기화합니다...")
                rag_engine = initialize_rag_engine(settings)
                st.success("RAG 엔진이 재초기화되었습니다.")

def run_cli_interface(settings: Settings):
    """CLI 인터페이스 실행"""
    rag_engine = initialize_rag_engine(settings)
    
    print(f"Welcome to {settings.app_name} v{settings.version}")
    print("Type 'exit' to quit")
    
    while True:
        try:
            query = input("\nEnter your query: ")
            if query.lower() == 'exit':
                break
            
            response = rag_engine.query(query)
            print("\nResponse:", response.answer)
            
            if response.sources:
                print("\nSources:")
                for source in response.sources:
                    print(f"- {source['file']}: {source['content'][:100]}...")
            
            print(f"\nConfidence: {response.confidence:.2%}")
            print(f"Response time: {response.response_time:.2f}s")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="UVM Debug Agent")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    args = parser.parse_args()
    
    setup_logging()
    settings = load_app_settings()
    
    if args.cli:
        run_cli_interface(settings)
    else:
        run_web_interface(settings)

if __name__ == "__main__":
    main() 