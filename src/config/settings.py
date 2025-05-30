import os
from typing import Dict, Any, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import yaml
from dataclasses import dataclass, field

@dataclass
class EmbeddingConfig:
    """임베딩 설정"""
    model: str = "text-embedding-3-small"
    provider: str = "openai"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    dimension: int = 1536
    batch_size: int = 32

@dataclass
class LLMConfig:
    """LLM 설정"""
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000

@dataclass
class RAGConfig:
    """RAG 설정"""
    chunk_size: int = 500
    chunk_overlap: int = 50
    context_window: int = 5
    min_similarity: float = 0.5
    rerank: bool = True

@dataclass
class VectorStoreConfig:
    """벡터 스토어 설정"""
    index_path: Path = Path("data/vectorstore")
    index_type: str = "IndexFlatIP"
    dimension: int = 1536

@dataclass
class UIConfig:
    """UI 설정"""
    theme: str = "light"
    max_width: int = 1200
    show_confidence: bool = True
    show_sources: bool = True
    group_by_file: bool = True

@dataclass
class Settings:
    """애플리케이션 설정"""
    # 기본 설정
    app_name: str = "UVM Debug Agent"
    version: str = "1.0.0"
    debug: bool = False
    
    # 데이터 디렉토리
    data_dir: Path = Path("data")
    log_dir: Path = Path("logs")
    report_dir: Path = Path("reports")
    
    # 임베딩 설정
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # LLM 설정
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # RAG 설정
    rag: RAGConfig = field(default_factory=RAGConfig)
    
    # 벡터 스토어 설정
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    
    # UI 설정
    ui: UIConfig = field(default_factory=UIConfig)
    
    @classmethod
    def from_yaml(cls, config: Dict[str, Any]) -> 'Settings':
        """YAML 설정에서 Settings 객체 생성"""
        settings = cls()
        
        # 기본 설정
        settings.app_name = config.get('app_name', settings.app_name)
        settings.version = config.get('version', settings.version)
        settings.debug = config.get('debug', settings.debug)
        
        # 디렉토리 설정
        settings.data_dir = Path(config.get('data_dir', settings.data_dir))
        settings.log_dir = Path(config.get('log_dir', settings.log_dir))
        settings.report_dir = Path(config.get('report_dir', settings.report_dir))
        
        # 임베딩 설정
        embedding_config = config.get('embedding', {})
        settings.embedding.model = embedding_config.get('model', settings.embedding.model)
        settings.embedding.provider = embedding_config.get('provider', settings.embedding.provider)
        settings.embedding.api_key = embedding_config.get('api_key') or os.getenv('EMBEDDING_API_KEY')
        settings.embedding.api_base_url = embedding_config.get('api_base_url') or os.getenv('EMBEDDING_API_BASE_URL')
        settings.embedding.dimension = embedding_config.get('dimension', settings.embedding.dimension)
        settings.embedding.batch_size = embedding_config.get('batch_size', settings.embedding.batch_size)
        
        # LLM 설정
        llm_config = config.get('llm', {})
        settings.llm.provider = llm_config.get('provider', settings.llm.provider)
        settings.llm.model = llm_config.get('model', settings.llm.model)
        settings.llm.api_key = llm_config.get('api_key') or os.getenv('LLM_API_KEY')
        settings.llm.api_base_url = llm_config.get('api_base_url') or os.getenv('LLM_API_BASE_URL')
        settings.llm.temperature = llm_config.get('temperature', settings.llm.temperature)
        settings.llm.max_tokens = llm_config.get('max_tokens', settings.llm.max_tokens)
        
        # RAG 설정
        rag_config = config.get('rag', {})
        settings.rag.chunk_size = rag_config.get('chunk_size', settings.rag.chunk_size)
        settings.rag.chunk_overlap = rag_config.get('chunk_overlap', settings.rag.chunk_overlap)
        settings.rag.context_window = rag_config.get('context_window', settings.rag.context_window)
        settings.rag.min_similarity = rag_config.get('min_similarity', settings.rag.min_similarity)
        settings.rag.rerank = rag_config.get('rerank', settings.rag.rerank)
        
        # 벡터 스토어 설정
        vector_store_config = config.get('vector_store', {})
        settings.vector_store.index_path = Path(vector_store_config.get('index_path', settings.vector_store.index_path))
        settings.vector_store.index_type = vector_store_config.get('index_type', settings.vector_store.index_type)
        settings.vector_store.dimension = vector_store_config.get('dimension', settings.vector_store.dimension)
        
        # UI 설정
        ui_config = config.get('ui', {})
        settings.ui.theme = ui_config.get('theme', settings.ui.theme)
        settings.ui.max_width = ui_config.get('max_width', settings.ui.max_width)
        settings.ui.show_confidence = ui_config.get('show_confidence', settings.ui.show_confidence)
        settings.ui.show_sources = ui_config.get('show_sources', settings.ui.show_sources)
        settings.ui.group_by_file = ui_config.get('group_by_file', settings.ui.group_by_file)
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Settings 객체를 딕셔너리로 변환"""
        return {
            'app_name': self.app_name,
            'version': self.version,
            'debug': self.debug,
            'data_dir': str(self.data_dir),
            'log_dir': str(self.log_dir),
            'report_dir': str(self.report_dir),
            'embedding': {
                'model': self.embedding.model,
                'provider': self.embedding.provider,
                'api_key': self.embedding.api_key,
                'api_base_url': self.embedding.api_base_url,
                'dimension': self.embedding.dimension,
                'batch_size': self.embedding.batch_size
            },
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'api_key': self.llm.api_key,
                'api_base_url': self.llm.api_base_url,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens
            },
            'rag': {
                'chunk_size': self.rag.chunk_size,
                'chunk_overlap': self.rag.chunk_overlap,
                'context_window': self.rag.context_window,
                'min_similarity': self.rag.min_similarity,
                'rerank': self.rag.rerank
            },
            'vector_store': {
                'index_path': str(self.vector_store.index_path),
                'index_type': self.vector_store.index_type,
                'dimension': self.vector_store.dimension
            },
            'ui': {
                'theme': self.ui.theme,
                'max_width': self.ui.max_width,
                'show_confidence': self.ui.show_confidence,
                'show_sources': self.ui.show_sources,
                'group_by_file': self.ui.group_by_file
            }
        }
    
    def validate(self):
        """설정 유효성 검사"""
        # API 키 검사
        if self.embedding.provider != "local" and not self.embedding.api_key:
            raise ValueError("임베딩 API 키가 필요합니다.")
        
        if self.llm.provider != "local" and not self.llm.api_key:
            raise ValueError("LLM API 키가 필요합니다.")
        
        # 디렉토리 생성
        try:
            # 데이터 디렉토리
            data_dir = Path(self.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # 로그 디렉토리
            log_dir = Path(self.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 보고서 디렉토리
            report_dir = Path(self.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 벡터 스토어 디렉토리
            vector_store_path = Path(self.vector_store.index_path)
            if vector_store_path.is_file():
                # 파일이 이미 존재하는 경우, 부모 디렉토리만 생성
                vector_store_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                # 디렉토리인 경우 생성
                vector_store_path.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            raise ValueError(f"디렉토리 생성 실패: {str(e)}")
        
        # 로그 디렉토리 검사
        for log_dir in self.log_directories:
            log_path = Path(log_dir)
            if not log_path.exists():
                log_path.mkdir(parents=True, exist_ok=True)
        
        # 알림 설정 검사
        if self.notifications.slack_webhook_url and not self.notifications.slack_webhook_url.startswith("https://"):
            raise ValueError("Slack 웹훅 URL이 올바르지 않습니다.")
        
        if self.notifications.smtp.username and not self.notifications.smtp.password:
            raise ValueError("SMTP 비밀번호가 필요합니다.") 