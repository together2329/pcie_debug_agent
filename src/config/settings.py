import os
from typing import Dict, Any, List, Optional
from pydantic import BaseSettings, Field
from pathlib import Path
import yaml

class Settings(BaseSettings):
    """전역 설정 관리"""
    
    # 앱 설정
    app_name: str = "UVM-RAG-Agent"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="APP_ENV")
    
    # 로그 수집
    log_directories: List[str] = Field(default_factory=list)
    error_patterns: Dict[str, str] = {
        "error": r"UVM_ERROR\s*:\s*(.+?)(?:\[|$)",
        "fatal": r"UVM_FATAL\s*:\s*(.+?)(?:\[|$)",
        "warning": r"UVM_WARNING\s*:\s*(.+?)(?:\[|$)"
    }
    
    # 임베딩 설정
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384
    
    # 벡터 DB 설정
    vector_store_type: str = "faiss"  # faiss or milvus
    faiss_index_path: str = "data/faiss_index"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    
    # LLM 설정
    llm_provider: str = "openai"  # openai or anthropic
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # RAG 설정
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 10
    max_iterations: int = 3
    
    # 배치 설정
    batch_size: int = 10
    parallel_workers: int = 4
    
    # 알림 설정
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        """YAML 파일에서 설정 로드"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data) 