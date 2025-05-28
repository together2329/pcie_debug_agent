import os
from typing import Dict, Any, List, Optional
from pydantic import BaseSettings, Field
from pathlib import Path
import yaml
from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    """Configuration for embedding models"""
    model: str
    batch_size: int
    dimension: int

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    provider: str
    model: str
    temperature: float
    max_tokens: int
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

@dataclass
class RAGConfig:
    """Configuration for RAG components"""
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int

@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    batch_size: int
    parallel_workers: int

@dataclass
class Settings:
    """Main settings class"""
    embedding: EmbeddingConfig
    llm: LLMConfig
    rag: RAGConfig
    log_directories: List[str]
    batch: BatchConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Settings':
        """Create Settings instance from dictionary"""
        return cls(
            embedding=EmbeddingConfig(**config_dict['embedding']),
            llm=LLMConfig(**config_dict['llm']),
            rag=RAGConfig(**config_dict['rag']),
            log_directories=config_dict['log_directories'],
            batch=BatchConfig(**config_dict['batch'])
        )
    
    def to_dict(self) -> Dict:
        """Convert Settings instance to dictionary"""
        return {
            'embedding': {
                'model': self.embedding.model,
                'batch_size': self.embedding.batch_size,
                'dimension': self.embedding.dimension
            },
            'llm': {
                'provider': self.llm.provider,
                'model': self.llm.model,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
                'openai_api_key': self.llm.openai_api_key,
                'anthropic_api_key': self.llm.anthropic_api_key
            },
            'rag': {
                'chunk_size': self.rag.chunk_size,
                'chunk_overlap': self.rag.chunk_overlap,
                'retrieval_top_k': self.rag.retrieval_top_k
            },
            'log_directories': self.log_directories,
            'batch': {
                'batch_size': self.batch.batch_size,
                'parallel_workers': self.batch.parallel_workers
            }
        }
    
    def validate(self) -> bool:
        """Validate settings"""
        # Check required directories
        for log_dir in self.log_directories:
            if not Path(log_dir).exists():
                raise ValueError(f"Log directory does not exist: {log_dir}")
        
        # Validate API keys
        if self.llm.provider == 'openai' and not self.llm.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI provider")
        elif self.llm.provider == 'anthropic' and not self.llm.anthropic_api_key:
            raise ValueError("Anthropic API key is required for Anthropic provider")
        
        # Validate numeric values
        if self.embedding.batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        if self.rag.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.rag.chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        if self.rag.retrieval_top_k <= 0:
            raise ValueError("Retrieval top-k must be positive")
        if self.batch.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.batch.parallel_workers <= 0:
            raise ValueError("Number of parallel workers must be positive")
        
        return True

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