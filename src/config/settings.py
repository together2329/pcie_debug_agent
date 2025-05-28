import os
from typing import Dict, Any, List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
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

class Settings(BaseSettings):
    """Main settings class"""
    # App settings
    app_name: str = "UVM-RAG-Agent"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", env="APP_ENV")
    
    # Log collection
    log_directories: List[str] = Field(default_factory=list)
    error_patterns: Dict[str, str] = {
        "error": r"UVM_ERROR\s*:\s*(.+?)(?:\[|$)",
        "fatal": r"UVM_FATAL\s*:\s*(.+?)(?:\[|$)",
        "warning": r"UVM_WARNING\s*:\s*(.+?)(?:\[|$)"
    }
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_dimension: int = 384
    
    # Vector store settings
    vector_store_type: str = "faiss"
    faiss_index_path: str = "data/faiss_index"
    
    # LLM settings
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    
    # RAG settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 10
    max_iterations: int = 3
    
    # Batch settings
    batch_size: int = 10
    parallel_workers: int = 4
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Settings":
        """Load settings from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def validate(self) -> bool:
        """Validate settings"""
        # Check required directories
        for log_dir in self.log_directories:
            if not Path(log_dir).exists():
                raise ValueError(f"Log directory does not exist: {log_dir}")
        
        # Validate API keys
        if self.llm_provider == 'openai' and not self.openai_api_key:
            raise ValueError("OpenAI API key is required for OpenAI provider")
        elif self.llm_provider == 'anthropic' and not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required for Anthropic provider")
        
        # Validate numeric values
        if self.embedding_batch_size <= 0:
            raise ValueError("Embedding batch size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap must be non-negative")
        if self.retrieval_top_k <= 0:
            raise ValueError("Retrieval top-k must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.parallel_workers <= 0:
            raise ValueError("Number of parallel workers must be positive")
        
        return True 