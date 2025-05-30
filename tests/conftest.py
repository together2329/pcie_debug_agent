"""Pytest configuration and shared fixtures"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings
from src.rag.vector_store import FAISSVectorStore
from src.rag.model_manager import ModelManager
from src.rag.enhanced_rag_engine import EnhancedRAGEngine


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_logs(test_data_dir) -> Dict[str, str]:
    """Sample log files for testing"""
    return {
        "pcie_error": """
[2024-01-15 10:23:45.123] ERROR: PCIe Device 0000:01:00.0 - Timeout waiting for completion
[2024-01-15 10:23:45.124] DEBUG: Register dump:
  DevCtl: 0x0010  DevSta: 0x0010
  LnkCap: 0x0042  LnkCtl: 0x0040
[2024-01-15 10:23:45.125] ERROR: PCIe AER: Uncorrectable Error detected
[2024-01-15 10:23:45.126] INFO: Attempting device reset...
        """.strip(),
        
        "pcie_init": """
[2024-01-15 09:00:00.000] INFO: PCIe subsystem initialization started
[2024-01-15 09:00:00.100] INFO: Scanning PCI bus 00
[2024-01-15 09:00:00.200] INFO: Found device: 0000:00:00.0 - Host Bridge
[2024-01-15 09:00:00.300] INFO: Found device: 0000:01:00.0 - Network Controller
[2024-01-15 09:00:00.400] INFO: PCIe initialization completed successfully
        """.strip(),
        
        "uvm_debug": """
# UVM_INFO @ 0: reporter [RNTST] Running test pcie_base_test...
# UVM_INFO test_pkg.sv(45) @ 100: uvm_test_top [CFGDB] Configuration database settings:
#   - pcie_agent.active = UVM_ACTIVE
#   - timeout = 10ms
# UVM_ERROR scoreboard.sv(123) @ 5000: uvm_test_top.env.sb [MISCOMPARE] Data mismatch:
#   Expected: 0xDEADBEEF
#   Actual:   0xCAFEBABE
# UVM_FATAL driver.sv(89) @ 10000: uvm_test_top.env.agent.driver [TIMEOUT] Transaction timeout after 5ms
        """.strip()
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_settings() -> Settings:
    """Mock settings for testing"""
    settings = Settings()
    
    # Configure with test values
    settings.app_name = "PCIe Debug Agent Test"
    settings.version = "1.0.0-test"
    
    # Embedding settings
    settings.embedding_provider = "local"
    settings.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    settings.embedding_config.api_key = "test-key"
    
    # LLM settings
    settings.llm_provider = "openai"
    settings.llm_model = "gpt-3.5-turbo"
    settings.llm_config.api_key = "test-llm-key"
    
    # RAG settings
    settings.chunk_size = 500
    settings.chunk_overlap = 50
    settings.context_window = 3
    settings.min_similarity = 0.7
    
    # Vector store settings
    settings.vector_store_path = "test_data/test_vectorstore"
    settings.index_type = "Flat"
    settings.embedding_dimension = 384
    
    return settings


@pytest.fixture
def mock_vector_store(temp_dir) -> FAISSVectorStore:
    """Mock vector store for testing"""
    store = FAISSVectorStore(
        index_path=str(temp_dir / "test_index"),
        index_type="Flat",
        dimension=384
    )
    return store


@pytest.fixture
def mock_model_manager() -> Mock:
    """Mock model manager for testing"""
    manager = Mock(spec=ModelManager)
    
    # Mock embedding generation
    manager.generate_embeddings.return_value = np.random.rand(5, 384).astype(np.float32)
    manager.generate_completion.return_value = "Test completion response"
    
    # Mock model info
    manager.embedding_model = "test-embedding-model"
    manager.embedding_dimension = 384
    
    return manager


@pytest.fixture
def mock_rag_engine(mock_vector_store, mock_model_manager) -> EnhancedRAGEngine:
    """Mock RAG engine for testing"""
    engine = EnhancedRAGEngine(
        vector_store=mock_vector_store,
        model_manager=mock_model_manager,
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000
    )
    return engine


@pytest.fixture
def sample_documents() -> list:
    """Sample documents for testing"""
    return [
        {
            "content": "PCIe error occurred during device initialization",
            "metadata": {
                "source": "test.log",
                "line": 10,
                "timestamp": "2024-01-15 10:00:00"
            }
        },
        {
            "content": "UVM testbench detected timeout in PCIe transaction",
            "metadata": {
                "source": "uvm_test.log",
                "line": 45,
                "severity": "ERROR"
            }
        },
        {
            "content": "Successfully completed PCIe enumeration",
            "metadata": {
                "source": "system.log",
                "line": 100,
                "severity": "INFO"
            }
        }
    ]


@pytest.fixture
def mock_llm_response() -> Dict[str, Any]:
    """Mock LLM response for testing"""
    return {
        "choices": [{
            "message": {
                "content": "Based on the PCIe logs, the error appears to be a timeout issue during device initialization."
            }
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test"""
    # Store original env vars
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Restore original env vars
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_api_responses():
    """Mock API responses for external services"""
    return {
        "openai_embedding": {
            "data": [{
                "embedding": [0.1] * 384,
                "index": 0
            }],
            "model": "text-embedding-3-small",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        },
        "openai_completion": {
            "id": "test-completion",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "This is a test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60
            }
        }
    }


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: Mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: Mark test as requiring external API"
    )