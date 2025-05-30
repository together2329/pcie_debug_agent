"""Helper utilities for testing"""

import os
import json
import time
import psutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from contextlib import contextmanager
from unittest.mock import Mock
import numpy as np
from functools import wraps


class TestHelpers:
    """Helper utilities for tests"""
    
    @staticmethod
    def create_temp_log_file(
        content: str,
        filename: str = "test.log",
        directory: Optional[Path] = None
    ) -> Path:
        """Create a temporary log file for testing"""
        if directory is None:
            directory = Path(tempfile.gettempdir())
        
        filepath = directory / filename
        filepath.write_text(content)
        return filepath
    
    @staticmethod
    def create_temp_config_file(
        config: Dict[str, Any],
        filename: str = "test_config.yaml",
        directory: Optional[Path] = None
    ) -> Path:
        """Create a temporary configuration file"""
        import yaml
        
        if directory is None:
            directory = Path(tempfile.gettempdir())
        
        filepath = directory / filename
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return filepath
    
    @staticmethod
    def mock_api_response(
        status_code: int = 200,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> Mock:
        """Create a mock API response"""
        response = Mock()
        response.status_code = status_code
        response.ok = 200 <= status_code < 300
        
        if data is not None:
            response.json.return_value = data
            response.text = json.dumps(data)
        elif error is not None:
            response.json.side_effect = ValueError("Invalid JSON")
            response.text = error
        else:
            response.json.return_value = {}
            response.text = "{}"
        
        return response
    
    @staticmethod
    def assert_embeddings_similar(
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float = 0.9
    ) -> bool:
        """Assert two embeddings are similar using cosine similarity"""
        # Normalize embeddings
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        assert similarity >= threshold, \
            f"Embeddings not similar enough: {similarity:.3f} < {threshold}"
        
        return True
    
    @staticmethod
    @contextmanager
    def measure_memory_usage():
        """Context manager to measure memory usage"""
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: {initial_memory:.2f} MB -> {final_memory:.2f} MB "
              f"(+{memory_increase:.2f} MB)")
    
    @staticmethod
    def measure_execution_time(func: Callable) -> Callable:
        """Decorator to measure function execution time"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"{func.__name__} took {execution_time:.3f} seconds")
            return result
        return wrapper
    
    @staticmethod
    @contextmanager
    def assert_max_duration(seconds: float):
        """Assert that code block executes within time limit"""
        start_time = time.time()
        yield
        duration = time.time() - start_time
        assert duration <= seconds, \
            f"Execution took {duration:.3f}s, exceeding limit of {seconds}s"
    
    @staticmethod
    def wait_for_condition(
        condition: Callable[[], bool],
        timeout: float = 5.0,
        interval: float = 0.1,
        message: str = "Condition not met"
    ):
        """Wait for a condition to become true"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition():
                return
            time.sleep(interval)
        
        raise TimeoutError(f"{message} after {timeout}s")
    
    @staticmethod
    def create_mock_streamlit():
        """Create mock Streamlit module for testing UI components"""
        st = Mock()
        
        # Mock session state
        st.session_state = {}
        
        # Mock common Streamlit functions
        st.title = Mock()
        st.header = Mock()
        st.subheader = Mock()
        st.write = Mock()
        st.markdown = Mock()
        st.text = Mock()
        st.code = Mock()
        st.error = Mock()
        st.warning = Mock()
        st.info = Mock()
        st.success = Mock()
        
        # Mock input widgets
        st.text_input = Mock(return_value="test input")
        st.text_area = Mock(return_value="test area")
        st.number_input = Mock(return_value=42)
        st.slider = Mock(return_value=0.5)
        st.selectbox = Mock(return_value="option1")
        st.multiselect = Mock(return_value=["option1", "option2"])
        st.checkbox = Mock(return_value=True)
        st.radio = Mock(return_value="choice1")
        st.button = Mock(return_value=False)
        st.form_submit_button = Mock(return_value=False)
        
        # Mock layout
        st.columns = Mock(return_value=[Mock(), Mock()])
        st.sidebar = Mock()
        st.container = Mock()
        st.expander = Mock()
        st.form = Mock()
        
        # Mock data display
        st.dataframe = Mock()
        st.table = Mock()
        st.metric = Mock()
        st.json = Mock()
        
        # Mock file handling
        st.file_uploader = Mock(return_value=None)
        st.download_button = Mock()
        
        # Mock progress
        st.progress = Mock()
        st.spinner = Mock()
        
        return st
    
    @staticmethod
    def assert_valid_json(data: str) -> Dict[str, Any]:
        """Assert that string is valid JSON and return parsed data"""
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise AssertionError(f"Invalid JSON: {e}")
    
    @staticmethod
    def assert_files_equal(file1: Path, file2: Path):
        """Assert that two files have identical content"""
        content1 = file1.read_text()
        content2 = file2.read_text()
        assert content1 == content2, \
            f"Files {file1} and {file2} have different content"
    
    @staticmethod
    def create_test_vector_index(
        num_vectors: int = 1000,
        dimension: int = 384,
        index_type: str = "Flat"
    ) -> Mock:
        """Create a mock FAISS index for testing"""
        index = Mock()
        index.d = dimension
        index.ntotal = num_vectors
        index.is_trained = True
        
        # Mock search method
        def mock_search(query, k):
            # Return mock distances and indices
            distances = np.random.rand(query.shape[0], k).astype(np.float32)
            indices = np.random.randint(0, num_vectors, size=(query.shape[0], k))
            return distances, indices
        
        index.search = mock_search
        index.add = Mock()
        index.remove_ids = Mock()
        
        return index
    
    @staticmethod
    def capture_logs(logger_name: str = "root") -> List[str]:
        """Capture log messages for testing"""
        import logging
        
        logs = []
        handler = logging.Handler()
        handler.emit = lambda record: logs.append(record.getMessage())
        
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        
        return logs
    
    @staticmethod
    def mock_async_function(return_value: Any = None):
        """Create a mock async function"""
        async def async_mock(*args, **kwargs):
            return return_value
        
        mock = Mock()
        mock.side_effect = async_mock
        return mock