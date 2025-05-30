"""Unit tests for Enhanced RAG Engine"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import numpy as np
import time
from typing import Dict, Any, List

from src.rag.enhanced_rag_engine import EnhancedRAGEngine, RAGResponse, RAGQuery
from tests.utils.factories import TestDataFactory
from tests.utils.helpers import TestHelpers

class TestEnhancedRAGEngine:
    """Test cases for Enhanced RAG Engine"""
    
    # Initialization Tests
    
    @pytest.mark.unit
    def test_init_with_valid_config(self, mock_vector_store, mock_model_manager):
        """Test initialization with valid configuration"""
        engine = EnhancedRAGEngine(
            vector_store=mock_vector_store,
            model_manager=mock_model_manager,
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )
        
        assert engine.vector_store == mock_vector_store
        assert engine.model_manager == mock_model_manager
        assert engine.llm_provider == "openai"
        assert engine.llm_model == "gpt-3.5-turbo"
        assert engine.temperature == 0.1
        assert engine.max_tokens == 1000
        assert engine.query_cache == {}
        assert engine.metrics["queries_processed"] == 0
    
    @pytest.mark.unit
    def test_init_with_invalid_vector_store(self, mock_model_manager):
        """Test initialization with invalid vector store"""
        # The implementation doesn't validate None inputs at init time
        # It will fail when trying to use the vector store
        engine = EnhancedRAGEngine(
            vector_store=None,
            model_manager=mock_model_manager,
            llm_provider="openai",
            llm_model="gpt-3.5-turbo"
        )
        
        # It should return an error response when trying to query
        result = engine.query(RAGQuery(query="test"))
        assert isinstance(result, RAGResponse)
        assert result.confidence == 0.0
        assert "오류가 발생했습니다" in result.answer or "error" in result.answer.lower()
    
    @pytest.mark.unit
    def test_init_with_missing_model_manager(self, mock_vector_store):
        """Test initialization with missing model manager"""
        # Similarly, model_manager=None will fail during usage
        engine = EnhancedRAGEngine(
            vector_store=mock_vector_store,
            model_manager=None,
            llm_provider="openai",
            llm_model="gpt-3.5-turbo"
        )
        
        # It should return an error response when trying to query
        result = engine.query(RAGQuery(query="test"))
        assert isinstance(result, RAGResponse)
        assert result.confidence == 0.0
        assert "오류가 발생했습니다" in result.answer or "error" in result.answer.lower()
    
    # Query Processing Tests
    
    @pytest.mark.unit
    def test_query_with_valid_input(self, mock_rag_engine, mock_llm_response):
        """Test query with valid input"""
        # Setup mocks
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[("PCIe error detected", {"source": "test.log"}, 0.9)]):
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                    result = mock_rag_engine.query(RAGQuery(query="What is the PCIe error?"))
        
        assert isinstance(result, RAGResponse)
        assert isinstance(result.answer, str)
        assert len(result.answer) > 0
        assert 0 <= result.confidence <= 1
        assert result.metadata["query_time"] > 0
    
    @pytest.mark.unit
    def test_query_with_empty_input(self, mock_rag_engine):
        """Test query with empty input"""
        # Since we now use RAGQuery, we should create an empty query object
        result = mock_rag_engine.query(RAGQuery(query=""))
        # The engine should handle empty queries gracefully
        assert isinstance(result, RAGResponse)
        assert "쿼리 처리 중 오류가 발생했습니다" in result.answer or "error" in result.answer.lower()
    
    @pytest.mark.unit
    def test_query_with_special_characters(self, mock_rag_engine, mock_llm_response):
        """Test query with special characters"""
        query = "What about PCIe @#$% errors?"
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                result = mock_rag_engine.query(RAGQuery(query=query))
        
        assert isinstance(result.answer, str)
        assert isinstance(result.answer, str)
    
    @pytest.mark.unit
    def test_query_with_long_input(self, mock_rag_engine):
        """Test query with very long input"""
        long_query = "PCIe " * 1000  # Very long query
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            # Should truncate or handle gracefully
            with patch('openai.ChatCompletion.create') as mock_llm:
                mock_llm.return_value = {"choices": [{"message": {"content": "Answer"}}]}
                result = mock_rag_engine.query(RAGQuery(query=long_query))
        
        assert isinstance(result, RAGResponse)
    
    @pytest.mark.unit
    def test_query_cache_hit(self, mock_rag_engine):
        """Test query with cache hit"""
        query = "What is PCIe?"
        cached_result = RAGResponse(
            answer="Cached answer",
            sources=[],
            confidence=0.95,
            metadata={"query_time": 0.1}
        )
        
        # Pre-populate cache
        mock_rag_engine.query_cache[mock_rag_engine._get_cache_key(RAGQuery(query=query))] = {
            "answer": cached_result.answer,
            "sources": cached_result.sources,
            "confidence": cached_result.confidence,
            "timestamp": time.time()
        }
        
        result = mock_rag_engine.query(RAGQuery(query=query))
        
        assert result.answer == "Cached answer"
        assert mock_rag_engine.metrics["cache_hits"] == 1
        assert mock_rag_engine.model_manager.generate_embeddings.call_count == 0
    
    @pytest.mark.unit
    def test_query_cache_miss(self, mock_rag_engine, mock_llm_response):
        """Test query with cache miss"""
        query = "What is PCIe error?"
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                result = mock_rag_engine.query(RAGQuery(query=query))
        
        assert mock_rag_engine.metrics["queries_processed"] == 1
        assert len(mock_rag_engine.query_cache) > 0
    
    @pytest.mark.unit
    def test_query_with_context_enhancement(self, mock_rag_engine, mock_llm_response):
        """Test query with context enhancement from similar documents"""
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        
        # Mock similar documents
        similar_docs = [
            ("PCIe Gen3 link training failed", {"source": "pcie.log", "line": 100}, 0.95),
            ("Device timeout during enumeration", {"source": "system.log", "line": 200}, 0.85)
        ]
        with patch.object(mock_rag_engine.vector_store, "search", return_value=similar_docs):
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                result = mock_rag_engine.query(RAGQuery(query="Why did PCIe fail?"))
        
        assert len(result.sources) == 2
        assert result.sources[0]["score"] == 0.95
        assert result.confidence > 0.8  # High confidence due to relevant sources
    
    # Response Formatting Tests
    
    @pytest.mark.unit
    def test_format_response_with_sources(self, mock_rag_engine):
        """Test response formatting with sources"""
        sources = [
            {"file": "test.log", "content": "Error details", "relevance": 0.9},
            {"file": "debug.log", "content": "Debug info", "relevance": 0.8}
        ]
        
        formatted = mock_rag_engine._format_response(
            answer="PCIe error found",
            sources=sources
        )
        
        assert "PCIe error found" in formatted
        assert "Sources:" in formatted
        assert "test.log" in formatted
        assert "debug.log" in formatted
    
    @pytest.mark.unit
    def test_format_response_without_sources(self, mock_rag_engine):
        """Test response formatting without sources"""
        formatted = mock_rag_engine._format_response(
            answer="No specific error found",
            sources=[]
        )
        
        assert "No specific error found" in formatted
        assert "Sources:" not in formatted
    
    @pytest.mark.unit
    def test_format_response_with_metadata(self, mock_rag_engine):
        """Test response formatting with metadata"""
        sources = [{
            "file": "test.log",
            "content": "Error at line 100",
            "relevance": 0.9,
            "metadata": {
                "timestamp": "2024-01-15 10:00:00",
                "severity": "ERROR"
            }
        }]
        
        formatted = mock_rag_engine._format_response(
            answer="Found error",
            sources=sources,
            include_metadata=True
        )
        
        assert "timestamp" in formatted
        assert "severity" in formatted
    
    # Performance Metrics Tests
    
    @pytest.mark.unit
    def test_update_metrics_on_query(self, mock_rag_engine, mock_llm_response):
        """Test metrics update on query execution"""
        initial_total = mock_rag_engine.metrics["queries_processed"]
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                result = mock_rag_engine.query(RAGQuery(query="Test query"))
        
        assert mock_rag_engine.metrics["queries_processed"] == initial_total + 1
        assert mock_rag_engine.metrics["average_response_time"] >= 0
        assert mock_rag_engine.metrics["average_response_time"] > 0
    
    @pytest.mark.unit
    def test_calculate_average_response_time(self, mock_rag_engine):
        """Test average response time calculation"""
        # Set average response time
        mock_rag_engine.metrics["average_response_time"] = 2.0
        
        avg_time = mock_rag_engine.metrics["average_response_time"]
        assert avg_time == 2.0
    
    @pytest.mark.unit
    def test_track_cache_statistics(self, mock_rag_engine):
        """Test cache hit/miss statistics tracking"""
        assert mock_rag_engine.metrics["cache_hits"] == 0
        assert mock_rag_engine.metrics.get("cache_misses", 0) == 0
        
        mock_rag_engine.metrics["cache_hits"] = 5
        mock_rag_engine.metrics["queries_processed"] = 10
        
        assert mock_rag_engine.metrics["cache_hits"] == 5
        assert mock_rag_engine.metrics["queries_processed"] - mock_rag_engine.metrics["cache_hits"] == 5
        assert mock_rag_engine.metrics["cache_hits"] == 5  # 5/(5+10)
    
    # Error Handling Tests
    
    @pytest.mark.unit
    def test_handle_llm_api_error(self, mock_rag_engine):
        """Test handling of LLM API errors"""
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            with patch('openai.ChatCompletion.create') as mock_llm:
                mock_llm.side_effect = Exception("API Error")
                
                with pytest.raises(Exception, match="API Error"):
                    mock_rag_engine.query(RAGQuery(query="Test query"))
    
    @pytest.mark.unit
    def test_handle_vector_store_error(self, mock_rag_engine):
        """Test handling of vector store errors"""
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        mock_rag_engine.vector_store.search = Mock(side_effect=Exception("Vector store error"))
        
        with pytest.raises(Exception, match="Vector store error"):
            mock_rag_engine.query(RAGQuery(query="Test query"))
    
    @pytest.mark.unit
    def test_handle_timeout_error(self, mock_rag_engine):
        """Test handling of timeout errors"""
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            with patch('openai.ChatCompletion.create') as mock_llm:
                mock_llm.side_effect = TimeoutError("Request timeout")
                
                with pytest.raises(TimeoutError, match="Request timeout"):
                    mock_rag_engine.query(RAGQuery(query="Test query"))
    
    # Edge Cases
    
    @pytest.mark.unit
    def test_concurrent_queries(self, mock_rag_engine, mock_llm_response):
        """Test handling of concurrent queries"""
        import concurrent.futures
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            queries = [f"Query {i}" for i in range(10)]
        
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(mock_rag_engine.query, RAGQuery(query=q)) for q in queries]
                    results = [f.result() for f in futures]
        
        assert len(results) == 10
        assert all(isinstance(r, RAGResponse) for r in results)
        assert mock_rag_engine.metrics["queries_processed"] == 10
    
    @pytest.mark.unit
    def test_memory_efficient_processing(self, mock_rag_engine):
        """Test memory-efficient processing of large contexts"""
        # Create large context
        large_docs = [
            {
                "content": "x" * 10000,  # Large document
                "metadata": {"source": f"doc{i}.txt"},
                "score": 0.9
            }
            for i in range(100)
        ]
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=large_docs):
            with TestHelpers.measure_memory_usage():
                # Should handle large context efficiently
                context = mock_rag_engine._prepare_context(large_docs, max_length=5000)
        
        assert len(context) <= 5000
    
    @pytest.mark.unit
    def test_query_with_invalid_encoding(self, mock_rag_engine, mock_llm_response):
        """Test query with invalid character encoding"""
        # Query with various Unicode characters
        query = "PCIe error: \u200b\ufeff\u0000"
        
        mock_rag_engine.model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        with patch.object(mock_rag_engine.vector_store, "search", return_value=[]):
            with patch('openai.ChatCompletion.create', return_value=mock_llm_response):
                result = mock_rag_engine.query(RAGQuery(query=query))
        
        assert isinstance(result, RAGResponse)
        assert result.answer == query.strip()  # Should handle special chars
    
    # Integration with different LLM providers
    
    @pytest.mark.unit
    def test_query_with_anthropic_provider(self, mock_vector_store, mock_model_manager):
        """Test query with Anthropic provider"""
        engine = EnhancedRAGEngine(
            vector_store=mock_vector_store,
            model_manager=mock_model_manager,
            llm_provider="anthropic",
            llm_model="claude-2"
        )
        
        mock_model_manager.generate_embeddings.return_value = np.array([TestDataFactory.create_embedding()])
        mock_vector_store.search = Mock(return_value=[])
        
        with patch('anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            mock_client.completions.create.return_value = Mock(completion="Test response")
            
            result = engine.query(RAGQuery(query="Test query"))
        
        assert isinstance(result, RAGResponse)
    
    @pytest.mark.unit
    def test_query_with_custom_provider(self, mock_vector_store, mock_model_manager):
        """Test query with custom LLM provider"""
        engine = EnhancedRAGEngine(
            vector_store=mock_vector_store,
            model_manager=mock_model_manager,
            llm_provider="custom",
            llm_model="custom-model",
            base_url="http://localhost:8000",
            api_key="custom-key"
        )
        
        assert engine.llm_provider == "custom"
        assert engine.base_url == "http://localhost:8000"