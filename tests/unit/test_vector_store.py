"""Unit tests for FAISS Vector Store"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import faiss

from src.rag.vector_store import FAISSVectorStore
from tests.utils.factories import TestDataFactory
from tests.utils.helpers import TestHelpers


class TestFAISSVectorStore:
    """Test cases for FAISS Vector Store"""
    
    # Initialization Tests
    
    @pytest.mark.unit
    def test_create_new_index(self, temp_dir):
        """Test creating a new vector store index"""
        store = FAISSVectorStore(
            index_path=str(temp_dir / "test_index"),
            index_type="Flat",
            dimension=384
        )
        
        assert store.dimension == 384
        assert store.index_type == "Flat"
        assert store.index is not None
        assert store.index.d == 384
        assert store.index.ntotal == 0
    
    @pytest.mark.unit
    def test_load_existing_index(self, temp_dir):
        """Test loading an existing index"""
        # Create and save an index
        index_path = str(temp_dir / "existing_index")
        store1 = FAISSVectorStore(index_path=index_path, dimension=384)
        
        # Add some vectors
        vectors = TestDataFactory.create_embeddings_batch(10, dimension=384)
        for i, vec in enumerate(vectors):
            store1.add_documents([{
                "id": f"doc_{i}",
                "content": f"Document {i}",
                "embedding": vec
            }])
        
        store1.save_index()
        
        # Load the saved index
        store2 = FAISSVectorStore(index_path=index_path, dimension=384)
        store2.load_index()
        
        assert store2.index.ntotal == 10
        assert len(store2.metadata) == 10
    
    @pytest.mark.unit
    def test_init_with_invalid_path(self):
        """Test initialization with invalid path"""
        with pytest.raises(Exception):
            FAISSVectorStore(
                index_path="/invalid/path/\0/index",
                dimension=384
            )
    
    @pytest.mark.unit
    def test_init_with_unsupported_index_type(self, temp_dir):
        """Test initialization with unsupported index type"""
        with pytest.raises(ValueError, match="Unsupported index type"):
            FAISSVectorStore(
                index_path=str(temp_dir / "test"),
                index_type="UnsupportedType",
                dimension=384
            )
    
    # Document Operations Tests
    
    @pytest.mark.unit
    def test_add_single_document(self, mock_vector_store):
        """Test adding a single document"""
        doc = {
            "id": "test_doc",
            "content": "This is a test document about PCIe",
            "embedding": TestDataFactory.create_embedding(dimension=384),
            "metadata": {"source": "test.log", "timestamp": "2024-01-15"}
        }
        
        mock_vector_store.add_documents([doc])
        
        # Verify document was added
        assert mock_vector_store.index.ntotal == 1
        assert "test_doc" in mock_vector_store.metadata
        assert mock_vector_store.metadata["test_doc"]["content"] == doc["content"]
    
    @pytest.mark.unit
    def test_add_batch_documents(self, mock_vector_store):
        """Test adding multiple documents in batch"""
        docs = []
        for i in range(100):
            docs.append({
                "id": f"doc_{i}",
                "content": f"Document {i} about PCIe errors",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i),
                "metadata": {"batch": "test_batch", "index": i}
            })
        
        mock_vector_store.add_documents(docs)
        
        assert mock_vector_store.index.ntotal == 100
        assert len(mock_vector_store.metadata) == 100
        assert all(f"doc_{i}" in mock_vector_store.metadata for i in range(100))
    
    @pytest.mark.unit
    def test_add_document_with_metadata(self, mock_vector_store):
        """Test adding document with rich metadata"""
        doc = {
            "id": "metadata_doc",
            "content": "PCIe link training failed",
            "embedding": TestDataFactory.create_embedding(dimension=384),
            "metadata": {
                "source": "pcie_debug.log",
                "timestamp": "2024-01-15T10:30:00",
                "severity": "ERROR",
                "device": "0000:01:00.0",
                "tags": ["pcie", "error", "link-training"]
            }
        }
        
        mock_vector_store.add_documents([doc])
        
        stored_metadata = mock_vector_store.metadata["metadata_doc"]["metadata"]
        assert stored_metadata["severity"] == "ERROR"
        assert stored_metadata["device"] == "0000:01:00.0"
        assert "link-training" in stored_metadata["tags"]
    
    @pytest.mark.unit
    def test_add_duplicate_document(self, mock_vector_store):
        """Test handling of duplicate document IDs"""
        doc1 = {
            "id": "duplicate_id",
            "content": "First document",
            "embedding": TestDataFactory.create_embedding(dimension=384, seed=1)
        }
        doc2 = {
            "id": "duplicate_id",
            "content": "Second document",
            "embedding": TestDataFactory.create_embedding(dimension=384, seed=2)
        }
        
        mock_vector_store.add_documents([doc1])
        mock_vector_store.add_documents([doc2])
        
        # Should update existing document
        assert mock_vector_store.index.ntotal == 1
        assert mock_vector_store.metadata["duplicate_id"]["content"] == "Second document"
    
    @pytest.mark.unit
    def test_update_existing_document(self, mock_vector_store):
        """Test updating an existing document"""
        # Add initial document
        doc = {
            "id": "update_test",
            "content": "Original content",
            "embedding": TestDataFactory.create_embedding(dimension=384, seed=1)
        }
        mock_vector_store.add_documents([doc])
        
        # Update document
        updated_doc = {
            "id": "update_test",
            "content": "Updated content",
            "embedding": TestDataFactory.create_embedding(dimension=384, seed=2)
        }
        mock_vector_store.update_document(updated_doc)
        
        assert mock_vector_store.metadata["update_test"]["content"] == "Updated content"
    
    @pytest.mark.unit
    def test_delete_document(self, mock_vector_store):
        """Test deleting a document"""
        # Add documents
        docs = [
            {
                "id": f"del_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(5)
        ]
        mock_vector_store.add_documents(docs)
        
        # Delete one document
        mock_vector_store.delete_document("del_doc_2")
        
        assert mock_vector_store.index.ntotal == 4
        assert "del_doc_2" not in mock_vector_store.metadata
        assert all(f"del_doc_{i}" in mock_vector_store.metadata for i in [0, 1, 3, 4])
    
    # Search Tests
    
    @pytest.mark.unit
    def test_similarity_search_basic(self, mock_vector_store):
        """Test basic similarity search"""
        # Add test documents
        docs = [
            {
                "id": "pcie_error",
                "content": "PCIe device timeout error detected",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=1)
            },
            {
                "id": "pcie_success",
                "content": "PCIe initialization successful",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=2)
            },
            {
                "id": "unrelated",
                "content": "CPU temperature normal",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=3)
            }
        ]
        mock_vector_store.add_documents(docs)
        
        # Search for similar documents
        query_embedding = TestDataFactory.create_embedding(dimension=384, seed=1)
        results = mock_vector_store.similarity_search(
            query_embedding=query_embedding,
            k=2
        )
        
        assert len(results) == 2
        assert results[0]["id"] == "pcie_error"  # Most similar
        assert results[0]["score"] > results[1]["score"]
    
    @pytest.mark.unit
    def test_similarity_search_with_filter(self, mock_vector_store):
        """Test similarity search with metadata filter"""
        # Add documents with different metadata
        docs = [
            {
                "id": f"doc_{i}",
                "content": f"PCIe log entry {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i),
                "metadata": {"severity": "ERROR" if i % 2 == 0 else "INFO"}
            }
            for i in range(10)
        ]
        mock_vector_store.add_documents(docs)
        
        # Search with filter
        query_embedding = TestDataFactory.create_embedding(dimension=384)
        results = mock_vector_store.similarity_search(
            query_embedding=query_embedding,
            k=5,
            filter={"severity": "ERROR"}
        )
        
        assert len(results) <= 5
        assert all(r["metadata"]["severity"] == "ERROR" for r in results)
    
    @pytest.mark.unit
    def test_similarity_search_with_threshold(self, mock_vector_store):
        """Test similarity search with score threshold"""
        # Add documents
        docs = [
            {
                "id": f"doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(20)
        ]
        mock_vector_store.add_documents(docs)
        
        # Search with similarity threshold
        query_embedding = TestDataFactory.create_embedding(dimension=384, seed=0)
        results = mock_vector_store.similarity_search(
            query_embedding=query_embedding,
            k=10,
            score_threshold=0.8
        )
        
        assert all(r["score"] >= 0.8 for r in results)
        assert len(results) <= 10
    
    @pytest.mark.unit
    def test_search_empty_index(self, mock_vector_store):
        """Test searching in empty index"""
        query_embedding = TestDataFactory.create_embedding(dimension=384)
        results = mock_vector_store.similarity_search(
            query_embedding=query_embedding,
            k=5
        )
        
        assert results == []
    
    @pytest.mark.unit
    def test_search_with_invalid_query(self, mock_vector_store):
        """Test search with invalid query embedding"""
        # Add some documents
        doc = {
            "id": "test",
            "content": "Test",
            "embedding": TestDataFactory.create_embedding(dimension=384)
        }
        mock_vector_store.add_documents([doc])
        
        # Invalid dimension
        with pytest.raises(ValueError):
            mock_vector_store.similarity_search(
                query_embedding=np.random.rand(256),  # Wrong dimension
                k=5
            )
        
        # Invalid type
        with pytest.raises(TypeError):
            mock_vector_store.similarity_search(
                query_embedding="not an array",
                k=5
            )
    
    # Index Management Tests
    
    @pytest.mark.unit
    def test_save_index(self, mock_vector_store, temp_dir):
        """Test saving index to disk"""
        # Add documents
        docs = [
            {
                "id": f"save_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(10)
        ]
        mock_vector_store.add_documents(docs)
        
        # Save index
        mock_vector_store.index_path = str(temp_dir / "saved_index")
        mock_vector_store.save_index()
        
        # Check files exist
        assert Path(mock_vector_store.index_path + ".faiss").exists()
        assert Path(mock_vector_store.index_path + ".meta").exists()
    
    @pytest.mark.unit
    def test_load_index(self, temp_dir):
        """Test loading index from disk"""
        # Create and save an index
        store1 = FAISSVectorStore(
            index_path=str(temp_dir / "load_test"),
            dimension=384
        )
        
        docs = [
            {
                "id": f"load_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(5)
        ]
        store1.add_documents(docs)
        store1.save_index()
        
        # Create new store and load
        store2 = FAISSVectorStore(
            index_path=str(temp_dir / "load_test"),
            dimension=384
        )
        store2.load_index()
        
        assert store2.index.ntotal == 5
        assert len(store2.metadata) == 5
        assert all(f"load_doc_{i}" in store2.metadata for i in range(5))
    
    @pytest.mark.unit
    def test_optimize_index(self, mock_vector_store):
        """Test index optimization"""
        # Add many documents
        docs = [
            {
                "id": f"opt_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(1000)
        ]
        mock_vector_store.add_documents(docs)
        
        # Optimize index
        initial_memory = mock_vector_store.get_memory_usage()
        mock_vector_store.optimize_index()
        optimized_memory = mock_vector_store.get_memory_usage()
        
        # Index should still work after optimization
        query = TestDataFactory.create_embedding(dimension=384)
        results = mock_vector_store.similarity_search(query, k=10)
        assert len(results) == 10
    
    @pytest.mark.unit
    def test_get_index_statistics(self, mock_vector_store):
        """Test getting index statistics"""
        # Add documents
        docs = [
            {
                "id": f"stat_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i),
                "metadata": {"type": "error" if i % 3 == 0 else "info"}
            }
            for i in range(30)
        ]
        mock_vector_store.add_documents(docs)
        
        stats = mock_vector_store.get_statistics()
        
        assert stats["total_documents"] == 30
        assert stats["dimension"] == 384
        assert stats["index_type"] == "Flat"
        assert "memory_usage" in stats
        assert "metadata_stats" in stats
        assert stats["metadata_stats"]["type"]["error"] == 10
        assert stats["metadata_stats"]["type"]["info"] == 20
    
    @pytest.mark.unit
    def test_clear_index(self, mock_vector_store):
        """Test clearing the index"""
        # Add documents
        docs = [
            {
                "id": f"clear_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(10)
        ]
        mock_vector_store.add_documents(docs)
        
        assert mock_vector_store.index.ntotal == 10
        
        # Clear index
        mock_vector_store.clear()
        
        assert mock_vector_store.index.ntotal == 0
        assert len(mock_vector_store.metadata) == 0
    
    # Performance Tests
    
    @pytest.mark.unit
    def test_batch_add_performance(self, mock_vector_store):
        """Test batch addition performance"""
        # Create large batch of documents
        batch_sizes = [10, 100, 1000]
        
        for batch_size in batch_sizes:
            docs = [
                {
                    "id": f"perf_doc_{i}",
                    "content": f"Document {i}",
                    "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
                }
                for i in range(batch_size)
            ]
            
            import time
            start_time = time.time()
            mock_vector_store.add_documents(docs)
            elapsed = time.time() - start_time
            
            # Should complete in reasonable time
            assert elapsed < batch_size * 0.001  # < 1ms per document
            
            mock_vector_store.clear()
    
    @pytest.mark.unit
    def test_search_performance_with_large_index(self, mock_vector_store):
        """Test search performance with large index"""
        # Add many documents
        num_docs = 10000
        docs = [
            {
                "id": f"large_doc_{i}",
                "content": f"Document {i}",
                "embedding": TestDataFactory.create_embedding(dimension=384, seed=i)
            }
            for i in range(num_docs)
        ]
        
        # Add in batches for efficiency
        batch_size = 1000
        for i in range(0, num_docs, batch_size):
            mock_vector_store.add_documents(docs[i:i+batch_size])
        
        # Test search performance
        query = TestDataFactory.create_embedding(dimension=384)
        
        import time
        start_time = time.time()
        results = mock_vector_store.similarity_search(query, k=100)
        elapsed = time.time() - start_time
        
        assert len(results) == 100
        assert elapsed < 0.1  # Should complete in < 100ms
    
    @pytest.mark.unit
    def test_memory_usage_optimization(self, mock_vector_store):
        """Test memory usage remains reasonable"""
        initial_memory = mock_vector_store.get_memory_usage()
        
        # Add documents incrementally
        for batch in range(10):
            docs = [
                {
                    "id": f"mem_doc_{batch}_{i}",
                    "content": f"Document {batch}_{i}",
                    "embedding": TestDataFactory.create_embedding(dimension=384, seed=batch*100+i)
                }
                for i in range(100)
            ]
            mock_vector_store.add_documents(docs)
        
        final_memory = mock_vector_store.get_memory_usage()
        memory_per_doc = (final_memory - initial_memory) / 1000
        
        # Memory usage should be reasonable (< 10KB per document)
        assert memory_per_doc < 10 * 1024  # bytes
    
    # Edge Cases
    
    @pytest.mark.unit
    def test_add_document_without_id(self, mock_vector_store):
        """Test adding document without ID"""
        doc = {
            "content": "Document without ID",
            "embedding": TestDataFactory.create_embedding(dimension=384)
        }
        
        # Should auto-generate ID
        mock_vector_store.add_documents([doc])
        
        assert mock_vector_store.index.ntotal == 1
        assert len(mock_vector_store.metadata) == 1
    
    @pytest.mark.unit
    def test_add_document_without_embedding(self, mock_vector_store):
        """Test adding document without embedding"""
        doc = {
            "id": "no_embedding",
            "content": "Document without embedding"
        }
        
        with pytest.raises(KeyError):
            mock_vector_store.add_documents([doc])
    
    @pytest.mark.unit
    def test_concurrent_operations(self, mock_vector_store):
        """Test concurrent add and search operations"""
        import concurrent.futures
        import threading
        
        # Lock for thread-safe operations
        lock = threading.Lock()
        
        def add_documents(start_idx):
            docs = [
                {
                    "id": f"concurrent_{start_idx}_{i}",
                    "content": f"Document {start_idx}_{i}",
                    "embedding": TestDataFactory.create_embedding(dimension=384, seed=start_idx+i)
                }
                for i in range(10)
            ]
            with lock:
                mock_vector_store.add_documents(docs)
        
        def search_documents():
            query = TestDataFactory.create_embedding(dimension=384)
            with lock:
                return mock_vector_store.similarity_search(query, k=5)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit mixed operations
            futures = []
            for i in range(5):
                futures.append(executor.submit(add_documents, i*10))
                futures.append(executor.submit(search_documents))
            
            # Wait for completion
            results = [f.result() for f in futures]
        
        # Verify final state
        assert mock_vector_store.index.ntotal == 50