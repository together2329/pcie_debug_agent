"""Test data factories for generating test fixtures"""

import random
import string
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
from faker import Faker

fake = Faker()


class TestDataFactory:
    """Factory for generating test data"""
    
    @staticmethod
    def create_log_entry(
        level: str = "ERROR",
        message: str = None,
        timestamp: datetime = None,
        source: str = "test.log",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a test log entry"""
        if message is None:
            messages = [
                "PCIe device timeout detected",
                "Transaction failed with error code 0x1234",
                "Device enumeration completed successfully",
                "Link training failed at Gen3 speed",
                "Memory allocation error in BAR region"
            ]
            message = random.choice(messages)
        
        if timestamp is None:
            timestamp = fake.date_time_between(start_date='-1d', end_date='now')
        
        entry = {
            "timestamp": timestamp.isoformat(),
            "level": level,
            "message": message,
            "source": source,
            "line_number": random.randint(1, 1000)
        }
        entry.update(kwargs)
        return entry
    
    @staticmethod
    def create_document(
        content: str = None,
        metadata: Dict[str, Any] = None,
        doc_type: str = "text",
        doc_id: str = None
    ) -> Dict[str, Any]:
        """Create a test document"""
        if content is None:
            content_templates = [
                "PCIe error analysis: {error}",
                "UVM test result: {result}",
                "Device configuration: {config}",
                "Performance metrics: {metrics}"
            ]
            template = random.choice(content_templates)
            content = template.format(
                error=fake.sentence(),
                result=random.choice(["PASS", "FAIL"]),
                config=fake.word(),
                metrics=f"{random.randint(1, 100)}ms"
            )
        
        if metadata is None:
            metadata = {
                "source": fake.file_name(extension="log"),
                "timestamp": fake.date_time().isoformat(),
                "author": fake.name(),
                "version": f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
            }
        
        if doc_id is None:
            doc_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))
        
        return {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "type": doc_type
        }
    
    @staticmethod
    def create_embedding(
        dimension: int = 384,
        normalized: bool = True,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Create a test embedding vector"""
        if seed is not None:
            np.random.seed(seed)
        
        embedding = np.random.randn(dimension).astype(np.float32)
        
        if normalized:
            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    @staticmethod
    def create_embeddings_batch(
        count: int,
        dimension: int = 384,
        normalized: bool = True
    ) -> List[np.ndarray]:
        """Create a batch of test embeddings"""
        return [
            TestDataFactory.create_embedding(dimension, normalized, seed=i)
            for i in range(count)
        ]
    
    @staticmethod
    def create_pcie_log_file(
        num_entries: int = 100,
        error_rate: float = 0.1,
        include_timestamps: bool = True
    ) -> str:
        """Create a complete PCIe log file content"""
        entries = []
        start_time = datetime.now() - timedelta(hours=1)
        
        for i in range(num_entries):
            timestamp = start_time + timedelta(seconds=i * 0.1)
            
            if random.random() < error_rate:
                level = "ERROR"
                messages = [
                    "PCIe AER: Uncorrectable Error detected",
                    "Device timeout waiting for completion",
                    "Link training failed",
                    "TLP poisoned bit set",
                    "Completion timeout"
                ]
            else:
                level = random.choice(["INFO", "DEBUG", "WARNING"])
                messages = [
                    "PCIe link up at Gen3 x16",
                    "Device enumeration in progress",
                    "Memory mapped to BAR0",
                    "MSI-X interrupts enabled",
                    "Power state changed to D0"
                ]
            
            message = random.choice(messages)
            
            if include_timestamps:
                entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] {level}: {message}"
            else:
                entry = f"{level}: {message}"
            
            entries.append(entry)
        
        return "\n".join(entries)
    
    @staticmethod
    def create_uvm_log_content(
        test_name: str = "pcie_base_test",
        include_errors: bool = True
    ) -> str:
        """Create UVM testbench log content"""
        lines = [
            f"# UVM_INFO @ 0: reporter [RNTST] Running test {test_name}...",
            "# UVM_INFO testbench.sv(10) @ 0: uvm_test_top [TEST_START] Starting test sequence",
            "# UVM_INFO pcie_agent.sv(45) @ 100: uvm_test_top.env.agent [AGENT] PCIe agent active",
        ]
        
        if include_errors:
            lines.extend([
                "# UVM_ERROR scoreboard.sv(123) @ 5000: uvm_test_top.env.sb [MISMATCH] Data mismatch detected",
                "#   Expected: 0xDEADBEEF",
                "#   Actual:   0xCAFEBABE",
                "# UVM_WARNING monitor.sv(67) @ 6000: uvm_test_top.env.mon [TIMEOUT] Transaction timeout",
                "# UVM_FATAL driver.sv(89) @ 10000: uvm_test_top.env.drv [FATAL] Critical error occurred"
            ])
        else:
            lines.extend([
                "# UVM_INFO scoreboard.sv(200) @ 5000: uvm_test_top.env.sb [MATCH] All transactions matched",
                "# UVM_INFO test_pkg.sv(300) @ 10000: uvm_test_top [TEST_DONE] Test completed successfully"
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def create_mock_llm_response(
        query: str,
        answer: str = None,
        confidence: float = 0.95,
        sources: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Create a mock LLM response"""
        if answer is None:
            answer = f"Based on the analysis of the logs, {fake.sentence()}"
        
        if sources is None:
            sources = [
                {
                    "file": fake.file_name(extension="log"),
                    "content": fake.sentence(),
                    "relevance": random.uniform(0.8, 1.0)
                }
                for _ in range(random.randint(1, 3))
            ]
        
        return {
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "metadata": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.1,
                "tokens_used": random.randint(100, 500),
                "response_time": random.uniform(0.5, 2.0)
            }
        }
    
    @staticmethod
    def create_vector_search_results(
        query_embedding: np.ndarray,
        num_results: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Create mock vector search results"""
        results = []
        
        for i in range(num_results):
            similarity = random.uniform(min_similarity, 1.0)
            results.append({
                "id": f"doc_{i}",
                "content": fake.sentence(),
                "metadata": {
                    "source": fake.file_name(),
                    "timestamp": fake.date_time().isoformat()
                },
                "similarity": similarity,
                "embedding": TestDataFactory.create_embedding()
            })
        
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
    
    @staticmethod
    def create_config_dict(
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a configuration dictionary"""
        config = {
            "app_name": "PCIe Debug Agent Test",
            "version": "1.0.0",
            "embedding": {
                "provider": "local",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "api_key": "test-embedding-key"
            },
            "llm": {
                "provider": provider,
                "model": model,
                "api_key": "test-llm-key",
                "temperature": 0.1,
                "max_tokens": 1000
            },
            "rag": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "context_window": 3,
                "min_similarity": 0.7
            },
            "vector_store": {
                "path": "test_vectorstore",
                "index_type": "Flat",
                "dimension": 384
            }
        }
        
        # Merge with any additional kwargs
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "llm.temperature"
                parts = key.split(".")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value
        
        return config