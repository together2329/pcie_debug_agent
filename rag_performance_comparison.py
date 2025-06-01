#!/usr/bin/env python3
"""
RAG Performance Comparison Tool
Compares OpenAI and Local embedding models for RAG performance
"""
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["GGML_METAL_LOG_LEVEL"] = "0"

from src.models.embedding_selector import get_embedding_selector
from src.vectorstore.faiss_store import FAISSVectorStore
from src.rag.enhanced_rag_engine import EnhancedRAGEngine
from src.models.model_selector import get_model_selector
from src.cli.utils.token_counter import TokenCounter

# Simple model wrapper for testing
class ModelWrapper:
    def __init__(self, selector, embedding_selector, rag_enabled=False):
        self.selector = selector
        self.embedding_selector = embedding_selector
        self.rag_enabled = rag_enabled
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ['provider', 'model']}
        return self.selector.generate_completion(prompt, **filtered_kwargs)
    
    def generate_embeddings(self, texts):
        if not self.rag_enabled:
            raise RuntimeError("RAG not enabled - no vector database")
        provider = self.embedding_selector.get_current_provider()
        return provider.encode(texts)

class RAGPerformanceComparator:
    """Compare RAG performance between different embedding models"""
    
    def __init__(self):
        self.embedding_selector = get_embedding_selector()
        self.model_selector = get_model_selector()
        self.token_counter = TokenCounter()
        self.test_queries = [
            "what causes AER errors?",
            "how to debug LTSSM timeout?", 
            "explain PCIe power management L0s L1 states",
            "why TLP malformed packets occur?",
            "signal integrity issues troubleshooting",
            "PCIe link training failure analysis",
            "completion timeout error handling",
            "how to configure AER registers?",
            "what is recovery state machine?",
            "bandwidth optimization techniques"
        ]
        self.results = {}
        
    def run_comparison(self):
        """Run comprehensive RAG performance comparison"""
        print("🔍 RAG Performance Comparison Tool")
        print("="*80)
        
        # Test models to compare
        models_to_test = [
            "text-embedding-3-small",    # OpenAI
            "all-MiniLM-L6-v2"          # Local
        ]
        
        for model in models_to_test:
            if self.embedding_selector.is_available(model):
                print(f"\n🧮 Testing {model}...")
                self.results[model] = self._test_model_performance(model)
            else:
                print(f"⚠️  {model} not available, skipping...")
        
        # Generate comparison report
        self._generate_report()
        
    def _test_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Test performance metrics for a specific embedding model"""
        print(f"   Switching to {model_name}...")
        
        # Switch to the model
        self.embedding_selector.switch_model(model_name)
        
        # Get model info
        model_info = self.embedding_selector.get_model_info()
        provider = self.embedding_selector.get_current_provider()
        
        results = {
            "model_info": model_info,
            "embedding_tests": [],
            "rag_tests": [],
            "vector_rebuild_time": 0,
            "vector_search_times": [],
            "relevance_scores": [],
            "response_qualities": []
        }
        
        # Test 1: Embedding Generation Speed
        print(f"   📊 Testing embedding generation speed...")
        embedding_times = self._test_embedding_speed(provider)
        results["embedding_tests"] = embedding_times
        
        # Test 2: Vector Database Rebuild (if needed)
        print(f"   🔧 Checking vector database compatibility...")
        vector_rebuild_time = self._ensure_compatible_vectordb(model_name)
        results["vector_rebuild_time"] = vector_rebuild_time
        
        # Test 3: RAG Query Performance
        print(f"   🔍 Testing RAG query performance...")
        for i, query in enumerate(self.test_queries, 1):
            print(f"      Query {i}/{len(self.test_queries)}: {query[:30]}...")
            
            rag_result = self._test_rag_query(query)
            results["rag_tests"].append(rag_result)
            results["vector_search_times"].append(rag_result["search_time"])
            if rag_result["relevance_score"] > 0:
                results["relevance_scores"].append(rag_result["relevance_score"])
        
        return results
    
    def _test_embedding_speed(self, provider) -> List[Dict[str, float]]:
        """Test embedding generation speed"""
        test_texts = [
            "PCIe link training failure",
            "Advanced Error Reporting configuration and monitoring",
            "Transaction Layer Packet malformed error analysis and debugging procedures",
            "Power management state transitions L0 L0s L1 L2 L3 and ASPM configuration",
            "Signal integrity troubleshooting for high-speed PCIe links with eye diagram analysis"
        ]
        
        embedding_times = []
        
        for text in test_texts:
            start_time = time.time()
            embeddings = provider.encode([text])
            end_time = time.time()
            
            embedding_times.append({
                "text_length": len(text),
                "embedding_time": end_time - start_time,
                "dimension": len(embeddings[0]) if len(embeddings) > 0 else 0
            })
        
        return embedding_times
    
    def _ensure_compatible_vectordb(self, model_name: str) -> float:
        """Ensure vector database is compatible with embedding model"""
        vector_db_path = Path("data/vectorstore")
        
        if not vector_db_path.exists():
            print(f"      Vector database not found, building...")
            return self._rebuild_vectordb()
        
        try:
            # Check compatibility
            store = FAISSVectorStore.load(str(vector_db_path))
            embedding_dim = self.embedding_selector.get_current_provider().get_dimension()
            
            if store.dimension != embedding_dim:
                print(f"      Dimension mismatch ({store.dimension}D vs {embedding_dim}D), rebuilding...")
                return self._rebuild_vectordb()
            else:
                print(f"      Vector database compatible ({embedding_dim}D)")
                return 0
                
        except Exception as e:
            print(f"      Error loading vector database, rebuilding...")
            return self._rebuild_vectordb()
    
    def _rebuild_vectordb(self) -> float:
        """Rebuild vector database with current embedding model"""
        import shutil
        from src.processors.document_chunker import DocumentChunker
        from src.cli.utils.output import print_info, print_error
        
        start_time = time.time()
        
        try:
            input_path = Path("data/knowledge_base")
            output_path = Path("data/vectorstore")
            
            # Remove existing database
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Create chunker
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
            
            # Process documents
            documents = []
            text_files = list(input_path.glob('**/*.md')) + list(input_path.glob('**/*.txt'))
            
            for file_path in text_files:
                chunks = chunker.chunk_documents(file_path)
                for chunk in chunks:
                    documents.append({
                        'content': chunk.content,
                        'source': str(file_path),
                        'chunk_id': chunk.chunk_id,
                        'metadata': chunk.metadata
                    })
            
            # Get embedding info
            embedding_info = self.embedding_selector.get_model_info()
            embedding_provider = self.embedding_selector.get_current_provider()
            
            # Create vector store
            store = FAISSVectorStore(dimension=embedding_info['dimension'])
            
            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                texts = [doc['content'] for doc in batch]
                metadatas = [{'source': doc.get('source', 'unknown')} for doc in batch]
                
                embeddings = embedding_provider.encode(texts)
                store.add_documents(embeddings, texts, metadatas)
            
            # Save the index
            store.save(str(output_path))
            
            end_time = time.time()
            return end_time - start_time
            
        except Exception as e:
            print(f"      Error rebuilding vector database: {e}")
            return -1
    
    def _test_rag_query(self, query: str) -> Dict[str, Any]:
        """Test RAG query performance and quality"""
        start_time = time.time()
        
        try:
            # Load vector store
            vector_store = FAISSVectorStore.load("data/vectorstore")
            
            # Create RAG engine
            model_wrapper = ModelWrapper(
                self.model_selector, 
                self.embedding_selector, 
                rag_enabled=True
            )
            rag_engine = EnhancedRAGEngine(
                vector_store=vector_store,
                model_manager=model_wrapper
            )
            
            # Time embedding generation
            embedding_start = time.time()
            provider = self.embedding_selector.get_current_provider()
            query_embedding = provider.encode([query])[0]
            embedding_time = time.time() - embedding_start
            
            # Time vector search
            search_start = time.time()
            search_results = vector_store.search(query_embedding, k=5)
            search_time = time.time() - search_start
            
            # Calculate relevance score
            relevance_score = 0
            if search_results:
                # Average of top results
                scores = [score for _, _, score in search_results]
                relevance_score = statistics.mean(scores) if scores else 0
            
            # Time LLM response (optional - can be slow)
            llm_start = time.time()
            # Skip actual LLM call for performance testing
            llm_time = 0  # time.time() - llm_start
            
            total_time = time.time() - start_time
            
            return {
                "query": query,
                "total_time": total_time,
                "embedding_time": embedding_time,
                "search_time": search_time,
                "llm_time": llm_time,
                "relevance_score": relevance_score,
                "sources_found": len(search_results) if search_results else 0,
                "success": True
            }
            
        except Exception as e:
            return {
                "query": query,
                "total_time": time.time() - start_time,
                "embedding_time": 0,
                "search_time": 0,
                "llm_time": 0,
                "relevance_score": 0,
                "sources_found": 0,
                "success": False,
                "error": str(e)
            }
    
    def _generate_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("📊 RAG Performance Comparison Report")
        print("="*80)
        
        if len(self.results) < 2:
            print("❌ Not enough models tested for comparison")
            return
        
        # Extract model names
        models = list(self.results.keys())
        openai_model = next((m for m in models if "text-embedding" in m), None)
        local_model = next((m for m in models if "MiniLM" in m or "mpnet" in m), None)
        
        if not openai_model or not local_model:
            print("❌ Missing OpenAI or Local model results")
            return
        
        print(f"\n🆚 Comparing: {openai_model} vs {local_model}")
        print("-" * 80)
        
        # Model Information
        self._compare_model_info(openai_model, local_model)
        
        # Embedding Performance
        self._compare_embedding_performance(openai_model, local_model)
        
        # Vector Operations Performance
        self._compare_vector_performance(openai_model, local_model)
        
        # RAG Quality Metrics
        self._compare_rag_quality(openai_model, local_model)
        
        # Overall Recommendation
        self._generate_recommendation(openai_model, local_model)
        
        # Save detailed results
        self._save_results()
    
    def _compare_model_info(self, openai_model: str, local_model: str):
        """Compare basic model information"""
        print("\n📋 Model Information:")
        print("-" * 50)
        
        openai_info = self.results[openai_model]["model_info"]
        local_info = self.results[local_model]["model_info"]
        
        print(f"OpenAI ({openai_model}):")
        print(f"  Provider: {openai_info.get('provider', 'unknown')}")
        print(f"  Dimension: {openai_info.get('dimension', 'unknown')}")
        print(f"  Cost: {openai_info.get('cost', 'unknown')}")
        
        print(f"\nLocal ({local_model}):")
        print(f"  Provider: {local_info.get('provider', 'unknown')}")
        print(f"  Dimension: {local_info.get('dimension', 'unknown')}")
        print(f"  Cost: {local_info.get('cost', 'unknown')}")
    
    def _compare_embedding_performance(self, openai_model: str, local_model: str):
        """Compare embedding generation performance"""
        print("\n⚡ Embedding Generation Performance:")
        print("-" * 50)
        
        openai_times = [t["embedding_time"] for t in self.results[openai_model]["embedding_tests"]]
        local_times = [t["embedding_time"] for t in self.results[local_model]["embedding_tests"]]
        
        openai_avg = statistics.mean(openai_times) if openai_times else 0
        local_avg = statistics.mean(local_times) if local_times else 0
        
        print(f"OpenAI average: {openai_avg:.3f}s")
        print(f"Local average: {local_avg:.3f}s")
        
        if openai_avg > 0 and local_avg > 0:
            speedup = openai_avg / local_avg
            winner = "Local" if local_avg < openai_avg else "OpenAI"
            print(f"🏆 Winner: {winner} ({speedup:.1f}x {'faster' if local_avg < openai_avg else 'slower'})")
    
    def _compare_vector_performance(self, openai_model: str, local_model: str):
        """Compare vector search performance"""
        print("\n🔍 Vector Search Performance:")
        print("-" * 50)
        
        openai_search_times = self.results[openai_model]["vector_search_times"]
        local_search_times = self.results[local_model]["vector_search_times"]
        
        openai_avg = statistics.mean(openai_search_times) if openai_search_times else 0
        local_avg = statistics.mean(local_search_times) if local_search_times else 0
        
        print(f"OpenAI search time: {openai_avg:.3f}s")
        print(f"Local search time: {local_avg:.3f}s")
        
        # Vector rebuild times
        openai_rebuild = self.results[openai_model]["vector_rebuild_time"]
        local_rebuild = self.results[local_model]["vector_rebuild_time"]
        
        if openai_rebuild > 0:
            print(f"OpenAI DB rebuild time: {openai_rebuild:.1f}s")
        if local_rebuild > 0:
            print(f"Local DB rebuild time: {local_rebuild:.1f}s")
    
    def _compare_rag_quality(self, openai_model: str, local_model: str):
        """Compare RAG quality metrics"""
        print("\n🎯 RAG Quality Metrics:")
        print("-" * 50)
        
        openai_scores = self.results[openai_model]["relevance_scores"]
        local_scores = self.results[local_model]["relevance_scores"]
        
        openai_avg_relevance = statistics.mean(openai_scores) if openai_scores else 0
        local_avg_relevance = statistics.mean(local_scores) if local_scores else 0
        
        print(f"OpenAI relevance score: {openai_avg_relevance:.3f}")
        print(f"Local relevance score: {local_avg_relevance:.3f}")
        
        # Success rates
        openai_successes = sum(1 for test in self.results[openai_model]["rag_tests"] if test["success"])
        local_successes = sum(1 for test in self.results[local_model]["rag_tests"] if test["success"])
        
        total_tests = len(self.test_queries)
        
        print(f"OpenAI success rate: {openai_successes}/{total_tests} ({openai_successes/total_tests*100:.1f}%)")
        print(f"Local success rate: {local_successes}/{total_tests} ({local_successes/total_tests*100:.1f}%)")
    
    def _generate_recommendation(self, openai_model: str, local_model: str):
        """Generate overall recommendation"""
        print("\n💡 Recommendations:")
        print("-" * 50)
        
        print("OpenAI Embedding Model (text-embedding-3-small):")
        print("  ✅ Pros: Higher dimensional vectors (1536D), potentially better semantic understanding")
        print("  ❌ Cons: API costs, requires internet, slower due to network latency")
        print("  🎯 Best for: Production environments with budget, highest quality requirements")
        
        print("\nLocal Embedding Model (all-MiniLM-L6-v2):")
        print("  ✅ Pros: Free, fast, no internet required, privacy-preserving")
        print("  ❌ Cons: Lower dimensional vectors (384D), potentially less nuanced")
        print("  🎯 Best for: Development, cost-sensitive deployments, offline environments")
    
    def _save_results(self):
        """Save detailed results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rag_performance_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {filename}")

def main():
    """Main function"""
    try:
        comparator = RAGPerformanceComparator()
        comparator.run_comparison()
    except KeyboardInterrupt:
        print("\n⏸️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()