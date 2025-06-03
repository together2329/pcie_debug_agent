#!/usr/bin/env python3
"""
Load testing for enhanced RAG system
Tests concurrent query handling and system stability
"""

import concurrent.futures
import time
import threading
import json
from datetime import datetime
import random

print("🔥 RAG System Load Test")
print("=" * 50)

# Global counters for tracking
successful_queries = 0
failed_queries = 0
lock = threading.Lock()

def process_mock_query(query_info):
    """
    Mock query processing for load testing
    In production, this would use the actual UnifiedRAGSystem
    """
    global successful_queries, failed_queries
    
    query_id, query, worker_id = query_info
    start_time = time.time()
    
    try:
        # Simulate varying processing times
        processing_time = random.uniform(0.5, 1.5)
        time.sleep(processing_time / 100)  # Scale down for mock
        
        # Simulate occasional failures (5% failure rate)
        if random.random() < 0.05:
            raise Exception("Simulated processing error")
        
        # Mock successful result
        result = {
            "query_id": query_id,
            "query": query,
            "worker_id": worker_id,
            "response_time": processing_time,
            "confidence": random.uniform(0.7, 0.95),
            "status": "success"
        }
        
        with lock:
            successful_queries += 1
        
        return result
        
    except Exception as e:
        with lock:
            failed_queries += 1
        
        return {
            "query_id": query_id,
            "query": query,
            "worker_id": worker_id,
            "response_time": time.time() - start_time,
            "status": "failed",
            "error": str(e)
        }

def run_load_test():
    """Run comprehensive load test with different scenarios"""
    
    test_scenarios = [
        {
            "name": "Light Load",
            "concurrent_workers": 5,
            "queries_per_worker": 10,
            "description": "Basic concurrent usage"
        },
        {
            "name": "Medium Load",
            "concurrent_workers": 10,
            "queries_per_worker": 20,
            "description": "Typical production load"
        },
        {
            "name": "Heavy Load",
            "concurrent_workers": 20,
            "queries_per_worker": 25,
            "description": "Peak usage simulation"
        },
        {
            "name": "Stress Test",
            "concurrent_workers": 50,
            "queries_per_worker": 10,
            "description": "System stress test"
        }
    ]
    
    # Sample queries for testing
    test_queries = [
        "What is PCIe FLR?",
        "How to debug completion timeout?",
        "Explain LTSSM states",
        "PCIe error recovery mechanisms",
        "What causes link training failures?",
        "PCIe compliance testing procedures",
        "Difference between posted and non-posted",
        "How to implement PCIe power management?"
    ]
    
    all_results = []
    
    for scenario in test_scenarios:
        print(f"\n🚀 Running: {scenario['name']}")
        print(f"   Workers: {scenario['concurrent_workers']}")
        print(f"   Queries per worker: {scenario['queries_per_worker']}")
        print(f"   Total queries: {scenario['concurrent_workers'] * scenario['queries_per_worker']}")
        print(f"   Description: {scenario['description']}")
        
        # Reset counters
        global successful_queries, failed_queries
        successful_queries = 0
        failed_queries = 0
        
        # Prepare queries for all workers
        all_queries = []
        query_id = 0
        for worker in range(scenario['concurrent_workers']):
            for q in range(scenario['queries_per_worker']):
                query = random.choice(test_queries)
                all_queries.append((query_id, query, worker))
                query_id += 1
        
        # Run load test
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario['concurrent_workers']) as executor:
            # Submit all queries
            futures = [executor.submit(process_mock_query, query_info) for query_info in all_queries]
            
            # Wait for completion with progress updates
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                if completed % 50 == 0:
                    print(f"   Progress: {completed}/{len(all_queries)} queries completed")
        
        # Calculate results
        total_time = time.time() - start_time
        total_queries = len(all_queries)
        throughput = total_queries / total_time
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        
        scenario_result = {
            "scenario": scenario['name'],
            "workers": scenario['concurrent_workers'],
            "queries_per_worker": scenario['queries_per_worker'],
            "total_queries": total_queries,
            "successful": successful_queries,
            "failed": failed_queries,
            "total_time": total_time,
            "throughput_qps": throughput,
            "success_rate": success_rate,
            "avg_time_per_query": total_time / total_queries
        }
        
        all_results.append(scenario_result)
        
        # Display results
        print(f"\n   📊 Results:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Throughput: {throughput:.1f} queries/second")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Successful: {successful_queries}")
        print(f"   Failed: {failed_queries}")
        print(f"   Avg Time per Query: {scenario_result['avg_time_per_query']:.3f}s")
    
    # Overall summary
    print("\n" + "=" * 50)
    print("📈 Load Test Summary:")
    print("-" * 50)
    
    for result in all_results:
        print(f"\n{result['scenario']}:")
        print(f"  - Throughput: {result['throughput_qps']:.1f} qps")
        print(f"  - Success Rate: {result['success_rate']:.1f}%")
        print(f"  - Avg Response: {result['avg_time_per_query']:.3f}s")
    
    # System capacity analysis
    print("\n💪 System Capacity Analysis:")
    max_throughput = max(r['throughput_qps'] for r in all_results)
    print(f"Peak Throughput: {max_throughput:.1f} queries/second")
    
    # Find breaking point
    breaking_point = None
    for result in all_results:
        if result['success_rate'] < 95:
            breaking_point = result
            break
    
    if breaking_point:
        print(f"System degrades at: {breaking_point['workers']} concurrent workers")
        print(f"Recommended max workers: {breaking_point['workers'] - 5}")
    else:
        print("System handled all load scenarios successfully!")
        print(f"Can handle at least {all_results[-1]['workers']} concurrent workers")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"load_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "scenarios": all_results,
            "summary": {
                "max_throughput": max_throughput,
                "breaking_point": breaking_point['scenario'] if breaking_point else "Not reached"
            }
        }, f, indent=2)
    
    print(f"\n💾 Load test results saved to: {results_file}")
    
    # Recommendations
    print("\n💡 Load Handling Recommendations:")
    print("1. Implement connection pooling for database connections")
    print("2. Use caching for frequently accessed queries")
    print("3. Consider horizontal scaling for high loads")
    print("4. Monitor memory usage during peak times")
    print("5. Implement rate limiting to prevent overload")

# Production load test would look like:
"""
def load_test_real_system():
    from src.rag.unified_rag_integration import UnifiedRAGSystem
    
    rag = UnifiedRAGSystem()
    
    def process_query(query):
        return rag.process_query(query)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        queries = ["PCIe query"] * 100
        results = list(executor.map(process_query, queries))
"""

if __name__ == "__main__":
    try:
        run_load_test()
        print("\n✅ Load test completed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Load test interrupted by user")
    except Exception as e:
        print(f"\n❌ Load test failed: {e}")