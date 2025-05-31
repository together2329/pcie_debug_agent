#!/usr/bin/env python3
"""
Model Comparison Test Script
Compare DeepSeek Q4_1 vs Current Llama Model
"""

import time
import subprocess
import psutil
import json
from datetime import datetime

class ModelTester:
    def __init__(self):
        self.test_queries = [
            "Explain PCIe TLP error detection and recovery mechanisms",
            "If a PCIe link shows 12 recovery cycles in 1 second, what are the likely root causes?",
            "Write a Python function to parse PCIe error logs",
            "Diagnose thermal throttling at 85°C with concurrent PCIe errors"
        ]
        self.results = {}
    
    def test_ollama_model(self, model_name: str) -> dict:
        """Test an Ollama model with benchmark queries"""
        print(f"🧪 Testing {model_name}...")
        
        results = {
            'model': model_name,
            'test_time': datetime.now().isoformat(),
            'responses': [],
            'performance': {}
        }
        
        for i, query in enumerate(self.test_queries):
            print(f"  Query {i+1}/4: {query[:50]}...")
            
            # Measure response time and system resources
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / 1024**3  # GB
            
            try:
                # Run Ollama command
                result = subprocess.run(
                    ['ollama', 'run', model_name, query],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / 1024**3  # GB
                
                response_data = {
                    'query': query,
                    'response': result.stdout,
                    'response_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'success': result.returncode == 0
                }
                
                results['responses'].append(response_data)
                
                print(f"    ✓ Response time: {response_data['response_time']:.2f}s")
                print(f"    ✓ Memory usage: {response_data['memory_delta']:.2f}GB")
                
            except subprocess.TimeoutExpired:
                print(f"    ❌ Timeout for query {i+1}")
                results['responses'].append({
                    'query': query,
                    'response': 'TIMEOUT',
                    'response_time': 60.0,
                    'memory_delta': 0,
                    'success': False
                })
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results['responses'].append({
                    'query': query,
                    'response': f'ERROR: {e}',
                    'response_time': 0,
                    'memory_delta': 0,
                    'success': False
                })
        
        # Calculate performance metrics
        successful_responses = [r for r in results['responses'] if r['success']]
        if successful_responses:
            results['performance'] = {
                'avg_response_time': sum(r['response_time'] for r in successful_responses) / len(successful_responses),
                'max_response_time': max(r['response_time'] for r in successful_responses),
                'avg_memory_usage': sum(r['memory_delta'] for r in successful_responses) / len(successful_responses),
                'success_rate': len(successful_responses) / len(self.test_queries) * 100
            }
        
        return results
    
    def compare_models(self, model1: str, model2: str):
        """Compare two models and generate report"""
        print("🔬 Model Comparison Test Starting...")
        print("=" * 60)
        
        # Test both models
        results1 = self.test_ollama_model(model1)
        results2 = self.test_ollama_model(model2)
        
        # Generate comparison report
        self.generate_comparison_report(results1, results2)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"model_comparison_{timestamp}.json", 'w') as f:
            json.dump({
                'comparison_time': timestamp,
                'model1_results': results1,
                'model2_results': results2
            }, f, indent=2)
        
        print(f"📊 Results saved to model_comparison_{timestamp}.json")
    
    def generate_comparison_report(self, results1: dict, results2: dict):
        """Generate detailed comparison report"""
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON REPORT")
        print("="*80)
        
        model1_name = results1['model']
        model2_name = results2['model']
        
        print(f"\n🥊 {model1_name} vs {model2_name}")
        print("-" * 60)
        
        # Performance comparison
        if 'performance' in results1 and 'performance' in results2:
            perf1 = results1['performance']
            perf2 = results2['performance']
            
            print(f"📈 Performance Metrics:")
            print(f"  Average Response Time:")
            print(f"    {model1_name}: {perf1.get('avg_response_time', 0):.2f}s")
            print(f"    {model2_name}: {perf2.get('avg_response_time', 0):.2f}s")
            
            print(f"  Memory Usage:")
            print(f"    {model1_name}: {perf1.get('avg_memory_usage', 0):.2f}GB")
            print(f"    {model2_name}: {perf2.get('avg_memory_usage', 0):.2f}GB")
            
            print(f"  Success Rate:")
            print(f"    {model1_name}: {perf1.get('success_rate', 0):.1f}%")
            print(f"    {model2_name}: {perf2.get('success_rate', 0):.1f}%")
        
        # Response quality comparison
        print(f"\n📝 Response Quality Analysis:")
        for i, (r1, r2) in enumerate(zip(results1['responses'], results2['responses'])):
            print(f"\n  Query {i+1}: {r1['query'][:50]}...")
            print(f"    {model1_name}: {len(r1.get('response', ''))} chars, {r1.get('response_time', 0):.2f}s")
            print(f"    {model2_name}: {len(r2.get('response', ''))} chars, {r2.get('response_time', 0):.2f}s")
        
        # Winner determination
        print(f"\n🏆 COMPARISON WINNER:")
        if 'performance' in results1 and 'performance' in results2:
            perf1 = results1['performance']
            perf2 = results2['performance']
            
            score1 = 0
            score2 = 0
            
            # Speed comparison (lower is better)
            if perf1.get('avg_response_time', float('inf')) < perf2.get('avg_response_time', float('inf')):
                score1 += 1
                print(f"  🏃 Speed Winner: {model1_name}")
            else:
                score2 += 1
                print(f"  🏃 Speed Winner: {model2_name}")
            
            # Memory efficiency (lower is better)
            if perf1.get('avg_memory_usage', float('inf')) < perf2.get('avg_memory_usage', float('inf')):
                score1 += 1
                print(f"  💾 Memory Winner: {model1_name}")
            else:
                score2 += 1
                print(f"  💾 Memory Winner: {model2_name}")
            
            # Success rate (higher is better)
            if perf1.get('success_rate', 0) > perf2.get('success_rate', 0):
                score1 += 1
                print(f"  ✅ Reliability Winner: {model1_name}")
            else:
                score2 += 1
                print(f"  ✅ Reliability Winner: {model2_name}")
            
            if score1 > score2:
                print(f"  🥇 Overall Winner: {model1_name} ({score1}/3 categories)")
            elif score2 > score1:
                print(f"  🥇 Overall Winner: {model2_name} ({score2}/3 categories)")
            else:
                print(f"  🤝 Tie: Both models performed equally well")


def main():
    """Main testing function"""
    tester = ModelTester()
    
    print("🔬 Model Comparison Tool")
    print("This will test DeepSeek Q4_1 vs your current Llama model")
    print()
    
    # Check available models
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode == 0:
            print("📋 Available Ollama models:")
            print(result.stdout)
        else:
            print("❌ Could not list Ollama models. Is Ollama installed?")
            return
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first:")
        print("   curl -fsSL https://ollama.com/install.sh | sh")
        return
    
    # Test models
    model1 = input("Enter first model name (e.g., llama2, llama3): ").strip()
    model2 = "deepseek-r1:8b-0528-qwen3-q4_1"
    
    if not model1:
        print("❌ Please provide a model name to compare against")
        return
    
    print(f"\n🔥 Comparing {model1} vs {model2}")
    tester.compare_models(model1, model2)


if __name__ == "__main__":
    main()