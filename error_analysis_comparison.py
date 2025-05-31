#!/usr/bin/env python3
"""
PCIe Error Analysis Performance Comparison: DeepSeek vs Llama
Focus on actual error analysis capabilities
"""

import sys
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class PCIeErrorAnalysisTest:
    """Test both models on specific PCIe error analysis scenarios"""
    
    def __init__(self):
        self.test_scenarios = [
            {
                "name": "Malformed TLP Analysis",
                "error_log": """
                PCIe Error Log:
                [12:34:56.789] PCIe: 0000:01:00.0 - Malformed TLP detected
                [12:34:56.790] PCIe: TLP Type: 0x7 (Invalid)
                [12:34:56.791] PCIe: Header: 0x40000001 0x12345678 0x9ABCDEF0 0x11223344
                [12:34:56.792] PCIe: ECRC mismatch - Expected: 0xABCD, Received: 0x1234
                [12:34:56.793] PCIe: Device reset initiated
                """,
                "question": "Analyze this malformed TLP error. What caused it and how should it be resolved?",
                "expected_keywords": ["TLP", "header", "ECRC", "format", "protocol", "reset", "device"]
            },
            {
                "name": "Link Training Failure",
                "error_log": """
                PCIe Error Log:
                [10:15:30.123] PCIe: 0000:02:00.0 - Link training failed
                [10:15:30.124] PCIe: LTSSM stuck in Recovery.RcvrLock state
                [10:15:30.125] PCIe: Signal integrity issues detected on lanes 0-3
                [10:15:30.126] PCIe: Equalization failed - coefficients out of range
                [10:15:30.127] PCIe: Link speed degraded from Gen3 to Gen1
                [10:15:30.128] PCIe: 247 recovery cycles in last 10 seconds
                """,
                "question": "Diagnose this link training failure. What are the root causes and debugging steps?",
                "expected_keywords": ["LTSSM", "recovery", "signal integrity", "equalization", "link training", "lanes", "Gen3"]
            },
            {
                "name": "Thermal Correlation Analysis",
                "error_log": """
                PCIe Error Log:
                [14:22:15.456] Thermal: GPU temperature reached 87°C
                [14:22:15.457] PCIe: 0000:01:00.0 - Completion timeout on config read
                [14:22:15.458] PCIe: Link errors increased to 15/second
                [14:22:15.459] Thermal: Throttling initiated at 89°C
                [14:22:15.460] PCIe: Device enumeration failed
                [14:22:15.461] PCIe: Link went down unexpectedly
                [14:22:15.462] Power: Card power consumption dropped to 5W
                """,
                "question": "Analyze the correlation between thermal events and PCIe errors. What's the failure sequence?",
                "expected_keywords": ["thermal", "temperature", "throttling", "correlation", "timeout", "enumeration", "power"]
            }
        ]
    
    def test_deepseek(self, scenario):
        """Test DeepSeek model on error analysis"""
        print(f"  🤖 Testing DeepSeek on: {scenario['name']}")
        
        prompt = f"""
PCIe Error Analysis Task:

{scenario['error_log']}

Question: {scenario['question']}

Please provide:
1. Root cause analysis
2. Failure sequence explanation  
3. Recommended debugging steps
4. Prevention strategies
"""
        
        try:
            start_time = time.time()
            result = subprocess.run(
                ['ollama', 'run', 'deepseek-r1:latest', prompt],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Analyze response quality
                keywords_found = sum(1 for kw in scenario['expected_keywords'] 
                                   if kw.lower() in response.lower())
                
                return {
                    "success": True,
                    "response": response,
                    "response_time": end_time - start_time,
                    "response_length": len(response),
                    "keywords_found": keywords_found,
                    "total_keywords": len(scenario['expected_keywords']),
                    "keyword_score": (keywords_found / len(scenario['expected_keywords'])) * 100,
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "response_time": end_time - start_time
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Timeout (5 minutes)",
                "response_time": 300
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def test_llama_fixed(self, scenario):
        """Test Llama model with fixed configuration"""
        print(f"  🦙 Testing Llama on: {scenario['name']}")
        
        try:
            # Try to import and use Llama with fixed settings
            from src.models.local_llm_provider import LocalLLMProvider
            
            # Create provider with larger context window
            provider = LocalLLMProvider(
                models_dir="models",
                n_ctx=16384,  # Larger context window
                n_gpu_layers=-1,
                verbose=False
            )
            
            if not provider.model_path.exists():
                return {
                    "success": False,
                    "error": "Llama model not found",
                    "response_time": 0
                }
            
            prompt = f"""PCIe Error Analysis:

{scenario['error_log']}

{scenario['question']}

Provide root cause analysis, failure sequence, debugging steps, and prevention strategies."""
            
            start_time = time.time()
            
            # Use direct completion without instruction formatting to avoid token issues
            if provider.llm is None:
                success = provider.load_model()
                if not success:
                    return {
                        "success": False,
                        "error": "Failed to load Llama model",
                        "response_time": time.time() - start_time
                    }
            
            # Generate response with simple prompt (no special formatting)
            response = provider.llm(
                prompt,
                max_tokens=1000,
                temperature=0.1,
                stop=["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            
            end_time = time.time()
            
            generated_text = response["choices"][0]["text"].strip()
            
            # Analyze response quality
            keywords_found = sum(1 for kw in scenario['expected_keywords'] 
                               if kw.lower() in generated_text.lower())
            
            return {
                "success": True,
                "response": generated_text,
                "response_time": end_time - start_time,
                "response_length": len(generated_text),
                "keywords_found": keywords_found,
                "total_keywords": len(scenario['expected_keywords']),
                "keyword_score": (keywords_found / len(scenario['expected_keywords'])) * 100,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response_time": 0
            }
    
    def run_error_analysis_comparison(self):
        """Run comprehensive error analysis comparison"""
        print("🔬 PCIe Error Analysis Performance Comparison")
        print("=" * 70)
        print("Testing both models on real PCIe debugging scenarios")
        print("=" * 70)
        
        results = {
            "deepseek": {},
            "llama": {},
            "comparison": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for i, scenario in enumerate(self.test_scenarios):
            print(f"\n📋 Test {i+1}/{len(self.test_scenarios)}: {scenario['name']}")
            print("-" * 50)
            
            # Test DeepSeek
            deepseek_result = self.test_deepseek(scenario)
            results["deepseek"][scenario['name']] = deepseek_result
            
            if deepseek_result['success']:
                print(f"    ✅ DeepSeek: {deepseek_result['response_time']:.1f}s, "
                      f"{deepseek_result['keyword_score']:.1f}% keyword match")
            else:
                print(f"    ❌ DeepSeek: {deepseek_result['error']}")
            
            # Test Llama
            llama_result = self.test_llama_fixed(scenario)
            results["llama"][scenario['name']] = llama_result
            
            if llama_result['success']:
                print(f"    ✅ Llama: {llama_result['response_time']:.1f}s, "
                      f"{llama_result['keyword_score']:.1f}% keyword match")
            else:
                print(f"    ❌ Llama: {llama_result['error']}")
        
        # Generate comprehensive analysis
        self.analyze_error_analysis_results(results)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"error_analysis_comparison_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        return results
    
    def analyze_error_analysis_results(self, results):
        """Analyze and compare error analysis results"""
        print("\n" + "=" * 70)
        print("📊 ERROR ANALYSIS PERFORMANCE COMPARISON")
        print("=" * 70)
        
        deepseek_results = results["deepseek"]
        llama_results = results["llama"]
        
        # Success rate analysis
        deepseek_successes = sum(1 for r in deepseek_results.values() if r.get('success'))
        llama_successes = sum(1 for r in llama_results.values() if r.get('success'))
        total_tests = len(self.test_scenarios)
        
        print(f"\n🎯 Success Rate:")
        print(f"  DeepSeek: {deepseek_successes}/{total_tests} ({deepseek_successes/total_tests*100:.1f}%)")
        print(f"  Llama:    {llama_successes}/{total_tests} ({llama_successes/total_tests*100:.1f}%)")
        
        # Performance metrics for successful tests
        if deepseek_successes > 0:
            deepseek_times = [r['response_time'] for r in deepseek_results.values() if r.get('success')]
            deepseek_avg_time = sum(deepseek_times) / len(deepseek_times)
            deepseek_keyword_scores = [r['keyword_score'] for r in deepseek_results.values() if r.get('success')]
            deepseek_avg_keywords = sum(deepseek_keyword_scores) / len(deepseek_keyword_scores)
        else:
            deepseek_avg_time = 0
            deepseek_avg_keywords = 0
        
        if llama_successes > 0:
            llama_times = [r['response_time'] for r in llama_results.values() if r.get('success')]
            llama_avg_time = sum(llama_times) / len(llama_times)
            llama_keyword_scores = [r['keyword_score'] for r in llama_results.values() if r.get('success')]
            llama_avg_keywords = sum(llama_keyword_scores) / len(llama_keyword_scores)
        else:
            llama_avg_time = 0
            llama_avg_keywords = 0
        
        print(f"\n⚡ Average Response Time:")
        print(f"  DeepSeek: {deepseek_avg_time:.1f}s")
        print(f"  Llama:    {llama_avg_time:.1f}s")
        if deepseek_avg_time > 0 and llama_avg_time > 0:
            faster = "Llama" if llama_avg_time < deepseek_avg_time else "DeepSeek"
            speedup = max(deepseek_avg_time, llama_avg_time) / min(deepseek_avg_time, llama_avg_time)
            print(f"  Winner:   {faster} ({speedup:.1f}x faster)")
        
        print(f"\n🧠 PCIe Knowledge Quality (Keyword Match):")
        print(f"  DeepSeek: {deepseek_avg_keywords:.1f}%")
        print(f"  Llama:    {llama_avg_keywords:.1f}%")
        if deepseek_avg_keywords > 0 and llama_avg_keywords > 0:
            better = "DeepSeek" if deepseek_avg_keywords > llama_avg_keywords else "Llama"
            print(f"  Winner:   {better}")
        
        # Scenario-specific analysis
        print(f"\n📋 Scenario-Specific Performance:")
        for scenario in self.test_scenarios:
            name = scenario['name']
            print(f"\n  {name}:")
            
            ds_result = deepseek_results.get(name, {})
            ll_result = llama_results.get(name, {})
            
            if ds_result.get('success'):
                print(f"    DeepSeek: {ds_result['response_time']:.1f}s, {ds_result['keyword_score']:.1f}% keywords")
            else:
                print(f"    DeepSeek: Failed - {ds_result.get('error', 'Unknown error')}")
            
            if ll_result.get('success'):
                print(f"    Llama:    {ll_result['response_time']:.1f}s, {ll_result['keyword_score']:.1f}% keywords")
            else:
                print(f"    Llama:    Failed - {ll_result.get('error', 'Unknown error')}")
        
        # Final recommendation
        print(f"\n🏆 ERROR ANALYSIS WINNER:")
        
        score_deepseek = 0
        score_llama = 0
        
        # Success rate scoring
        if deepseek_successes > llama_successes:
            score_deepseek += 1
            print(f"  ✅ Reliability: DeepSeek ({deepseek_successes} vs {llama_successes} successful)")
        elif llama_successes > deepseek_successes:
            score_llama += 1
            print(f"  ✅ Reliability: Llama ({llama_successes} vs {deepseek_successes} successful)")
        else:
            print(f"  🤝 Reliability: Tie ({deepseek_successes} successful each)")
        
        # Speed scoring
        if deepseek_avg_time > 0 and llama_avg_time > 0:
            if deepseek_avg_time < llama_avg_time:
                score_deepseek += 1
                print(f"  ⚡ Speed: DeepSeek ({deepseek_avg_time:.1f}s vs {llama_avg_time:.1f}s)")
            elif llama_avg_time < deepseek_avg_time:
                score_llama += 1
                print(f"  ⚡ Speed: Llama ({llama_avg_time:.1f}s vs {deepseek_avg_time:.1f}s)")
            else:
                print(f"  ⚡ Speed: Tie")
        
        # Quality scoring
        if deepseek_avg_keywords > 0 and llama_avg_keywords > 0:
            if deepseek_avg_keywords > llama_avg_keywords:
                score_deepseek += 1
                print(f"  🧠 Quality: DeepSeek ({deepseek_avg_keywords:.1f}% vs {llama_avg_keywords:.1f}%)")
            elif llama_avg_keywords > deepseek_avg_keywords:
                score_llama += 1
                print(f"  🧠 Quality: Llama ({llama_avg_keywords:.1f}% vs {deepseek_avg_keywords:.1f}%)")
            else:
                print(f"  🧠 Quality: Tie")
        
        print(f"\n🎊 FINAL SCORE:")
        print(f"  DeepSeek: {score_deepseek}/3 categories")
        print(f"  Llama:    {score_llama}/3 categories")
        
        if score_deepseek > score_llama:
            print(f"\n🥇 WINNER: DeepSeek Q4_1")
            print(f"   Better for PCIe error analysis tasks")
        elif score_llama > score_deepseek:
            print(f"\n🥇 WINNER: Llama 3.2 3B")
            print(f"   Better for PCIe error analysis tasks")
        else:
            print(f"\n🤝 RESULT: TIE")
            print(f"   Both models perform equally well")

def main():
    """Run the error analysis comparison"""
    tester = PCIeErrorAnalysisTest()
    tester.run_error_analysis_comparison()

if __name__ == "__main__":
    main()