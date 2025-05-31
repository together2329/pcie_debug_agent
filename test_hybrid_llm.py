#!/usr/bin/env python3
"""
Test Hybrid LLM Provider for PCIe Error Analysis
Tests both quick (Llama) and detailed (DeepSeek) analysis modes
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.hybrid_llm_provider import HybridLLMProvider, AnalysisRequest

def test_hybrid_provider():
    """Test the hybrid LLM provider"""
    print("🔬 Testing Hybrid LLM Provider for PCIe Error Analysis")
    print("=" * 70)
    
    # Initialize hybrid provider
    provider = HybridLLMProvider(models_dir="models")
    
    # Check model status
    status = provider.get_model_status()
    print(f"\n📋 Model Status:")
    print(f"  Llama 3.2 3B: {'✅ Available' if status['llama']['available'] else '❌ Not Available'}")
    print(f"  DeepSeek Q4_1: {'✅ Available' if status['deepseek']['available'] else '❌ Not Available'}")
    print(f"  Hybrid Ready: {'✅ Yes' if status['hybrid_ready'] else '❌ No'}")
    
    if not status['hybrid_ready']:
        print("\n❌ No models available for testing")
        return
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Quick Analysis Test",
            "type": "quick",
            "query": "What does this PCIe error mean?",
            "error_log": "[10:15:30] PCIe: 0000:01:00.0 - Link training failed\n[10:15:31] PCIe: LTSSM stuck in Recovery state",
            "expected_time": 15.0
        },
        {
            "name": "Detailed Analysis Test", 
            "type": "detailed",
            "query": "Provide comprehensive root cause analysis of this complex PCIe failure",
            "error_log": """
[12:34:56] PCIe: 0000:01:00.0 - Multiple errors detected
[12:34:57] PCIe: Malformed TLP with invalid type 0x7
[12:34:58] PCIe: ECRC mismatch - Expected: 0xABCD, Received: 0x1234
[12:34:59] Thermal: GPU temperature 89°C
[12:35:00] PCIe: Link errors: 15/second
[12:35:01] PCIe: Device enumeration failed
[12:35:02] Power: Consumption dropped to 5W
""",
            "context": "System under high load, recent BIOS update, thermal issues reported",
            "expected_time": 180.0
        },
        {
            "name": "Auto Analysis Test",
            "type": "auto", 
            "query": "Analyze this link recovery issue",
            "error_log": "[14:22:15] PCIe: Recovery cycles: 247 in 10 seconds",
            "expected_time": 30.0
        }
    ]
    
    results = {}
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*50}")
        print(f"🧪 Test {i+1}/{len(test_scenarios)}: {scenario['name']}")
        print(f"{'='*50}")
        
        # Create analysis request
        request = AnalysisRequest(
            query=scenario['query'],
            error_log=scenario['error_log'],
            analysis_type=scenario['type'],
            max_response_time=scenario['expected_time'],
            context=scenario.get('context', '')
        )
        
        print(f"📝 Query: {scenario['query']}")
        print(f"⚡ Expected: {scenario['type']} analysis")
        print(f"⏱️  Timeout: {scenario['expected_time']}s")
        
        # Run analysis
        start_time = time.time()
        try:
            response = provider.analyze_pcie_error(request)
            actual_time = time.time() - start_time
            
            # Display results
            print(f"\n📊 Results:")
            print(f"  Model Used: {response.model_used}")
            print(f"  Analysis Type: {response.analysis_type}")
            print(f"  Response Time: {response.response_time:.1f}s")
            print(f"  Confidence: {response.confidence_score:.2f}")
            print(f"  Fallback Used: {response.fallback_used}")
            
            if response.error:
                print(f"  ❌ Error: {response.error}")
                success = False
            else:
                print(f"  ✅ Success: {len(response.response)} characters")
                print(f"  📄 Preview: {response.response[:150]}...")
                success = True
            
            # Performance evaluation
            time_acceptable = actual_time <= scenario['expected_time'] * 1.5  # 50% tolerance
            print(f"\n⚡ Performance:")
            print(f"  Expected: ≤ {scenario['expected_time']}s")
            print(f"  Actual: {actual_time:.1f}s")
            print(f"  Status: {'✅ Good' if time_acceptable else '⚠️ Slow'}")
            
            # Store results
            results[scenario['name']] = {
                "success": success,
                "model_used": response.model_used,
                "analysis_type": response.analysis_type,
                "response_time": response.response_time,
                "confidence_score": response.confidence_score,
                "fallback_used": response.fallback_used,
                "error": response.error,
                "time_acceptable": time_acceptable,
                "response_length": len(response.response) if response.response else 0
            }
            
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results[scenario['name']] = {
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time
            }
    
    # Generate comprehensive test summary
    generate_test_summary(results, status)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"hybrid_llm_test_results_{timestamp}.json"
    
    test_data = {
        "timestamp": timestamp,
        "model_status": status,
        "test_results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")

def generate_test_summary(results, status):
    """Generate comprehensive test summary"""
    print("\n" + "=" * 70)
    print("📊 HYBRID LLM PROVIDER TEST SUMMARY")
    print("=" * 70)
    
    successful_tests = [name for name, result in results.items() if result.get('success')]
    total_tests = len(results)
    
    print(f"\n🎯 Overall Results:")
    print(f"  Tests Passed: {len(successful_tests)}/{total_tests}")
    print(f"  Success Rate: {len(successful_tests)/total_tests*100:.1f}%")
    
    # Model usage analysis
    model_usage = {}
    for result in results.values():
        if result.get('success'):
            model = result.get('model_used', 'unknown')
            model_usage[model] = model_usage.get(model, 0) + 1
    
    print(f"\n🤖 Model Usage:")
    for model, count in model_usage.items():
        print(f"  {model}: {count} tests")
    
    # Performance analysis
    response_times = [r.get('response_time', 0) for r in results.values() if r.get('success')]
    if response_times:
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        print(f"\n⚡ Performance Analysis:")
        print(f"  Average Response Time: {avg_time:.1f}s")
        print(f"  Fastest Response: {min_time:.1f}s")
        print(f"  Slowest Response: {max_time:.1f}s")
    
    # Confidence analysis
    confidence_scores = [r.get('confidence_score', 0) for r in results.values() if r.get('success')]
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"\n🧠 Response Quality:")
        print(f"  Average Confidence: {avg_confidence:.2f}")
        print(f"  High Confidence (>0.7): {sum(1 for c in confidence_scores if c > 0.7)} tests")
    
    # Test-specific results
    print(f"\n📋 Test-Specific Results:")
    for test_name, result in results.items():
        status_icon = "✅" if result.get('success') else "❌"
        time_str = f"{result.get('response_time', 0):.1f}s"
        model_str = result.get('model_used', 'none')
        
        print(f"  {status_icon} {test_name}: {model_str} in {time_str}")
        
        if result.get('fallback_used'):
            print(f"    ⚠️  Fallback model used")
        
        if result.get('error'):
            print(f"    ❌ Error: {result['error']}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if status['llama']['available'] and status['deepseek']['available']:
        print("  ✅ Full hybrid capability available")
        print("  🚀 Use quick analysis for interactive debugging")
        print("  🔬 Use detailed analysis for complex issues")
    elif status['llama']['available']:
        print("  ⚡ Llama available - good for fast analysis")
        print("  ⚠️  Consider installing DeepSeek for detailed analysis")
    elif status['deepseek']['available']:
        print("  🧠 DeepSeek available - good for detailed analysis")
        print("  ⚠️  Consider fixing Llama for fast interactive analysis")
    else:
        print("  ❌ No models available - setup required")
    
    # Quick vs Detailed recommendations
    quick_tests = [r for r in results.values() if r.get('analysis_type') == 'quick']
    detailed_tests = [r for r in results.values() if r.get('analysis_type') == 'detailed']
    
    if quick_tests:
        quick_success = sum(1 for r in quick_tests if r.get('success'))
        quick_avg_time = sum(r.get('response_time', 0) for r in quick_tests if r.get('success'))
        quick_avg_time = quick_avg_time / len(quick_tests) if quick_tests else 0
        
        print(f"\n🏃 Quick Analysis Performance:")
        print(f"  Success Rate: {quick_success}/{len(quick_tests)}")
        print(f"  Average Time: {quick_avg_time:.1f}s")
        print(f"  Recommendation: {'✅ Ready for interactive use' if quick_avg_time < 15 else '⚠️ May be slow for interactive use'}")
    
    if detailed_tests:
        detailed_success = sum(1 for r in detailed_tests if r.get('success'))
        detailed_avg_time = sum(r.get('response_time', 0) for r in detailed_tests if r.get('success'))
        detailed_avg_time = detailed_avg_time / len(detailed_tests) if detailed_tests else 0
        
        print(f"\n🔬 Detailed Analysis Performance:")
        print(f"  Success Rate: {detailed_success}/{len(detailed_tests)}")
        print(f"  Average Time: {detailed_avg_time:.1f}s")
        print(f"  Recommendation: {'✅ Good for comprehensive analysis' if detailed_avg_time < 300 else '⚠️ Very slow, consider batch use only'}")

def test_convenience_methods():
    """Test convenience methods"""
    print("\n" + "=" * 50)
    print("🧪 Testing Convenience Methods")
    print("=" * 50)
    
    provider = HybridLLMProvider()
    
    # Test quick analysis
    print("\n⚡ Testing quick_analysis()...")
    try:
        response = provider.quick_analysis(
            "What caused this error?",
            "[10:15:30] PCIe: Link training failed"
        )
        print(f"  ✅ Quick analysis: {response.model_used} in {response.response_time:.1f}s")
    except Exception as e:
        print(f"  ❌ Quick analysis failed: {e}")
    
    # Test detailed analysis
    print("\n🔬 Testing detailed_analysis()...")
    try:
        response = provider.detailed_analysis(
            "Provide comprehensive analysis",
            "[12:34:56] PCIe: Multiple TLP errors detected",
            "High-performance GPU workload"
        )
        print(f"  ✅ Detailed analysis: {response.model_used} in {response.response_time:.1f}s")
    except Exception as e:
        print(f"  ❌ Detailed analysis failed: {e}")

if __name__ == "__main__":
    # Run main tests
    test_hybrid_provider()
    
    # Run convenience method tests
    test_convenience_methods()
    
    print("\n🎉 Hybrid LLM Provider testing completed!")