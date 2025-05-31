#!/usr/bin/env python3
"""
Quick PCIe Error Analysis Comparison
Focus on one test case to get concrete results
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_error_analysis_scenario():
    """Test a single PCIe error analysis scenario"""
    
    # Simple but realistic PCIe error scenario
    error_log = """
PCIe Error Analysis:

Error Log:
[10:15:30] PCIe: 0000:01:00.0 - Link training failed
[10:15:31] PCIe: LTSSM stuck in Recovery.RcvrLock state  
[10:15:32] PCIe: 15 recovery cycles in last 5 seconds
[10:15:33] PCIe: Link speed degraded from Gen3 to Gen1

Question: What caused this link training failure and how to fix it?
"""
    
    print("🔬 PCIe Error Analysis Performance Test")
    print("=" * 50)
    print("Testing: Link Training Failure Analysis")
    print("=" * 50)
    
    # Test DeepSeek (with timeout)
    print("\n🤖 Testing DeepSeek Q4_1...")
    deepseek_result = test_deepseek_quick(error_log)
    
    # Test Llama (with fixes)
    print("\n🦙 Testing Llama 3.2 3B...")
    llama_result = test_llama_quick(error_log)
    
    # Compare results
    print("\n" + "=" * 50)
    print("📊 ERROR ANALYSIS COMPARISON RESULTS")
    print("=" * 50)
    
    print(f"\n⚡ Performance:")
    if deepseek_result['success'] and llama_result['success']:
        print(f"  DeepSeek: {deepseek_result['time']:.1f}s")
        print(f"  Llama:    {llama_result['time']:.1f}s")
        faster = "Llama" if llama_result['time'] < deepseek_result['time'] else "DeepSeek"
        speedup = max(deepseek_result['time'], llama_result['time']) / min(deepseek_result['time'], llama_result['time'])
        print(f"  Winner:   {faster} ({speedup:.1f}x faster)")
    else:
        print(f"  DeepSeek: {'✅' if deepseek_result['success'] else '❌'}")
        print(f"  Llama:    {'✅' if llama_result['success'] else '❌'}")
    
    print(f"\n📝 Response Quality:")
    if deepseek_result['success']:
        print(f"  DeepSeek: {len(deepseek_result['response'])} characters")
        print(f"  Preview:  {deepseek_result['response'][:100]}...")
    else:
        print(f"  DeepSeek: Failed - {deepseek_result['error']}")
    
    if llama_result['success']:
        print(f"  Llama:    {len(llama_result['response'])} characters")
        print(f"  Preview:  {llama_result['response'][:100]}...")
    else:
        print(f"  Llama:    Failed - {llama_result['error']}")
    
    # PCIe knowledge check
    pcie_keywords = ['LTSSM', 'recovery', 'link training', 'signal', 'equalization']
    
    if deepseek_result['success'] and llama_result['success']:
        deepseek_keywords = sum(1 for kw in pcie_keywords if kw.lower() in deepseek_result['response'].lower())
        llama_keywords = sum(1 for kw in pcie_keywords if kw.lower() in llama_result['response'].lower())
        
        print(f"\n🧠 PCIe Knowledge (keywords found):")
        print(f"  DeepSeek: {deepseek_keywords}/{len(pcie_keywords)} keywords")
        print(f"  Llama:    {llama_keywords}/{len(pcie_keywords)} keywords")
        
        if deepseek_keywords > llama_keywords:
            print(f"  Winner:   DeepSeek (better technical knowledge)")
        elif llama_keywords > deepseek_keywords:
            print(f"  Winner:   Llama (better technical knowledge)")
        else:
            print(f"  Winner:   Tie (equal technical knowledge)")
    
    # Final recommendation
    print(f"\n🎯 RECOMMENDATION FOR PCIe ERROR ANALYSIS:")
    
    if deepseek_result['success'] and llama_result['success']:
        if llama_result['time'] < deepseek_result['time'] * 0.5:  # Llama significantly faster
            print(f"  🚀 Use Llama 3.2 3B for interactive debugging")
            print(f"     Fast responses ({llama_result['time']:.1f}s) good for real-time analysis")
        
        if deepseek_result['time'] > 60:  # DeepSeek very slow
            print(f"  ⏰ DeepSeek too slow for interactive use ({deepseek_result['time']:.1f}s)")
            print(f"     Consider for batch analysis only")
        else:
            print(f"  🤖 DeepSeek suitable for detailed analysis")
    
    elif llama_result['success']:
        print(f"  ✅ Use Llama 3.2 3B - working and fast")
    elif deepseek_result['success']:
        print(f"  ⏳ Use DeepSeek - working but slow")
    else:
        print(f"  ❌ Both models need configuration fixes")

def test_deepseek_quick(prompt):
    """Quick test of DeepSeek with shorter timeout"""
    try:
        start_time = time.time()
        result = subprocess.run(
            ['ollama', 'run', 'deepseek-r1:latest', prompt],
            capture_output=True,
            text=True,
            timeout=90  # 90 second timeout
        )
        end_time = time.time()
        
        if result.returncode == 0:
            return {
                "success": True,
                "response": result.stdout.strip(),
                "time": end_time - start_time,
                "error": None
            }
        else:
            return {
                "success": False,
                "error": f"Return code {result.returncode}: {result.stderr}",
                "time": end_time - start_time
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Timeout (90 seconds)",
            "time": 90
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": 0
        }

def test_llama_quick(prompt):
    """Quick test of Llama with simple approach"""
    try:
        from src.models.local_llm_provider import LocalLLMProvider
        
        start_time = time.time()
        
        # Simple approach - avoid complex formatting
        provider = LocalLLMProvider(
            models_dir="models",
            n_ctx=4096,  # Smaller context to avoid issues
            n_gpu_layers=-1
        )
        
        if not provider.model_path.exists():
            return {
                "success": False,
                "error": "Model file not found",
                "time": 0
            }
        
        # Load model if needed
        if provider.llm is None:
            provider.load_model()
        
        if provider.llm is None:
            return {
                "success": False,
                "error": "Failed to load model",
                "time": time.time() - start_time
            }
        
        # Simple completion without special formatting
        response = provider.llm(
            prompt,
            max_tokens=300,
            temperature=0.1,
            echo=False
        )
        
        end_time = time.time()
        
        generated_text = response["choices"][0]["text"].strip()
        
        return {
            "success": True,
            "response": generated_text,
            "time": end_time - start_time,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start_time if 'start_time' in locals() else 0
        }

if __name__ == "__main__":
    test_error_analysis_scenario()