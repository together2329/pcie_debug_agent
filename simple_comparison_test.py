#!/usr/bin/env python3
"""
Simple DeepSeek vs Llama comparison test
"""

import sys
import time
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_deepseek_model():
    """Test DeepSeek model with a simple PCIe query"""
    print("🤖 Testing DeepSeek Q4_1...")
    
    query = "What is PCIe and what are common error types?"
    
    try:
        start_time = time.time()
        result = subprocess.run(
            ['ollama', 'run', 'deepseek-r1:latest', query],
            capture_output=True,
            text=True,
            timeout=60
        )
        end_time = time.time()
        
        if result.returncode == 0:
            response = result.stdout.strip()
            print(f"✅ DeepSeek Response ({end_time - start_time:.2f}s):")
            print(f"Length: {len(response)} characters")
            print(f"Preview: {response[:200]}...")
            return True, response, end_time - start_time
        else:
            print(f"❌ DeepSeek failed: {result.stderr}")
            return False, None, 0
            
    except subprocess.TimeoutExpired:
        print("❌ DeepSeek timeout")
        return False, None, 60
    except Exception as e:
        print(f"❌ DeepSeek error: {e}")
        return False, None, 0

def test_llama_model():
    """Test Llama model with existing setup"""
    print("🦙 Testing Llama 3.2 3B...")
    
    try:
        from src.models.local_llm_provider import LocalLLMProvider
        
        provider = LocalLLMProvider(models_dir="models")
        
        if not provider.is_available():
            print("❌ Llama model not available")
            return False, None, 0
        
        query = "What is PCIe and what are common error types?"
        
        start_time = time.time()
        response = provider.generate_completion(
            prompt=query,
            max_tokens=500,
            temperature=0.1
        )
        end_time = time.time()
        
        print(f"✅ Llama Response ({end_time - start_time:.2f}s):")
        print(f"Length: {len(response)} characters")
        print(f"Preview: {response[:200]}...")
        return True, response, end_time - start_time
        
    except Exception as e:
        print(f"❌ Llama error: {e}")
        return False, None, 0

def compare_responses():
    """Compare both models on PCIe knowledge"""
    print("🔬 DeepSeek Q4_1 vs Llama 3.2 3B Simple Comparison")
    print("=" * 60)
    
    # Test both models
    deepseek_success, deepseek_response, deepseek_time = test_deepseek_model()
    print()
    llama_success, llama_response, llama_time = test_llama_model()
    
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    # Performance comparison
    print(f"\n⚡ Performance:")
    if deepseek_success and llama_success:
        faster_model = "DeepSeek" if deepseek_time < llama_time else "Llama"
        print(f"  DeepSeek: {deepseek_time:.2f}s")
        print(f"  Llama:    {llama_time:.2f}s")
        print(f"  Winner:   {faster_model} (faster)")
    else:
        print(f"  DeepSeek: {'✅' if deepseek_success else '❌'}")
        print(f"  Llama:    {'✅' if llama_success else '❌'}")
    
    # Response quality
    print(f"\n📝 Response Quality:")
    if deepseek_success and llama_success:
        print(f"  DeepSeek: {len(deepseek_response)} characters")
        print(f"  Llama:    {len(llama_response)} characters")
        
        # Check for PCIe keywords
        pcie_keywords = ['PCIe', 'PCI Express', 'error', 'TLP', 'link', 'protocol']
        deepseek_keywords = sum(1 for kw in pcie_keywords if kw.lower() in deepseek_response.lower())
        llama_keywords = sum(1 for kw in pcie_keywords if kw.lower() in llama_response.lower())
        
        print(f"  PCIe keywords - DeepSeek: {deepseek_keywords}, Llama: {llama_keywords}")
        
        if deepseek_keywords > llama_keywords:
            print(f"  Winner: DeepSeek (better PCIe terminology)")
        elif llama_keywords > deepseek_keywords:
            print(f"  Winner: Llama (better PCIe terminology)")
        else:
            print(f"  Winner: Tie (equal PCIe terminology)")
    
    # Overall recommendation
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if deepseek_success and llama_success:
        if deepseek_time < llama_time:
            print("  🚀 DeepSeek appears faster for this query")
        if len(deepseek_response) > len(llama_response):
            print("  📝 DeepSeek provides more detailed responses")
        print("  ✅ Both models are operational and can handle PCIe queries")
        print("  🔄 Consider running the full comparison test for comprehensive analysis")
    elif deepseek_success:
        print("  🤖 Only DeepSeek is working - use DeepSeek for PCIe analysis")
    elif llama_success:
        print("  🦙 Only Llama is working - continue using Llama for PCIe analysis")
    else:
        print("  ❌ Neither model is working - check your setup")

if __name__ == "__main__":
    compare_responses()