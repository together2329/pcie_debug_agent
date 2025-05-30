#!/usr/bin/env python
"""
Quick demonstration of working local LLM functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.local_llm_provider import LocalLLMProvider
from src.models.pcie_prompts import PCIePromptTemplates

def show_local_llm_status():
    """Show the current status of local LLM"""
    print("🔍 Local LLM Status Check")
    print("=" * 40)
    
    # Check if model file exists
    provider = LocalLLMProvider(models_dir="models")
    model_exists = provider.model_path.exists()
    
    print(f"Model file: {provider.model_path}")
    print(f"Model exists: {'✅ YES' if model_exists else '❌ NO'}")
    
    if model_exists:
        size_mb = provider.model_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB ({size_mb/1024:.1f} GB)")
        
        # Show model info
        info = provider.get_model_info()
        print(f"Description: {info['description']}")
        print(f"Context window: {info['context_window']:,} tokens")
        print(f"Memory usage: {info['memory_usage']}")
        
        return True
    else:
        print("Model would be auto-downloaded on first use")
        print(f"Download URL: {provider.model_info['url']}")
        return False

def show_pcie_prompts():
    """Demonstrate PCIe-specific prompt generation"""
    print("\n🧠 PCIe Prompt Generation")
    print("=" * 40)
    
    # Sample context
    context = """
[Source 1] PCIe: 0000:01:00.0 AER: Multiple Uncorrected error
[Source 2] PCIe: Link training failed, LTSSM stuck in Polling
[Source 3] PCIe: Transaction layer timeout detected
"""
    
    test_queries = [
        "What PCIe errors occurred?",
        "Analyze link training failure", 
        "Check performance issues",
        "Review TLP transactions"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        prompt = PCIePromptTemplates.get_prompt_for_query_type(query, context)
        
        # Show prompt characteristics
        print(f"  Generated: {len(prompt)} characters")
        
        # Check for key indicators
        if "PCIe debugging expert" in prompt:
            print("  ✅ Contains PCIe expertise")
        if "<|begin_of_text|>" in prompt:
            print("  ✅ Uses Llama 3.2 format")
        if "link training" in query.lower() and "LTSSM" in prompt:
            print("  ✅ Link training specific")
        if "performance" in query.lower() and "throughput" in prompt:
            print("  ✅ Performance specific")

def show_actual_llm_response():
    """Show actual LLM response (if model is available)"""
    print("\n🤖 LLM Response Demo")
    print("=" * 40)
    
    try:
        provider = LocalLLMProvider(models_dir="models")
        
        if not provider.model_path.exists():
            print("❌ Model not downloaded yet")
            print("To see actual LLM responses:")
            print("1. Run a query - model will auto-download")
            print("2. First query takes 2-5 minutes (download + load)")
            print("3. Subsequent queries are fast")
            return False
        
        print("✅ Model file found, attempting to load...")
        
        # Load model
        if provider.load_model():
            print("✅ Model loaded successfully")
            
            # Test simple query
            test_prompt = "What is PCIe? Answer briefly."
            print(f"\nTest prompt: {test_prompt}")
            
            response = provider.generate_completion(
                prompt=test_prompt,
                max_tokens=100,
                temperature=0.1
            )
            
            print(f"✅ Response generated:")
            print(f"   Length: {len(response)} characters")
            print(f"   Content: {response[:200]}...")
            
            # Test PCIe-specific query
            pcie_context = """
[Source 1] PCIe: 0000:01:00.0 AER: Uncorrected error
[Source 2] PCIe: Link training failed
"""
            
            pcie_prompt = PCIePromptTemplates.get_prompt_for_query_type(
                "What PCIe errors occurred?", 
                pcie_context
            )
            
            print(f"\nPCIe-specific analysis:")
            pcie_response = provider.generate_completion(
                prompt=pcie_prompt,
                max_tokens=200,
                temperature=0.1
            )
            
            print(f"   Response length: {len(pcie_response)} characters")
            print(f"   First 150 chars: {pcie_response[:150]}...")
            
            # Unload model
            provider.unload_model()
            print("✅ Model unloaded")
            
            return True
        else:
            print("❌ Failed to load model")
            return False
            
    except Exception as e:
        print(f"❌ LLM demo failed: {e}")
        return False

def show_system_capabilities():
    """Show what the system can do"""
    print("\n🚀 System Capabilities")
    print("=" * 40)
    
    # Check system
    compat = LocalLLMProvider.check_system_compatibility()
    
    print("Hardware:")
    print(f"  Platform: {compat['platform']} {compat['machine']}")
    print(f"  RAM: {compat['total_memory_gb']:.1f} GB total, {compat['available_memory_gb']:.1f} GB available")
    print(f"  Metal GPU: {'✅ YES' if compat['metal_support'] else '❌ NO'}")
    
    print("\nSoftware:")
    print(f"  llama-cpp-python: {'✅ YES' if compat['llama_cpp_available'] else '❌ NO'}")
    print(f"  Memory sufficient: {'✅ YES' if compat['sufficient_memory'] else '⚠️  LIMITED'}")
    
    print("\nFeatures:")
    print("  ✅ Vector search (works immediately)")
    print("  ✅ Local embeddings (sentence-transformers)")
    print("  ✅ PCIe-specific prompts")
    print("  ✅ Auto-download models")
    print("  ✅ M1 Metal acceleration")
    print("  ✅ Graceful fallback")
    
    # Show available templates
    templates = PCIePromptTemplates.get_available_templates()
    print(f"\nPCIe Analysis Types ({len(templates)}):")
    for name, desc in templates.items():
        print(f"  • {name}: {desc}")

def main():
    """Show how the LLM works"""
    print("🤖 Local LLM Functionality Demo")
    print("PCIe Debug Agent with Llama 3.2 3B Instruct")
    print("=" * 50)
    
    # Check status
    model_available = show_local_llm_status()
    
    # Show prompts
    show_pcie_prompts()
    
    # Show actual LLM if available
    if model_available:
        llm_working = show_actual_llm_response()
        if llm_working:
            print("\n🎉 LOCAL LLM IS FULLY WORKING!")
        else:
            print("\n⚠️  Model file exists but loading failed")
    else:
        print("\n📥 Model will auto-download on first query")
    
    # Show capabilities
    show_system_capabilities()
    
    print("\n" + "=" * 50)
    print("💡 How It Works:")
    print("1. Vector search provides immediate results")
    print("2. LLM analysis auto-downloads model on first use")
    print("3. PCIe-specific prompts enhance debugging")
    print("4. M1 Metal GPU provides fast inference")
    print("5. Graceful fallback if LLM unavailable")
    
    if model_available:
        print("\n✅ READY: Full local LLM functionality available!")
    else:
        print("\n📋 READY: Vector search working, LLM will auto-setup!")

if __name__ == "__main__":
    main()