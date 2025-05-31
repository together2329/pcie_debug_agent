#!/bin/bash

echo "🚀 DeepSeek Q4_1 vs Llama 3.2 3B Setup and Test Script"
echo "========================================================="
echo ""

# Check system info
echo "📱 System Information:"
system_profiler SPHardwareDataType | grep "Model Name\|Memory\|Chip"
echo ""

# Check if Ollama is installed
echo "🔍 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed: $(ollama --version)"
else
    echo "❌ Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "✅ Ollama installed"
fi
echo ""

# Check current Ollama models
echo "📋 Current Ollama models:"
ollama list
echo ""

# Check if DeepSeek model is available
echo "🔍 Checking DeepSeek Q4_1 model..."
if ollama list | grep -q "deepseek-r1:8b-0528-qwen3-q4_1"; then
    echo "✅ DeepSeek Q4_1 model is already available"
else
    echo "📥 DeepSeek Q4_1 model not found. Downloading..."
    echo "⚠️  This will download ~5.2 GB. Continue? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🔄 Downloading DeepSeek Q4_1 model..."
        ollama pull deepseek-r1:8b-0528-qwen3-q4_1
        echo "✅ DeepSeek Q4_1 model downloaded"
    else
        echo "⏭️ Skipping DeepSeek download"
    fi
fi
echo ""

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python3 -c "import psutil; print('✅ psutil available')" 2>/dev/null || echo "❌ psutil not available - install with: pip install psutil"

# Check if local Llama is available
echo "🔍 Checking local Llama setup..."
cd /Users/brian/Desktop/Project/pcie_debug_agent
python3 -c "
try:
    from src.models.local_llm_provider import LocalLLMProvider
    provider = LocalLLMProvider()
    if provider.is_available():
        print('✅ Llama 3.2 3B model is available')
    else:
        print('⚠️ Llama 3.2 3B model not found - will auto-download on first use')
except Exception as e:
    print(f'❌ Llama setup issue: {e}')
"
echo ""

# Quick test DeepSeek
echo "🧪 Quick DeepSeek test..."
if ollama list | grep -q "deepseek-r1:8b-0528-qwen3-q4_1"; then
    echo "Testing DeepSeek with a simple query..."
    echo "What is PCIe?" | ollama run deepseek-r1:8b-0528-qwen3-q4_1 | head -3
    echo "✅ DeepSeek test completed"
else
    echo "⏭️ Skipping DeepSeek test - model not available"
fi
echo ""

# Run the comprehensive comparison
echo "🔬 Ready to run comprehensive comparison?"
echo "This will test both models on 6 PCIe debugging scenarios"
echo "Continue? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "🚀 Running comprehensive comparison..."
    python3 src/simulators/deepseek_vs_llama_test.py
else
    echo "⏭️ Skipping comprehensive test"
    echo ""
    echo "To run manually later:"
    echo "  cd /Users/brian/Desktop/Project/pcie_debug_agent"
    echo "  python3 src/simulators/deepseek_vs_llama_test.py"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 What's been set up:"
echo "  ✅ Ollama installation verified"
echo "  ✅ DeepSeek Q4_1 model available (if downloaded)"
echo "  ✅ Comprehensive test script ready"
echo ""
echo "🚀 Next steps:"
echo "  1. Run: python3 src/simulators/deepseek_vs_llama_test.py"
echo "  2. Compare the detailed results"
echo "  3. Choose the best model for your PCIe debugging needs"