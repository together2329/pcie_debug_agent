#!/bin/bash
# Run PCIe Debug Agent UI with Hybrid LLM support

echo "🚀 Starting PCIe Debug Agent with Hybrid LLM..."
echo "🦙 Llama 3.2 3B + 🤖 DeepSeek Q4_1"
echo ""

streamlit run src/ui/app_hybrid.py --server.port 8501 --server.address localhost
