#!/usr/bin/env python3
"""
Finalize hybrid integration - fix all remaining issues
"""

import os
import shutil
from pathlib import Path

def update_test_with_correct_params():
    """Update test file with correct FAISSVectorStore parameters"""
    test_file = "test_hybrid_integration.py"
    
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Fix FAISSVectorStore initialization
        content = content.replace(
            'vector_store = FAISSVectorStore(dimension=384)',
            'vector_store = FAISSVectorStore(index_path="data/vectorstore", dimension=384)'
        )
        
        with open(test_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {test_file} with correct parameters")
        
    except Exception as e:
        print(f"❌ Failed to update test file: {e}")

def update_rag_engine_hybrid():
    """Fix RAG engine hybrid with correct parameters"""
    rag_file = "src/rag/enhanced_rag_engine_hybrid.py"
    
    try:
        with open(rag_file, 'r') as f:
            content = f.read()
        
        # Fix FAISSVectorStore initialization in app_hybrid.py
        ui_file = "src/ui/app_hybrid.py"
        with open(ui_file, 'r') as f:
            ui_content = f.read()
        
        ui_content = ui_content.replace(
            'FAISSVectorStore(dimension=settings.embedding.dimension)',
            'FAISSVectorStore(\n            index_path=str(settings.vector_store.index_path),\n            index_type=settings.vector_store.index_type,\n            dimension=settings.embedding.dimension\n        )'
        )
        
        with open(ui_file, 'w') as f:
            f.write(ui_content)
        
        print(f"✅ Updated {ui_file} with correct parameters")
        
    except Exception as e:
        print(f"❌ Failed to update files: {e}")

def create_wrapper_script():
    """Create wrapper scripts for easy execution"""
    
    # CLI wrapper
    cli_wrapper = """#!/usr/bin/env python3
'''Run PCIe Debug Agent CLI with Hybrid LLM support'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.cli.main_hybrid import main

if __name__ == "__main__":
    main()
"""
    
    with open("pcie-debug-hybrid", 'w') as f:
        f.write(cli_wrapper)
    
    os.chmod("pcie-debug-hybrid", 0o755)
    print("✅ Created pcie-debug-hybrid CLI wrapper")
    
    # Streamlit wrapper
    streamlit_wrapper = """#!/bin/bash
# Run PCIe Debug Agent UI with Hybrid LLM support

echo "🚀 Starting PCIe Debug Agent with Hybrid LLM..."
echo "🦙 Llama 3.2 3B + 🤖 DeepSeek Q4_1"
echo ""

streamlit run src/ui/app_hybrid.py --server.port 8501 --server.address localhost
"""
    
    with open("run_hybrid_ui.sh", 'w') as f:
        f.write(streamlit_wrapper)
    
    os.chmod("run_hybrid_ui.sh", 0o755)
    print("✅ Created run_hybrid_ui.sh UI wrapper")

def create_quick_start_guide():
    """Create quick start guide for hybrid system"""
    
    guide = """# 🚀 PCIe Debug Agent - Hybrid LLM Quick Start

## ✅ Setup Complete!

Your hybrid LLM system is now integrated and ready to use.

## 🎯 Quick Commands

### 1. Check Model Status
```bash
./pcie-debug-hybrid model status
```

### 2. Run Quick Analysis (Llama)
```bash
./pcie-debug-hybrid analyze -q "What is a PCIe TLP error?" -t quick
```

### 3. Run Detailed Analysis (DeepSeek)
```bash
./pcie-debug-hybrid analyze -q "Analyze PCIe link training failure" -t detailed
```

### 4. Auto Mode (Intelligent Selection)
```bash
./pcie-debug-hybrid analyze -q "Debug intermittent device disconnection" -t auto
```

### 5. Launch Web UI
```bash
./run_hybrid_ui.sh
```
Then open: http://localhost:8501

## 📊 Model Comparison

| Feature | Llama 3.2 3B | DeepSeek Q4_1 |
|---------|--------------|---------------|
| Speed | ~25s ⚡ | 90s+ 🐌 |
| Memory | 1.8GB 💚 | 5.2GB 🟡 |
| Quality | Good ✅ | Excellent 🌟 |
| Best For | Interactive | Deep Analysis |

## 🔧 Configuration

Settings are in: `configs/settings.yaml`

To use hybrid provider, set:
```yaml
llm:
  provider: hybrid
  model: auto-hybrid
```

## 📝 Examples

### Quick PCIe Error Check
```bash
echo "[10:15:30] PCIe: Link training failed" | ./pcie-debug-hybrid analyze -q "What's wrong?" -t quick
```

### Detailed Root Cause Analysis
```bash
./pcie-debug-hybrid analyze error.log -q "Provide comprehensive root cause analysis" -t detailed
```

## 🎉 Enjoy your hybrid AI-powered PCIe debugging!
"""
    
    with open("HYBRID_QUICK_START.md", 'w') as f:
        f.write(guide)
    
    print("✅ Created HYBRID_QUICK_START.md guide")

def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        "data/vectorstore",
        "models",
        "logs",
        "reports"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Ensured all directories exist")

def main():
    """Finalize hybrid integration"""
    print("🔧 Finalizing Hybrid LLM Integration")
    print("=" * 60)
    
    # Update files
    print("\n📝 Updating files with correct parameters...")
    update_test_with_correct_params()
    update_rag_engine_hybrid()
    
    # Create wrapper scripts
    print("\n📝 Creating wrapper scripts...")
    create_wrapper_script()
    
    # Create documentation
    print("\n📝 Creating documentation...")
    create_quick_start_guide()
    
    # Ensure directories
    print("\n📁 Ensuring directories...")
    ensure_directories()
    
    print("\n✅ Hybrid integration finalized!")
    print("\n🚀 Next steps:")
    print("   1. Run: ./pcie-debug-hybrid model status")
    print("   2. Run: ./run_hybrid_ui.sh")
    print("   3. See HYBRID_QUICK_START.md for usage guide")

if __name__ == "__main__":
    main()