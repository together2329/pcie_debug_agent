# Enhanced RAG v3 Dependencies Guide

## Required Dependencies for Full v3 Functionality

### Core Dependencies
```bash
# Install required packages for Enhanced RAG v3
pip install numpy>=1.21.0
pip install sentence-transformers>=2.2.0
pip install rank-bm25>=0.2.2
pip install difflib  # Usually built-in
```

### Quick Setup Script
```bash
#!/bin/bash
# setup_rag_v3.sh
echo "🚀 Setting up Enhanced RAG v3 dependencies..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install Python and pip first."
    exit 1
fi

# Install dependencies
echo "📦 Installing numpy..."
pip install numpy>=1.21.0

echo "📦 Installing sentence-transformers..."
pip install sentence-transformers>=2.2.0

echo "📦 Installing rank-bm25..."
pip install rank-bm25>=0.2.2

echo "✅ Enhanced RAG v3 dependencies installed!"
echo "🔧 Run 'pcie-debug vectordb build' to initialize vector database"
```

### Dependency Check Function
Add to `src/config/dependencies.py`:
```python
def check_rag_v3_dependencies():
    """Check if Enhanced RAG v3 dependencies are available"""
    dependencies = {
        'numpy': False,
        'sentence_transformers': False, 
        'rank_bm25': False
    }
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
    except ImportError:
        pass
        
    try:
        import rank_bm25
        dependencies['rank_bm25'] = True
    except ImportError:
        pass
    
    return dependencies

def get_rag_v3_readiness():
    """Get RAG v3 readiness status"""
    deps = check_rag_v3_dependencies()
    available = sum(deps.values())
    total = len(deps)
    
    if available == total:
        return "FULL", "All dependencies available - Enhanced RAG v3 ready"
    elif available >= 2:
        return "PARTIAL", f"{available}/{total} dependencies available - Limited functionality"
    else:
        return "BASIC", f"Only {available}/{total} dependencies - Fallback to basic RAG"
```

### Graceful Feature Detection
Update initialization in `interactive.py`:
```python
def _check_v3_capabilities(self):
    """Check Enhanced RAG v3 capabilities"""
    from src.config.dependencies import get_rag_v3_readiness
    
    status, message = get_rag_v3_readiness()
    
    if status == "FULL":
        self.v3_available = True
        if self.verbose:
            print("🚀 Enhanced RAG v3 fully available")
    elif status == "PARTIAL": 
        self.v3_available = "limited"
        if self.verbose:
            print("⚠️ Enhanced RAG v3 partially available")
    else:
        self.v3_available = False
        if self.verbose:
            print("ℹ️ Using basic RAG (install dependencies for v3)")
```