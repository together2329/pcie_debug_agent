# OpenAI Embeddings Setup Guide for Unified RAG

## Overview
This guide helps you upgrade from local embeddings to OpenAI embeddings for better quality PCIe debugging assistance.

## Current Status
- ✅ **Unified RAG is working** with local embeddings (sentence-transformers/all-MiniLM-L6-v2)
- ✅ **1,746 documents indexed** from 9 PCIe specification PDFs
- ⏳ **Ready to upgrade** to OpenAI embeddings for better quality

## Benefits of OpenAI Embeddings
- **Higher quality**: Better semantic understanding of PCIe concepts
- **Larger dimensions**: 1536 vs 384 dimensions (4x more detail)
- **Better context**: Trained on more technical documentation
- **Improved accuracy**: Better at understanding PCIe-specific terminology

## Setup Steps

### 1. Get OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Sign up or log in
3. Create a new API key
4. Copy the key (starts with `sk-`)

### 2. Set Environment Variable
```bash
# Set your API key (replace with your actual key)
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Verify it's set
echo $OPENAI_API_KEY
```

### 3. Run Setup Script
```bash
# Make sure you're in the project directory
cd /Users/brian/Desktop/Project/pcie_debug_agent

# Run the setup script
./setup_openai_embeddings.sh
```

This will:
- Detect your OpenAI API key
- Re-process all PCIe PDFs with OpenAI embeddings
- Create a new vector store at `data/vectorstore/unified_openai_1536d/`
- Update configuration to use OpenAI by default

### 4. Test the Setup
```bash
# Test that OpenAI is working
python3 test_openai_embeddings.py

# Start the interactive CLI
source venv/bin/activate
export PYTHONPATH=.
python3 src/cli/interactive.py
```

## Usage Examples

Once set up, try these queries:

```
# Basic queries
What is PCIe link training?
Explain PCIe Gen5 vs Gen6 differences
How does PCIe flow control work?

# Advanced queries with Unified RAG
/urag "Debug PCIe link training failure at L0"
/urag "Compare PCIe 4.0 and 5.0 error handling"
/urag_status  # Check which embedding model is active
```

## Cost Considerations

OpenAI embedding costs (as of 2024):
- **text-embedding-3-small**: $0.02 per 1M tokens
- Processing all PCIe specs (~3.6M characters) ≈ $0.02-0.03
- Very affordable for the quality improvement

## Switching Between Models

You can switch between embedding models:

```bash
# In the interactive CLI
/rag_model text-embedding-3-small    # OpenAI
/rag_model all-MiniLM-L6-v2          # Local (free)
/rag_model list                       # See all options
```

## Troubleshooting

### API Key Issues
```bash
# Check if key is set
echo $OPENAI_API_KEY

# Test API key
python3 -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
try:
    response = openai.models.list()
    print('✅ API key is valid')
except Exception as e:
    print(f'❌ API key error: {e}')
"
```

### Rate Limits
If you hit rate limits during setup:
1. Wait a few seconds and retry
2. The setup script handles retries automatically
3. Consider using a paid OpenAI account for higher limits

### Fallback to Local
If OpenAI fails, the system automatically falls back to local embeddings, so you're never blocked.

## Performance Comparison

| Feature | Local (MiniLM) | OpenAI (text-embedding-3-small) |
|---------|----------------|----------------------------------|
| Dimensions | 384 | 1536 |
| Quality | Good | Excellent |
| Speed | Very Fast | Fast (API calls) |
| Cost | Free | ~$0.02 per setup |
| Offline | Yes | No |
| PCIe Understanding | Good | Excellent |

## Next Steps

After setting up OpenAI embeddings:

1. **Test queries** to see the quality improvement
2. **Compare results** between local and OpenAI models
3. **Use hybrid mode** for best of both worlds
4. **Monitor costs** in your OpenAI dashboard

## Support

- Check setup status: `python3 test_openai_embeddings.py`
- View logs: `cat logs/app.log`
- Reset to local: Delete `OPENAI_API_KEY` and restart