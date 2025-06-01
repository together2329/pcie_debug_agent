# Incremental RAG Documentation

This directory contains comprehensive documentation and examples for implementing Incremental RAG (Retrieval-Augmented Generation) updates in the PCIe Debug Agent.

## 📁 Directory Contents

### Documentation
- **[Complete Guide](../INCREMENTAL_RAG_GUIDE.md)** - Comprehensive implementation guide
- **[Quick Start](../INCREMENTAL_RAG_QUICKSTART.md)** - Get started in 5 minutes

### Code Examples
- **[implementation_example.py](./implementation_example.py)** - Complete reference implementation
- **[simple_example.py](./simple_example.py)** - Minimal working example
- **[cli_integration.py](./cli_integration.py)** - CLI command integration
- **[auto_update.py](./auto_update.py)** - Automatic update examples

### Configuration Templates
- **[config_template.yaml](./config_template.yaml)** - Configuration template
- **[metadata_schema.json](./metadata_schema.json)** - Metadata structure schema

## 🚀 Quick Start

### 1. Basic Implementation
```python
from incremental_rag import IncrementalRAGManager

# Initialize
manager = IncrementalRAGManager("data/vectorstore")

# Perform update
stats = manager.perform_incremental_update("data/knowledge_base")

print(f"Added {stats.new_documents} new documents")
```

### 2. CLI Usage
```bash
# Update incrementally
pcie-debug vectordb update

# Add specific files
pcie-debug vectordb add path/to/new/docs/

# Check status
pcie-debug vectordb status
```

## 🎯 Key Concepts

### Document Tracking
- Each document is tracked by content hash
- Changes are detected automatically
- Version history is maintained

### Update Process
1. **Detection**: Find new/modified/deleted files
2. **Processing**: Generate embeddings only for changes
3. **Integration**: Add to existing vector store
4. **Persistence**: Save metadata and state

### Performance Benefits
- **Speed**: 10-100x faster than full rebuild
- **Efficiency**: Only process what changed
- **Scalability**: Handle large knowledge bases

## 📊 Architecture

```
IncrementalRAGManager
├── Document Tracker (hash-based change detection)
├── Version Manager (track document history)
├── Vector Store Manager (FAISS integration)
└── Metadata Persistence (JSON storage)
```

## 🛠️ Implementation Checklist

- [ ] Set up IncrementalRAGManager class
- [ ] Implement document hash tracking
- [ ] Add change detection logic
- [ ] Create incremental update methods
- [ ] Add CLI commands
- [ ] Set up automatic updates
- [ ] Add monitoring and logging
- [ ] Test with your documents

## 📚 Further Reading

- [Performance Optimization Tips](../performance.md)
- [Troubleshooting Guide](../troubleshooting.md)
- [API Reference](../api_reference.md)

## 💡 Tips

1. **Start Simple**: Use the minimal example first
2. **Test Incrementally**: Add documents one at a time initially
3. **Monitor Performance**: Track update times and sizes
4. **Backup Regularly**: Enable automatic backups
5. **Version Control**: Track metadata files in git

## 🤝 Contributing

Have improvements or examples to share? Please contribute:
1. Fork the repository
2. Add your examples to this directory
3. Update this README
4. Submit a pull request

---

For questions or issues, please open a GitHub issue.