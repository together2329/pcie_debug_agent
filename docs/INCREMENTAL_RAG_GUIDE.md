# Incremental RAG Database Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementation Guide](#implementation-guide)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

### What is Incremental RAG?

Incremental RAG (Retrieval-Augmented Generation) allows you to update your vector database without rebuilding it from scratch. Instead of processing all documents every time, it only processes new or modified documents, significantly improving performance and efficiency.

### Key Benefits

- **⚡ Performance**: 10-100x faster updates for large knowledge bases
- **💾 Efficiency**: Only process changed documents
- **📊 History Tracking**: Full audit trail of document changes
- **🔄 Flexibility**: Add, update, or remove documents on-the-fly
- **🎯 Scalability**: Handle growing knowledge bases efficiently

### Use Cases

1. **Continuous Knowledge Updates**: Add new documentation as it's created
2. **Living Documentation**: Update existing docs without full rebuilds
3. **Multi-Source Integration**: Merge knowledge from different sources
4. **Version Control**: Track document changes over time
5. **A/B Testing**: Test different document versions

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Incremental RAG System                    │
├─────────────────────┬───────────────────┬──────────────────┤
│   Document Tracker  │   Vector Manager  │  Metadata Store  │
├─────────────────────┼───────────────────┼──────────────────┤
│ • Hash calculation  │ • FAISS index     │ • Document info  │
│ • Change detection  │ • Embeddings      │ • Chunk mapping  │
│ • Version tracking  │ • Search/Add/Del  │ • Update history │
└─────────────────────┴───────────────────┴──────────────────┘
```

### Data Flow

```
1. Document Input
   ↓
2. Hash Calculation → Compare with existing
   ↓                    ↓
3. New/Modified?    Unchanged → Skip
   ↓ Yes
4. Chunk Document
   ↓
5. Generate Embeddings
   ↓
6. Add to Vector Store
   ↓
7. Update Metadata
   ↓
8. Save State
```

### Metadata Structure

```json
{
  "documents": {
    "path/to/doc.md": {
      "hash": "sha256_hash",
      "last_modified": "2024-01-15T10:30:00",
      "version": 1,
      "chunks": ["chunk_id_1", "chunk_id_2"],
      "embedding_model": "text-embedding-3-small",
      "stats": {
        "size_bytes": 1024,
        "chunk_count": 2,
        "avg_chunk_size": 512
      }
    }
  },
  "index_info": {
    "total_documents": 50,
    "total_chunks": 500,
    "last_update": "2024-01-15T10:30:00",
    "vector_dimension": 1536
  }
}
```

## Implementation Guide

### Step 1: Create Incremental Manager

```python
from pathlib import Path
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Tuple

class IncrementalRAGManager:
    def __init__(self, vector_store_path: str):
        self.vector_store_path = Path(vector_store_path)
        self.metadata_path = self.vector_store_path / "incremental_metadata.json"
        self.document_tracker = self.load_metadata()
        
    def load_metadata(self) -> Dict:
        """Load existing metadata or create new"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {
            "documents": {},
            "index_info": {
                "total_documents": 0,
                "total_chunks": 0,
                "last_update": None
            }
        }
```

### Step 2: Implement Change Detection

```python
def calculate_file_hash(self, file_path: Path) -> str:
    """Calculate SHA256 hash of file content"""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def detect_changes(self, source_dir: Path) -> Tuple[List[Path], List[Path], List[str]]:
    """Detect new, modified, and deleted files"""
    new_files = []
    modified_files = []
    deleted_files = []
    
    # Check existing files
    current_files = set()
    for file_path in source_dir.glob("**/*.md"):
        relative_path = str(file_path.relative_to(source_dir))
        current_files.add(relative_path)
        
        file_hash = self.calculate_file_hash(file_path)
        
        if relative_path not in self.document_tracker["documents"]:
            new_files.append(file_path)
        elif self.document_tracker["documents"][relative_path]["hash"] != file_hash:
            modified_files.append(file_path)
    
    # Check for deleted files
    for doc_path in self.document_tracker["documents"]:
        if doc_path not in current_files:
            deleted_files.append(doc_path)
    
    return new_files, modified_files, deleted_files
```

### Step 3: Process Documents Incrementally

```python
def add_documents(self, files: List[Path], chunker, embedding_provider, vector_store):
    """Add new documents to vector store"""
    for file_path in files:
        # Read document
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk document
        chunks = chunker.chunk_text(content)
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = embedding_provider.encode(chunk_texts)
        
        # Create metadata
        chunk_ids = []
        metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path.name}_chunk_{i}_{int(time.time())}"
            chunk_ids.append(chunk_id)
            metadata_list.append({
                'source': str(file_path),
                'chunk_id': chunk_id,
                'chunk_index': i,
                'timestamp': datetime.now().isoformat()
            })
        
        # Add to vector store
        vector_store.add_documents(
            embeddings=embeddings.tolist(),
            documents=chunk_texts,
            metadata=metadata_list
        )
        
        # Update tracker
        relative_path = str(file_path.relative_to(self.source_dir))
        self.document_tracker["documents"][relative_path] = {
            "hash": self.calculate_file_hash(file_path),
            "last_modified": datetime.now().isoformat(),
            "chunks": chunk_ids,
            "version": self.get_next_version(relative_path)
        }
```

### Step 4: Handle Updates and Deletions

```python
def update_documents(self, files: List[Path], vector_store):
    """Update modified documents"""
    for file_path in files:
        relative_path = str(file_path.relative_to(self.source_dir))
        
        # Remove old chunks
        old_chunks = self.document_tracker["documents"][relative_path]["chunks"]
        self.remove_chunks_from_index(old_chunks, vector_store)
        
        # Add updated document
        self.add_documents([file_path], chunker, embedding_provider, vector_store)

def remove_documents(self, doc_paths: List[str], vector_store):
    """Remove deleted documents from index"""
    for doc_path in doc_paths:
        if doc_path in self.document_tracker["documents"]:
            chunks = self.document_tracker["documents"][doc_path]["chunks"]
            self.remove_chunks_from_index(chunks, vector_store)
            del self.document_tracker["documents"][doc_path]
```

## Usage Examples

### Basic Incremental Update

```python
# Initialize manager
manager = IncrementalRAGManager("data/vectorstore")

# Detect changes
new, modified, deleted = manager.detect_changes(Path("data/knowledge_base"))

# Process changes
if new:
    print(f"Adding {len(new)} new documents...")
    manager.add_documents(new, chunker, embedding_provider, vector_store)

if modified:
    print(f"Updating {len(modified)} modified documents...")
    manager.update_documents(modified, vector_store)

if deleted:
    print(f"Removing {len(deleted)} deleted documents...")
    manager.remove_documents(deleted, vector_store)

# Save state
manager.save_metadata()
vector_store.save("data/vectorstore")
```

### CLI Usage

```bash
# Add new documents
pcie-debug vectordb add data/new_docs/

# Update all changed documents
pcie-debug vectordb update

# Remove specific documents
pcie-debug vectordb remove "*.tmp"

# Show database status
pcie-debug vectordb status

# View update history
pcie-debug vectordb history --limit 10
```

### Scheduled Updates

```python
import schedule
import time

def incremental_update_job():
    """Run incremental update"""
    manager = IncrementalRAGManager("data/vectorstore")
    vector_store = FAISSVectorStore.load("data/vectorstore")
    
    new, modified, deleted = manager.detect_changes(Path("data/knowledge_base"))
    
    if new or modified or deleted:
        print(f"Processing {len(new)} new, {len(modified)} modified, {len(deleted)} deleted")
        # Process changes...
        manager.save_metadata()
        vector_store.save("data/vectorstore")

# Schedule hourly updates
schedule.every().hour.do(incremental_update_job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## API Reference

### IncrementalRAGManager

#### Constructor
```python
__init__(self, vector_store_path: str, source_dir: Optional[str] = None)
```

#### Methods

##### detect_changes
```python
detect_changes(self, source_dir: Path) -> Tuple[List[Path], List[Path], List[str]]
```
Returns: (new_files, modified_files, deleted_file_paths)

##### add_documents
```python
add_documents(self, files: List[Path], chunker, embedding_provider, vector_store) -> int
```
Returns: Number of chunks added

##### update_documents
```python
update_documents(self, files: List[Path], vector_store) -> int
```
Returns: Number of documents updated

##### remove_documents
```python
remove_documents(self, doc_paths: List[str], vector_store) -> int
```
Returns: Number of documents removed

##### get_statistics
```python
get_statistics(self) -> Dict[str, Any]
```
Returns: Database statistics

### CLI Commands

#### vectordb add
```bash
pcie-debug vectordb add [OPTIONS] PATH

Options:
  --recursive        Include subdirectories
  --pattern TEXT     File pattern (e.g., "*.md")
  --force           Override existing documents
```

#### vectordb update
```bash
pcie-debug vectordb update [OPTIONS]

Options:
  --source PATH     Source directory (default: data/knowledge_base)
  --dry-run        Show what would be updated without doing it
```

#### vectordb remove
```bash
pcie-debug vectordb remove [OPTIONS] PATTERN

Options:
  --confirm        Skip confirmation prompt
```

#### vectordb status
```bash
pcie-debug vectordb status [OPTIONS]

Options:
  --detailed       Show per-document statistics
  --json          Output as JSON
```

## Best Practices

### 1. Document Organization

```
knowledge_base/
├── core/           # Core documentation (rarely changes)
├── guides/         # User guides (occasional updates)
├── api/            # API docs (frequent updates)
└── examples/       # Code examples (very frequent updates)
```

### 2. Update Strategies

#### Continuous Updates
- Best for: Frequently changing documentation
- Run: Every commit or push
- Example: CI/CD pipeline integration

#### Batch Updates
- Best for: Stable documentation
- Run: Daily or weekly
- Example: Scheduled cron job

#### On-Demand Updates
- Best for: Manual documentation management
- Run: When explicitly triggered
- Example: After documentation review

### 3. Performance Optimization

```python
# Use batch processing
BATCH_SIZE = 100
for i in range(0, len(files), BATCH_SIZE):
    batch = files[i:i + BATCH_SIZE]
    manager.add_documents(batch, chunker, embedding_provider, vector_store)

# Enable parallel processing
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file, f) for f in files]
```

### 4. Error Handling

```python
def safe_incremental_update(manager, source_dir):
    """Update with error handling and rollback"""
    backup_metadata = manager.document_tracker.copy()
    
    try:
        new, modified, deleted = manager.detect_changes(source_dir)
        # Process changes...
        manager.save_metadata()
    except Exception as e:
        # Rollback on error
        manager.document_tracker = backup_metadata
        manager.save_metadata()
        raise e
```

## Troubleshooting

### Common Issues

#### 1. Dimension Mismatch
**Problem**: "Dimension mismatch between embeddings and index"
**Solution**: Ensure same embedding model is used
```python
# Check current model
current_model = embedding_selector.get_current_model()
stored_model = metadata["index_info"].get("embedding_model")
if current_model != stored_model:
    print(f"Model mismatch: {current_model} vs {stored_model}")
```

#### 2. Memory Issues
**Problem**: "Out of memory when processing large files"
**Solution**: Use streaming and chunked processing
```python
# Process in smaller chunks
def process_large_file(file_path, chunk_size=1000):
    with open(file_path, 'r') as f:
        while chunk := f.read(chunk_size):
            yield chunk
```

#### 3. Corrupted Index
**Problem**: "Failed to load vector index"
**Solution**: Implement backup and recovery
```python
# Regular backups
def backup_index(vector_store_path):
    backup_path = f"{vector_store_path}_backup_{datetime.now().strftime('%Y%m%d')}"
    shutil.copytree(vector_store_path, backup_path)
```

### Debug Mode

Enable verbose logging for troubleshooting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via CLI
pcie-debug vectordb update --verbose --dry-run
```

### Performance Metrics

Monitor update performance:
```python
import time

start_time = time.time()
chunks_processed = manager.add_documents(files, ...)
elapsed = time.time() - start_time

print(f"Performance: {chunks_processed / elapsed:.2f} chunks/second")
```

## Advanced Topics

### Multi-Version Support

Track multiple versions of documents:
```python
"documents": {
    "guide.md": {
        "current_version": 3,
        "versions": {
            "1": {"hash": "abc123", "date": "2024-01-01"},
            "2": {"hash": "def456", "date": "2024-01-10"},
            "3": {"hash": "ghi789", "date": "2024-01-15"}
        }
    }
}
```

### Distributed Updates

For large-scale deployments:
```python
# Use Redis for distributed locking
import redis
r = redis.Redis()

with r.lock("vectordb_update"):
    # Perform update
    manager.incremental_update()
```

### Custom Chunking Strategies

Implement document-specific chunking:
```python
def get_chunker_for_document(file_path):
    if file_path.suffix == '.md':
        return MarkdownChunker()
    elif file_path.suffix == '.py':
        return CodeChunker()
    else:
        return DefaultChunker()
```

## Conclusion

Incremental RAG updates provide a powerful way to maintain and scale your knowledge base efficiently. By following this guide, you can implement a robust system that handles continuous updates while maintaining performance and reliability.

For more information and updates, see:
- [GitHub Repository](https://github.com/together2329/pcie_debug_agent)
- [API Documentation](/docs/api/)
- [Examples](/examples/incremental_rag/)