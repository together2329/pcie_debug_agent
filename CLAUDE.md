# CLAUDE.md - Project Context for AI Assistant

## Project Overview
PCIe Debug Agent - An advanced AI-powered tool for analyzing PCIe debug logs and troubleshooting PCIe issues using RAG (Retrieval-Augmented Generation) technology with multi-modal search capabilities.

## Key Components

### 1. Core Modules
- **src/cli/interactive.py**: Main interactive shell with Claude Code-style interface
- **src/rag/enhanced_rag_engine.py**: Main RAG engine for processing and analyzing logs
- **src/rag/hybrid_search.py**: Hybrid search engine combining BM25 and semantic search
- **src/vectorstore/faiss_store.py**: FAISS vector database management
- **src/vectorstore/multi_model_manager.py**: Multi-model vector database manager
- **src/collectors/log_collector.py**: Collects and processes PCIe debug logs
- **src/processors/**: Document processing and embedding modules

### 2. UI Components
- **src/ui/app.py**: Streamlit-based web interface
- **src/ui/interactive_chat.py**: Chat interface for querying logs
- **src/ui/semantic_search.py**: Semantic search functionality

### 3. Configuration
- **src/config/settings.py**: Application configuration management
- **configs/settings.yaml**: YAML configuration file

### 4. Interactive Shell Commands
The interactive shell (`src/cli/interactive.py`) provides Claude Code-style commands:

#### Model Management
- `/model [model-id]` - List or switch AI models (gpt-4o-mini, llama-3.2-3b, etc.)
- `/rag_model [model]` - Switch RAG embedding models
- `/rag_mode [mode]` - Select search mode (semantic/hybrid/keyword)

#### RAG Features
- `/rag [on/off]` - Toggle RAG functionality
- `/rag_status` - Show detailed RAG and vector DB status
- `/rag_files` - Show which files are indexed in each RAG database
- `/rag_check` - Quick check of current RAG database readiness
- `/knowledge_base` or `/kb` - Show knowledge base content and status

#### Session Management
- `/cost` - Show session cost and duration analysis
- `/tokens` - Show detailed token usage
- `/clear` - Clear conversation history
- `/session [save/load]` - Manage conversation sessions
- `/memory [set/del]` - Manage persistent memory

#### System Features
- `/doctor` - Comprehensive system health check
- `/status` - Show system status
- `/config` - Show configuration
- `/verbose [on/off]` - Toggle verbose analysis mode
- `/stream [on/off]` - Toggle real-time streaming responses

#### Other Commands
- `/analyze <log>` - Analyze PCIe log file
- `/search <query>` - Search knowledge base
- `/help` - Show all available commands
- `/` - Show command suggestions (like Claude Code)
- `/exit` or `/quit` - Exit the shell

### 5. Key Features Added
- **Tab Auto-completion**: Press TAB to complete commands and arguments
- **Command Suggestions**: Type `/` to see all available commands
- **Multi-modal Search**: Semantic, hybrid, and keyword search modes
- **Cost Tracking**: Real-time token usage and cost estimation
- **Health Monitoring**: System health checks with `/doctor`
- **Multi-model Support**: Switch between different embedding models
- **Streaming Responses**: Real-time response streaming
- **Session Management**: Save and load conversation sessions

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guide
- Use type hints for function parameters and returns
- Keep functions focused and modular

### Testing
- Run tests before committing changes
- Test command: `python -m pytest tests/` (if tests exist)

### Linting
- Use `flake8` or `pylint` for code quality checks
- Command: `flake8 src/` or `pylint src/`

## Common Tasks

### Running the Interactive Shell
```bash
# Start the interactive PCIe Debug Agent
python src/cli/main.py

# Or with a specific model
python src/cli/main.py --model gpt-4o-mini

# With verbose mode
python src/cli/main.py --verbose
```

### Building RAG Database
```bash
# Build vector database for current embedding model
pcie-debug vectordb build

# Force rebuild
pcie-debug vectordb build --force

# Build for specific model
pcie-debug vectordb build --model text-embedding-3-small
```

### Adding New Features
1. Update relevant modules in `src/`
2. Add slash commands in `src/cli/interactive.py`
3. Update tab completion in `_complete_command_args()`
4. Test thoroughly with all search modes
5. Update CLAUDE.md documentation

### Debugging
- Check logs in `logs/` directory
- Use `/verbose on` for detailed analysis steps
- Use `/doctor` for system health check
- Use `/rag_check` to verify RAG database status

## Dependencies
- Main dependencies are listed in `requirements.txt`
- Key libraries: streamlit, faiss-cpu, sentence-transformers, rank-bm25, openai
- Python 3.8+ required

## Docker Support
- `Dockerfile` and `docker-compose.yml` available for containerized deployment
- Build: `docker-compose build`
- Run: `docker-compose up`

## Recent Updates (2025-06-01)

### New Commands
1. **`/rag_mode`** - Switch between semantic/hybrid/keyword search modes
2. **`/cost`** - Show session cost and token usage with detailed breakdown
3. **`/doctor`** - Comprehensive system health check (dependencies, memory, disk)
4. **`/rag_files`** - Show which files are indexed in each RAG database
5. **`/rag_check`** - Quick check of current database readiness

### Bug Fixes
1. Fixed hybrid search mode initialization error
2. Fixed RAG document retrieval (lowered similarity threshold from 0.5 to 0.1)
3. Suppressed empty error messages in vector store search
4. Fixed tab completion for slash commands

### Improvements
1. Added Claude Code-style command suggestions when typing `/`
2. Enhanced verbose mode with detailed search statistics
3. Added multi-model vector database support
4. Improved error handling and user feedback
5. Added file status indicators in RAG commands

## Important Notes

### Search Mode Differences
- **Semantic**: Cosine similarity (0.0-1.0), best for conceptual queries
- **Hybrid**: Normalized combination (0.0-1.0), best overall performance
- **Keyword**: BM25 scores (0.0-10.0+), best for exact term matches

### RAG System
- Default similarity threshold: 0.1 (lowered from 0.5 for better recall)
- Supports multiple embedding models simultaneously
- Automatic BM25 index creation for hybrid/keyword search
- Each embedding model has its own vector database

### Testing Commands
Always test new features with:
1. All three search modes (semantic/hybrid/keyword)
2. Both verbose on/off states
3. Different embedding models
4. Tab completion functionality