# CLAUDE.md - Project Context for AI Assistant

## Project Overview
PCIe Debug Agent - A tool for analyzing PCIe debug logs and code using RAG (Retrieval-Augmented Generation) technology.

## Key Components

### 1. Core Modules
- **src/main.py**: Entry point for the application
- **src/rag/enhanced_rag_engine.py**: Main RAG engine for processing and analyzing logs
- **src/rag/vector_store.py**: Vector database management for embeddings
- **src/collectors/log_collector.py**: Collects and processes PCIe debug logs
- **src/processors/**: Document processing and embedding modules

### 2. UI Components
- **src/ui/app.py**: Streamlit-based web interface
- **src/ui/interactive_chat.py**: Chat interface for querying logs
- **src/ui/semantic_search.py**: Semantic search functionality

### 3. Configuration
- **src/config/settings.py**: Application configuration management
- **configs/settings.yaml**: YAML configuration file

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

### Adding New Features
1. Update relevant modules in `src/`
2. Update configuration if needed
3. Test thoroughly
4. Update documentation

### Debugging
- Check logs in `logs/` directory
- Use debug mode in Streamlit: `streamlit run src/ui/app.py --server.runOnSave true`

## Dependencies
- Main dependencies are listed in `requirements.txt`
- Key libraries: streamlit, faiss, langchain, transformers

## Docker Support
- `Dockerfile` and `docker-compose.yml` available for containerized deployment
- Build: `docker-compose build`
- Run: `docker-compose up`