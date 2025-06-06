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
- `/model [model-id]` - List or switch AI models (default: gpt-4o-mini, llama-3.2-3b, etc.)
- `/rag_model [model]` - Switch RAG embedding models
- `/rag_mode [mode]` - Select search mode (semantic/hybrid/keyword/unified/pcie)

#### Unified RAG System (2025-06-03)
- `/rag <query>` - **Primary RAG command** with intelligent PCIe analysis and auto-testing
- `/rag <query> --engine [production|v3|standard]` - Query with specific engine
- `/rag --test` - Run comprehensive quality test suite (10 PCIe scenarios)
- `/rag --status` - Show system health, quality metrics, and performance
- `/rag --engines` - List available engines and current priority
- `/rag --config` - Display unified RAG configuration

#### Legacy RAG Commands (Deprecated - Use `/rag` instead)
- `/rag_status` - Show detailed RAG and vector DB status
- `/rag_files` - Show which files are indexed in each RAG database
- `/rag_check` - Quick check of current RAG database readiness
- `/rag_analyze` - Enhanced analysis (replaced by `/rag`)
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
- **Unified RAG System**: Single `/rag` command with auto-testing and quality monitoring
- **Auto-Quality Testing**: Automated background testing every hour with 10 PCIe scenarios
- **Multi-Engine Fallback**: Production, V3, and Standard engines with automatic switching
- **PCIe Compliance Intelligence**: Instant recognition of FLR/CRS violations and debugging
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
- **Automated RAG Testing**: `python automated_rag_test_suite.py` - Comprehensive quality testing
- **Interactive Testing**: `/rag --test` - Run quality test suite from within shell
- **Unit Tests**: `python -m pytest tests/` (if tests exist)
- **Quality Benchmarks**: Aim for >70% overall score, >80% for critical compliance tests

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

# Deploy unified RAG system (one-time setup)
python deploy_unified_rag.py
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
4. **Test with unified RAG**: Run `/rag --test` to ensure quality
5. **Add test cases**: Update `automated_rag_test_suite.py` for new features
6. Test thoroughly with all search modes
7. Update CLAUDE.md documentation

### Unified RAG Quality Management
```bash
# Check current system quality
/rag --status

# Run comprehensive test suite (10 PCIe scenarios)
/rag --test

# Monitor performance over time
/rag --engines

# Test specific compliance scenarios
/rag "FLR compliance test query"
```

### Debugging
- **Unified RAG Issues**: Use `/rag --status` for comprehensive system health
- **Quality Problems**: Run `/rag --test` to identify specific test failures
- **Performance Issues**: Check `/rag --engines` for engine switching patterns
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

## Recent Updates

### Latest Release: PCIe Adaptive RAG Mode (2025-06-05)

#### New PCIe-Optimized RAG Architecture
1. **`/rag_mode pcie`** - PCIe-specific adaptive chunking with 1000-word target chunks
2. **Semantic Boundary Preservation** - Respects headers, procedures, and specification sections
3. **PCIe Concept Extraction** - Automatically identifies LTSSM states, TLP types, error conditions, power states
4. **Technical Level Classification** - Filters content by complexity (1=basic, 2=intermediate, 3=advanced)
5. **Adaptive Chunking Strategy** - Intelligent splitting that maintains technical coherence

#### PCIe Mode Features
1. **`/rag_mode pcie`** - Switch to PCIe-optimized chunking mode
2. **Technical Level Filtering** - Query by complexity level for precise results
3. **PCIe Layer Filtering** - Filter by physical, transaction, data_link, power_management layers
4. **Concept Boosting** - Automatic boosting of results with matching PCIe concepts
5. **Enhanced Metadata** - Rich context including semantic type, technical level, PCIe concepts
6. **Smart Detection** - Auto-detects PCIe queries and suggests PCIe mode
7. **Full CLI Integration** - Complete integration with pcie-debug command line tool

#### Interactive Usage (Primary Method)
```bash
# Start interactive shell
./pcie-debug

# System auto-detects PCIe queries and suggests PCIe mode
🔧 > What are LTSSM states?
🔧 Detected PCIe-related query!
💡 Consider switching to PCIe mode for better results:
👉 Type '/rag_mode pcie' for optimized PCIe answers

# Switch to PCIe mode
🔧 > /rag_mode pcie
✅ PCIe adaptive RAG engine ready!
📊 Vectors: 1,896
🧮 Chunking: adaptive
🎯 Concept boosting: enabled

# Ask PCIe questions directly
🔧 > What are LTSSM timeout conditions?
🔧 > How does FLR work?
🔧 > Explain malformed TLP conditions?
```

#### Command Line Tools
```bash
# Build PCIe knowledge base
./pcie-debug pcie build --force --target-size 1000

# Query with advanced filters  
./pcie-debug pcie query "LTSSM states" --top-k 3 --technical-level 2
./pcie-debug pcie query "Power management" --layer power_management
./pcie-debug pcie query "TLP examples" --semantic-type example

# Get comprehensive statistics
./pcie-debug pcie stats

# One-shot PCIe queries
./pcie-debug -p "What causes PCIe completion timeout errors?"

# Run comprehensive test suite
python test_pcie_adaptive_rag.py
python test_pcie_integration.py  # Integration tests
```

#### Smart PCIe Detection
The system automatically detects PCIe-related queries using 25+ keywords:
- **Core PCIe**: pcie, ltssm, tlp, flr, aer, aspm
- **Technical**: completion timeout, link training, hot reset, malformed tlp
- **Layers**: physical layer, transaction layer, data link layer, system architecture
- **Components**: endpoint, root complex, switch, bridge, config space
- **Advanced**: ecrc, lcrc, dllp, ordered set, equalization, eye diagram

#### Technical Implementation
- **Adaptive Chunker**: `src/processors/pcie_adaptive_chunker.py` - 1000-word target with semantic boundaries
- **PCIe RAG Engine**: `src/rag/pcie_rag_engine.py` - Specialized engine with concept boosting
- **CLI Integration**: Enhanced `/rag_mode` command with PCIe option plus full CLI commands
- **Interactive Integration**: Auto-detection and suggestion system in interactive mode
- **Vector Database**: Separate `pcie_adaptive_*` databases for optimized storage

#### Performance Improvements
1. **Better Context Preservation** - 1000-word chunks maintain complete procedures
2. **Semantic Coherence** - Chunks respect natural document boundaries
3. **PCIe Intelligence** - Concept extraction boosts relevant results by 10-30%
4. **Technical Precision** - Level filtering reduces noise for complex queries
5. **Smart Detection** - Auto-suggests PCIe mode for relevant queries
6. **Fast Queries** - 0.3-0.7s response times with rich metadata

#### Quality Metrics (100% Test Pass Rate)
- **Adaptive Chunking**: ✅ 15 chunks from transaction layer, optimal size distribution
- **RAG Engine**: ✅ 1,896 vectors built successfully with comprehensive coverage
- **Query Functionality**: ✅ All 5 test queries return relevant results (0.64+ scores)
- **Concept Boosting**: ✅ 28+ PCIe concepts correctly identified and boosted
- **Technical Filtering**: ✅ Complexity-based filtering working across all levels
- **Integration**: ✅ Both interactive and CLI modes fully functional

#### Files Added
- `src/processors/pcie_adaptive_chunker.py` - PCIe-optimized adaptive document chunker
- `src/rag/pcie_rag_engine.py` - Specialized RAG engine with PCIe intelligence
- `src/cli/commands/pcie_rag.py` - Command-line tools for PCIe mode
- `test_pcie_adaptive_rag.py` - Comprehensive test suite for PCIe mode
- `test_pcie_integration.py` - Integration test suite for interactive/CLI modes

### Major Release: Unified RAG System (2025-06-03)

#### New Unified RAG Architecture
1. **`/rag <query>`** - Single command replacing all fragmented RAG commands
2. **Auto-Testing Suite** - 10 PCIe test scenarios with quality monitoring
3. **Multi-Engine System** - Production, V3, and Standard engines with auto-fallback
4. **PCIe Intelligence** - Instant recognition of FLR/CRS compliance violations
5. **Quality Metrics** - Real-time confidence, response time, and success rate tracking

#### Quality Management Commands
1. **`/rag --test`** - Comprehensive test suite (FLR/CRS, completion timeout, LTSSM, etc.)
2. **`/rag --status`** - System health with quality metrics and performance data
3. **`/rag --engines`** - Engine status and auto-switching behavior
4. **`/rag --config`** - Unified configuration display

#### Performance Improvements
1. **10x faster** responses for known PCIe patterns (FLR/CRS scenarios)
2. **Background testing** every hour with automatic quality monitoring
3. **Smart engine switching** when quality drops below 70% threshold
4. **Instant compliance checking** with spec reference citations

#### Files Added
- `automated_rag_test_suite.py` - Comprehensive quality testing framework
- `unified_rag_integration.py` - Multi-engine RAG system with auto-testing
- `deploy_unified_rag.py` - One-command deployment and setup
- `production_rag_fix.py` - High-performance PCIe-specific RAG engine

### Previous Updates (2025-06-01)

#### Legacy Commands (Now Deprecated)
1. **`/rag_mode`** - Switch between semantic/hybrid/keyword/unified/pcie search modes
2. **`/cost`** - Show session cost and token usage with detailed breakdown
3. **`/doctor`** - Comprehensive system health check (dependencies, memory, disk)
4. **`/rag_files`** - Show which files are indexed in each RAG database
5. **`/rag_check`** - Quick check of current database readiness

## Important Notes

### Search Mode Differences
- **Semantic**: Cosine similarity (0.0-1.0), best for conceptual queries
- **Hybrid**: Normalized combination (0.0-1.0), best overall performance
- **Keyword**: BM25 scores (0.0-10.0+), best for exact term matches

### Unified RAG System (Current)
- **Multi-Engine Architecture**: Production, V3, and Standard engines with automatic fallback
- **Auto-Quality Testing**: 10 PCIe test scenarios including FLR/CRS compliance
- **Performance Monitoring**: Real-time confidence, response time, and success rate tracking
- **PCIe Intelligence**: Instant pattern recognition for common compliance violations
- **Background Testing**: Hourly quality assessments with automatic engine switching

### Legacy RAG System (Deprecated)
- Default similarity threshold: 0.1 (lowered from 0.5 for better recall)
- Supports multiple embedding models simultaneously
- Automatic BM25 index creation for hybrid/keyword search
- Each embedding model has its own vector database

### Testing Commands (Updated)
Always test new features with:
1. **Primary**: `/rag --test` - Comprehensive quality test suite
2. **Engine Testing**: Test with different engines using `--engine` flag
3. **PCIe Mode Testing**: `python test_pcie_adaptive_rag.py` - Test adaptive chunking
4. **Quality Monitoring**: Check `/rag --status` for performance metrics
5. All search modes (semantic/hybrid/keyword/unified/pcie) - comprehensive testing
6. Both verbose on/off states
7. Different embedding models
8. Tab completion functionality

### Quality Benchmarks
- **Overall Score**: >70% for production use, >80% for critical compliance
- **Response Time**: <5s for complex queries, <1s for known patterns
- **Confidence**: >60% minimum, >80% for compliance-critical responses
- **Success Rate**: >90% for automated testing scenarios