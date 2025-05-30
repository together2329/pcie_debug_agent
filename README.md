# PCIe Debug Agent 🔍

[![Test Suite](https://github.com/yourusername/pcie_debug_agent/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/pcie_debug_agent/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/yourusername/pcie_debug_agent/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/pcie_debug_agent)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/yourusername/pcie-debug-agent)

An AI-powered PCIe log analysis tool using Retrieval-Augmented Generation (RAG) technology. Quickly analyze complex PCIe debug logs, identify errors, and get intelligent insights using state-of-the-art language models.

## 🚀 Features

- **🤖 AI-Powered Analysis**: Leverage GPT-4, Claude, and other LLMs to understand complex PCIe errors
- **🔍 Semantic Search**: Find relevant log entries using natural language queries
- **📊 Smart Indexing**: Efficiently index and retrieve information from large log files
- **💻 CLI & Web Interface**: Use via command line or interactive web dashboard
- **📈 Performance Metrics**: Track analysis performance and accuracy
- **🔄 Real-time Processing**: Analyze logs as they're generated
- **📄 Comprehensive Reports**: Generate detailed HTML, Markdown, or JSON reports
- **🐳 Docker Support**: Easy deployment with Docker containers

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [CLI Usage](#cli-usage)
- [Web Interface](#web-interface)
- [Configuration](#configuration)
- [Development](#development)
- [Testing](#testing)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🏃 Quick Start

```bash
# Install via pip
pip install pcie-debug-agent

# Initialize configuration
pcie-debug config init

# Index your log files
pcie-debug index build /path/to/logs

# Analyze logs
pcie-debug analyze --query "What PCIe errors occurred?"

# Start web interface
pcie-debug web
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Optional: Docker for containerized deployment

### Install from PyPI

```bash
pip install pcie-debug-agent
```

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/pcie_debug_agent.git
cd pcie_debug_agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Docker Installation

```bash
# Pull the image
docker pull yourusername/pcie-debug-agent:latest

# Run the container
docker run -p 8501:8501 -v /path/to/logs:/app/logs pcie-debug-agent
```

## 🖥️ CLI Usage

### Configuration

```bash
# Initialize configuration interactively
pcie-debug config init

# Validate configuration
pcie-debug config validate

# Show current configuration
pcie-debug config show

# Set specific values
pcie-debug config set llm.model gpt-4
pcie-debug config set chunk_size 1000 --type int
```

### Indexing

```bash
# Build index from log files
pcie-debug index build /path/to/logs --recursive

# Update existing index
pcie-debug index update /new/logs

# Show index statistics
pcie-debug index stats

# Optimize index for better performance
pcie-debug index optimize
```

### Analysis

```bash
# Analyze with a query
pcie-debug analyze --query "Find all PCIe timeout errors"

# Analyze specific log file
pcie-debug analyze /path/to/specific.log --query "What went wrong?"

# Use different output formats
pcie-debug analyze -q "Error analysis" --output json > results.json

# Set confidence threshold
pcie-debug analyze -q "Critical errors" --confidence 0.8
```

### Search

```bash
# Semantic search
pcie-debug search "link training failed"

# Search with filters
pcie-debug search "timeout" --filter severity=ERROR --filter source=pcie.log

# Limit results
pcie-debug search "device error" --limit 5 --similarity 0.8
```

### Report Generation

```bash
# Generate HTML report
pcie-debug report --query "Analyze all errors" --output report.html

# Generate report with multiple queries
pcie-debug report -q "Timeouts" -q "Link failures" -q "Recovery actions"

# Include statistics and timeline
pcie-debug report --include-stats --include-timeline --format markdown
```

### Testing

```bash
# Test connectivity to APIs
pcie-debug test connectivity

# Run performance benchmarks
pcie-debug test performance

# Run test suite
pcie-debug test suite --unit --integration --coverage
```

## 🌐 Web Interface

### Starting the Web Interface

```bash
# Using CLI
pcie-debug web

# Using Streamlit directly
streamlit run src/ui/app.py

# Using Docker
docker-compose up
```

### Features

- **Interactive Chat**: Ask questions about your logs in natural language
- **Semantic Search**: Find relevant log entries with advanced search
- **Visual Analytics**: View error distributions and patterns
- **Real-time Analysis**: Process logs as they arrive
- **Report Generation**: Create and download analysis reports

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
EMBEDDING_API_KEY=your_embedding_key

# Optional: Custom endpoints
LLM_API_BASE_URL=https://your-custom-endpoint.com
EMBEDDING_API_BASE_URL=https://your-embedding-endpoint.com

# Application settings
APP_ENV=production
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_SIZE_MB=1000
```

### Configuration File

Edit `configs/settings.yaml`:

```yaml
app_name: PCIe Debug Agent
version: 1.0.0

embedding:
  provider: openai
  model: text-embedding-3-small
  
llm:
  provider: openai
  model: gpt-4
  temperature: 0.1
  max_tokens: 2000

rag:
  chunk_size: 500
  chunk_overlap: 50
  context_window: 3
  min_similarity: 0.7

vector_store:
  path: data/vectorstore
  index_type: Flat
  dimension: 1536
```

## 🔧 Development

### Project Structure

```
pcie_debug_agent/
├── src/
│   ├── cli/              # CLI implementation
│   │   ├── commands/     # CLI commands
│   │   └── utils/        # CLI utilities
│   ├── rag/              # RAG engine components
│   ├── collectors/       # Log collectors
│   ├── processors/       # Document processors
│   ├── ui/               # Web interface
│   └── config/           # Configuration management
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── performance/     # Performance tests
├── docs/                # Documentation
├── docker/              # Docker files
└── scripts/             # Utility scripts
```

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run linters
black src tests
flake8 src tests
mypy src

# Run tests
pytest tests/unit -v
pytest tests/integration -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Adding New Features

1. Create a feature branch
2. Implement your feature with tests
3. Run the test suite
4. Submit a pull request

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit
pytest tests/integration
pytest tests/performance

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run with markers
pytest -m "not slow"
pytest -m "requires_api"
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical operations
- **E2E Tests**: Test complete workflows

## 📚 API Reference

### Python API

```python
from pcie_debug_agent import PCIeDebugAgent

# Initialize agent
agent = PCIeDebugAgent(config_path="configs/settings.yaml")

# Index logs
agent.index_directory("/path/to/logs")

# Analyze
result = agent.analyze("What errors occurred?")
print(result.answer)
print(f"Confidence: {result.confidence}")

# Search
results = agent.search("timeout", limit=10)
for r in results:
    print(f"{r.score}: {r.content}")
```

### REST API (Coming Soon)

```bash
# Start API server
pcie-debug api serve

# Query endpoint
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "What errors occurred?"}'
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all public functions
- Add unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Uses [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- Powered by [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), and other LLM providers
- CLI built with [Click](https://click.palletsprojects.com/)
- Testing with [pytest](https://pytest.org/)

## 📞 Support

- 📧 Email: support@pcie-debug.example.com
- 💬 Discord: [Join our community](https://discord.gg/pcie-debug)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/pcie_debug_agent/issues)
- 📖 Docs: [Documentation](https://pcie-debug-agent.readthedocs.io/)

---

Made with ❤️ by the PCIe Debug Team