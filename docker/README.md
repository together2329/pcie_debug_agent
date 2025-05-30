# Docker Configuration for PCIe Debug Agent

## Environment Variables

The application reads configuration from environment variables, which take precedence over values in the config file.

### Required Environment Variables

Set these in your `.env` file:

```bash
# API Keys (choose based on your provider)
OPENAI_API_KEY=your_openai_api_key      # For OpenAI models
ANTHROPIC_API_KEY=your_anthropic_key    # For Claude models

# Or use generic keys
EMBEDDING_API_KEY=your_embedding_key    # Override for embedding service
LLM_API_KEY=your_llm_key               # Override for LLM service
```

### Optional Environment Variables

```bash
# Custom API endpoints
EMBEDDING_API_BASE_URL=https://custom-embedding-api.com
LLM_API_BASE_URL=https://custom-llm-api.com

# Application settings
APP_ENV=production
LOG_LEVEL=INFO
MAX_WORKERS=4
CACHE_SIZE_MB=1000
```

## Quick Start

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your API keys:**
   ```bash
   # At minimum, add one of these:
   OPENAI_API_KEY=sk-...
   # or
   ANTHROPIC_API_KEY=sk-ant-...
   ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up
   ```

## How Environment Variables Work

1. **Priority Order:**
   - Environment variables (highest priority)
   - Config file values
   - Default values (lowest priority)

2. **Provider-Specific Keys:**
   - If using OpenAI for both embedding and LLM, just set `OPENAI_API_KEY`
   - If using different providers, set `EMBEDDING_API_KEY` and `LLM_API_KEY`

3. **Base URLs:**
   - Only set if using custom endpoints
   - Leave empty to use default provider endpoints

## Examples

### Example 1: Using OpenAI for Everything
```bash
OPENAI_API_KEY=sk-...
```

### Example 2: Mixed Providers
```bash
EMBEDDING_API_KEY=your-embedding-key
LLM_API_KEY=sk-ant-...
EMBEDDING_API_BASE_URL=https://custom-embed.com
```

### Example 3: Custom Deployment
```bash
LLM_API_KEY=custom-key
LLM_API_BASE_URL=https://your-llm-server.com/v1
EMBEDDING_API_KEY=custom-key
EMBEDDING_API_BASE_URL=https://your-embed-server.com/v1
```