#!/usr/bin/env python3
"""
Script to check environment configuration for PCIe Debug Agent
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import Settings

def check_environment():
    """Check environment variables and configuration"""
    print("🔍 Checking PCIe Debug Agent Environment Configuration\n")
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found (using system environment variables)")
    
    print("\n📋 Environment Variables:")
    print("-" * 50)
    
    # Check API keys
    env_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "EMBEDDING_API_KEY": os.getenv("EMBEDDING_API_KEY"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY"),
        "EMBEDDING_API_BASE_URL": os.getenv("EMBEDDING_API_BASE_URL"),
        "LLM_API_BASE_URL": os.getenv("LLM_API_BASE_URL"),
    }
    
    for var, value in env_vars.items():
        if value:
            if "KEY" in var:
                # Mask API keys
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✅ {var}: {masked}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
    
    print("\n🔧 Loading Settings:")
    print("-" * 50)
    
    try:
        # Try to load settings
        config_path = Path("configs/settings.yaml")
        if config_path.exists():
            print(f"✅ Config file found: {config_path}")
            settings = Settings.load_settings(str(config_path))
        else:
            print("⚠️  Config file not found, using defaults with env vars")
            settings = Settings.load_settings(None)
        
        print("\n📊 Resolved Configuration:")
        print("-" * 50)
        print(f"Embedding Provider: {settings.embedding_provider}")
        print(f"Embedding Model: {settings.embedding_model}")
        print(f"Embedding API Key: {'✅ Set' if settings.embedding_config.api_key else '❌ Not set'}")
        print(f"Embedding Base URL: {settings.embedding_config.api_base_url or 'Default'}")
        print()
        print(f"LLM Provider: {settings.llm_provider}")
        print(f"LLM Model: {settings.llm_model}")
        print(f"LLM API Key: {'✅ Set' if settings.llm_config.api_key else '❌ Not set'}")
        print(f"LLM Base URL: {settings.llm_config.api_base_url or 'Default'}")
        
        # Validate settings
        print("\n🔍 Validating Configuration:")
        print("-" * 50)
        try:
            settings.validate()
            print("✅ Configuration is valid!")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading settings: {e}")
        return False
    
    print("\n✨ Environment check complete!")
    return True

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)