"""
Model selection command for PCIe Debug Agent
"""

import click
from pathlib import Path
import json
from typing import Dict, List, Optional
import sys

from ..utils.output import print_success, print_error, print_info, print_warning

# Available models configuration
AVAILABLE_MODELS = {
    "gpt-4o-mini": {
        "name": "GPT-4o-Mini (Default)",
        "description": "OpenAI's efficient mini model - 2.5M tokens/day limit",
        "api": "openai",
        "context_size": 16384,
        "speed": "very-fast",
        "quality": "good",
        "local": False,
        "requires": "OPENAI_API_KEY",
        "token_limit": "2.5M/day"
    },
    "llama-3.2-3b": {
        "name": "Llama 3.2 3B Instruct",
        "description": "Fast, efficient model for quick analysis",
        "model_file": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "context_size": 8192,
        "speed": "fast",
        "quality": "good",
        "local": True
    },
    "mock-llm": {
        "name": "Mock LLM",
        "description": "Built-in model for testing and offline use",
        "model_file": "models/mock-model.gguf",
        "context_size": 8192,
        "speed": "instant",
        "quality": "good",
        "local": True,
        "always_available": True
    },
    "deepseek-r1-7b": {
        "name": "DeepSeek R1 7B (Ollama)",
        "description": "Detailed analysis with reasoning capabilities via Ollama",
        "model_name": "deepseek-r1:latest",
        "context_size": 16384,
        "speed": "medium",
        "quality": "excellent",
        "local": True,
        "provider": "ollama"
    },
    "gpt-4o": {
        "name": "GPT-4o (Latest)",
        "description": "OpenAI's latest and most capable model (requires API key)",
        "api": "openai",
        "context_size": 128000,
        "speed": "fast",
        "quality": "best",
        "local": False,
        "requires": "OPENAI_API_KEY"
    },
    "gpt-4": {
        "name": "GPT-4",
        "description": "OpenAI's most capable model (requires API key)",
        "api": "openai",
        "context_size": 128000,
        "speed": "slow",
        "quality": "best",
        "local": False,
        "requires": "OPENAI_API_KEY"
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus",
        "description": "Anthropic's most capable model (requires API key)",
        "api": "anthropic",
        "context_size": 200000,
        "speed": "slow",
        "quality": "best",
        "local": False,
        "requires": "ANTHROPIC_API_KEY"
    }
}

# Model settings file
SETTINGS_FILE = Path.home() / ".pcie_debug" / "model_settings.json"

def load_model_settings() -> Dict:
    """Load current model settings"""
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {"current_model": "gpt-4o-mini", "history": []}

def save_model_settings(settings: Dict):
    """Save model settings"""
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

@click.group()
def model():
    """Manage model selection for analysis"""
    pass

@model.command(name="list")
def list_models():
    """List all available models"""
    settings = load_model_settings()
    current_model = settings.get("current_model", "llama-3.2-3b")
    
    print_info("\n📊 Available Models:")
    print("=" * 70)
    
    for model_id, info in AVAILABLE_MODELS.items():
        is_current = model_id == current_model
        status = "✓ CURRENT" if is_current else ""
        local_tag = "🏠 LOCAL" if info["local"] else "☁️  API"
        
        # Check availability for local models
        available = True
        if info["local"] and not info.get("always_available", False):
            model_path = Path(info["model_file"])
            available = model_path.exists()
        
        availability_tag = "✅ READY" if available else "❌ NOT AVAILABLE"
        
        print(f"\n{model_id:20} {status:10} {local_tag} {availability_tag}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Context: {info['context_size']:,} tokens")
        print(f"  Speed: {info['speed']}, Quality: {info['quality']}")
        
        if not available and info["local"]:
            print(f"  📥 Model file needed: {info['model_file']}")
        elif not info["local"] and "requires" in info:
            print(f"  ⚠️  Requires: {info['requires']} environment variable")
    
    print("\n" + "=" * 70)
    print(f"Current model: {current_model}")
    print("\nUse 'pcie-debug model set <model-id>' to change the model")

@model.command(name="set")
@click.argument('model_id')
def set_model(model_id: str):
    """Set the active model for analysis"""
    if model_id not in AVAILABLE_MODELS:
        print_error(f"Unknown model: {model_id}")
        print("Available models:", ", ".join(AVAILABLE_MODELS.keys()))
        return
    
    model_info = AVAILABLE_MODELS[model_id]
    settings = load_model_settings()
    
    # Check if model is available
    if model_info["local"]:
        # Mock model is always available
        if model_info.get("always_available", False):
            pass  # Mock model doesn't need file check
        else:
            model_path = Path(model_info["model_file"])
            if not model_path.exists():
                print_error(f"Model file not found: {model_path}")
                print_info("Please download the model first using 'pcie-debug model download'")
                return
    else:
        # Check for API key
        import os
        if "requires" in model_info and not os.getenv(model_info["requires"]):
            print_error(f"API key not found: {model_info['requires']}")
            print_info(f"Please set the {model_info['requires']} environment variable")
            return
    
    # Update settings
    old_model = settings.get("current_model")
    settings["current_model"] = model_id
    
    # Add to history
    if "history" not in settings:
        settings["history"] = []
    settings["history"].append({
        "from": old_model,
        "to": model_id,
        "timestamp": str(Path.ctime(Path.cwd()))
    })
    
    save_model_settings(settings)
    
    print_success(f"✅ Model switched to: {model_info['name']}")
    print(f"   Context size: {model_info['context_size']:,} tokens")
    print(f"   Speed: {model_info['speed']}, Quality: {model_info['quality']}")

@model.command(name="info")
@click.argument('model_id', required=False)
def model_info(model_id: Optional[str] = None):
    """Show detailed information about a model"""
    if model_id is None:
        settings = load_model_settings()
        model_id = settings.get("current_model", "llama-3.2-3b")
        print_info(f"Showing info for current model: {model_id}")
    
    if model_id not in AVAILABLE_MODELS:
        print_error(f"Unknown model: {model_id}")
        return
    
    info = AVAILABLE_MODELS[model_id]
    
    print(f"\n📋 Model Information: {model_id}")
    print("=" * 60)
    print(f"Name: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Type: {'Local' if info['local'] else 'API-based'}")
    print(f"Context Window: {info['context_size']:,} tokens")
    print(f"Performance: Speed={info['speed']}, Quality={info['quality']}")
    
    if info["local"]:
        if info.get("always_available", False):
            print(f"Model Type: Built-in (always available)")
            print(f"Status: ✅ Ready to use")
        else:
            model_path = Path(info["model_file"])
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                print(f"Model File: {model_path}")
                print(f"File Size: {size_mb:.1f} MB")
                print(f"Status: ✅ Available")
            else:
                print(f"Model File: {model_path}")
                print(f"Status: ❌ Not downloaded")
    else:
        print(f"API Provider: {info['api']}")
        if "requires" in info:
            import os
            has_key = bool(os.getenv(info["requires"]))
            print(f"API Key ({info['requires']}): {'✅ Set' if has_key else '❌ Not set'}")
    
    print("\n" + "=" * 60)

@model.command(name="download")
@click.argument('model_id', required=False)
def download_model(model_id: Optional[str] = None):
    """Download a local model"""
    if model_id is None:
        # Show downloadable models
        print_info("\n📥 Downloadable Models:")
        for mid, info in AVAILABLE_MODELS.items():
            if info["local"]:
                model_path = Path(info["model_file"])
                status = "✅ Downloaded" if model_path.exists() else "❌ Not downloaded"
                print(f"  {mid}: {info['name']} - {status}")
        return
    
    if model_id not in AVAILABLE_MODELS:
        print_error(f"Unknown model: {model_id}")
        return
    
    info = AVAILABLE_MODELS[model_id]
    if not info["local"]:
        print_error(f"{model_id} is an API-based model and doesn't need downloading")
        return
    
    model_path = Path(info["model_file"])
    if model_path.exists():
        print_info(f"Model already downloaded: {model_path}")
        return
    
    # TODO: Implement actual download logic
    print_warning("Model download not yet implemented")
    print_info(f"Please manually download {info['name']} to: {model_path}")
    print_info("Download links will be added in a future update")

@model.command(name="compare")
@click.argument('model1')
@click.argument('model2')
def compare_models(model1: str, model2: str):
    """Compare two models"""
    if model1 not in AVAILABLE_MODELS or model2 not in AVAILABLE_MODELS:
        print_error("Both models must be valid model IDs")
        return
    
    info1 = AVAILABLE_MODELS[model1]
    info2 = AVAILABLE_MODELS[model2]
    
    print(f"\n📊 Model Comparison: {model1} vs {model2}")
    print("=" * 70)
    
    # Create comparison table
    attributes = [
        ("Name", "name"),
        ("Type", lambda x: "Local" if x["local"] else "API"),
        ("Context Size", lambda x: f"{x['context_size']:,} tokens"),
        ("Speed", "speed"),
        ("Quality", "quality"),
    ]
    
    for attr_name, attr_key in attributes:
        if callable(attr_key):
            val1 = attr_key(info1)
            val2 = attr_key(info2)
        else:
            val1 = info1.get(attr_key, "N/A")
            val2 = info2.get(attr_key, "N/A")
        
        print(f"{attr_name:15} | {str(val1):25} | {str(val2):25}")
    
    print("=" * 70)
    
    # Recommendations
    print("\n💡 Recommendations:")
    if info1["speed"] == "fast" and info2["speed"] != "fast":
        print(f"  - Use {model1} for quick analysis and debugging")
    elif info2["speed"] == "fast" and info1["speed"] != "fast":
        print(f"  - Use {model2} for quick analysis and debugging")
    
    if info1["quality"] == "best" and info2["quality"] != "best":
        print(f"  - Use {model1} for complex issues requiring deep analysis")
    elif info2["quality"] == "best" and info1["quality"] != "best":
        print(f"  - Use {model2} for complex issues requiring deep analysis")
    
    if info1["local"] and not info2["local"]:
        print(f"  - Use {model1} for offline work or sensitive data")
    elif info2["local"] and not info1["local"]:
        print(f"  - Use {model2} for offline work or sensitive data")

if __name__ == "__main__":
    model()