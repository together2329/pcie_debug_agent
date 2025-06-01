"""
Model selector for dynamic model switching
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any, List
from abc import ABC, abstractmethod

from src.models.local_llm_provider import LocalLLMProvider
try:
    from src.models.mock_llm_provider import MockLLMProvider
except ImportError:
    MockLLMProvider = None

try:
    from src.models.ollama_provider import OllamaModelProvider
except ImportError:
    OllamaModelProvider = None

try:
    from src.models.openai_provider import OpenAIModelProvider
except ImportError:
    OpenAIModelProvider = None


class BaseModelProvider(ABC):
    """Base class for model providers"""
    
    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion from prompt"""
        pass
    
    def generate_completion_stream(self, prompt: str, **kwargs):
        """Generate streaming completion from prompt"""
        # Default implementation: fallback to non-streaming
        result = self.generate_completion(prompt, **kwargs)
        yield result
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class LocalModelProvider(BaseModelProvider):
    """Provider for local GGUF models"""
    
    def __init__(self, model_path: str, context_size: int = 8192):
        self.model_path = model_path
        self.context_size = context_size
        # Use mock provider for testing
        if "mock" in model_path:
            try:
                from src.models.mock_llm_provider import MockLLMProvider as MockProvider
                self.provider = MockProvider(
                    model_path=model_path,
                    n_ctx=context_size,
                    verbose=False
                )
            except ImportError:
                self.provider = LocalLLMProvider(
                    model_path=model_path,
                    n_ctx=context_size,
                    verbose=False
                )
        else:
            self.provider = LocalLLMProvider(
                model_path=model_path,
                n_ctx=context_size,
                verbose=False
            )
        self.provider.load_model()
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        return self.provider.generate_completion(prompt, **kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "local",
            "model_path": self.model_path,
            "context_size": self.context_size
        }


class OpenAIModelProvider(BaseModelProvider):
    """Provider for OpenAI API models"""
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def generate_completion_stream(self, prompt: str, **kwargs):
        """Generate streaming completion"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1000),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise RuntimeError(f"OpenAI API streaming error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "openai",
            "model": self.model_name
        }


class AnthropicModelProvider(BaseModelProvider):
    """Provider for Anthropic API models"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        return {
            "type": "anthropic",
            "model": self.model_name
        }


class ModelSelector:
    """Manages model selection and switching"""
    
    # Model configurations
    MODEL_CONFIGS = {
        "gpt-4o-mini": {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "context_size": 16384
        },
        "mock-llm": {
            "provider": "local",
            "model_path": "models/mock-model.gguf",
            "context_size": 8192
        },
        "llama-3.2-3b": {
            "provider": "local",
            "model_path": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "context_size": 8192
        },
        "deepseek-r1-7b": {
            "provider": "ollama",
            "model_name": "deepseek-r1:latest",
            "context_size": 16384
        },
        "gpt-4o": {
            "provider": "openai",
            "model_name": "gpt-4o"
        },
        "gpt-4": {
            "provider": "openai",
            "model_name": "gpt-4"
        },
        "claude-3-opus": {
            "provider": "anthropic",
            "model_name": "claude-3-opus-20240229"
        }
    }
    
    def __init__(self):
        self.settings_file = Path.home() / ".pcie_debug" / "model_settings.json"
        self.current_model_id = None
        self.current_provider = None
        self._load_settings()
    
    def _load_settings(self):
        """Load model settings"""
        if self.settings_file.exists():
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.current_model_id = settings.get("current_model", "mock-llm")
        else:
            # Default to gpt-4o-mini if API key is available, otherwise mock-llm
            import os
            if os.getenv("OPENAI_API_KEY"):
                self.current_model_id = "gpt-4o-mini"
            else:
                self.current_model_id = "mock-llm"
    
    def get_current_model(self) -> str:
        """Get current model ID"""
        return self.current_model_id
    
    def get_provider(self, model_id: Optional[str] = None) -> BaseModelProvider:
        """Get model provider instance"""
        if model_id is None:
            model_id = self.current_model_id
        
        if model_id not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_id}")
        
        config = self.MODEL_CONFIGS[model_id]
        
        if config["provider"] == "local":
            # Check if model file exists
            model_path = Path(config["model_path"])
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}\nPlease download the model first.")
            
            # Extract model name from path for LocalLLMProvider
            model_name = config.get("model_name", model_id)
            if model_id == "mock-llm":
                return MockLLMProvider()
            else:
                return LocalLLMProvider(
                    model_name=model_name,
                    models_dir="models",
                    n_ctx=config.get("context_size", 8192)
                )
        elif config["provider"] == "openai":
            if OpenAIModelProvider is None:
                raise ImportError("OpenAI provider not available")
            return OpenAIModelProvider(model_name=config["model_name"])
        elif config["provider"] == "anthropic":
            return AnthropicModelProvider(model_name=config["model_name"])
        elif config["provider"] == "ollama":
            if OllamaModelProvider is None:
                raise ImportError("Ollama provider not available")
            return OllamaModelProvider(model_name=config["model_name"])
        else:
            raise ValueError(f"Unknown provider: {config['provider']}")
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model"""
        if model_id not in self.MODEL_CONFIGS:
            return False
        
        # Test if model is accessible
        try:
            provider = self.get_provider(model_id)
            # Simple test
            provider.get_info()
            
            # Save settings
            self.current_model_id = model_id
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump({"current_model": model_id}, f)
            
            return True
        except Exception as e:
            print(f"Failed to switch to {model_id}: {e}")
            return False
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion using current model with fallback"""
        try:
            provider = self.get_provider()
            return provider.generate_completion(prompt, **kwargs)
        except (FileNotFoundError, ValueError) as e:
            # If current model fails, try to fall back to an available model
            if "Model file not found" in str(e) or "API key not found" in str(e):
                fallback_models = self._get_available_models()
                
                if fallback_models:
                    print(f"⚠️ Current model unavailable, falling back to {fallback_models[0]}")
                    if self.switch_model(fallback_models[0]):
                        provider = self.get_provider()
                        return provider.generate_completion(prompt, **kwargs)
                
                # If no fallbacks work, re-raise original error
                raise e
            else:
                raise e
    
    def generate_completion_stream(self, prompt: str, **kwargs):
        """Generate streaming completion using current model"""
        try:
            provider = self.get_provider()
            yield from provider.generate_completion_stream(prompt, **kwargs)
        except (FileNotFoundError, ValueError) as e:
            # If current model fails, try to fall back to an available model
            if "Model file not found" in str(e) or "API key not found" in str(e):
                fallback_models = self._get_available_models()
                
                if fallback_models:
                    print(f"⚠️ Current model unavailable, falling back to {fallback_models[0]}")
                    if self.switch_model(fallback_models[0]):
                        provider = self.get_provider()
                        yield from provider.generate_completion_stream(prompt, **kwargs)
                        return
                
                # If no fallbacks work, re-raise original error
                raise e
            else:
                raise e
    
    def _get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        available = []
        
        for model_id, config in self.MODEL_CONFIGS.items():
            try:
                if config["provider"] == "local":
                    model_path = Path(config["model_path"])
                    if model_path.exists():
                        available.append(model_id)
                else:
                    # Check if API key is available
                    import os
                    if "requires" in config:
                        api_key_var = config["requires"]
                        if os.getenv(api_key_var):
                            available.append(model_id)
            except:
                continue
        
        return available


# Global instance
_model_selector = None

def get_model_selector() -> ModelSelector:
    """Get global model selector instance"""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector()
    return _model_selector