"""
Ollama Model Provider for PCIe Debug Agent
Integrates with local Ollama installation
"""

import json
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class BaseModelProvider(ABC):
    """Base class for model providers"""
    
    @abstractmethod
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """Generate completion for the given prompt"""
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """Get provider information"""
        pass


class OllamaModelProvider(BaseModelProvider):
    """Provider for Ollama models"""
    
    def __init__(self, model_name: str = "deepseek-r1:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama Model Provider
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        
        # Check if Ollama is running and model is available
        self._check_availability()
    
    def _check_availability(self):
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama server not responding")
            
            # Check if specific model is available
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if self.model_name not in model_names:
                raise ValueError(f"Model {self.model_name} not found. Available: {model_names}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama server not running. Start with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama availability check failed: {e}")
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate completion using Ollama
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        try:
            # Prepare request data
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "num_predict": kwargs.get("max_tokens", 300),  # Very small for memory constraints
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": 10,  # Further limit vocabulary 
                    "repeat_penalty": 1.1,
                    "num_ctx": 2048,  # Reduce context window to save memory
                    "num_gpu": 0,  # Force CPU-only to avoid GPU memory issues
                    "stop": kwargs.get("stop_sequences", [])
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.api_url}/generate",
                json=data,
                timeout=300  # 5 minutes for large models like DeepSeek R1
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return result.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out (DeepSeek R1 is slow - try shorter prompts or use llama-3.2-3b for faster responses)")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Lost connection to Ollama server")
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        try:
            # Get model info from Ollama
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            models = response.json().get("models", [])
            
            model_info = None
            for model in models:
                if model["name"] == self.model_name:
                    model_info = model
                    break
            
            return {
                "name": "Ollama Provider",
                "model": self.model_name,
                "type": "ollama",
                "status": "ready",
                "description": f"Ollama {self.model_name}",
                "size": model_info.get("size", "unknown") if model_info else "unknown",
                "modified": model_info.get("modified", "unknown") if model_info else "unknown",
                "base_url": self.base_url
            }
        except Exception:
            return {
                "name": "Ollama Provider",
                "model": self.model_name,
                "type": "ollama",
                "status": "error",
                "description": f"Ollama {self.model_name}",
                "base_url": self.base_url
            }
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available Ollama models"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            models = response.json().get("models", [])
            
            return {
                model["name"]: {
                    "name": model["name"],
                    "size": model.get("size", "unknown"),
                    "modified": model.get("modified", "unknown"),
                    "digest": model.get("digest", "unknown")
                }
                for model in models
            }
        except Exception as e:
            return {"error": f"Failed to list models: {e}"}
    
    @staticmethod
    def check_ollama_status() -> Dict[str, Any]:
        """Check Ollama server status"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return {
                "running": response.status_code == 200,
                "models_count": len(response.json().get("models", [])),
                "url": "http://localhost:11434"
            }
        except Exception:
            return {
                "running": False,
                "error": "Ollama server not accessible",
                "url": "http://localhost:11434"
            }