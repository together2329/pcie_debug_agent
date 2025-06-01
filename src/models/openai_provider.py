"""
OpenAI Model Provider for PCIe Debug Agent
Integrates with OpenAI API for GPT models
"""

import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, environment variables should be set manually
    pass

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


class OpenAIModelProvider(BaseModelProvider):
    """Provider for OpenAI models"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize OpenAI Model Provider
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
    
    def generate_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate completion using OpenAI API
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated text response
        """
        try:
            # Use appropriate parameter based on model
            params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", 0.1),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "presence_penalty": kwargs.get("presence_penalty", 0)
            }
            
            # gpt-4o-mini and newer models use max_completion_tokens instead of max_tokens
            if self.model_name in ["gpt-4o-mini", "o1-mini", "o3-mini", "o4-mini"]:
                params["max_completion_tokens"] = kwargs.get("max_tokens", 2000)
            else:
                params["max_tokens"] = kwargs.get("max_tokens", 2000)
            
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "name": "OpenAI Provider",
            "model": self.model_name,
            "type": "api",
            "status": "ready",
            "description": f"OpenAI {self.model_name}",
            "api_key_set": bool(self.api_key),
            "endpoint": "https://api.openai.com/v1/chat/completions"
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the OpenAI API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return {
                "status": "success",
                "model": self.model_name,
                "response_length": len(response.choices[0].message.content)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @staticmethod
    def list_available_models() -> Dict[str, str]:
        """List available OpenAI models"""
        return {
            # Mini models (2.5M tokens/day limit)
            "gpt-4o-mini": "GPT-4o-Mini - Efficient version (2.5M tokens/day)",
            "o1-mini": "O1-Mini - Reasoning model (2.5M tokens/day)",
            # Note: o4-mini, o3-mini will be available when released
            # Standard models (250K tokens/day limit)
            "gpt-4o": "GPT-4o - Latest and most capable (250K tokens/day)",
            "gpt-4": "GPT-4 - High quality reasoning (250K tokens/day)",
            "o1": "O1 - Advanced reasoning (250K tokens/day)",
            # Other models
            "gpt-4-turbo": "GPT-4 Turbo - Fast and capable",
            "gpt-3.5-turbo": "GPT-3.5 Turbo - Fast and efficient"
        }