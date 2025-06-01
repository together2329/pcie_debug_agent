"""
Embedding model selector for RAG functionality
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers"""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        pass


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Local sentence-transformers provider"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the model"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            # Get dimension from first encoding
            test_embedding = self._model.encode(["test"])
            self._dimension = test_embedding.shape[1]
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        self._load_model()
        embeddings = self._model.encode(texts)
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        return {
            "provider": "sentence-transformers",
            "model": self.model_name,
            "dimension": self.get_dimension(),
            "type": "local",
            "cost": "free",
            "speed": "fast" if "MiniLM" in self.model_name else "medium"
        }


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = []
            for item in response.data:
                embeddings.append(item.embedding)
            
            return np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding error: {e}")
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimensions.get(self.model_name, 1536)
    
    def get_info(self) -> Dict[str, Any]:
        """Get provider information"""
        costs = {
            "text-embedding-3-small": "$0.020 per 1M tokens",
            "text-embedding-3-large": "$0.130 per 1M tokens", 
            "text-embedding-ada-002": "$0.100 per 1M tokens"
        }
        
        return {
            "provider": "openai",
            "model": self.model_name,
            "dimension": self.get_dimension(),
            "type": "api",
            "cost": costs.get(self.model_name, "unknown"),
            "speed": "fast" if "small" in self.model_name else "medium"
        }


class EmbeddingSelector:
    """Manages embedding model selection and switching"""
    
    def __init__(self):
        self.settings_file = Path.home() / ".pcie_debug" / "embedding_settings.json"
        self.settings_file.parent.mkdir(exist_ok=True)
        
        # Available models
        self.available_models = {
            # Local models (sentence-transformers)
            "all-MiniLM-L6-v2": {
                "provider": SentenceTransformerProvider,
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Fast local model (384 dim)",
                "dimension": 384,
                "speed": "fast",
                "cost": "free"
            },
            "all-mpnet-base-v2": {
                "provider": SentenceTransformerProvider,
                "model_name": "sentence-transformers/all-mpnet-base-v2", 
                "description": "High quality local model (768 dim)",
                "dimension": 768,
                "speed": "medium",
                "cost": "free"
            },
            "multi-qa-MiniLM-L6-cos-v1": {
                "provider": SentenceTransformerProvider,
                "model_name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "description": "Question-answering optimized (384 dim)",
                "dimension": 384,
                "speed": "fast", 
                "cost": "free"
            },
            # OpenAI models
            "text-embedding-3-small": {
                "provider": OpenAIEmbeddingProvider,
                "model_name": "text-embedding-3-small",
                "description": "OpenAI small embedding model (1536 dim)",
                "dimension": 1536,
                "speed": "fast",
                "cost": "$0.020 per 1M tokens"
            },
            "text-embedding-3-large": {
                "provider": OpenAIEmbeddingProvider,
                "model_name": "text-embedding-3-large",
                "description": "OpenAI large embedding model (3072 dim)",
                "dimension": 3072,
                "speed": "medium",
                "cost": "$0.130 per 1M tokens"
            },
            "text-embedding-ada-002": {
                "provider": OpenAIEmbeddingProvider,
                "model_name": "text-embedding-ada-002",
                "description": "OpenAI legacy embedding model (1536 dim)",
                "dimension": 1536,
                "speed": "medium", 
                "cost": "$0.100 per 1M tokens"
            }
        }
        
        self.current_model = None
        self.current_provider = None
        self._load_settings()
    
    def _load_settings(self):
        """Load embedding model settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    model_id = settings.get("current_model", self._get_default_model())
            else:
                model_id = self._get_default_model()
            
            self.switch_model(model_id)
        except Exception:
            # Fallback to default
            self.switch_model(self._get_default_model())
    
    def _get_default_model(self) -> str:
        """Get the default embedding model with fallback logic"""
        # Try OpenAI text-embedding-3-small first if API key available
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key.strip():
            try:
                # Test if we can create the provider
                provider = OpenAIEmbeddingProvider("text-embedding-3-small")
                # Try a simple test to ensure the API key works
                test_result = provider.get_info()
                if test_result.get("api_key_set", False):
                    return "text-embedding-3-small"
            except Exception:
                pass  # Fall back to local model
        
        # Fallback to local model
        return "all-MiniLM-L6-v2"
    
    def _save_settings(self):
        """Save embedding model settings to file"""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump({"current_model": self.current_model}, f)
        except Exception:
            pass  # Ignore save errors
    
    def list_models(self) -> Dict[str, Dict]:
        """List all available embedding models"""
        return self.available_models
    
    def get_current_model(self) -> str:
        """Get current embedding model ID"""
        return self.current_model
    
    def get_current_provider(self) -> BaseEmbeddingProvider:
        """Get current embedding provider instance"""
        return self.current_provider
    
    def switch_model(self, model_id: str) -> bool:
        """Switch to a different embedding model"""
        if model_id not in self.available_models:
            return False
        
        try:
            model_config = self.available_models[model_id]
            provider_class = model_config["provider"]
            model_name = model_config["model_name"]
            
            # Create provider instance
            self.current_provider = provider_class(model_name)
            self.current_model = model_id
            
            # Save settings
            self._save_settings()
            return True
        except Exception as e:
            print(f"Failed to switch to {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model"""
        if model_id is None:
            model_id = self.current_model
        
        if model_id not in self.available_models:
            return {}
        
        config = self.available_models[model_id].copy()
        
        # Add runtime info if this is the current model
        if model_id == self.current_model and self.current_provider:
            try:
                runtime_info = self.current_provider.get_info()
                config.update(runtime_info)
            except:
                pass
        
        return config
    
    def is_available(self, model_id: str) -> bool:
        """Check if a model is available"""
        if model_id not in self.available_models:
            return False
        
        config = self.available_models[model_id]
        
        # Check if OpenAI models have API key
        if config["provider"] == OpenAIEmbeddingProvider:
            return bool(os.getenv("OPENAI_API_KEY"))
        
        return True


# Global instance
_embedding_selector = None

def get_embedding_selector() -> EmbeddingSelector:
    """Get the global embedding selector instance"""
    global _embedding_selector
    if _embedding_selector is None:
        _embedding_selector = EmbeddingSelector()
    return _embedding_selector