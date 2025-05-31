"""
Model Manager for handling different embedding and LLM models
"""

import os
from typing import Dict, Any, List, Optional, Union
import logging
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information"""
    name: str
    provider: str
    description: str
    dimension: Optional[int] = None
    context_window: Optional[int] = None
    cost_per_1k: Optional[float] = None
    memory_usage: Optional[str] = None

class EmbeddingModelManager:
    """Manages different embedding models"""
    
    SUPPORTED_MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": ModelInfo(
            name="MiniLM-L6-v2",
            provider="sentence-transformers",
            description="Fast and efficient for general use",
            dimension=384,
            memory_usage="~80MB"
        ),
        "sentence-transformers/all-mpnet-base-v2": ModelInfo(
            name="MPNet-base-v2",
            provider="sentence-transformers",
            description="Better quality with moderate speed",
            dimension=768,
            memory_usage="~420MB"
        ),
        "sentence-transformers/all-MiniLM-L12-v2": ModelInfo(
            name="MiniLM-L12-v2",
            provider="sentence-transformers",
            description="Higher quality than L6, still fast",
            dimension=384,
            memory_usage="~120MB"
        ),
        "sentence-transformers/multi-qa-mpnet-base-dot-v1": ModelInfo(
            name="Multi-QA MPNet",
            provider="sentence-transformers",
            description="Optimized for question-answering tasks",
            dimension=768,
            memory_usage="~420MB"
        ),
        "sentence-transformers/all-roberta-large-v1": ModelInfo(
            name="RoBERTa Large",
            provider="sentence-transformers",
            description="Highest quality, slower performance",
            dimension=1024,
            memory_usage="~1.4GB"
        )
    }
    
    def __init__(self):
        self._models = {}
        self._current_model = None
        
    def load_model(self, model_name: str, device: Optional[str] = None) -> SentenceTransformer:
        """Load an embedding model"""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
            
        if model_name in self._models:
            logger.info(f"Using cached model: {model_name}")
            self._current_model = model_name
            return self._models[model_name]
            
        logger.info(f"Loading embedding model: {model_name}")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        try:
            model = SentenceTransformer(model_name, device=device)
            self._models[model_name] = model
            self._current_model = model_name
            
            logger.info(f"Successfully loaded {model_name} on {device}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get information about a model"""
        return self.SUPPORTED_MODELS.get(model_name)
        
    def get_current_model(self) -> Optional[SentenceTransformer]:
        """Get the currently loaded model"""
        if self._current_model:
            return self._models.get(self._current_model)
        return None
        
    def clear_cache(self):
        """Clear all cached models"""
        self._models.clear()
        self._current_model = None
        torch.cuda.empty_cache()
        logger.info("Model cache cleared")

class LLMModelManager:
    """Manages different LLM models"""
    
    SUPPORTED_MODELS = {
        "openai": {
            "gpt-4": ModelInfo(
                name="GPT-4",
                provider="openai",
                description="Most capable, best for complex analysis",
                context_window=8192,
                cost_per_1k=0.03
            ),
            "gpt-4-turbo": ModelInfo(
                name="GPT-4 Turbo",
                provider="openai",
                description="Faster and cheaper than GPT-4",
                context_window=128000,
                cost_per_1k=0.01
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="GPT-3.5 Turbo",
                provider="openai",
                description="Fast and cost-effective",
                context_window=16385,
                cost_per_1k=0.001
            ),
            "gpt-3.5-turbo-16k": ModelInfo(
                name="GPT-3.5 Turbo 16K",
                provider="openai",
                description="Extended context window",
                context_window=16385,
                cost_per_1k=0.003
            )
        },
        "anthropic": {
            "claude-3-opus": ModelInfo(
                name="Claude 3 Opus",
                provider="anthropic",
                description="Most capable Claude model",
                context_window=200000,
                cost_per_1k=0.015
            ),
            "claude-3-sonnet": ModelInfo(
                name="Claude 3 Sonnet",
                provider="anthropic",
                description="Balanced performance and cost",
                context_window=200000,
                cost_per_1k=0.003
            ),
            "claude-3-haiku": ModelInfo(
                name="Claude 3 Haiku",
                provider="anthropic",
                description="Fast and efficient",
                context_window=200000,
                cost_per_1k=0.00025
            )
        },
        "local": {
            "llama-3.2-3b-instruct": ModelInfo(
                name="Llama 3.2 3B Instruct",
                provider="local",
                description="Optimized for M1 Mac with Q4_K_M quantization",
                context_window=131072,
                memory_usage="~1.8GB"
            ),
            "llama2-7b": ModelInfo(
                name="Llama 2 7B",
                provider="local",
                description="Small local model",
                context_window=4096,
                memory_usage="~13GB"
            ),
            "llama2-13b": ModelInfo(
                name="Llama 2 13B",
                provider="local",
                description="Medium local model",
                context_window=4096,
                memory_usage="~26GB"
            ),
            "mistral-7b": ModelInfo(
                name="Mistral 7B",
                provider="local",
                description="Efficient local model",
                context_window=8192,
                memory_usage="~13GB"
            ),
            "codellama-7b": ModelInfo(
                name="Code Llama 7B",
                provider="local",
                description="Optimized for code analysis",
                context_window=16384,
                memory_usage="~13GB"
            )
,
        "hybrid": {
            "llama-3.2-3b-quick": ModelInfo(
                name="Llama 3.2 3B (Quick Analysis)",
                provider="hybrid",
                description="Fast PCIe error analysis",
                context_window=16384,
                memory_usage="~1.8GB"
            ),
            "deepseek-r1-detailed": ModelInfo(
                name="DeepSeek R1 (Detailed Analysis)",
                provider="hybrid",
                description="Comprehensive PCIe analysis",
                context_window=32768,
                memory_usage="~5.2GB"
            ),
            "auto-hybrid": ModelInfo(
                name="Auto Hybrid Selection",
                provider="hybrid",
                description="Automatically selects best model",
                context_window=16384,
                memory_usage="~1.8-5.2GB"
            )
        }        }
    }
    
    def __init__(self):
        self._clients = {}
        self._current_provider = None
        self._current_model = None
        
    def initialize_provider(self, provider: str, api_key: Optional[str] = None, **kwargs):
        """Initialize a provider with credentials"""
        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required")
            from openai import OpenAI
            self._clients["openai"] = OpenAI(api_key=api_key)
            
        elif provider == "anthropic":
            if not api_key:
                raise ValueError("Anthropic API key required")
            self._clients["anthropic"] = Anthropic(api_key=api_key)
            
        elif provider == "local":
            # Initialize local model handler
            try:
                from src.models.local_llm_provider import LocalLLMProvider
                self._clients["local"] = LocalLLMProvider(**kwargs)
            except ImportError as e:
                logger.warning(f"Local LLM not available: {e}")
                self._clients["local"] = None
                
        
        elif provider == "hybrid":
            # Initialize hybrid model handler
            try:
                from src.models.hybrid_llm_provider import HybridLLMProvider
                self._clients["hybrid"] = HybridLLMProvider(**kwargs)
            except ImportError as e:
                logger.warning(f"Hybrid LLM not available: {e}")
                self._clients["hybrid"] = None
                
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        self._current_provider = provider
        logger.info(f"Initialized {provider} provider")
        
    def generate_completion(self,
                          provider: str,
                          model: str,
                          prompt: str,
                          temperature: float = 0.1,
                          max_tokens: int = 2000,
                          **kwargs) -> str:
        """Generate completion using specified model"""
        
        if provider not in self._clients:
            raise ValueError(f"Provider {provider} not initialized")
            
        try:
            if provider == "openai":
                client = self._clients["openai"]
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a UVM verification expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.choices[0].message.content
                
            elif provider == "anthropic":
                client = self._clients["anthropic"]
                response = client.messages.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response.content[0].text
                
            elif provider == "local":
                if self._clients["local"] is None:
                    raise RuntimeError("Local models not available")
                    
                # Use LocalLLMProvider
                local_provider = self._clients["local"]
                
                # Ensure the model matches what LocalLLMProvider expects
                if model not in local_provider.MODELS:
                    logger.warning(f"Model {model} not supported by LocalLLMProvider, using default")
                    
                return local_provider.generate_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
            
            elif provider == "hybrid":
                if self._clients["hybrid"] is None:
                    raise RuntimeError("Hybrid models not available")
                    
                # Use HybridLLMProvider
                hybrid_provider = self._clients["hybrid"]
                
                # Determine analysis type from model name or kwargs
                analysis_type = "auto"
                if "quick" in model.lower():
                    analysis_type = "quick"
                elif "detailed" in model.lower():
                    analysis_type = "detailed"
                
                from src.models.hybrid_llm_provider import AnalysisRequest
                request = AnalysisRequest(
                    query=prompt,
                    analysis_type=analysis_type,
                    max_response_time=kwargs.get("timeout", 30.0)
                )
                
                response = hybrid_provider.analyze_pcie_error(request)
                return response.response if response.response else "Analysis failed"
                
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error generating completion with {provider}/{model}: {e}")
            raise
            
    def get_model_info(self, provider: str, model: str) -> ModelInfo:
        """Get information about a model"""
        provider_models = self.SUPPORTED_MODELS.get(provider, {})
        return provider_models.get(model)
        
    def estimate_cost(self, provider: str, model: str, tokens: int) -> float:
        """Estimate cost for token usage"""
        model_info = self.get_model_info(provider, model)
        if model_info and model_info.cost_per_1k:
            return (tokens / 1000) * model_info.cost_per_1k
        return 0.0
        
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """List available models"""
        if provider:
            return {provider: list(self.SUPPORTED_MODELS.get(provider, {}).keys())}
        
        return {
            p: list(models.keys())
            for p, models in self.SUPPORTED_MODELS.items()
        }

class ModelManager:
    """Unified model manager for both embedding and LLM models"""
    
    def __init__(self):
        self.embedding_manager = EmbeddingModelManager()
        self.llm_manager = LLMModelManager()
        self._usage_stats = {
            "embeddings_generated": 0,
            "llm_calls": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0
        }
        
    def load_embedding_model(self, model_name: str, **kwargs) -> SentenceTransformer:
        """Load an embedding model"""
        return self.embedding_manager.load_model(model_name, **kwargs)
        
    def initialize_llm(self, provider: str, api_key: Optional[str] = None, **kwargs):
        """Initialize an LLM provider"""
        self.llm_manager.initialize_provider(provider, api_key, **kwargs)
        
    def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
        """Generate embeddings for texts"""
        if model_name:
            model = self.embedding_manager.load_model(model_name)
        else:
            model = self.embedding_manager.get_current_model()
            
        if not model:
            raise RuntimeError("No embedding model loaded")
            
        embeddings = model.encode(texts, convert_to_numpy=True)
        self._usage_stats["embeddings_generated"] += len(texts)
        
        return embeddings
        
    def generate_completion(self, provider: str, model: str, prompt: str, **kwargs) -> str:
        """Generate LLM completion"""
        response = self.llm_manager.generate_completion(provider, model, prompt, **kwargs)
        
        # Update usage stats
        self._usage_stats["llm_calls"] += 1
        self._usage_stats["total_tokens"] += kwargs.get("max_tokens", 2000)
        self._usage_stats["estimated_cost"] += self.llm_manager.estimate_cost(
            provider, model, kwargs.get("max_tokens", 2000)
        )
        
        return response
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self._usage_stats.copy()
        
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self._usage_stats = {
            "embeddings_generated": 0,
            "llm_calls": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0
        }
        
    
    def list_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """List available models across all providers"""
        return self.llm_manager.list_available_models(provider)
    
    def get_model_recommendations(self, use_case: str) -> Dict[str, Any]:
        """Get model recommendations for specific use case"""
        recommendations = {
            "fast_analysis": {
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_provider": "openai",
                "llm_model": "gpt-3.5-turbo",
                "reason": "Optimized for speed and cost-effectiveness"
            },
            "quality_analysis": {
                "embedding": "sentence-transformers/all-mpnet-base-v2",
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "reason": "Best quality for complex error analysis"
            },
            "offline_analysis": {
                "embedding": "sentence-transformers/all-MiniLM-L12-v2",
                "llm_provider": "local",
                "llm_model": "llama-3.2-3b-instruct",
                "reason": "Works without internet connection, optimized for M1 Mac"
            },
            "hybrid_analysis": {
                "embedding": "sentence-transformers/all-MiniLM-L12-v2",
                "llm_provider": "hybrid",
                "llm_model": "auto-hybrid",
                "reason": "Best of both worlds - fast for simple, detailed for complex"
            },
            "cost_optimized": {
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_provider": "anthropic",
                "llm_model": "claude-3-haiku",
                "reason": "Lowest cost per analysis"
            }
        }
        
        return recommendations.get(use_case, recommendations["fast_analysis"]) 