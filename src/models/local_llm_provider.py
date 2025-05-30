"""
Local LLM Provider using llama.cpp with Metal acceleration for M1 Macs
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from urllib.parse import urlparse
import hashlib

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Local LLM will not be available.")


class LocalLLMProvider:
    """Local LLM Provider for running Llama models locally with M1 optimization"""
    
    MODELS = {
        "llama-3.2-3b-instruct": {
            "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "size_gb": 1.8,
            "context_length": 131072,
            "description": "Llama 3.2 3B Instruct model optimized for M1 with Q4_K_M quantization"
        }
    }
    
    def __init__(self, 
                 model_name: str = "llama-3.2-3b-instruct",
                 models_dir: str = "models",
                 n_ctx: int = 8192,
                 n_gpu_layers: int = -1,  # Use all GPU layers on M1
                 verbose: bool = False):
        """
        Initialize Local LLM Provider
        
        Args:
            model_name: Name of the model to use
            models_dir: Directory to store models
            n_ctx: Context window size
            n_gpu_layers: Number of GPU layers (-1 for all on M1)
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.llm = None
        
        # Create models directory
        self.models_dir.mkdir(exist_ok=True)
        
        # Check if llama-cpp-python is available
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for local LLM. "
                "Install with: pip install llama-cpp-python"
            )
        
        # Validate model
        if model_name not in self.MODELS:
            available = ", ".join(self.MODELS.keys())
            raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
        
        self.model_info = self.MODELS[model_name]
        self.model_path = self.models_dir / self.model_info["filename"]
        
    def ensure_model_downloaded(self) -> bool:
        """
        Ensure the model is downloaded and ready to use
        
        Returns:
            True if model is ready, False otherwise
        """
        if self.model_path.exists():
            logger.info(f"Model {self.model_name} already exists at {self.model_path}")
            return True
        
        logger.info(f"Downloading model {self.model_name}...")
        return self._download_model()
    
    def _download_model(self) -> bool:
        """
        Download the model from HuggingFace
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            url = self.model_info["url"]
            logger.info(f"Downloading from: {url}")
            
            # Get file size for progress tracking
            response = requests.head(url, allow_redirects=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress tracking
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            downloaded = 0
            chunk_size = 8192
            
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (chunk_size * 1000) == 0:  # Log every ~8MB
                                logger.info(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Model downloaded successfully to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Clean up partial download
            if self.model_path.exists():
                self.model_path.unlink()
            return False
    
    def load_model(self) -> bool:
        """
        Load the model into memory
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.ensure_model_downloaded():
            return False
        
        try:
            logger.info(f"Loading model {self.model_name}...")
            
            # Configure for M1 Mac optimization
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,  # Use Metal GPU on M1
                verbose=self.verbose,
                n_threads=None,  # Auto-detect optimal threads
                f16_kv=True,  # Use half precision for key/value cache
                use_mlock=True,  # Keep model in memory
                use_mmap=True,  # Memory map the model file
            )
            
            logger.info(f"Model {self.model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.llm = None
            return False
    
    def generate_completion(self, 
                          prompt: str,
                          max_tokens: int = 2000,
                          temperature: float = 0.1,
                          top_p: float = 0.95,
                          stop_sequences: Optional[List[str]] = None) -> str:
        """
        Generate completion for the given prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Stop sequences
            
        Returns:
            Generated text completion
        """
        if self.llm is None:
            if not self.load_model():
                raise RuntimeError("Failed to load local LLM model")
        
        try:
            # Format prompt for Llama 3.2 Instruct
            formatted_prompt = self._format_instruct_prompt(prompt)
            
            # Generate response
            response = self.llm(
                formatted_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop_sequences or ["<|eot_id|>", "<|end_of_text|>"],
                echo=False
            )
            
            # Extract generated text
            generated_text = response["choices"][0]["text"].strip()
            
            # Remove any remaining special tokens
            generated_text = self._clean_generated_text(generated_text)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            raise
    
    def _format_instruct_prompt(self, prompt: str) -> str:
        """
        Format prompt for Llama 3.2 Instruct model
        
        Args:
            prompt: Raw prompt
            
        Returns:
            Formatted prompt with proper instruction tags
        """
        # Llama 3.2 Instruct format
        formatted = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant specialized in PCIe debugging and analysis. Provide clear, accurate, and technical responses based on the given context.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return formatted
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean generated text by removing special tokens and formatting issues
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Remove special tokens that might leak through
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
            "<|im_start|>",
            "<|im_end|>"
        ]
        
        for token in special_tokens:
            text = text.replace(token, "")
        
        # Clean up extra whitespace
        text = text.strip()
        
        return text
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = self.model_info.copy()
        info.update({
            "model_path": str(self.model_path),
            "loaded": self.llm is not None,
            "context_window": self.n_ctx,
            "gpu_layers": self.n_gpu_layers,
            "memory_usage": f"~{self.model_info['size_gb']}GB"
        })
        
        if self.model_path.exists():
            info["file_size_mb"] = self.model_path.stat().st_size / (1024 * 1024)
        
        return info
    
    def unload_model(self):
        """Unload the model from memory"""
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info(f"Model {self.model_name} unloaded from memory")
    
    def is_available(self) -> bool:
        """Check if local LLM is available"""
        return LLAMA_CPP_AVAILABLE and (self.llm is not None or self.model_path.exists())
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return cls.MODELS.copy()
    
    @staticmethod
    def get_memory_requirements() -> Dict[str, float]:
        """Get memory requirements for different models (in GB)"""
        return {
            name: info["size_gb"] 
            for name, info in LocalLLMProvider.MODELS.items()
        }
    
    @staticmethod
    def check_system_compatibility() -> Dict[str, Any]:
        """Check system compatibility for local LLM"""
        import platform
        import psutil
        
        system_info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "llama_cpp_available": LLAMA_CPP_AVAILABLE,
            "metal_support": platform.machine() == "arm64" and platform.system() == "Darwin"
        }
        
        # Check if sufficient memory
        min_memory_gb = min(LocalLLMProvider.get_memory_requirements().values())
        system_info["sufficient_memory"] = system_info["available_memory_gb"] > min_memory_gb * 1.5
        
        return system_info