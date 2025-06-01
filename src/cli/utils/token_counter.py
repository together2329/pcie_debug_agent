"""
Token counting utilities for LLM interactions
"""
import tiktoken
from typing import Optional, Dict, Tuple

class TokenCounter:
    """Count tokens for various LLM models"""
    
    def __init__(self):
        self.encoders = {}
        self.model_encodings = {
            # OpenAI models
            "gpt-4": "cl100k_base",
            "gpt-4o": "o200k_base",
            "gpt-4o-mini": "o200k_base", 
            "gpt-3.5-turbo": "cl100k_base",
            # Default for others
            "default": "cl100k_base"
        }
    
    def get_encoder(self, model_name: str):
        """Get the appropriate encoder for the model"""
        # Find the encoding for this model
        encoding_name = self.model_encodings.get(model_name, self.model_encodings["default"])
        
        # Cache encoders
        if encoding_name not in self.encoders:
            try:
                self.encoders[encoding_name] = tiktoken.get_encoding(encoding_name)
            except:
                # Fallback to cl100k_base if encoding not found
                self.encoders[encoding_name] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[encoding_name]
    
    def count_tokens(self, text: str, model_name: str = "gpt-4") -> int:
        """Count tokens in text for a specific model"""
        try:
            encoder = self.get_encoder(model_name)
            return len(encoder.encode(text))
        except Exception:
            # Fallback: estimate based on word count
            return self.estimate_tokens(text)
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count based on word count"""
        # Rough estimate: 1 token ≈ 0.75 words
        word_count = len(text.split())
        return int(word_count * 1.33)
    
    def count_messages_tokens(self, messages: list, model_name: str = "gpt-4") -> int:
        """Count tokens for a list of messages (for chat models)"""
        total = 0
        for message in messages:
            if isinstance(message, dict):
                # Add tokens for role and content
                total += self.count_tokens(message.get("role", ""), model_name)
                total += self.count_tokens(message.get("content", ""), model_name)
                # Add formatting tokens (approximate)
                total += 4  # <|im_start|>, <|im_end|>, etc.
        return total
    
    def format_token_count(self, count: int) -> str:
        """Format token count for display"""
        if count >= 1000:
            return f"{count:,} ({count/1000:.1f}k)"
        return str(count)
    
    def get_model_limits(self, model_name: str) -> Dict[str, int]:
        """Get token limits for different models"""
        limits = {
            "gpt-4": {"context": 8192, "max_output": 4096},
            "gpt-4o": {"context": 128000, "max_output": 16384},
            "gpt-4o-mini": {"context": 128000, "max_output": 16384},
            "gpt-3.5-turbo": {"context": 16385, "max_output": 4096},
            "claude-3-opus": {"context": 200000, "max_output": 4096},
            "llama-3.2-3b": {"context": 8192, "max_output": 2048},
            "deepseek-r1-7b": {"context": 32768, "max_output": 4096},
        }
        return limits.get(model_name, {"context": 4096, "max_output": 2048})
    
    def check_token_usage(self, input_tokens: int, output_tokens: int, 
                         model_name: str) -> Dict[str, any]:
        """Check if token usage is within model limits"""
        limits = self.get_model_limits(model_name)
        total_tokens = input_tokens + output_tokens
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "context_limit": limits["context"],
            "output_limit": limits["max_output"],
            "context_usage_pct": (total_tokens / limits["context"]) * 100,
            "output_usage_pct": (output_tokens / limits["max_output"]) * 100,
            "within_limits": total_tokens <= limits["context"] and output_tokens <= limits["max_output"]
        }