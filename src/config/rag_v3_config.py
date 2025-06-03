"""
Configuration management for Enhanced RAG v3
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml

@dataclass
class RAGv3Config:
    """Configuration for Enhanced RAG v3 features"""
    
    # Core features
    enable_hybrid_search: bool = True
    enable_answer_verification: bool = True
    enable_question_normalization: bool = True
    
    # Confidence and quality settings
    min_confidence_threshold: float = 0.3
    verification_threshold: float = 0.5
    max_results: int = 10
    
    # Performance settings
    enable_caching: bool = True
    cache_size: int = 100
    processing_timeout: float = 30.0
    
    # Search configuration
    hybrid_search_alpha: float = 0.7  # Weight for semantic vs keyword
    semantic_similarity_threshold: float = 0.1
    
    # Knowledge classification
    enable_domain_classification: bool = True
    difficulty_assessment: bool = True
    fact_extraction: bool = True
    
    # Answer verification methods
    verification_methods: List[str] = field(default_factory=lambda: [
        "exact_match", "keyword_match", "semantic_similarity", "fact_verification"
    ])
    
    # Question normalization
    enable_intent_classification: bool = True
    enable_question_suggestions: bool = True
    max_suggestions: int = 5
    
    # Logging and monitoring
    enable_performance_monitoring: bool = True
    log_confidence_scores: bool = False
    log_processing_times: bool = True
    
    # Advanced features
    enable_streaming_responses: bool = True
    enable_progressive_enhancement: bool = True
    fallback_to_basic_rag: bool = True

@dataclass 
class PCIeDomainConfig:
    """PCIe domain-specific configuration"""
    
    # Categories to prioritize
    priority_categories: List[str] = field(default_factory=lambda: [
        "error_handling", "ltssm", "tlp", "power_management"
    ])
    
    # Fact extraction patterns
    extract_register_offsets: bool = True
    extract_bit_fields: bool = True
    extract_timeouts: bool = True
    extract_error_codes: bool = True
    
    # Question templates
    enable_pcie_templates: bool = True
    template_confidence_boost: float = 0.1

class RAGv3ConfigManager:
    """Manages Enhanced RAG v3 configuration"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("configs/rag_v3.yaml")
        self.rag_config = RAGv3Config()
        self.domain_config = PCIeDomainConfig()
        
        if self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update RAG config
            rag_data = config_data.get('rag_v3', {})
            for key, value in rag_data.items():
                if hasattr(self.rag_config, key):
                    setattr(self.rag_config, key, value)
            
            # Update domain config
            domain_data = config_data.get('pcie_domain', {})
            for key, value in domain_data.items():
                if hasattr(self.domain_config, key):
                    setattr(self.domain_config, key, value)
                    
        except Exception as e:
            print(f"⚠️ Failed to load RAG v3 config: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Ensure config directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = {
                'rag_v3': {
                    'enable_hybrid_search': self.rag_config.enable_hybrid_search,
                    'enable_answer_verification': self.rag_config.enable_answer_verification,
                    'enable_question_normalization': self.rag_config.enable_question_normalization,
                    'min_confidence_threshold': self.rag_config.min_confidence_threshold,
                    'verification_threshold': self.rag_config.verification_threshold,
                    'max_results': self.rag_config.max_results,
                    'enable_caching': self.rag_config.enable_caching,
                    'cache_size': self.rag_config.cache_size,
                    'processing_timeout': self.rag_config.processing_timeout,
                    'hybrid_search_alpha': self.rag_config.hybrid_search_alpha,
                    'semantic_similarity_threshold': self.rag_config.semantic_similarity_threshold,
                    'enable_domain_classification': self.rag_config.enable_domain_classification,
                    'difficulty_assessment': self.rag_config.difficulty_assessment,
                    'fact_extraction': self.rag_config.fact_extraction,
                    'verification_methods': self.rag_config.verification_methods,
                    'enable_intent_classification': self.rag_config.enable_intent_classification,
                    'enable_question_suggestions': self.rag_config.enable_question_suggestions,
                    'max_suggestions': self.rag_config.max_suggestions,
                    'enable_performance_monitoring': self.rag_config.enable_performance_monitoring,
                    'log_confidence_scores': self.rag_config.log_confidence_scores,
                    'log_processing_times': self.rag_config.log_processing_times,
                    'enable_streaming_responses': self.rag_config.enable_streaming_responses,
                    'enable_progressive_enhancement': self.rag_config.enable_progressive_enhancement,
                    'fallback_to_basic_rag': self.rag_config.fallback_to_basic_rag
                },
                'pcie_domain': {
                    'priority_categories': self.domain_config.priority_categories,
                    'extract_register_offsets': self.domain_config.extract_register_offsets,
                    'extract_bit_fields': self.domain_config.extract_bit_fields,
                    'extract_timeouts': self.domain_config.extract_timeouts,
                    'extract_error_codes': self.domain_config.extract_error_codes,
                    'enable_pcie_templates': self.domain_config.enable_pcie_templates,
                    'template_confidence_boost': self.domain_config.template_confidence_boost
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to save RAG v3 config: {e}")
    
    def get_rag_config(self) -> RAGv3Config:
        """Get RAG configuration"""
        return self.rag_config
    
    def get_domain_config(self) -> PCIeDomainConfig:
        """Get domain configuration"""
        return self.domain_config
    
    def update_config(self, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(self.rag_config, key):
                setattr(self.rag_config, key, value)
            elif hasattr(self.domain_config, key):
                setattr(self.domain_config, key, value)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.rag_config = RAGv3Config()
        self.domain_config = PCIeDomainConfig()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Validate thresholds
        if not 0.0 <= self.rag_config.min_confidence_threshold <= 1.0:
            issues.append("min_confidence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.rag_config.verification_threshold <= 1.0:
            issues.append("verification_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.rag_config.hybrid_search_alpha <= 1.0:
            issues.append("hybrid_search_alpha must be between 0.0 and 1.0")
        
        # Validate counts
        if self.rag_config.max_results <= 0:
            issues.append("max_results must be positive")
        
        if self.rag_config.cache_size <= 0:
            issues.append("cache_size must be positive")
        
        if self.rag_config.processing_timeout <= 0:
            issues.append("processing_timeout must be positive")
        
        return issues

# Global config manager instance
_config_manager = None

def get_rag_v3_config_manager() -> RAGv3ConfigManager:
    """Get global RAG v3 configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = RAGv3ConfigManager()
    return _config_manager

def get_rag_v3_config() -> RAGv3Config:
    """Get RAG v3 configuration"""
    return get_rag_v3_config_manager().get_rag_config()

def get_pcie_domain_config() -> PCIeDomainConfig:
    """Get PCIe domain configuration"""
    return get_rag_v3_config_manager().get_domain_config()