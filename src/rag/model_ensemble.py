#!/usr/bin/env python3
"""
Model Ensemble System for Phase 2 Advanced Features

Combines multiple embedding models for better coverage and implements
specialized model routing for different query types.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

from .pcie_knowledge_classifier import PCIeCategory, QueryIntent
from .query_expansion_engine import QueryExpansionEngine

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of embedding models"""
    SEMANTIC = "semantic"           # General semantic understanding
    TECHNICAL = "technical"         # Technical documentation
    DOMAIN_SPECIFIC = "domain"      # PCIe domain specific
    MULTILINGUAL = "multilingual"   # Multi-language support
    CODE_AWARE = "code"            # Code and configuration aware

@dataclass
class ModelConfig:
    """Configuration for an embedding model"""
    model_id: str
    model_type: ModelType
    embedding_dim: int
    strengths: List[str]
    weaknesses: List[str]
    optimal_categories: List[PCIeCategory]
    weight: float = 1.0
    available: bool = True

@dataclass
class EnsembleResult:
    """Result from ensemble model processing"""
    combined_embedding: np.ndarray
    model_contributions: Dict[str, float]
    confidence: float
    primary_model: str
    fallback_models: List[str]

class ModelEnsemble:
    """Ensemble system for combining multiple embedding models"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.query_expander = QueryExpansionEngine()
        
        # Define available models and their characteristics
        self.model_configs = self._initialize_model_configs()
        self.routing_rules = self._initialize_routing_rules()
        
        # Performance tracking
        self.performance_history = {}
        
    def _initialize_model_configs(self) -> Dict[str, ModelConfig]:
        """Initialize configuration for available embedding models"""
        return {
            'text-embedding-3-small': ModelConfig(
                model_id='text-embedding-3-small',
                model_type=ModelType.SEMANTIC,
                embedding_dim=1536,
                strengths=['general semantic understanding', 'fast processing', 'good coverage'],
                weaknesses=['limited technical depth', 'generic responses'],
                optimal_categories=[PCIeCategory.GENERAL, PCIeCategory.ARCHITECTURE],
                weight=0.8
            ),
            'text-embedding-3-large': ModelConfig(
                model_id='text-embedding-3-large',
                model_type=ModelType.SEMANTIC,
                embedding_dim=3072,
                strengths=['deep semantic understanding', 'nuanced concepts', 'high accuracy'],
                weaknesses=['slower processing', 'higher cost'],
                optimal_categories=[PCIeCategory.ERROR_HANDLING, PCIeCategory.COMPLIANCE],
                weight=1.0
            ),
            'text-embedding-ada-002': ModelConfig(
                model_id='text-embedding-ada-002',
                model_type=ModelType.SEMANTIC,
                embedding_dim=1536,
                strengths=['balanced performance', 'reliable', 'widely tested'],
                weaknesses=['older architecture', 'less specialized'],
                optimal_categories=[PCIeCategory.GENERAL, PCIeCategory.TLP],
                weight=0.9
            ),
            # Technical-specific models (hypothetical - would need actual implementations)
            'technical-doc-embeddings': ModelConfig(
                model_id='technical-doc-embeddings',
                model_type=ModelType.TECHNICAL,
                embedding_dim=1024,
                strengths=['technical terminology', 'specification documents', 'precise matching'],
                weaknesses=['limited general knowledge', 'narrow scope'],
                optimal_categories=[PCIeCategory.COMPLIANCE, PCIeCategory.DEBUGGING],
                weight=1.2,
                available=False  # Placeholder for future implementation
            ),
            'pcie-domain-embeddings': ModelConfig(
                model_id='pcie-domain-embeddings',
                model_type=ModelType.DOMAIN_SPECIFIC,
                embedding_dim=768,
                strengths=['PCIe expertise', 'domain terminology', 'context understanding'],
                weaknesses=['limited general knowledge', 'specialized only'],
                optimal_categories=[
                    PCIeCategory.ERROR_HANDLING, PCIeCategory.LTSSM, 
                    PCIeCategory.FLOW_CONTROL, PCIeCategory.POWER_MANAGEMENT
                ],
                weight=1.5,
                available=False  # Placeholder for future implementation
            )
        }
    
    def _initialize_routing_rules(self) -> Dict[str, Dict]:
        """Initialize model routing rules based on query characteristics"""
        return {
            'category_routing': {
                PCIeCategory.ERROR_HANDLING: ['text-embedding-3-large', 'pcie-domain-embeddings'],
                PCIeCategory.COMPLIANCE: ['text-embedding-3-large', 'technical-doc-embeddings'],
                PCIeCategory.DEBUGGING: ['technical-doc-embeddings', 'text-embedding-3-large'],
                PCIeCategory.LTSSM: ['pcie-domain-embeddings', 'text-embedding-3-large'],
                PCIeCategory.TLP: ['pcie-domain-embeddings', 'text-embedding-3-small'],
                PCIeCategory.POWER_MANAGEMENT: ['pcie-domain-embeddings', 'text-embedding-3-small'],
                PCIeCategory.PHYSICAL_LAYER: ['text-embedding-3-large', 'pcie-domain-embeddings'],
                PCIeCategory.ARCHITECTURE: ['text-embedding-3-small', 'text-embedding-ada-002'],
                PCIeCategory.CONFIGURATION: ['text-embedding-3-large', 'technical-doc-embeddings'],
                PCIeCategory.FLOW_CONTROL: ['pcie-domain-embeddings', 'text-embedding-3-large'],
                PCIeCategory.ADVANCED_FEATURES: ['text-embedding-3-large', 'text-embedding-3-small'],
                PCIeCategory.PERFORMANCE: ['text-embedding-3-large', 'technical-doc-embeddings'],
                PCIeCategory.SECURITY: ['text-embedding-3-large', 'text-embedding-3-small'],
                PCIeCategory.TIMING: ['technical-doc-embeddings', 'text-embedding-3-large'],
                PCIeCategory.TESTING: ['technical-doc-embeddings', 'text-embedding-3-small'],
                PCIeCategory.GENERAL: ['text-embedding-3-small', 'text-embedding-ada-002']
            },
            'intent_routing': {
                QueryIntent.LEARN: ['text-embedding-3-small', 'text-embedding-ada-002'],
                QueryIntent.TROUBLESHOOT: ['text-embedding-3-large', 'pcie-domain-embeddings'],
                QueryIntent.IMPLEMENT: ['technical-doc-embeddings', 'text-embedding-3-large'],
                QueryIntent.VERIFY: ['text-embedding-3-large', 'technical-doc-embeddings'],
                QueryIntent.OPTIMIZE: ['text-embedding-3-large', 'technical-doc-embeddings'],
                QueryIntent.REFERENCE: ['text-embedding-3-small', 'technical-doc-embeddings']
            },
            'complexity_routing': {
                'simple': ['text-embedding-3-small', 'text-embedding-ada-002'],
                'medium': ['text-embedding-3-large', 'text-embedding-3-small'],
                'complex': ['text-embedding-3-large', 'pcie-domain-embeddings', 'technical-doc-embeddings']
            }
        }
    
    def select_optimal_models(self, query: str, category: PCIeCategory = None, 
                            intent: QueryIntent = None, max_models: int = 2) -> List[str]:
        """Select optimal models for a given query"""
        
        # Expand query to get better classification
        expanded = self.query_expander.expand_query(query)
        if not category:
            category = expanded.category
        if not intent:
            intent = expanded.intent
        
        # Determine query complexity
        complexity = self._assess_query_complexity(expanded.expanded_query)
        
        # Get candidate models from routing rules
        candidate_models = set()
        
        # Add models based on category
        category_models = self.routing_rules['category_routing'].get(category, [])
        candidate_models.update(category_models)
        
        # Add models based on intent
        intent_models = self.routing_rules['intent_routing'].get(intent, [])
        candidate_models.update(intent_models)
        
        # Add models based on complexity
        complexity_models = self.routing_rules['complexity_routing'].get(complexity, [])
        candidate_models.update(complexity_models)
        
        # Filter available models
        available_models = [
            model_id for model_id in candidate_models 
            if model_id in self.model_configs and self.model_configs[model_id].available
        ]
        
        # Score and rank models
        model_scores = {}
        for model_id in available_models:
            config = self.model_configs[model_id]
            score = self._calculate_model_score(config, category, intent, complexity)
            model_scores[model_id] = score
        
        # Sort by score and return top models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        selected_models = [model_id for model_id, _ in sorted_models[:max_models]]
        
        # Ensure we have at least one model
        if not selected_models:
            selected_models = ['text-embedding-3-small']  # Default fallback
        
        logger.debug(f"Selected models for query '{query[:50]}...': {selected_models}")
        return selected_models
    
    def _assess_query_complexity(self, query: str) -> str:
        """Assess complexity of query"""
        # Simple heuristics for complexity assessment
        word_count = len(query.split())
        technical_terms = ['register', 'specification', 'compliance', 'protocol', 'implementation']
        tech_term_count = sum(1 for term in technical_terms if term in query.lower())
        
        if word_count > 20 or tech_term_count >= 3:
            return 'complex'
        elif word_count > 10 or tech_term_count >= 1:
            return 'medium'
        else:
            return 'simple'
    
    def _calculate_model_score(self, config: ModelConfig, category: PCIeCategory, 
                             intent: QueryIntent, complexity: str) -> float:
        """Calculate score for model selection"""
        score = config.weight
        
        # Category match bonus
        if category in config.optimal_categories:
            score += 0.5
        
        # Model type bonuses
        if complexity == 'complex' and config.model_type in [ModelType.TECHNICAL, ModelType.DOMAIN_SPECIFIC]:
            score += 0.3
        elif complexity == 'simple' and config.model_type == ModelType.SEMANTIC:
            score += 0.2
        
        # Intent-specific bonuses
        intent_bonuses = {
            QueryIntent.TROUBLESHOOT: {ModelType.DOMAIN_SPECIFIC: 0.3, ModelType.TECHNICAL: 0.2},
            QueryIntent.VERIFY: {ModelType.TECHNICAL: 0.3, ModelType.DOMAIN_SPECIFIC: 0.2},
            QueryIntent.LEARN: {ModelType.SEMANTIC: 0.2},
            QueryIntent.IMPLEMENT: {ModelType.TECHNICAL: 0.3, ModelType.CODE_AWARE: 0.2}
        }
        
        if intent in intent_bonuses:
            bonus = intent_bonuses[intent].get(config.model_type, 0)
            score += bonus
        
        # Performance history bonus (if available)
        if config.model_id in self.performance_history:
            avg_performance = np.mean(self.performance_history[config.model_id])
            score += (avg_performance - 0.5) * 0.4  # Scale performance to [-0.2, 0.2]
        
        return score
    
    def generate_ensemble_embedding(self, query: str, selected_models: List[str]) -> EnsembleResult:
        """Generate ensemble embedding from multiple models"""
        
        # Generate embeddings from each model
        model_embeddings = {}
        model_weights = {}
        
        for model_id in selected_models:
            try:
                # Generate embedding using model manager
                embedding = self.model_manager.generate_embeddings([query], model=model_id)[0]
                model_embeddings[model_id] = np.array(embedding)
                
                # Get model weight
                config = self.model_configs.get(model_id, ModelConfig(
                    model_id=model_id, model_type=ModelType.SEMANTIC, 
                    embedding_dim=len(embedding), strengths=[], weaknesses=[], 
                    optimal_categories=[]
                ))
                model_weights[model_id] = config.weight
                
                logger.debug(f"Generated embedding for {model_id}: shape {embedding.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to generate embedding for {model_id}: {e}")
                continue
        
        if not model_embeddings:
            raise ValueError("No models available for embedding generation")
        
        # Combine embeddings using weighted average
        combined_embedding = self._combine_embeddings(model_embeddings, model_weights)
        
        # Calculate confidence based on model agreement
        confidence = self._calculate_ensemble_confidence(model_embeddings)
        
        # Determine primary model and fallbacks
        primary_model = max(model_weights.items(), key=lambda x: x[1])[0]
        fallback_models = [m for m in selected_models if m != primary_model]
        
        return EnsembleResult(
            combined_embedding=combined_embedding,
            model_contributions=model_weights,
            confidence=confidence,
            primary_model=primary_model,
            fallback_models=fallback_models
        )
    
    def _combine_embeddings(self, embeddings: Dict[str, np.ndarray], 
                          weights: Dict[str, float]) -> np.ndarray:
        """Combine multiple embeddings using weighted average"""
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Find target dimension (use the largest dimension)
        max_dim = max(emb.shape[0] for emb in embeddings.values())
        
        # Combine embeddings
        combined = np.zeros(max_dim)
        
        for model_id, embedding in embeddings.items():
            weight = normalized_weights[model_id]
            
            # Handle different embedding dimensions
            if embedding.shape[0] == max_dim:
                combined += weight * embedding
            else:
                # Pad or truncate to match target dimension
                if embedding.shape[0] < max_dim:
                    padded = np.pad(embedding, (0, max_dim - embedding.shape[0]), 'constant')
                    combined += weight * padded
                else:
                    truncated = embedding[:max_dim]
                    combined += weight * truncated
        
        # Normalize the combined embedding
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined
    
    def _calculate_ensemble_confidence(self, embeddings: Dict[str, np.ndarray]) -> float:
        """Calculate confidence based on embedding similarity"""
        if len(embeddings) < 2:
            return 0.8  # Single model baseline confidence
        
        # Calculate pairwise similarities
        similarities = []
        embedding_list = list(embeddings.values())
        
        for i in range(len(embedding_list)):
            for j in range(i + 1, len(embedding_list)):
                emb1, emb2 = embedding_list[i], embedding_list[j]
                
                # Handle different dimensions by padding/truncating
                min_dim = min(emb1.shape[0], emb2.shape[0])
                emb1_norm = emb1[:min_dim] / np.linalg.norm(emb1[:min_dim])
                emb2_norm = emb2[:min_dim] / np.linalg.norm(emb2[:min_dim])
                
                similarity = np.dot(emb1_norm, emb2_norm)
                similarities.append(similarity)
        
        # Average similarity as confidence indicator
        avg_similarity = np.mean(similarities) if similarities else 0.5
        
        # Map similarity to confidence (higher similarity = higher confidence)
        confidence = 0.5 + 0.5 * avg_similarity  # Scale to [0.5, 1.0]
        
        return min(max(confidence, 0.1), 1.0)
    
    def update_performance(self, model_id: str, performance_score: float):
        """Update performance history for a model"""
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []
        
        self.performance_history[model_id].append(performance_score)
        
        # Keep only recent history (last 100 scores)
        if len(self.performance_history[model_id]) > 100:
            self.performance_history[model_id] = self.performance_history[model_id][-100:]
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble system statistics"""
        available_models = [
            model_id for model_id, config in self.model_configs.items() 
            if config.available
        ]
        
        return {
            'total_models': len(self.model_configs),
            'available_models': len(available_models),
            'model_types': len(set(config.model_type for config in self.model_configs.values())),
            'routing_rules': {
                'categories': len(self.routing_rules['category_routing']),
                'intents': len(self.routing_rules['intent_routing']),
                'complexity_levels': len(self.routing_rules['complexity_routing'])
            },
            'performance_tracked': len(self.performance_history),
            'available_model_list': available_models
        }