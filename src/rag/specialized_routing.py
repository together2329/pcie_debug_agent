#!/usr/bin/env python3
"""
Specialized Model Routing System

Routes queries to optimal processing pipelines based on:
- Query complexity and intent
- Required expertise level
- Performance requirements
- Resource availability
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import logging

from .pcie_knowledge_classifier import PCIeCategory, QueryIntent
from .query_expansion_engine import QueryExpansionEngine, ExpandedQuery
from .compliance_intelligence import ComplianceIntelligence
from .model_ensemble import ModelEnsemble

logger = logging.getLogger(__name__)

class ProcessingTier(Enum):
    """Processing tiers for different query types"""
    FAST_LOOKUP = "fast_lookup"          # Simple reference lookups
    STANDARD_ANALYSIS = "standard"       # Standard analysis
    DEEP_ANALYSIS = "deep"              # Complex technical analysis
    COMPLIANCE_CHECK = "compliance"      # Compliance-specific processing
    EXPERT_CONSULTATION = "expert"       # Most complex queries

class RoutingDecision(Enum):
    """Routing decision types"""
    DIRECT = "direct"                    # Direct processing
    ENSEMBLE = "ensemble"                # Multi-model ensemble
    HYBRID = "hybrid"                   # Hybrid approach
    SPECIALIZED = "specialized"          # Specialized pipeline
    ESCALATED = "escalated"             # Escalate to higher tier

@dataclass
class RoutingResult:
    """Result of routing decision"""
    tier: ProcessingTier
    decision: RoutingDecision
    selected_models: List[str]
    processing_pipeline: List[str]
    estimated_time: float
    confidence_target: float
    reasoning: str

@dataclass
class ProcessingPipeline:
    """Definition of a processing pipeline"""
    name: str
    tier: ProcessingTier
    models: List[str]
    processors: List[str]
    max_time: float
    min_confidence: float
    resource_cost: float

class SpecializedRouter:
    """Routes queries to optimal processing pipelines"""
    
    def __init__(self, model_ensemble: ModelEnsemble):
        self.model_ensemble = model_ensemble
        self.query_expander = QueryExpansionEngine()
        self.compliance_engine = ComplianceIntelligence()
        
        # Initialize processing pipelines
        self.pipelines = self._initialize_pipelines()
        
        # Performance tracking
        self.routing_history = []
        self.performance_stats = {
            'total_queries': 0,
            'tier_usage': {tier.value: 0 for tier in ProcessingTier},
            'average_response_time': {},
            'success_rate': {}
        }
    
    def _initialize_pipelines(self) -> Dict[ProcessingTier, List[ProcessingPipeline]]:
        """Initialize processing pipelines for each tier"""
        return {
            ProcessingTier.FAST_LOOKUP: [
                ProcessingPipeline(
                    name="Quick Reference",
                    tier=ProcessingTier.FAST_LOOKUP,
                    models=["text-embedding-3-small"],
                    processors=["basic_retrieval", "simple_formatting"],
                    max_time=1.0,
                    min_confidence=0.6,
                    resource_cost=0.1
                ),
                ProcessingPipeline(
                    name="Cached Lookup",
                    tier=ProcessingTier.FAST_LOOKUP,
                    models=["text-embedding-ada-002"],
                    processors=["cache_check", "basic_retrieval"],
                    max_time=0.5,
                    min_confidence=0.5,
                    resource_cost=0.05
                )
            ],
            
            ProcessingTier.STANDARD_ANALYSIS: [
                ProcessingPipeline(
                    name="Standard PCIe Analysis",
                    tier=ProcessingTier.STANDARD_ANALYSIS,
                    models=["text-embedding-3-large"],
                    processors=["query_expansion", "semantic_retrieval", "context_building", "llm_generation"],
                    max_time=5.0,
                    min_confidence=0.7,
                    resource_cost=0.3
                ),
                ProcessingPipeline(
                    name="Hybrid Analysis",
                    tier=ProcessingTier.STANDARD_ANALYSIS,
                    models=["text-embedding-3-small", "text-embedding-3-large"],
                    processors=["query_expansion", "hybrid_search", "reranking", "llm_generation"],
                    max_time=7.0,
                    min_confidence=0.75,
                    resource_cost=0.4
                )
            ],
            
            ProcessingTier.DEEP_ANALYSIS: [
                ProcessingPipeline(
                    name="Deep Technical Analysis",
                    tier=ProcessingTier.DEEP_ANALYSIS,
                    models=["text-embedding-3-large", "pcie-domain-embeddings"],
                    processors=["advanced_expansion", "ensemble_retrieval", "technical_analysis", "verification", "llm_generation"],
                    max_time=15.0,
                    min_confidence=0.85,
                    resource_cost=0.7
                ),
                ProcessingPipeline(
                    name="Multi-Model Ensemble",
                    tier=ProcessingTier.DEEP_ANALYSIS,
                    models=["text-embedding-3-large", "text-embedding-3-small", "technical-doc-embeddings"],
                    processors=["query_classification", "ensemble_search", "cross_validation", "confidence_scoring", "llm_generation"],
                    max_time=20.0,
                    min_confidence=0.9,
                    resource_cost=0.9
                )
            ],
            
            ProcessingTier.COMPLIANCE_CHECK: [
                ProcessingPipeline(
                    name="Compliance Analysis",
                    tier=ProcessingTier.COMPLIANCE_CHECK,
                    models=["text-embedding-3-large", "technical-doc-embeddings"],
                    processors=["compliance_detection", "spec_reference", "violation_analysis", "remediation_suggestions"],
                    max_time=10.0,
                    min_confidence=0.8,
                    resource_cost=0.5
                ),
                ProcessingPipeline(
                    name="Specification Lookup",
                    tier=ProcessingTier.COMPLIANCE_CHECK,
                    models=["technical-doc-embeddings"],
                    processors=["spec_search", "reference_extraction", "context_verification"],
                    max_time=5.0,
                    min_confidence=0.75,
                    resource_cost=0.3
                )
            ],
            
            ProcessingTier.EXPERT_CONSULTATION: [
                ProcessingPipeline(
                    name="Expert Analysis",
                    tier=ProcessingTier.EXPERT_CONSULTATION,
                    models=["text-embedding-3-large", "pcie-domain-embeddings", "technical-doc-embeddings"],
                    processors=["comprehensive_expansion", "multi_model_ensemble", "expert_analysis", "cross_verification", "detailed_explanation"],
                    max_time=30.0,
                    min_confidence=0.95,
                    resource_cost=1.5
                )
            ]
        }
    
    def route_query(self, query: str, user_preferences: Dict[str, Any] = None) -> RoutingResult:
        """Route query to optimal processing pipeline"""
        
        # Expand and classify query
        expanded_query = self.query_expander.expand_query(query)
        
        # Analyze query characteristics
        query_analysis = self._analyze_query(expanded_query)
        
        # Determine optimal tier
        tier = self._determine_tier(query_analysis, user_preferences or {})
        
        # Select specific pipeline within tier
        pipeline = self._select_pipeline(tier, query_analysis, user_preferences or {})
        
        # Make routing decision
        decision = self._make_routing_decision(pipeline, query_analysis)
        
        # Select models based on pipeline and decision
        selected_models = self._select_models_for_pipeline(pipeline, expanded_query)
        
        # Estimate processing time and confidence
        estimated_time = self._estimate_processing_time(pipeline, query_analysis)
        confidence_target = pipeline.min_confidence
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(tier, pipeline, query_analysis)
        
        result = RoutingResult(
            tier=tier,
            decision=decision,
            selected_models=selected_models,
            processing_pipeline=pipeline.processors,
            estimated_time=estimated_time,
            confidence_target=confidence_target,
            reasoning=reasoning
        )
        
        # Update statistics
        self._update_routing_stats(result)
        
        logger.info(f"Routed query to {tier.value}/{pipeline.name}: {reasoning}")
        
        return result
    
    def _analyze_query(self, expanded_query: ExpandedQuery) -> Dict[str, Any]:
        """Analyze query characteristics for routing"""
        original = expanded_query.original_query
        expanded = expanded_query.expanded_query
        
        analysis = {
            'category': expanded_query.category,
            'intent': expanded_query.intent,
            'complexity': self._assess_complexity(expanded),
            'urgency': self._assess_urgency(original),
            'specificity': self._assess_specificity(expanded),
            'technical_depth': self._assess_technical_depth(expanded),
            'compliance_related': self._is_compliance_related(expanded),
            'error_related': self._is_error_related(expanded),
            'implementation_related': self._is_implementation_related(expanded),
            'word_count': len(expanded.split()),
            'technical_terms': len(expanded_query.acronyms_expanded),
            'expansion_confidence': expanded_query.confidence
        }
        
        return analysis
    
    def _assess_complexity(self, query: str) -> str:
        """Assess query complexity"""
        indicators = {
            'simple': ['what is', 'define', 'list', 'show', 'quick'],
            'medium': ['how to', 'explain', 'difference', 'compare', 'analyze'],
            'complex': ['implement', 'troubleshoot', 'optimize', 'debug', 'compliance', 'specification']
        }
        
        query_lower = query.lower()
        scores = {}
        
        for level, terms in indicators.items():
            score = sum(1 for term in terms if term in query_lower)
            scores[level] = score
        
        # Additional complexity factors
        if len(query.split()) > 20:
            scores['complex'] += 2
        if len(re.findall(r'0x[0-9a-f]+', query.lower())) > 0:
            scores['complex'] += 1
        
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'simple'
    
    def _assess_urgency(self, query: str) -> str:
        """Assess query urgency"""
        urgent_terms = ['urgent', 'critical', 'failing', 'broken', 'not working', 'error', 'problem']
        normal_terms = ['understand', 'learn', 'explain', 'define']
        
        query_lower = query.lower()
        
        urgent_count = sum(1 for term in urgent_terms if term in query_lower)
        normal_count = sum(1 for term in normal_terms if term in query_lower)
        
        if urgent_count > normal_count:
            return 'high'
        elif normal_count > 0:
            return 'low'
        else:
            return 'medium'
    
    def _assess_specificity(self, query: str) -> str:
        """Assess how specific the query is"""
        specific_indicators = ['register', 'offset', 'bit', 'field', '0x', 'specification', 'section']
        general_indicators = ['how', 'what', 'why', 'general', 'overview']
        
        query_lower = query.lower()
        
        specific_count = sum(1 for term in specific_indicators if term in query_lower)
        general_count = sum(1 for term in general_indicators if term in query_lower)
        
        if specific_count > general_count * 2:
            return 'high'
        elif general_count > specific_count:
            return 'low'
        else:
            return 'medium'
    
    def _assess_technical_depth(self, query: str) -> str:
        """Assess required technical depth"""
        deep_terms = ['implementation', 'protocol', 'specification', 'compliance', 'register', 'bit field']
        surface_terms = ['overview', 'introduction', 'basic', 'simple']
        
        query_lower = query.lower()
        
        deep_count = sum(1 for term in deep_terms if term in query_lower)
        surface_count = sum(1 for term in surface_terms if term in query_lower)
        
        if deep_count > 0:
            return 'high'
        elif surface_count > 0:
            return 'low'
        else:
            return 'medium'
    
    def _is_compliance_related(self, query: str) -> bool:
        """Check if query is compliance-related"""
        compliance_terms = ['compliance', 'specification', 'standard', 'conformance', 'violation', 'requirement']
        return any(term in query.lower() for term in compliance_terms)
    
    def _is_error_related(self, query: str) -> bool:
        """Check if query is error-related"""
        error_terms = ['error', 'timeout', 'failure', 'problem', 'issue', 'debug', 'troubleshoot']
        return any(term in query.lower() for term in error_terms)
    
    def _is_implementation_related(self, query: str) -> bool:
        """Check if query is implementation-related"""
        impl_terms = ['implement', 'configure', 'setup', 'program', 'code', 'build']
        return any(term in query.lower() for term in impl_terms)
    
    def _determine_tier(self, analysis: Dict[str, Any], preferences: Dict[str, Any]) -> ProcessingTier:
        """Determine optimal processing tier"""
        
        # User preference override
        if 'tier' in preferences:
            return ProcessingTier(preferences['tier'])
        
        # Compliance-specific routing
        if analysis['compliance_related']:
            if analysis['complexity'] == 'complex' or analysis['technical_depth'] == 'high':
                return ProcessingTier.COMPLIANCE_CHECK
            else:
                return ProcessingTier.STANDARD_ANALYSIS
        
        # Error/troubleshooting routing
        if analysis['error_related'] and analysis['urgency'] == 'high':
            if analysis['complexity'] == 'complex':
                return ProcessingTier.DEEP_ANALYSIS
            else:
                return ProcessingTier.STANDARD_ANALYSIS
        
        # Complexity-based routing
        if analysis['complexity'] == 'complex' and analysis['technical_depth'] == 'high':
            if analysis['specificity'] == 'high':
                return ProcessingTier.EXPERT_CONSULTATION
            else:
                return ProcessingTier.DEEP_ANALYSIS
        
        # Simple lookups
        if (analysis['complexity'] == 'simple' and 
            analysis['intent'] == QueryIntent.REFERENCE and
            analysis['specificity'] == 'low'):
            return ProcessingTier.FAST_LOOKUP
        
        # Default to standard analysis
        return ProcessingTier.STANDARD_ANALYSIS
    
    def _select_pipeline(self, tier: ProcessingTier, analysis: Dict[str, Any], 
                        preferences: Dict[str, Any]) -> ProcessingPipeline:
        """Select specific pipeline within tier"""
        
        available_pipelines = self.pipelines[tier]
        
        if len(available_pipelines) == 1:
            return available_pipelines[0]
        
        # Score pipelines based on query characteristics
        best_pipeline = None
        best_score = -1
        
        for pipeline in available_pipelines:
            score = self._score_pipeline(pipeline, analysis, preferences)
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
        
        return best_pipeline or available_pipelines[0]
    
    def _score_pipeline(self, pipeline: ProcessingPipeline, analysis: Dict[str, Any], 
                       preferences: Dict[str, Any]) -> float:
        """Score pipeline suitability for query"""
        score = 0.0
        
        # Time preference
        max_time_pref = preferences.get('max_time', 30.0)
        if pipeline.max_time <= max_time_pref:
            score += 1.0
        else:
            score -= 0.5
        
        # Confidence requirement
        min_conf_pref = preferences.get('min_confidence', 0.7)
        if pipeline.min_confidence >= min_conf_pref:
            score += 0.5
        
        # Resource consideration
        if preferences.get('optimize_cost', False):
            score -= pipeline.resource_cost * 0.3
        
        # Specific pipeline characteristics
        if 'compliance' in pipeline.name.lower() and analysis['compliance_related']:
            score += 1.0
        
        if 'ensemble' in pipeline.name.lower() and analysis['complexity'] == 'complex':
            score += 0.5
        
        if 'quick' in pipeline.name.lower() and analysis['urgency'] == 'high':
            score += 0.3
        
        return score
    
    def _make_routing_decision(self, pipeline: ProcessingPipeline, 
                             analysis: Dict[str, Any]) -> RoutingDecision:
        """Make specific routing decision within pipeline"""
        
        # Determine if ensemble approach is beneficial
        if (analysis['complexity'] == 'complex' and 
            analysis['technical_depth'] == 'high' and
            len(pipeline.models) > 1):
            return RoutingDecision.ENSEMBLE
        
        # Use specialized processing for compliance
        if analysis['compliance_related'] and pipeline.tier == ProcessingTier.COMPLIANCE_CHECK:
            return RoutingDecision.SPECIALIZED
        
        # Use hybrid for medium complexity
        if analysis['complexity'] == 'medium' and len(pipeline.models) > 1:
            return RoutingDecision.HYBRID
        
        # Default to direct processing
        return RoutingDecision.DIRECT
    
    def _select_models_for_pipeline(self, pipeline: ProcessingPipeline, 
                                  expanded_query: ExpandedQuery) -> List[str]:
        """Select specific models for pipeline execution"""
        
        # Start with pipeline's default models
        pipeline_models = [model for model in pipeline.models 
                          if self.model_ensemble.model_configs.get(model, {}).get('available', True)]
        
        # Use ensemble selection if no models or need optimization
        if not pipeline_models or len(pipeline_models) < len(pipeline.models):
            optimal_models = self.model_ensemble.select_optimal_models(
                expanded_query.original_query,
                expanded_query.category,
                expanded_query.intent,
                max_models=max(2, len(pipeline.models))
            )
            pipeline_models.extend(optimal_models)
        
        # Remove duplicates while preserving order
        seen = set()
        selected_models = []
        for model in pipeline_models:
            if model not in seen:
                seen.add(model)
                selected_models.append(model)
        
        return selected_models[:3]  # Limit to 3 models max
    
    def _estimate_processing_time(self, pipeline: ProcessingPipeline, 
                                analysis: Dict[str, Any]) -> float:
        """Estimate processing time for pipeline"""
        
        base_time = pipeline.max_time * 0.6  # Conservative estimate
        
        # Adjust for complexity
        complexity_multipliers = {'simple': 0.7, 'medium': 1.0, 'complex': 1.3}
        base_time *= complexity_multipliers.get(analysis['complexity'], 1.0)
        
        # Adjust for number of models
        if len(pipeline.models) > 1:
            base_time *= 1.2
        
        # Adjust for technical depth
        if analysis['technical_depth'] == 'high':
            base_time *= 1.1
        
        return min(base_time, pipeline.max_time)
    
    def _generate_routing_reasoning(self, tier: ProcessingTier, 
                                  pipeline: ProcessingPipeline,
                                  analysis: Dict[str, Any]) -> str:
        """Generate human-readable routing reasoning"""
        
        reasons = []
        
        # Tier reasoning
        if tier == ProcessingTier.FAST_LOOKUP:
            reasons.append("simple reference query")
        elif tier == ProcessingTier.COMPLIANCE_CHECK:
            reasons.append("compliance-related analysis required")
        elif tier == ProcessingTier.DEEP_ANALYSIS:
            reasons.append("complex technical analysis needed")
        elif tier == ProcessingTier.EXPERT_CONSULTATION:
            reasons.append("expert-level analysis required")
        
        # Analysis factors
        if analysis['complexity'] == 'complex':
            reasons.append("high complexity detected")
        if analysis['urgency'] == 'high':
            reasons.append("urgent response needed")
        if analysis['technical_depth'] == 'high':
            reasons.append("deep technical knowledge required")
        
        return f"Routed to {tier.value} ({pipeline.name}): " + ", ".join(reasons)
    
    def _update_routing_stats(self, result: RoutingResult):
        """Update routing statistics"""
        self.performance_stats['total_queries'] += 1
        self.performance_stats['tier_usage'][result.tier.value] += 1
        
        # Store routing decision for analysis
        self.routing_history.append({
            'timestamp': time.time(),
            'tier': result.tier.value,
            'decision': result.decision.value,
            'models': result.selected_models,
            'estimated_time': result.estimated_time
        })
        
        # Keep recent history only
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        total = self.performance_stats['total_queries']
        
        return {
            'total_queries_routed': total,
            'tier_distribution': {
                tier: (count / total * 100) if total > 0 else 0
                for tier, count in self.performance_stats['tier_usage'].items()
            },
            'available_tiers': len(self.pipelines),
            'total_pipelines': sum(len(pipelines) for pipelines in self.pipelines.values()),
            'recent_decisions': self.routing_history[-10:] if self.routing_history else []
        }