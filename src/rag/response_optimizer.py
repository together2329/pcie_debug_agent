#!/usr/bin/env python3
"""
Response Optimization System for Phase 3 Intelligence Layer

Optimizes responses based on user feedback, performance metrics,
and adaptive learning to continuously improve response quality.
"""

import time
import json
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import math

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"  # 1-5 scale
    EXPLICIT_FEEDBACK = "explicit_feedback"  # Text feedback
    IMPLICIT_POSITIVE = "implicit_positive"  # User action indicating satisfaction
    IMPLICIT_NEGATIVE = "implicit_negative"  # User action indicating dissatisfaction

class OptimizationStrategy(Enum):
    """Response optimization strategies"""
    QUALITY_FOCUSED = "quality_focused"
    SPEED_FOCUSED = "speed_focused"
    BALANCED = "balanced"
    USER_PREFERENCE = "user_preference"
    ADAPTIVE = "adaptive"

@dataclass
class UserFeedback:
    """User feedback data structure"""
    feedback_id: str
    query_id: str
    session_id: str
    feedback_type: FeedbackType
    value: Any  # Can be boolean, number, or text
    timestamp: datetime
    user_context: Dict[str, Any]
    response_metadata: Dict[str, Any]
    
@dataclass
class ResponseProfile:
    """Profile for response optimization"""
    query_pattern: str
    category: str
    intent: str
    optimal_parameters: Dict[str, Any]
    performance_score: float
    feedback_score: float
    usage_count: int
    last_updated: datetime

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    recommendation_id: str
    optimization_type: str
    current_parameter: str
    suggested_parameter: str
    expected_improvement: float
    confidence: float
    supporting_evidence: List[str]
    timestamp: datetime

class ResponseOptimizer:
    """Response optimization system with adaptive learning"""
    
    def __init__(self, quality_monitor=None, analytics=None):
        self.quality_monitor = quality_monitor
        self.analytics = analytics
        
        # Feedback storage
        self.user_feedback = deque(maxlen=10000)  # Last 10k feedback items
        self.feedback_by_query = defaultdict(list)  # query_id -> feedback list
        self.feedback_by_session = defaultdict(list)  # session_id -> feedback list
        
        # Response profiles for optimization
        self.response_profiles = {}  # pattern_key -> ResponseProfile
        self.optimization_history = deque(maxlen=1000)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.min_feedback_count = 5
        
        # Optimization strategies
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.strategy_weights = {
            'quality': 0.4,
            'speed': 0.3,
            'user_satisfaction': 0.3
        }
        
        # Adaptive parameters
        self.adaptive_parameters = {
            'retrieval_count': {'min': 3, 'max': 10, 'current': 5},
            'confidence_threshold': {'min': 0.3, 'max': 0.9, 'current': 0.6},
            'response_length': {'min': 100, 'max': 1000, 'current': 500},
            'processing_tier': {'options': ['fast', 'standard', 'deep'], 'current': 'standard'}
        }
        
    def record_feedback(self, query_id: str, session_id: str, 
                       feedback_type: FeedbackType, value: Any,
                       user_context: Dict[str, Any] = None,
                       response_metadata: Dict[str, Any] = None):
        """Record user feedback for response optimization"""
        
        feedback = UserFeedback(
            feedback_id=f"feedback_{int(time.time() * 1000)}",
            query_id=query_id,
            session_id=session_id,
            feedback_type=feedback_type,
            value=value,
            timestamp=datetime.now(),
            user_context=user_context or {},
            response_metadata=response_metadata or {}
        )
        
        # Store feedback
        self.user_feedback.append(feedback)
        self.feedback_by_query[query_id].append(feedback)
        self.feedback_by_session[session_id].append(feedback)
        
        # Trigger optimization if enough feedback collected
        if len(self.user_feedback) % 10 == 0:  # Every 10 pieces of feedback
            self._trigger_optimization_analysis()
        
        logger.info(f"Recorded feedback: {feedback_type.value} for query {query_id}")
    
    def record_implicit_feedback(self, query_id: str, session_id: str,
                                action: str, context: Dict[str, Any] = None):
        """Record implicit feedback based on user actions"""
        
        # Map actions to feedback types
        positive_actions = [
            'copy_response', 'bookmark_response', 'share_response',
            'follow_up_question', 'expand_details', 'time_spent_reading'
        ]
        
        negative_actions = [
            'quick_exit', 'reformulate_query', 'skip_response',
            'error_report', 'ask_for_clarification'
        ]
        
        if action in positive_actions:
            feedback_type = FeedbackType.IMPLICIT_POSITIVE
            value = True
        elif action in negative_actions:
            feedback_type = FeedbackType.IMPLICIT_NEGATIVE
            value = False
        else:
            return  # Unknown action
        
        self.record_feedback(
            query_id=query_id,
            session_id=session_id,
            feedback_type=feedback_type,
            value=value,
            user_context={'action': action},
            response_metadata=context or {}
        )
    
    def optimize_response_parameters(self, query: str, category: str, intent: str,
                                   current_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize response parameters based on learning"""
        
        # Get response profile for this query pattern
        pattern_key = self._get_pattern_key(query, category, intent)
        profile = self.response_profiles.get(pattern_key)
        
        if profile and profile.usage_count >= self.min_feedback_count:
            # Use learned optimal parameters
            optimized_params = current_params.copy()
            optimized_params.update(profile.optimal_parameters)
            
            logger.debug(f"Applied optimized parameters for pattern {pattern_key}")
            return optimized_params
        
        # Fallback to adaptive parameter adjustment
        return self._apply_adaptive_optimization(current_params, category, intent)
    
    def _get_pattern_key(self, query: str, category: str, intent: str) -> str:
        """Generate pattern key for response profile"""
        
        # Extract key terms from query
        words = query.lower().split()
        key_terms = []
        
        # PCIe-specific key terms
        pcie_terms = ['pcie', 'flr', 'crs', 'ltssm', 'aer', 'error', 'timeout', 
                     'completion', 'configuration', 'compliance', 'debug']
        
        for word in words:
            if word in pcie_terms or len(word) > 6:  # Long words likely important
                key_terms.append(word)
        
        # Limit to top 3 terms to avoid over-specificity
        key_terms = key_terms[:3]
        
        return f"{category}_{intent}_{'_'.join(key_terms)}"
    
    def _apply_adaptive_optimization(self, current_params: Dict[str, Any],
                                   category: str, intent: str) -> Dict[str, Any]:
        """Apply adaptive optimization based on current strategy"""
        
        optimized_params = current_params.copy()
        
        # Strategy-specific optimizations
        if self.current_strategy == OptimizationStrategy.QUALITY_FOCUSED:
            optimized_params.update({
                'retrieval_count': max(7, self.adaptive_parameters['retrieval_count']['current']),
                'confidence_threshold': max(0.7, self.adaptive_parameters['confidence_threshold']['current']),
                'processing_tier': 'deep'
            })
        
        elif self.current_strategy == OptimizationStrategy.SPEED_FOCUSED:
            optimized_params.update({
                'retrieval_count': min(3, self.adaptive_parameters['retrieval_count']['current']),
                'confidence_threshold': min(0.5, self.adaptive_parameters['confidence_threshold']['current']),
                'processing_tier': 'fast'
            })
        
        elif self.current_strategy == OptimizationStrategy.BALANCED:
            optimized_params.update({
                'retrieval_count': self.adaptive_parameters['retrieval_count']['current'],
                'confidence_threshold': self.adaptive_parameters['confidence_threshold']['current'],
                'processing_tier': self.adaptive_parameters['processing_tier']['current']
            })
        
        # Category-specific adjustments
        if category == 'error_handling':
            optimized_params['confidence_threshold'] = max(0.7, optimized_params.get('confidence_threshold', 0.6))
        elif category == 'compliance':
            optimized_params['retrieval_count'] = max(5, optimized_params.get('retrieval_count', 5))
        
        # Intent-specific adjustments
        if intent == 'troubleshoot':
            optimized_params['processing_tier'] = 'deep'
        elif intent == 'reference':
            optimized_params['processing_tier'] = 'fast'
        
        return optimized_params
    
    def _trigger_optimization_analysis(self):
        """Trigger comprehensive optimization analysis"""
        
        # Analyze recent feedback patterns
        recent_feedback = list(self.user_feedback)[-100:]  # Last 100 feedback items
        
        if len(recent_feedback) < 10:
            return
        
        # Calculate satisfaction metrics
        satisfaction_scores = self._calculate_satisfaction_scores(recent_feedback)
        
        # Update response profiles
        self._update_response_profiles(recent_feedback)
        
        # Adjust adaptive parameters
        self._adjust_adaptive_parameters(satisfaction_scores)
        
        # Generate optimization recommendations
        recommendations = self._generate_optimization_recommendations()
        
        logger.info(f"Completed optimization analysis. Generated {len(recommendations)} recommendations.")
    
    def _calculate_satisfaction_scores(self, feedback_list: List[UserFeedback]) -> Dict[str, float]:
        """Calculate satisfaction scores from feedback"""
        
        scores = {}
        
        # Overall satisfaction
        positive_feedback = 0
        total_feedback = 0
        
        for feedback in feedback_list:
            total_feedback += 1
            
            if feedback.feedback_type == FeedbackType.THUMBS_UP:
                positive_feedback += 1
            elif feedback.feedback_type == FeedbackType.THUMBS_DOWN:
                positive_feedback += 0
            elif feedback.feedback_type == FeedbackType.RATING:
                positive_feedback += (feedback.value - 1) / 4  # Normalize 1-5 to 0-1
            elif feedback.feedback_type == FeedbackType.IMPLICIT_POSITIVE:
                positive_feedback += 0.7  # Weight implicit feedback lower
            elif feedback.feedback_type == FeedbackType.IMPLICIT_NEGATIVE:
                positive_feedback += 0.3
        
        scores['overall'] = positive_feedback / total_feedback if total_feedback > 0 else 0.5
        
        # Category-specific satisfaction
        category_feedback = defaultdict(list)
        for feedback in feedback_list:
            category = feedback.response_metadata.get('category', 'unknown')
            category_feedback[category].append(feedback)
        
        for category, cat_feedback in category_feedback.items():
            cat_positive = sum(1 for f in cat_feedback 
                             if f.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.IMPLICIT_POSITIVE])
            scores[f'category_{category}'] = cat_positive / len(cat_feedback) if cat_feedback else 0.5
        
        return scores
    
    def _update_response_profiles(self, feedback_list: List[UserFeedback]):
        """Update response profiles based on feedback"""
        
        # Group feedback by pattern
        pattern_feedback = defaultdict(list)
        
        for feedback in feedback_list:
            metadata = feedback.response_metadata
            category = metadata.get('category', 'unknown')
            intent = metadata.get('intent', 'unknown')
            query = metadata.get('query', '')
            
            pattern_key = self._get_pattern_key(query, category, intent)
            pattern_feedback[pattern_key].append(feedback)
        
        # Update profiles
        for pattern_key, pattern_fb in pattern_feedback.items():
            if len(pattern_fb) < 3:  # Need minimum feedback
                continue
            
            # Calculate performance metrics
            positive_count = sum(1 for f in pattern_fb 
                               if f.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.IMPLICIT_POSITIVE])
            feedback_score = positive_count / len(pattern_fb)
            
            # Extract response parameters from feedback
            response_times = []
            confidences = []
            parameters = defaultdict(list)
            
            for feedback in pattern_fb:
                metadata = feedback.response_metadata
                if 'response_time' in metadata:
                    response_times.append(metadata['response_time'])
                if 'confidence' in metadata:
                    confidences.append(metadata['confidence'])
                
                # Extract parameters
                for key, value in metadata.items():
                    if key.startswith('param_'):
                        param_name = key[6:]  # Remove 'param_' prefix
                        parameters[param_name].append(value)
            
            # Calculate optimal parameters
            optimal_params = {}
            
            # For numerical parameters, find values that correlate with positive feedback
            for param_name, values in parameters.items():
                if len(values) == len(pattern_fb) and values:
                    # Correlate parameter values with feedback scores
                    param_feedback_pairs = []
                    for i, feedback in enumerate(pattern_fb):
                        fb_score = 1.0 if feedback.feedback_type in [FeedbackType.THUMBS_UP, FeedbackType.IMPLICIT_POSITIVE] else 0.0
                        param_feedback_pairs.append((values[i], fb_score))
                    
                    # Find parameter value with best average feedback
                    if param_feedback_pairs:
                        # Group by parameter value and calculate average feedback
                        param_groups = defaultdict(list)
                        for param_val, fb_score in param_feedback_pairs:
                            param_groups[param_val].append(fb_score)
                        
                        best_param = max(param_groups.items(), 
                                       key=lambda x: statistics.mean(x[1]))[0]
                        optimal_params[param_name] = best_param
            
            # Update or create profile
            if pattern_key in self.response_profiles:
                profile = self.response_profiles[pattern_key]
                profile.feedback_score = feedback_score
                profile.usage_count += len(pattern_fb)
                profile.optimal_parameters.update(optimal_params)
                profile.last_updated = datetime.now()
            else:
                # Create new profile
                parts = pattern_key.split('_')
                category = parts[0] if len(parts) > 0 else 'unknown'
                intent = parts[1] if len(parts) > 1 else 'unknown'
                
                profile = ResponseProfile(
                    query_pattern=pattern_key,
                    category=category,
                    intent=intent,
                    optimal_parameters=optimal_params,
                    performance_score=statistics.mean(confidences) if confidences else 0.5,
                    feedback_score=feedback_score,
                    usage_count=len(pattern_fb),
                    last_updated=datetime.now()
                )
                
                self.response_profiles[pattern_key] = profile
    
    def _adjust_adaptive_parameters(self, satisfaction_scores: Dict[str, float]):
        """Adjust adaptive parameters based on satisfaction scores"""
        
        overall_satisfaction = satisfaction_scores.get('overall', 0.5)
        
        # Adjust parameters based on satisfaction
        if overall_satisfaction < 0.6:  # Low satisfaction
            # Increase quality parameters
            self._adjust_parameter('retrieval_count', +1)
            self._adjust_parameter('confidence_threshold', +0.05)
            if self.adaptive_parameters['processing_tier']['current'] == 'fast':
                self.adaptive_parameters['processing_tier']['current'] = 'standard'
            elif self.adaptive_parameters['processing_tier']['current'] == 'standard':
                self.adaptive_parameters['processing_tier']['current'] = 'deep'
        
        elif overall_satisfaction > 0.8:  # High satisfaction
            # Can potentially optimize for speed
            recent_response_times = self._get_recent_response_times()
            if recent_response_times and statistics.mean(recent_response_times) > 5.0:
                # Response times are slow, optimize for speed
                self._adjust_parameter('retrieval_count', -1)
                if self.adaptive_parameters['processing_tier']['current'] == 'deep':
                    self.adaptive_parameters['processing_tier']['current'] = 'standard'
        
        logger.debug(f"Adjusted adaptive parameters. Overall satisfaction: {overall_satisfaction:.2%}")
    
    def _adjust_parameter(self, param_name: str, adjustment: float):
        """Adjust a parameter within its bounds"""
        
        if param_name not in self.adaptive_parameters:
            return
        
        param_config = self.adaptive_parameters[param_name]
        current = param_config['current']
        
        if 'min' in param_config and 'max' in param_config:
            new_value = max(param_config['min'], 
                          min(param_config['max'], current + adjustment))
            param_config['current'] = new_value
    
    def _get_recent_response_times(self) -> List[float]:
        """Get recent response times from feedback metadata"""
        
        response_times = []
        recent_feedback = list(self.user_feedback)[-50:]  # Last 50 feedback items
        
        for feedback in recent_feedback:
            if 'response_time' in feedback.response_metadata:
                response_times.append(feedback.response_metadata['response_time'])
        
        return response_times
    
    def _generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Analyze response profiles for recommendations
        for pattern_key, profile in self.response_profiles.items():
            if profile.usage_count < self.min_feedback_count:
                continue
            
            # Check if profile shows room for improvement
            if profile.feedback_score < 0.7:
                # Recommend quality improvements
                if 'confidence_threshold' in profile.optimal_parameters:
                    current_threshold = self.adaptive_parameters['confidence_threshold']['current']
                    optimal_threshold = profile.optimal_parameters['confidence_threshold']
                    
                    if optimal_threshold > current_threshold + 0.1:
                        rec = OptimizationRecommendation(
                            recommendation_id=f"opt_{int(time.time())}",
                            optimization_type="confidence_threshold",
                            current_parameter=str(current_threshold),
                            suggested_parameter=str(optimal_threshold),
                            expected_improvement=(optimal_threshold - current_threshold) * profile.feedback_score,
                            confidence=0.8,
                            supporting_evidence=[f"Pattern {pattern_key} shows {profile.feedback_score:.2%} satisfaction with threshold {optimal_threshold}"],
                            timestamp=datetime.now()
                        )
                        recommendations.append(rec)
        
        # Global optimization recommendations
        recent_satisfaction = self._calculate_satisfaction_scores(list(self.user_feedback)[-100:])
        overall_sat = recent_satisfaction.get('overall', 0.5)
        
        if overall_sat < 0.6:
            rec = OptimizationRecommendation(
                recommendation_id=f"global_opt_{int(time.time())}",
                optimization_type="strategy_change",
                current_parameter=self.current_strategy.value,
                suggested_parameter=OptimizationStrategy.QUALITY_FOCUSED.value,
                expected_improvement=0.15,
                confidence=0.7,
                supporting_evidence=[f"Overall satisfaction ({overall_sat:.2%}) below target"],
                timestamp=datetime.now()
            )
            recommendations.append(rec)
        
        return recommendations
    
    def apply_optimization_recommendation(self, recommendation_id: str) -> bool:
        """Apply an optimization recommendation"""
        
        # Find recommendation in history
        recommendation = None
        for rec in self.optimization_history:
            if hasattr(rec, 'recommendation_id') and rec.recommendation_id == recommendation_id:
                recommendation = rec
                break
        
        if not recommendation:
            logger.warning(f"Recommendation {recommendation_id} not found")
            return False
        
        try:
            # Apply recommendation based on type
            if recommendation.optimization_type == "confidence_threshold":
                new_value = float(recommendation.suggested_parameter)
                self.adaptive_parameters['confidence_threshold']['current'] = new_value
                
            elif recommendation.optimization_type == "strategy_change":
                new_strategy = OptimizationStrategy(recommendation.suggested_parameter)
                self.current_strategy = new_strategy
                
            elif recommendation.optimization_type == "retrieval_count":
                new_value = int(recommendation.suggested_parameter)
                self.adaptive_parameters['retrieval_count']['current'] = new_value
            
            logger.info(f"Applied optimization recommendation: {recommendation.optimization_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply recommendation {recommendation_id}: {e}")
            return False
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        
        recent_feedback = list(self.user_feedback)[-100:]
        satisfaction_scores = self._calculate_satisfaction_scores(recent_feedback)
        
        # Profile statistics
        profile_stats = {
            'total_profiles': len(self.response_profiles),
            'active_profiles': len([p for p in self.response_profiles.values() 
                                  if p.usage_count >= self.min_feedback_count]),
            'avg_feedback_score': statistics.mean([p.feedback_score for p in self.response_profiles.values()]) 
                                if self.response_profiles else 0.0
        }
        
        # Recent optimization activity
        recent_optimizations = len([rec for rec in self.optimization_history 
                                  if rec.timestamp > datetime.now() - timedelta(hours=24)])
        
        return {
            'current_strategy': self.current_strategy.value,
            'adaptive_parameters': self.adaptive_parameters,
            'satisfaction_scores': satisfaction_scores,
            'profile_statistics': profile_stats,
            'total_feedback_items': len(self.user_feedback),
            'recent_feedback_items': len(recent_feedback),
            'recent_optimizations': recent_optimizations,
            'learning_status': {
                'profiles_learned': len(self.response_profiles),
                'min_feedback_for_learning': self.min_feedback_count,
                'learning_rate': self.learning_rate
            }
        }
    
    def generate_user_satisfaction_report(self, timeframe_hours: int = 24) -> Dict[str, Any]:
        """Generate user satisfaction report"""
        
        cutoff_time = datetime.now() - timedelta(hours=timeframe_hours)
        period_feedback = [f for f in self.user_feedback if f.timestamp >= cutoff_time]
        
        if not period_feedback:
            return {'status': 'no_data', 'message': 'No feedback data for specified timeframe'}
        
        # Calculate satisfaction metrics
        satisfaction_scores = self._calculate_satisfaction_scores(period_feedback)
        
        # Feedback type distribution
        feedback_distribution = defaultdict(int)
        for feedback in period_feedback:
            feedback_distribution[feedback.feedback_type.value] += 1
        
        # Session-based analysis
        session_satisfaction = {}
        for session_id, session_feedback in self.feedback_by_session.items():
            session_period_feedback = [f for f in session_feedback if f.timestamp >= cutoff_time]
            if session_period_feedback:
                session_scores = self._calculate_satisfaction_scores(session_period_feedback)
                session_satisfaction[session_id] = session_scores.get('overall', 0.5)
        
        # Trend analysis
        satisfaction_trend = "stable"
        if len(period_feedback) > 10:
            first_half = period_feedback[:len(period_feedback)//2]
            second_half = period_feedback[len(period_feedback)//2:]
            
            first_satisfaction = self._calculate_satisfaction_scores(first_half).get('overall', 0.5)
            second_satisfaction = self._calculate_satisfaction_scores(second_half).get('overall', 0.5)
            
            change = (second_satisfaction - first_satisfaction) / first_satisfaction * 100
            if change > 5:
                satisfaction_trend = "improving"
            elif change < -5:
                satisfaction_trend = "declining"
        
        return {
            'timeframe_hours': timeframe_hours,
            'total_feedback_items': len(period_feedback),
            'satisfaction_scores': satisfaction_scores,
            'feedback_distribution': dict(feedback_distribution),
            'session_satisfaction': session_satisfaction,
            'satisfaction_trend': satisfaction_trend,
            'top_issues': self._identify_top_issues(period_feedback),
            'improvement_opportunities': self._identify_improvement_opportunities(period_feedback)
        }
    
    def _identify_top_issues(self, feedback_list: List[UserFeedback]) -> List[str]:
        """Identify top issues from negative feedback"""
        
        issues = []
        negative_feedback = [f for f in feedback_list 
                           if f.feedback_type in [FeedbackType.THUMBS_DOWN, FeedbackType.IMPLICIT_NEGATIVE]]
        
        if not negative_feedback:
            return ["No significant issues identified"]
        
        # Analyze patterns in negative feedback
        response_times = []
        confidences = []
        categories = defaultdict(int)
        
        for feedback in negative_feedback:
            metadata = feedback.response_metadata
            
            if 'response_time' in metadata:
                response_times.append(metadata['response_time'])
            if 'confidence' in metadata:
                confidences.append(metadata['confidence'])
            if 'category' in metadata:
                categories[metadata['category']] += 1
        
        # Identify issues
        if response_times and statistics.mean(response_times) > 8.0:
            issues.append(f"Slow response times (avg: {statistics.mean(response_times):.1f}s)")
        
        if confidences and statistics.mean(confidences) < 0.5:
            issues.append(f"Low confidence responses (avg: {statistics.mean(confidences):.2%})")
        
        if categories:
            most_problematic = max(categories.items(), key=lambda x: x[1])
            issues.append(f"Issues with {most_problematic[0]} category ({most_problematic[1]} complaints)")
        
        return issues if issues else ["No specific issues identified"]
    
    def _identify_improvement_opportunities(self, feedback_list: List[UserFeedback]) -> List[str]:
        """Identify improvement opportunities"""
        
        opportunities = []
        
        # Analyze response profiles
        underperforming_profiles = [
            p for p in self.response_profiles.values() 
            if p.feedback_score < 0.7 and p.usage_count >= self.min_feedback_count
        ]
        
        if underperforming_profiles:
            opportunities.append(f"Optimize {len(underperforming_profiles)} underperforming query patterns")
        
        # Check parameter optimization potential
        recent_params = defaultdict(list)
        for feedback in feedback_list:
            for key, value in feedback.response_metadata.items():
                if key.startswith('param_'):
                    recent_params[key].append(value)
        
        # Suggest parameter tuning if values are consistently at boundaries
        for param, values in recent_params.items():
            if len(set(values)) == 1:  # All same value
                opportunities.append(f"Consider tuning {param[6:]} parameter (currently static)")
        
        return opportunities if opportunities else ["System performing optimally"]
    
    def export_optimization_data(self, filename: str):
        """Export optimization data for analysis"""
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_strategy': self.current_strategy.value,
            'adaptive_parameters': self.adaptive_parameters,
            'response_profiles': {k: asdict(v) for k, v in self.response_profiles.items()},
            'user_feedback': [asdict(f) for f in self.user_feedback],
            'optimization_history': [asdict(r) for r in self.optimization_history],
            'satisfaction_report': self.generate_user_satisfaction_report(168)  # Last week
        }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=convert_datetime)
        
        logger.info(f"Optimization data exported to {filename}")


# Integration helper
class OptimizerIntegration:
    """Integration helper for response optimization"""
    
    def __init__(self, quality_monitor=None, analytics=None):
        self.optimizer = ResponseOptimizer(quality_monitor, analytics)
        
    def optimize_query_parameters(self, query: str, category: str, intent: str,
                                 base_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize query parameters before execution"""
        
        return self.optimizer.optimize_response_parameters(query, category, intent, base_parameters)
    
    def record_query_feedback(self, query_id: str, session_id: str,
                            user_action: str, response_metadata: Dict[str, Any]):
        """Record user feedback after query execution"""
        
        # Map user actions to feedback
        if user_action in ['thumbs_up', 'helpful', 'correct']:
            self.optimizer.record_feedback(
                query_id, session_id, FeedbackType.THUMBS_UP, True, 
                response_metadata=response_metadata
            )
        elif user_action in ['thumbs_down', 'unhelpful', 'incorrect']:
            self.optimizer.record_feedback(
                query_id, session_id, FeedbackType.THUMBS_DOWN, False,
                response_metadata=response_metadata
            )
        else:
            # Record as implicit feedback
            self.optimizer.record_implicit_feedback(
                query_id, session_id, user_action, response_metadata
            )
    
    def get_optimization_dashboard(self) -> Dict[str, Any]:
        """Get optimization dashboard data"""
        
        status = self.optimizer.get_optimization_status()
        satisfaction_report = self.optimizer.generate_user_satisfaction_report()
        
        return {
            'optimization_status': status,
            'satisfaction_report': satisfaction_report,
            'learning_progress': {
                'profiles_created': len(self.optimizer.response_profiles),
                'feedback_collected': len(self.optimizer.user_feedback),
                'optimization_active': status['satisfaction_scores'].get('overall', 0.5) > 0.5
            }
        }