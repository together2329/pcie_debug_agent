#!/usr/bin/env python3
"""
Context Memory System for Phase 3 Intelligence Layer

Provides session-aware context memory for improved responses
and user experience continuity across conversation sessions.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context information"""
    USER_PREFERENCE = "user_preference"
    QUERY_HISTORY = "query_history"
    DOMAIN_EXPERTISE = "domain_expertise"
    SESSION_STATE = "session_state"
    PROBLEM_CONTEXT = "problem_context"
    LEARNING_CONTEXT = "learning_context"

@dataclass
class ContextItem:
    """Individual context memory item"""
    item_id: str
    context_type: ContextType
    key: str
    value: Any
    confidence: float
    session_id: str
    timestamp: datetime
    expiry: Optional[datetime] = None
    usage_count: int = 0
    
@dataclass
class SessionContext:
    """Complete session context"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    query_count: int
    context_items: Dict[str, ContextItem]
    session_summary: str
    expertise_level: str
    primary_domain: str

class ContextMemory:
    """Session-aware context memory system"""
    
    def __init__(self, max_sessions: int = 1000, max_items_per_session: int = 100):
        self.max_sessions = max_sessions
        self.max_items_per_session = max_items_per_session
        
        # Memory storage
        self.sessions: Dict[str, SessionContext] = {}  # session_id -> context
        self.global_context: Dict[str, ContextItem] = {}  # Global persistent context
        self.user_profiles: Dict[str, Dict] = {}  # user_id -> profile
        
        # Context patterns for learning
        self.context_patterns = defaultdict(lambda: {'count': 0, 'success_rate': 0.0})
        
        # Memory management
        self.cleanup_interval = timedelta(hours=1)
        self.last_cleanup = datetime.now()
        
    def create_session(self, session_id: str, user_id: str = None) -> SessionContext:
        """Create a new session context"""
        
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            query_count=0,
            context_items={},
            session_summary="",
            expertise_level="intermediate",  # Default
            primary_domain="general"
        )
        
        self.sessions[session_id] = session
        
        # Load user profile if available
        if user_id and user_id in self.user_profiles:
            self._load_user_profile(session)
        
        logger.info(f"Created session context: {session_id}")
        return session
    
    def update_context(self, session_id: str, context_type: ContextType,
                      key: str, value: Any, confidence: float = 1.0,
                      expiry_hours: int = None):
        """Update context information for a session"""
        
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now()
        
        # Create context item
        item_id = f"{session_id}_{context_type.value}_{key}_{int(time.time())}"
        expiry = datetime.now() + timedelta(hours=expiry_hours) if expiry_hours else None
        
        context_item = ContextItem(
            item_id=item_id,
            context_type=context_type,
            key=key,
            value=value,
            confidence=confidence,
            session_id=session_id,
            timestamp=datetime.now(),
            expiry=expiry,
            usage_count=0
        )
        
        # Store in session context
        session.context_items[key] = context_item
        
        # Manage session size
        if len(session.context_items) > self.max_items_per_session:
            self._trim_session_context(session)
        
        # Update session metadata
        self._update_session_metadata(session, context_type, key, value)
        
        logger.debug(f"Updated context: {session_id} -> {key}: {value}")
    
    def get_context(self, session_id: str, key: str = None,
                   context_type: ContextType = None) -> Optional[Any]:
        """Retrieve context information"""
        
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if key:
            # Get specific context item
            if key in session.context_items:
                item = session.context_items[key]
                
                # Check expiry
                if item.expiry and datetime.now() > item.expiry:
                    del session.context_items[key]
                    return None
                
                # Update usage
                item.usage_count += 1
                return item.value
            
            # Check global context
            if key in self.global_context:
                item = self.global_context[key]
                if not item.expiry or datetime.now() <= item.expiry:
                    item.usage_count += 1
                    return item.value
            
            return None
        
        elif context_type:
            # Get all items of specific type
            items = {}
            for item_key, item in session.context_items.items():
                if item.context_type == context_type:
                    if not item.expiry or datetime.now() <= item.expiry:
                        items[item_key] = item.value
                        item.usage_count += 1
            return items
        
        else:
            # Get all valid context
            context = {}
            for item_key, item in session.context_items.items():
                if not item.expiry or datetime.now() <= item.expiry:
                    context[item_key] = item.value
                    item.usage_count += 1
            return context
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""
        
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        # Query patterns
        query_history = self.get_context(session_id, context_type=ContextType.QUERY_HISTORY) or {}
        
        # User preferences
        preferences = self.get_context(session_id, context_type=ContextType.USER_PREFERENCE) or {}
        
        # Domain expertise
        expertise = self.get_context(session_id, context_type=ContextType.DOMAIN_EXPERTISE) or {}
        
        # Problem context
        problems = self.get_context(session_id, context_type=ContextType.PROBLEM_CONTEXT) or {}
        
        return {
            'session_id': session_id,
            'duration': (datetime.now() - session.start_time).total_seconds(),
            'query_count': session.query_count,
            'expertise_level': session.expertise_level,
            'primary_domain': session.primary_domain,
            'query_patterns': query_history,
            'user_preferences': preferences,
            'domain_expertise': expertise,
            'current_problems': problems,
            'context_items_count': len(session.context_items)
        }
    
    def enhance_query_with_context(self, session_id: str, query: str,
                                 category: str = None, intent: str = None) -> Dict[str, Any]:
        """Enhance query with relevant context"""
        
        enhanced_context = {
            'original_query': query,
            'enhanced_query': query,
            'context_additions': [],
            'user_context': {}
        }
        
        if session_id not in self.sessions:
            return enhanced_context
        
        session = self.sessions[session_id]
        session.query_count += 1
        
        # Add user preferences context
        preferences = self.get_context(session_id, context_type=ContextType.USER_PREFERENCE) or {}
        if preferences:
            enhanced_context['user_context']['preferences'] = preferences
            
            # Apply preference-based enhancements
            if preferences.get('detail_level') == 'detailed':
                enhanced_context['enhanced_query'] += " with detailed explanation"
                enhanced_context['context_additions'].append("detailed explanation requested")
            elif preferences.get('detail_level') == 'concise':
                enhanced_context['enhanced_query'] += " concise answer"
                enhanced_context['context_additions'].append("concise response preferred")
        
        # Add domain expertise context
        expertise = self.get_context(session_id, context_type=ContextType.DOMAIN_EXPERTISE) or {}
        if expertise:
            enhanced_context['user_context']['expertise'] = expertise
            
            # Adjust complexity based on expertise
            if session.expertise_level == 'beginner':
                enhanced_context['enhanced_query'] += " explain simply"
                enhanced_context['context_additions'].append("beginner-level explanation")
            elif session.expertise_level == 'expert':
                enhanced_context['enhanced_query'] += " technical details"
                enhanced_context['context_additions'].append("expert-level technical details")
        
        # Add problem context if relevant
        problems = self.get_context(session_id, context_type=ContextType.PROBLEM_CONTEXT) or {}
        if problems and category in ['error_handling', 'debugging']:
            current_problem = problems.get('current_issue')
            if current_problem:
                enhanced_context['enhanced_query'] += f" related to {current_problem}"
                enhanced_context['context_additions'].append(f"problem context: {current_problem}")
                enhanced_context['user_context']['current_problem'] = current_problem
        
        # Add query history context
        query_history = self.get_context(session_id, context_type=ContextType.QUERY_HISTORY) or {}
        if query_history:
            recent_queries = list(query_history.values())[-3:]  # Last 3 queries
            enhanced_context['user_context']['recent_queries'] = recent_queries
            
            # Check for follow-up patterns
            if any('error' in q.lower() for q in recent_queries) and 'fix' in query.lower():
                enhanced_context['enhanced_query'] += " continuation of error troubleshooting"
                enhanced_context['context_additions'].append("follow-up to error investigation")
        
        # Store this query in history
        self.update_context(session_id, ContextType.QUERY_HISTORY, 
                          f"query_{session.query_count}", query)
        
        return enhanced_context
    
    def learn_from_interaction(self, session_id: str, query: str, response: Dict[str, Any],
                             user_feedback: str = None, success: bool = True):
        """Learn from user interaction to improve context"""
        
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Learn user preferences from successful interactions
        if success and response:
            # Response length preference
            response_length = len(response.get('answer', ''))
            if response_length < 200:
                self._update_preference(session_id, 'detail_level', 'concise', 0.1)
            elif response_length > 800:
                self._update_preference(session_id, 'detail_level', 'detailed', 0.1)
            
            # Confidence preference (infer from response acceptance)
            confidence = response.get('confidence', 0.5)
            if confidence > 0.8:
                self._update_preference(session_id, 'confidence_threshold', 'high', 0.1)
            
            # Category expertise learning
            category = response.get('category', 'general')
            if category != 'general':
                self._update_expertise(session_id, category, 0.05)
        
        # Learn from explicit feedback
        if user_feedback:
            if 'too long' in user_feedback.lower():
                self._update_preference(session_id, 'detail_level', 'concise', 0.2)
            elif 'more detail' in user_feedback.lower():
                self._update_preference(session_id, 'detail_level', 'detailed', 0.2)
            elif 'too technical' in user_feedback.lower():
                self._adjust_expertise_level(session_id, -1)
            elif 'more technical' in user_feedback.lower():
                self._adjust_expertise_level(session_id, +1)
        
        # Update session summary
        self._update_session_summary(session)
    
    def _update_preference(self, session_id: str, preference_key: str, 
                          preference_value: str, weight: float):
        """Update user preference with weighted learning"""
        
        current_pref = self.get_context(session_id, preference_key, ContextType.USER_PREFERENCE)
        
        if current_pref and isinstance(current_pref, dict):
            # Update existing preference weights
            if preference_value in current_pref:
                current_pref[preference_value] += weight
            else:
                current_pref[preference_value] = weight
        else:
            # Create new preference
            current_pref = {preference_value: weight}
        
        # Normalize weights
        total_weight = sum(current_pref.values())
        if total_weight > 0:
            normalized_pref = {k: v/total_weight for k, v in current_pref.items()}
            
            # Get dominant preference
            dominant_pref = max(normalized_pref.items(), key=lambda x: x[1])
            if dominant_pref[1] > 0.6:  # 60% confidence threshold
                final_value = dominant_pref[0]
            else:
                final_value = normalized_pref  # Keep weighted options
            
            self.update_context(session_id, ContextType.USER_PREFERENCE, 
                              preference_key, final_value, dominant_pref[1])
    
    def _update_expertise(self, session_id: str, domain: str, increment: float):
        """Update domain expertise level"""
        
        current_expertise = self.get_context(session_id, domain, ContextType.DOMAIN_EXPERTISE) or 0.0
        new_expertise = min(1.0, current_expertise + increment)
        
        self.update_context(session_id, ContextType.DOMAIN_EXPERTISE, 
                          domain, new_expertise, new_expertise)
        
        # Update primary domain
        session = self.sessions[session_id]
        all_expertise = self.get_context(session_id, context_type=ContextType.DOMAIN_EXPERTISE) or {}
        if all_expertise:
            primary_domain = max(all_expertise.items(), key=lambda x: x[1])[0]
            session.primary_domain = primary_domain
    
    def _adjust_expertise_level(self, session_id: str, adjustment: int):
        """Adjust overall expertise level"""
        
        session = self.sessions[session_id]
        levels = ['beginner', 'intermediate', 'advanced', 'expert']
        current_index = levels.index(session.expertise_level) if session.expertise_level in levels else 1
        
        new_index = max(0, min(len(levels)-1, current_index + adjustment))
        session.expertise_level = levels[new_index]
        
        self.update_context(session_id, ContextType.USER_PREFERENCE, 
                          'expertise_level', session.expertise_level)
    
    def _update_session_summary(self, session: SessionContext):
        """Update session summary with key insights"""
        
        # Analyze context for summary
        query_count = session.query_count
        duration = (datetime.now() - session.start_time).total_seconds() / 60  # minutes
        
        # Get dominant categories
        query_history = self.get_context(session.session_id, context_type=ContextType.QUERY_HISTORY) or {}
        categories = []
        
        for query in query_history.values():
            if isinstance(query, str):
                # Simple category detection
                if 'error' in query.lower() or 'problem' in query.lower():
                    categories.append('troubleshooting')
                elif 'how to' in query.lower() or 'implement' in query.lower():
                    categories.append('implementation')
                elif 'what' in query.lower() or 'define' in query.lower():
                    categories.append('learning')
                else:
                    categories.append('general')
        
        dominant_category = max(set(categories), key=categories.count) if categories else 'general'
        
        # Generate summary
        session.session_summary = f"{query_count} queries in {duration:.1f}m, focus: {dominant_category}, level: {session.expertise_level}"
    
    def _load_user_profile(self, session: SessionContext):
        """Load user profile into session context"""
        
        if not session.user_id or session.user_id not in self.user_profiles:
            return
        
        profile = self.user_profiles[session.user_id]
        
        # Load preferences
        for key, value in profile.get('preferences', {}).items():
            self.update_context(session.session_id, ContextType.USER_PREFERENCE, 
                              key, value, confidence=0.8)
        
        # Load expertise
        for domain, level in profile.get('expertise', {}).items():
            self.update_context(session.session_id, ContextType.DOMAIN_EXPERTISE, 
                              domain, level, confidence=0.9)
        
        # Set session metadata
        session.expertise_level = profile.get('overall_expertise', 'intermediate')
        session.primary_domain = profile.get('primary_domain', 'general')
    
    def save_user_profile(self, session_id: str):
        """Save session learning to user profile"""
        
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        if not session.user_id:
            return
        
        # Create or update user profile
        if session.user_id not in self.user_profiles:
            self.user_profiles[session.user_id] = {
                'preferences': {},
                'expertise': {},
                'overall_expertise': 'intermediate',
                'primary_domain': 'general',
                'last_updated': datetime.now().isoformat()
            }
        
        profile = self.user_profiles[session.user_id]
        
        # Update preferences
        preferences = self.get_context(session_id, context_type=ContextType.USER_PREFERENCE) or {}
        for key, value in preferences.items():
            if isinstance(value, (str, int, float, bool)):
                profile['preferences'][key] = value
        
        # Update expertise
        expertise = self.get_context(session_id, context_type=ContextType.DOMAIN_EXPERTISE) or {}
        for domain, level in expertise.items():
            if isinstance(level, (int, float)):
                profile['expertise'][domain] = level
        
        # Update metadata
        profile['overall_expertise'] = session.expertise_level
        profile['primary_domain'] = session.primary_domain
        profile['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Saved user profile for {session.user_id}")
    
    def _trim_session_context(self, session: SessionContext):
        """Trim session context to stay within limits"""
        
        # Sort items by usage and recency
        items = list(session.context_items.items())
        
        # Score items (higher is better)
        scored_items = []
        for key, item in items:
            age_hours = (datetime.now() - item.timestamp).total_seconds() / 3600
            score = item.usage_count / max(1, age_hours) * item.confidence
            scored_items.append((score, key, item))
        
        # Keep top items
        scored_items.sort(reverse=True)
        keep_count = int(self.max_items_per_session * 0.8)  # Keep 80% of limit
        
        new_context = {}
        for i, (score, key, item) in enumerate(scored_items):
            if i < keep_count:
                new_context[key] = item
        
        session.context_items = new_context
        logger.debug(f"Trimmed session {session.session_id} context to {len(new_context)} items")
    
    def cleanup_expired_context(self):
        """Clean up expired context items"""
        
        if datetime.now() - self.last_cleanup < self.cleanup_interval:
            return
        
        cleanup_count = 0
        
        # Clean session contexts
        for session in self.sessions.values():
            expired_keys = []
            for key, item in session.context_items.items():
                if item.expiry and datetime.now() > item.expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del session.context_items[key]
                cleanup_count += 1
        
        # Clean global context
        expired_keys = []
        for key, item in self.global_context.items():
            if item.expiry and datetime.now() > item.expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.global_context[key]
            cleanup_count += 1
        
        # Remove old sessions (inactive for > 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        old_sessions = [sid for sid, session in self.sessions.items() 
                       if session.last_activity < cutoff_time]
        
        for session_id in old_sessions:
            # Save profile before removing session
            self.save_user_profile(session_id)
            del self.sessions[session_id]
            cleanup_count += 1
        
        self.last_cleanup = datetime.now()
        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired context items/sessions")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get memory system status"""
        
        active_sessions = len(self.sessions)
        total_context_items = sum(len(s.context_items) for s in self.sessions.values())
        
        # Session statistics
        if self.sessions:
            avg_session_duration = statistics.mean([
                (datetime.now() - s.start_time).total_seconds() / 60 
                for s in self.sessions.values()
            ])
            avg_queries_per_session = statistics.mean([s.query_count for s in self.sessions.values()])
        else:
            avg_session_duration = 0
            avg_queries_per_session = 0
        
        # User profiles
        total_users = len(self.user_profiles)
        
        return {
            'active_sessions': active_sessions,
            'total_context_items': total_context_items,
            'global_context_items': len(self.global_context),
            'total_user_profiles': total_users,
            'avg_session_duration_minutes': avg_session_duration,
            'avg_queries_per_session': avg_queries_per_session,
            'memory_limits': {
                'max_sessions': self.max_sessions,
                'max_items_per_session': self.max_items_per_session
            },
            'last_cleanup': self.last_cleanup.isoformat()
        }


# Integration helper
class ContextMemoryIntegration:
    """Integration helper for context memory"""
    
    def __init__(self):
        self.memory = ContextMemory()
        
    def enhance_query_with_memory(self, session_id: str, query: str, 
                                 category: str = None, intent: str = None) -> Tuple[str, Dict[str, Any]]:
        """Enhance query with context memory"""
        
        enhanced = self.memory.enhance_query_with_context(session_id, query, category, intent)
        return enhanced['enhanced_query'], enhanced['user_context']
    
    def record_interaction(self, session_id: str, query: str, response: Dict[str, Any],
                         execution_time: float, user_feedback: str = None):
        """Record interaction for learning"""
        
        success = 'error' not in response and response.get('confidence', 0) > 0.5
        
        # Record session state
        self.memory.update_context(session_id, ContextType.SESSION_STATE, 
                                 'last_response_time', execution_time)
        
        # Record problem context if error-related
        if response.get('category') == 'error_handling':
            problem_desc = query[:100]  # First 100 chars as problem description
            self.memory.update_context(session_id, ContextType.PROBLEM_CONTEXT,
                                     'current_issue', problem_desc, expiry_hours=4)
        
        # Learn from interaction
        self.memory.learn_from_interaction(session_id, query, response, user_feedback, success)
    
    def get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """Get session insights for dashboard"""
        
        summary = self.memory.get_session_summary(session_id)
        status = self.memory.get_memory_status()
        
        return {
            'session_summary': summary,
            'memory_status': status,
            'context_available': len(summary.get('user_preferences', {})) > 0
        }