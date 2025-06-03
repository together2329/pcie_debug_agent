#!/usr/bin/env python3
"""
Query Expansion Engine for Phase 2 Advanced Features

Provides intelligent query expansion with:
- Technical acronym expansion
- PCIe domain synonym detection
- Context-based query rewriting
- Intent-driven enhancement
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .pcie_knowledge_classifier import (
    PCIeKnowledgeClassifier, 
    PCIeCategory, 
    QueryIntent
)

@dataclass
class ExpandedQuery:
    """Expanded query with metadata"""
    original_query: str
    expanded_query: str
    category: PCIeCategory
    intent: QueryIntent
    context_hints: List[str]
    acronyms_expanded: List[str]
    synonyms_added: List[str]
    confidence: float

class QueryExpansionEngine:
    """Intelligent query expansion for PCIe domain"""
    
    def __init__(self):
        self.classifier = PCIeKnowledgeClassifier()
        
        # Domain-specific synonyms for query expansion
        self.pcie_synonyms = {
            # Common variations
            'pci express': ['pcie', 'pci-e', 'pci-express'],
            'pcie': ['pci express', 'pci-express'],
            
            # Error terms
            'timeout': ['time out', 'timed out', 'timing out'],
            'error': ['failure', 'fault', 'problem', 'issue'],
            'completion': ['complete', 'completed', 'completing'],
            
            # Technical terms
            'header': ['hdr', 'packet header', 'tlp header'],
            'register': ['reg', 'configuration register'],
            'capability': ['cap', 'extended capability'],
            'endpoint': ['ep', 'device'],
            'root complex': ['rc', 'root'],
            
            # States
            'training': ['link training', 'training sequence'],
            'configuration': ['config', 'configuration process'],
            'polling': ['poll', 'polling state'],
            'detection': ['detect', 'detection state'],
            
            # Operations
            'reset': ['function reset', 'device reset'],
            'initialization': ['init', 'setup', 'initialization process'],
            'enumeration': ['discovery', 'device enumeration'],
            
            # Analysis terms
            'debug': ['debugging', 'troubleshoot', 'diagnose'],
            'analyze': ['analysis', 'examination', 'investigation'],
            'verify': ['verification', 'validation', 'check'],
        }
        
        # Context-specific query patterns
        self.query_patterns = {
            'compliance_check': [
                r'\b(?:is|does|should|must|required?)\b.*?\b(?:compliant?|spec|standard)\b',
                r'\b(?:verify|check|validate)\b.*?\b(?:compliance|conformance)\b',
                r'\b(?:according to|per|as per)\b.*?\bspec\b'
            ],
            'troubleshooting': [
                r'\b(?:why|how to fix|what causes?)\b.*?\b(?:error|problem|issue|fail)\b',
                r'\b(?:debug|troubleshoot|solve|resolve)\b',
                r'\b(?:not working|broken|failing)\b'
            ],
            'implementation': [
                r'\b(?:how to|implement|build|create|setup)\b',
                r'\b(?:configure|enable|program|code)\b',
                r'\b(?:step by step|procedure|process)\b'
            ],
            'reference_lookup': [
                r'\b(?:what is|define|list|show|values?)\b',
                r'\b(?:register|offset|bit|field)\b.*?\b(?:value|meaning)\b',
                r'\b(?:quick|reference|lookup|table)\b'
            ]
        }
        
    def expand_query(self, query: str, max_expansion_ratio: float = 2.0) -> ExpandedQuery:
        """
        Expand query with domain intelligence
        
        Args:
            query: Original user query
            max_expansion_ratio: Maximum allowed expansion (expanded/original length)
            
        Returns:
            ExpandedQuery with enhanced information
        """
        # Classify query first
        category = self.classifier._categorize_content(query)
        intent = self.classifier.classify_query_intent(query)
        
        # Start with original query
        expanded_query = query.strip()
        acronyms_expanded = []
        synonyms_added = []
        
        # 1. Expand technical acronyms
        expanded_query, expanded_acronyms = self._expand_acronyms(expanded_query)
        acronyms_expanded.extend(expanded_acronyms)
        
        # 2. Add domain synonyms
        expanded_query, added_synonyms = self._add_domain_synonyms(expanded_query, category)
        synonyms_added.extend(added_synonyms)
        
        # 3. Add context-specific terms
        expanded_query = self._add_context_terms(expanded_query, category, intent)
        
        # 4. Apply query-pattern specific enhancements
        expanded_query = self._apply_pattern_enhancements(expanded_query, intent)
        
        # 5. Check expansion ratio and trim if necessary
        if len(expanded_query) > len(query) * max_expansion_ratio:
            expanded_query = self._trim_expansion(query, expanded_query, max_expansion_ratio)
        
        # Generate context hints
        context_hints = self.classifier.generate_context_hints(query, category, intent)
        
        # Calculate expansion confidence
        confidence = self._calculate_expansion_confidence(
            query, expanded_query, category, intent
        )
        
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            category=category,
            intent=intent,
            context_hints=context_hints,
            acronyms_expanded=acronyms_expanded,
            synonyms_added=synonyms_added,
            confidence=confidence
        )
    
    def _expand_acronyms(self, query: str) -> Tuple[str, List[str]]:
        """Expand technical acronyms in query"""
        expanded_query = query
        expanded_acronyms = []
        
        for acronym, expansions in self.classifier.acronym_expansions.items():
            # Look for whole word matches (case insensitive)
            pattern = r'\b' + re.escape(acronym) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                primary_expansion = expansions[0]
                # Add expansion if not already present
                if primary_expansion.lower() not in expanded_query.lower():
                    expanded_query += f" {primary_expansion}"
                    expanded_acronyms.append(f"{acronym} → {primary_expansion}")
        
        return expanded_query, expanded_acronyms
    
    def _add_domain_synonyms(self, query: str, category: PCIeCategory) -> Tuple[str, List[str]]:
        """Add relevant domain synonyms"""
        expanded_query = query
        added_synonyms = []
        query_lower = query.lower()
        
        for base_term, synonyms in self.pcie_synonyms.items():
            if base_term in query_lower:
                # Add most relevant synonym based on category
                best_synonym = self._select_best_synonym(base_term, synonyms, category)
                if best_synonym and best_synonym.lower() not in expanded_query.lower():
                    expanded_query += f" {best_synonym}"
                    added_synonyms.append(f"{base_term} + {best_synonym}")
        
        return expanded_query, added_synonyms
    
    def _select_best_synonym(self, base_term: str, synonyms: List[str], category: PCIeCategory) -> Optional[str]:
        """Select most relevant synonym based on category"""
        # Category-specific synonym preferences
        category_preferences = {
            PCIeCategory.ERROR_HANDLING: {
                'error': 'failure',
                'timeout': 'timed out',
                'completion': 'completed'
            },
            PCIeCategory.DEBUGGING: {
                'debug': 'troubleshoot',
                'analyze': 'investigation'
            },
            PCIeCategory.CONFIGURATION: {
                'register': 'configuration register',
                'capability': 'extended capability'
            },
            PCIeCategory.COMPLIANCE: {
                'verify': 'validation',
                'check': 'compliance check'
            }
        }
        
        preferences = category_preferences.get(category, {})
        preferred = preferences.get(base_term)
        
        if preferred and preferred in synonyms:
            return preferred
        
        # Default to first synonym
        return synonyms[0] if synonyms else None
    
    def _add_context_terms(self, query: str, category: PCIeCategory, intent: QueryIntent) -> str:
        """Add context-specific terms based on category and intent"""
        expanded_query = query
        query_lower = query.lower()
        
        # Category-specific context additions
        context_additions = {
            PCIeCategory.ERROR_HANDLING: {
                'common_terms': ['error handling', 'recovery', 'reporting'],
                'intent_specific': {
                    QueryIntent.TROUBLESHOOT: ['root cause', 'debug steps'],
                    QueryIntent.VERIFY: ['error checking', 'validation']
                }
            },
            PCIeCategory.COMPLIANCE: {
                'common_terms': ['specification', 'standard', 'conformance'],
                'intent_specific': {
                    QueryIntent.VERIFY: ['compliance testing', 'validation'],
                    QueryIntent.REFERENCE: ['spec reference', 'requirements']
                }
            },
            PCIeCategory.LTSSM: {
                'common_terms': ['state machine', 'link training'],
                'intent_specific': {
                    QueryIntent.LEARN: ['state transitions', 'training sequence'],
                    QueryIntent.TROUBLESHOOT: ['training failure', 'state analysis']
                }
            },
            PCIeCategory.DEBUGGING: {
                'common_terms': ['analysis', 'investigation', 'diagnostics'],
                'intent_specific': {
                    QueryIntent.TROUBLESHOOT: ['debug tools', 'troubleshooting steps'],
                    QueryIntent.IMPLEMENT: ['debug setup', 'monitoring']
                }
            }
        }
        
        if category in context_additions:
            category_data = context_additions[category]
            
            # Add common terms
            for term in category_data.get('common_terms', []):
                if term not in query_lower and len(expanded_query.split()) < 20:
                    expanded_query += f" {term}"
            
            # Add intent-specific terms
            intent_terms = category_data.get('intent_specific', {}).get(intent, [])
            for term in intent_terms:
                if term not in query_lower and len(expanded_query.split()) < 20:
                    expanded_query += f" {term}"
        
        return expanded_query
    
    def _apply_pattern_enhancements(self, query: str, intent: QueryIntent) -> str:
        """Apply query pattern-specific enhancements"""
        expanded_query = query
        query_lower = query.lower()
        
        # Intent-specific enhancements
        if intent == QueryIntent.TROUBLESHOOT:
            if 'error' in query_lower and 'how to fix' not in query_lower:
                expanded_query += " troubleshooting solution"
            if 'problem' in query_lower and 'debug' not in query_lower:
                expanded_query += " debug analysis"
                
        elif intent == QueryIntent.VERIFY:
            if 'compliance' not in query_lower:
                expanded_query += " compliance verification"
            if 'specification' not in query_lower and 'spec' not in query_lower:
                expanded_query += " specification requirements"
                
        elif intent == QueryIntent.IMPLEMENT:
            if 'setup' not in query_lower and 'configuration' not in query_lower:
                expanded_query += " implementation guide"
            if 'steps' not in query_lower:
                expanded_query += " step-by-step"
                
        elif intent == QueryIntent.REFERENCE:
            if 'register' in query_lower and 'offset' not in query_lower:
                expanded_query += " register offset"
            if 'value' in query_lower and 'meaning' not in query_lower:
                expanded_query += " field meaning"
        
        return expanded_query
    
    def _trim_expansion(self, original: str, expanded: str, max_ratio: float) -> str:
        """Trim expansion to stay within ratio limits"""
        max_length = int(len(original) * max_ratio)
        
        if len(expanded) <= max_length:
            return expanded
        
        # Split into original and added parts
        added_part = expanded[len(original):].strip()
        added_words = added_part.split()
        
        # Keep most important added words (prioritize technical terms)
        technical_words = []
        other_words = []
        
        for word in added_words:
            if any(term in word.lower() for term in ['pcie', 'flr', 'crs', 'ltssm', 'error', 'timeout']):
                technical_words.append(word)
            else:
                other_words.append(word)
        
        # Combine original with most important additions
        remaining_space = max_length - len(original) - 1  # -1 for space
        
        trimmed_additions = []
        current_length = 0
        
        # Add technical words first
        for word in technical_words:
            if current_length + len(word) + 1 <= remaining_space:
                trimmed_additions.append(word)
                current_length += len(word) + 1
        
        # Add other words if space allows
        for word in other_words:
            if current_length + len(word) + 1 <= remaining_space:
                trimmed_additions.append(word)
                current_length += len(word) + 1
        
        if trimmed_additions:
            return f"{original} {' '.join(trimmed_additions)}"
        else:
            return original
    
    def _calculate_expansion_confidence(self, original: str, expanded: str, 
                                      category: PCIeCategory, intent: QueryIntent) -> float:
        """Calculate confidence in the expansion quality"""
        confidence = 0.5  # Base confidence
        
        # Length factor (moderate expansion is good)
        expansion_ratio = len(expanded) / len(original)
        if 1.2 <= expansion_ratio <= 2.0:
            confidence += 0.2
        elif expansion_ratio < 1.2:
            confidence += 0.1  # Minimal expansion
        else:
            confidence -= 0.1  # Over-expansion
        
        # Category confidence (some categories expand better)
        category_confidence = {
            PCIeCategory.ERROR_HANDLING: 0.2,
            PCIeCategory.COMPLIANCE: 0.15,
            PCIeCategory.DEBUGGING: 0.15,
            PCIeCategory.LTSSM: 0.1,
            PCIeCategory.CONFIGURATION: 0.1
        }
        confidence += category_confidence.get(category, 0.0)
        
        # Intent confidence
        intent_confidence = {
            QueryIntent.TROUBLESHOOT: 0.15,
            QueryIntent.VERIFY: 0.1,
            QueryIntent.LEARN: 0.1,
            QueryIntent.IMPLEMENT: 0.05
        }
        confidence += intent_confidence.get(intent, 0.0)
        
        # Technical term presence
        technical_terms = ['pcie', 'flr', 'crs', 'ltssm', 'aer', 'tlp', 'error', 'timeout']
        terms_found = sum(1 for term in technical_terms if term in expanded.lower())
        confidence += min(terms_found * 0.05, 0.15)
        
        return min(confidence, 1.0)
    
    def get_expansion_statistics(self) -> Dict[str, int]:
        """Get statistics about available expansions"""
        return {
            'acronym_expansions': len(self.classifier.acronym_expansions),
            'synonym_groups': len(self.pcie_synonyms),
            'categories': len(list(PCIeCategory)),
            'intents': len(list(QueryIntent)),
            'query_patterns': sum(len(patterns) for patterns in self.query_patterns.values())
        }