"""
PCIe Domain-Specific Knowledge Classification
Categorizes content and extracts PCIe-specific facts for enhanced RAG
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class PCIeCategory(Enum):
    """PCIe knowledge categories"""
    ERROR_HANDLING = "error_handling"
    LTSSM = "ltssm"
    TLP = "tlp"
    POWER_MANAGEMENT = "power_management"
    PHYSICAL_LAYER = "physical_layer"
    ARCHITECTURE = "architecture"
    CONFIGURATION = "configuration"
    FLOW_CONTROL = "flow_control"
    ADVANCED_FEATURES = "advanced_features"
    GENERAL = "general"

class QuestionType(Enum):
    """Question types for PCIe content"""
    DEFINITION = "definition"
    TECHNICAL_DETAILS = "technical_details"
    PROCESS = "process"
    TROUBLESHOOTING = "troubleshooting"
    ERROR_SPECIFIC = "error_specific"
    REGISTER_SPECIFIC = "register_specific"
    STATE_TRANSITION = "state_transition"
    COMPARISON = "comparison"

@dataclass
class PCIeFact:
    """Structured PCIe fact"""
    content: str
    fact_type: str  # register, error_code, timeout, definition, etc.
    category: PCIeCategory
    confidence: float
    metadata: Dict[str, str]

@dataclass
class PCIeKnowledgeItem:
    """Enhanced knowledge item with PCIe domain expertise"""
    content: str
    category: PCIeCategory
    question_types: List[QuestionType]
    facts: List[PCIeFact]
    keywords: List[str]
    difficulty: str  # basic, intermediate, advanced, expert
    source_metadata: Dict[str, str]

class PCIeKnowledgeClassifier:
    """Classifies and extracts PCIe domain knowledge"""
    
    def __init__(self):
        self.category_keywords = {
            PCIeCategory.ERROR_HANDLING: [
                'error', 'aer', 'correctable', 'uncorrectable', 'fatal', 'recovery',
                'err_cor', 'err_fatal', 'err_nonfatal', 'poisoned', 'malformed',
                'completion timeout', 'receiver overflow', 'timeout', 'response timeout',
                'transaction timeout', 'completion', 'cto', 'ur', 'ca', 'advisory', 'ecrc',
                'crs', 'configuration request retry', 'retry status', 'request retry'
            ],
            PCIeCategory.LTSSM: [
                'ltssm', 'state', 'detect', 'polling', 'configuration', 'l0', 'l1', 'l2',
                'training', 'link training', 'recovery', 'hot reset', 'compliance'
            ],
            PCIeCategory.TLP: [
                'tlp', 'transaction', 'packet', 'header', 'payload', 'memory', 'io', 
                'config', 'completion', 'message', 'fmt', 'type', 'length'
            ],
            PCIeCategory.POWER_MANAGEMENT: [
                'power', 'l0s', 'l1', 'l2', 'l3', 'aspm', 'clkreq', 'd0', 'd1', 'd2', 'd3',
                'power state', 'device power', 'link power'
            ],
            PCIeCategory.PHYSICAL_LAYER: [
                'physical', 'signal', 'integrity', 'lane', 'speed', 'generation', 'encoding',
                'scrambling', 'receiver', 'transmitter', 'differential'
            ],
            PCIeCategory.ARCHITECTURE: [
                'root', 'endpoint', 'switch', 'bridge', 'hierarchy', 'topology',
                'root complex', 'pci bridge', 'bus', 'device', 'function'
            ],
            PCIeCategory.CONFIGURATION: [
                'config', 'space', 'capability', 'register', 'base', 'address', 'bar',
                'vendor id', 'device id', 'command', 'status', 'flr', 'function level reset',
                'reset', 'function reset', 'secondary bus reset'
            ],
            PCIeCategory.FLOW_CONTROL: [
                'flow', 'control', 'credit', 'posted', 'non-posted', 'completion',
                'fc', 'updatefc', 'dllp', 'buffer'
            ],
            PCIeCategory.ADVANCED_FEATURES: [
                'sr-iov', 'ats', 'ari', 'tph', 'ide', 'doe', 'cxl', 'msi', 'msi-x',
                'virtual', 'atomic', 'extended capability'
            ]
        }
        
        # Fact extraction patterns
        self.fact_patterns = {
            'register_offset': r'Offset\s+(0x[0-9A-Fa-f]+):\s*(.+)',
            'bit_field': r'\[(\d+):(\d+)\]\s*(.+)',
            'single_bit': r'\[(\d+)\]\s*(.+)',
            'error_code': r'(\w+\s*Error|Error\s+\w+):\s*(.+)',
            'timeout': r'(\d+(?:\.\d+)?)\s*(ms|μs|us|seconds?|ns)\s*(timeout|delay|wait|timer)',
            'completion_timeout': r'(completion\s+timeout|CTO)\s*(?:errors?)?.*?(?:during|when|at)\s*(.+)',
            'flr_behavior': r'(flr|function\s+level\s+reset)\s*(?:during|when|at|causes?|results?)\s*(.+)',
            'crs_response': r'(crs|configuration\s+request\s+retry)\s*(?:status|return|response)\s*(.+)',
            'timeout_value': r'timeout\s*(?:of|=|:)\s*(\d+(?:\.\d+)?)\s*(ms|μs|us|seconds?|ns)',
            'error_during': r'(error|timeout|failure)\s+(?:during|when|at)\s+(.+)',
            'hex_value': r'(0x[0-9A-Fa-f]+)h?',
            'state_name': r'([A-Z][a-z]+(?:\.[A-Z][a-z]+)*)\s*state',
            'capability_id': r'Capability\s+ID\s*=\s*(0x[0-9A-Fa-f]+)',
            'version': r'Version\s*=\s*(\d+)h?'
        }
    
    def classify_content(self, content: str, metadata: Dict[str, str] = None) -> PCIeKnowledgeItem:
        """Classify content and extract PCIe knowledge"""
        
        # Determine primary category
        category = self._categorize_content(content)
        
        # Extract facts
        facts = self._extract_facts(content, category)
        
        # Determine question types
        question_types = self._identify_question_types(content, facts)
        
        # Extract keywords
        keywords = self._extract_keywords(content, category)
        
        # Determine difficulty
        difficulty = self._assess_difficulty(content, facts)
        
        return PCIeKnowledgeItem(
            content=content,
            category=category,
            question_types=question_types,
            facts=facts,
            keywords=keywords,
            difficulty=difficulty,
            source_metadata=metadata or {}
        )
    
    def _categorize_content(self, content: str) -> PCIeCategory:
        """Determine primary category based on keyword analysis with query context priority"""
        content_lower = content.lower()
        scores = {}
        
        # Check if this is contextual content (query | retrieved_content)
        is_contextual = '|' in content
        
        if is_contextual:
            # Split into query and content parts
            parts = content.split('|', 1)
            query_part = parts[0].strip().lower() if len(parts) > 0 else ""
            content_part = parts[1].strip().lower() if len(parts) > 1 else ""
            
            # Score with priority to query intent
            for category, keywords in self.category_keywords.items():
                query_score = sum(query_part.count(keyword.lower()) for keyword in keywords)
                content_score = sum(content_part.count(keyword.lower()) for keyword in keywords)
                
                # Give higher weight to query keywords, especially for specific categories
                if category == PCIeCategory.ERROR_HANDLING and query_score > 0:
                    # Strong boost for error handling in queries
                    total_score = query_score * 3 + content_score
                elif query_score > 0:
                    # Moderate boost for other categories in queries
                    total_score = query_score * 2 + content_score
                else:
                    # Regular scoring for content-only matches
                    total_score = content_score
                    
                scores[category] = total_score
        else:
            # Regular scoring for non-contextual content
            for category, keywords in self.category_keywords.items():
                score = sum(content_lower.count(keyword.lower()) for keyword in keywords)
                scores[category] = score
        
        if not scores or max(scores.values()) == 0:
            return PCIeCategory.GENERAL
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _extract_facts(self, content: str, category: PCIeCategory) -> List[PCIeFact]:
        """Extract structured facts from content"""
        facts = []
        
        for fact_type, pattern in self.fact_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                fact_content = match.group(0)
                confidence = self._calculate_fact_confidence(fact_type, match)
                
                metadata = {
                    'pattern': fact_type,
                    'groups': list(match.groups()),
                    'position': f"{match.start()}-{match.end()}"
                }
                
                facts.append(PCIeFact(
                    content=fact_content,
                    fact_type=fact_type,
                    category=category,
                    confidence=confidence,
                    metadata=metadata
                ))
        
        return facts
    
    def _identify_question_types(self, content: str, facts: List[PCIeFact]) -> List[QuestionType]:
        """Identify potential question types for this content"""
        question_types = []
        content_lower = content.lower()
        
        # Definition questions
        if any(word in content_lower for word in ['is', 'define', 'definition', 'means']):
            question_types.append(QuestionType.DEFINITION)
        
        # Technical details
        if facts or any(word in content_lower for word in ['bit', 'field', 'register', 'offset']):
            question_types.append(QuestionType.TECHNICAL_DETAILS)
        
        # Process/procedure
        if any(word in content_lower for word in ['step', 'process', 'procedure', 'sequence', 'how']):
            question_types.append(QuestionType.PROCESS)
        
        # Troubleshooting
        if any(word in content_lower for word in ['troubleshoot', 'debug', 'analyze', 'solve', 'diagnose']):
            question_types.append(QuestionType.TROUBLESHOOTING)
        
        # Error-specific
        if any(fact.fact_type == 'error_code' for fact in facts):
            question_types.append(QuestionType.ERROR_SPECIFIC)
        
        # Register-specific
        if any(fact.fact_type in ['register_offset', 'bit_field'] for fact in facts):
            question_types.append(QuestionType.REGISTER_SPECIFIC)
        
        # State transition
        if any(word in content_lower for word in ['transition', 'state', 'entry', 'exit']):
            question_types.append(QuestionType.STATE_TRANSITION)
        
        # Default to technical details if no specific type identified
        if not question_types:
            question_types.append(QuestionType.TECHNICAL_DETAILS)
        
        return question_types
    
    def _extract_keywords(self, content: str, category: PCIeCategory) -> List[str]:
        """Extract relevant keywords for search optimization"""
        keywords = []
        
        # Add category-specific keywords that appear in content
        if category in self.category_keywords:
            content_lower = content.lower()
            for keyword in self.category_keywords[category]:
                if keyword.lower() in content_lower:
                    keywords.append(keyword)
        
        # Extract technical terms (ALL CAPS acronyms)
        acronyms = re.findall(r'\b[A-Z]{2,}\b', content)
        keywords.extend(acronyms)
        
        # Extract hex values
        hex_values = re.findall(r'0x[0-9A-Fa-f]+', content)
        keywords.extend(hex_values)
        
        # Extract register names
        register_names = re.findall(r'\b\w+\s+Register\b', content, re.IGNORECASE)
        keywords.extend(register_names)
        
        return list(set(keywords))  # Remove duplicates
    
    def _assess_difficulty(self, content: str, facts: List[PCIeFact]) -> str:
        """Assess difficulty level of content"""
        # Calculate complexity score
        score = 0
        
        # Technical facts increase difficulty
        score += len(facts) * 2
        
        # Specific patterns indicate higher difficulty
        if re.search(r'\[\d+:\d+\]', content):  # Bit fields
            score += 5
        if re.search(r'0x[0-9A-Fa-f]+', content):  # Hex values
            score += 3
        if len(re.findall(r'\b[A-Z]{3,}\b', content)) > 2:  # Multiple acronyms
            score += 4
        
        # Length and complexity
        word_count = len(content.split())
        if word_count > 200:
            score += 3
        if word_count > 500:
            score += 5
        
        # Categorize difficulty
        if score < 5:
            return "basic"
        elif score < 15:
            return "intermediate" 
        elif score < 25:
            return "advanced"
        else:
            return "expert"
    
    def _calculate_fact_confidence(self, fact_type: str, match) -> float:
        """Calculate confidence score for extracted fact"""
        # Base confidence by fact type
        base_confidence = {
            'register_offset': 0.9,
            'bit_field': 0.85,
            'error_code': 0.8,
            'timeout': 0.75,
            'completion_timeout': 0.9,
            'timeout_value': 0.85,
            'error_during': 0.8,
            'hex_value': 0.7,
            'capability_id': 0.95
        }
        
        confidence = base_confidence.get(fact_type, 0.6)
        
        # Adjust based on match context
        match_text = match.group(0)
        if len(match_text) < 10:  # Very short matches are less reliable
            confidence *= 0.8
        if len(match_text) > 100:  # Very long matches might be overly broad
            confidence *= 0.9
            
        return min(confidence, 1.0)