"""
Question Normalization and Intent Classification
Handles multiple question forms for the same PCIe concepts
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from difflib import SequenceMatcher

class QuestionIntent(Enum):
    """Question intent categories"""
    DEFINITION = "definition"
    EXPLANATION = "explanation"
    PROCEDURE = "procedure"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    SPECIFICATION = "specification"
    EXAMPLE = "example"
    LIST = "list"

@dataclass
class NormalizedQuestion:
    """Normalized question with intent and keywords"""
    original_question: str
    normalized_form: str
    intent: QuestionIntent
    key_concepts: List[str]
    question_type: str
    confidence: float
    metadata: Dict[str, any]

@dataclass
class QuestionTemplate:
    """Template for question variations"""
    canonical_form: str
    variations: List[str]
    intent: QuestionIntent
    key_concepts: List[str]
    answer_keywords: List[str]

class QuestionNormalizer:
    """Normalizes questions to handle multiple forms of the same query"""
    
    def __init__(self):
        self.question_templates = self._build_question_templates()
        self.intent_patterns = self._build_intent_patterns()
        self.concept_synonyms = self._build_concept_synonyms()
    
    def normalize_question(self, question: str) -> NormalizedQuestion:
        """Normalize question to standard form"""
        
        # Clean and preprocess
        cleaned_question = self._clean_question(question)
        
        # Find matching template
        template_match = self._find_matching_template(cleaned_question)
        
        if template_match:
            template, confidence = template_match
            return NormalizedQuestion(
                original_question=question,
                normalized_form=template.canonical_form,
                intent=template.intent,
                key_concepts=template.key_concepts,
                question_type="template_match",
                confidence=confidence,
                metadata={"template": template.canonical_form}
            )
        
        # Fallback to intent-based normalization
        return self._intent_based_normalization(question, cleaned_question)
    
    def _build_question_templates(self) -> List[QuestionTemplate]:
        """Build database of question templates and variations"""
        return [
            # LTSSM Questions
            QuestionTemplate(
                canonical_form="What are the PCIe LTSSM states and their timeouts?",
                variations=[
                    "detail all pcie ltssm states with exact timeout values",
                    "list all ltssm states with timeouts",
                    "what are the pcie link training states and timeouts",
                    "describe ltssm state machine with timing",
                    "explain ltssm states and transition timeouts",
                    "what are all the ltssm states in pcie",
                    "ltssm state timeouts",
                    "link training state machine",
                    "pcie ltssm overview"
                ],
                intent=QuestionIntent.SPECIFICATION,
                key_concepts=["ltssm", "states", "timeouts", "link training"],
                answer_keywords=["detect", "polling", "configuration", "l0", "timeout", "ms"]
            ),
            
            # TLP Header Questions
            QuestionTemplate(
                canonical_form="What is the structure of a PCIe TLP header?",
                variations=[
                    "what are the exact bit field definitions for a 3dw memory request tlp header",
                    "describe the 3dw memory request tlp header format with exact bit positions",
                    "what is the structure of a 3dw memory tlp header",
                    "list all fields in a 3dw memory request tlp header",
                    "how is a 3dw memory request header formatted",
                    "tlp header format",
                    "memory request tlp structure",
                    "3dw tlp header",
                    "tlp header bit fields"
                ],
                intent=QuestionIntent.SPECIFICATION,
                key_concepts=["tlp", "header", "memory", "request", "bit fields"],
                answer_keywords=["dw0", "dw1", "dw2", "fmt", "type", "bit"]
            ),
            
            # Power States Questions
            QuestionTemplate(
                canonical_form="What are the PCIe power states and their characteristics?",
                variations=[
                    "detail all pcie power states with exact timing requirements",
                    "list all pcie power states with timing and power specs",
                    "what are pcie d-states and l-states",
                    "describe pcie power management states",
                    "explain pcie device and link power states",
                    "power states in pcie",
                    "device power states",
                    "link power states",
                    "aspm power states"
                ],
                intent=QuestionIntent.SPECIFICATION,
                key_concepts=["power", "states", "d-states", "l-states", "aspm"],
                answer_keywords=["d0", "d1", "d2", "d3", "l0", "l1", "l2", "aspm"]
            ),
            
            # Flow Control Questions
            QuestionTemplate(
                canonical_form="How does PCIe flow control work?",
                variations=[
                    "explain pcie flow control mechanism with all credit types",
                    "describe pcie flow control credit types",
                    "how does pcie credit-based flow control work",
                    "what are the 6 pcie credit types",
                    "explain pcie credit management",
                    "flow control in pcie",
                    "credit based flow control",
                    "pcie credits"
                ],
                intent=QuestionIntent.EXPLANATION,
                key_concepts=["flow control", "credits", "posted", "non-posted"],
                answer_keywords=["ph", "pd", "nph", "npd", "cplh", "cpld", "dllp"]
            ),
            
            # Error Handling Questions
            QuestionTemplate(
                canonical_form="How does PCIe error handling work?",
                variations=[
                    "provide the complete pcie aer capability register map",
                    "list all aer capability registers with bit definitions",
                    "what is the complete aer register layout",
                    "describe aer capability structure registers",
                    "explain complete aer register map",
                    "pcie error handling",
                    "aer capability",
                    "error reporting",
                    "correctable errors",
                    "uncorrectable errors"
                ],
                intent=QuestionIntent.SPECIFICATION,
                key_concepts=["error", "aer", "correctable", "uncorrectable"],
                answer_keywords=["aer", "error", "status", "mask", "severity", "capability"]
            ),
            
            # Configuration Space Questions
            QuestionTemplate(
                canonical_form="What is the PCIe configuration space layout?",
                variations=[
                    "provide complete pcie configuration space header layout",
                    "describe pcie configuration space header layout",
                    "what are all the fields in pcie config header",
                    "list pcie configuration header type 0 format",
                    "explain pcie endpoint configuration header",
                    "config space layout",
                    "configuration registers",
                    "pci config header"
                ],
                intent=QuestionIntent.SPECIFICATION,
                key_concepts=["configuration", "space", "header", "registers"],
                answer_keywords=["vendor", "device", "command", "status", "bar"]
            )
        ]
    
    def _build_intent_patterns(self) -> Dict[QuestionIntent, List[str]]:
        """Build patterns for intent classification"""
        return {
            QuestionIntent.DEFINITION: [
                r"what is", r"define", r"definition of", r"meaning of"
            ],
            QuestionIntent.EXPLANATION: [
                r"how does", r"explain", r"describe", r"tell me about"
            ],
            QuestionIntent.PROCEDURE: [
                r"how to", r"steps to", r"procedure", r"process"
            ],
            QuestionIntent.TROUBLESHOOTING: [
                r"troubleshoot", r"debug", r"fix", r"solve", r"diagnose"
            ],
            QuestionIntent.COMPARISON: [
                r"difference", r"compare", r"vs", r"versus", r"contrast"
            ],
            QuestionIntent.SPECIFICATION: [
                r"complete", r"all", r"list", r"exact", r"detailed", r"specification"
            ],
            QuestionIntent.EXAMPLE: [
                r"example", r"sample", r"instance", r"demonstrate"
            ],
            QuestionIntent.LIST: [
                r"list", r"enumerate", r"all", r"types of"
            ]
        }
    
    def _build_concept_synonyms(self) -> Dict[str, List[str]]:
        """Build synonyms for PCIe concepts"""
        return {
            "ltssm": ["link training", "state machine", "link states"],
            "tlp": ["transaction layer packet", "packet", "transaction"],
            "dllp": ["data link layer packet", "link layer packet"],
            "power": ["power management", "power states", "pm"],
            "error": ["error handling", "error reporting", "aer"],
            "config": ["configuration", "config space", "configuration space"],
            "flow": ["flow control", "credit", "credits"],
            "header": ["packet header", "tlp header"],
            "register": ["reg", "registers", "register map"]
        }
    
    def _clean_question(self, question: str) -> str:
        """Clean and standardize question text"""
        # Convert to lowercase
        cleaned = question.lower().strip()
        
        # Remove punctuation except meaningful ones
        cleaned = re.sub(r'[^\w\s\-\?\.]', ' ', cleaned)
        
        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove common question words at the end
        cleaned = re.sub(r'\s*\?+\s*$', '', cleaned)
        
        return cleaned
    
    def _find_matching_template(self, question: str) -> Optional[Tuple[QuestionTemplate, float]]:
        """Find best matching question template"""
        best_match = None
        best_score = 0.0
        
        question_words = set(re.findall(r'\w+', question))
        
        for template in self.question_templates:
            max_template_score = 0.0
            
            # Check canonical form
            canonical_score = self._calculate_similarity(question, template.canonical_form)
            max_template_score = max(max_template_score, canonical_score)
            
            # Check variations
            for variation in template.variations:
                variation_score = self._calculate_similarity(question, variation)
                max_template_score = max(max_template_score, variation_score)
            
            # Boost score for concept matches
            concept_boost = 0.0
            for concept in template.key_concepts:
                if concept in question:
                    concept_boost += 0.1
            
            total_score = max_template_score + concept_boost
            
            if total_score > best_score and total_score >= 0.5:
                best_score = total_score
                best_match = template
        
        return (best_match, best_score) if best_match else None
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Sequence similarity
        seq_sim = SequenceMatcher(None, text1, text2).ratio()
        
        # Word overlap similarity
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            word_sim = 0.0
        else:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            word_sim = intersection / union
        
        # Combined similarity
        return (seq_sim * 0.4) + (word_sim * 0.6)
    
    def _intent_based_normalization(self, original: str, cleaned: str) -> NormalizedQuestion:
        """Fallback normalization based on intent patterns"""
        
        # Classify intent
        intent = self._classify_intent(cleaned)
        
        # Extract key concepts
        concepts = self._extract_concepts(cleaned)
        
        # Generate normalized form
        normalized = self._generate_normalized_form(cleaned, intent, concepts)
        
        return NormalizedQuestion(
            original_question=original,
            normalized_form=normalized,
            intent=intent,
            key_concepts=concepts,
            question_type="intent_based",
            confidence=0.6,  # Lower confidence for fallback
            metadata={"method": "intent_classification"}
        )
    
    def _classify_intent(self, question: str) -> QuestionIntent:
        """Classify question intent based on patterns"""
        question_lower = question.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return intent
        
        # Default intent
        return QuestionIntent.EXPLANATION
    
    def _extract_concepts(self, question: str) -> List[str]:
        """Extract key PCIe concepts from question"""
        concepts = []
        question_lower = question.lower()
        
        # Direct concept matches
        pcie_concepts = [
            "ltssm", "tlp", "dllp", "power", "error", "aer", "config", "configuration",
            "flow", "control", "credit", "header", "register", "state", "timeout",
            "memory", "io", "completion", "message", "capability", "bar", "vendor",
            "device", "command", "status", "training", "link", "lane", "speed",
            "multi-function", "multi function", "multifunction", "ari", "flr", 
            "function", "shared resource", "arbitration"
        ]
        
        for concept in pcie_concepts:
            if concept in question_lower:
                concepts.append(concept)
        
        # Synonym expansion
        for concept in concepts.copy():
            if concept in self.concept_synonyms:
                for synonym in self.concept_synonyms[concept]:
                    if synonym in question_lower and synonym not in concepts:
                        concepts.append(synonym)
        
        return concepts
    
    def _generate_normalized_form(self, question: str, intent: QuestionIntent, concepts: List[str]) -> str:
        """Generate normalized question form"""
        
        if not concepts:
            return question
        
        # For better normalization, combine related concepts
        if len(concepts) >= 2:
            # Look for meaningful combinations
            combined_concepts = []
            if "completion" in concepts and "timeout" in concepts:
                combined_concepts.append("completion timeout")
            elif "error" in concepts and any(c in concepts for c in ["timeout", "tlp", "ltssm"]):
                error_type = next((c for c in concepts if c in ["timeout", "tlp", "ltssm"]), "")
                if error_type:
                    combined_concepts.append(f"{error_type} error")
            
            if combined_concepts:
                primary_concept = combined_concepts[0]
            else:
                # Use most specific concept (longer names are usually more specific)
                primary_concept = max(concepts, key=len)
        else:
            primary_concept = concepts[0] if concepts else "PCIe"
        
        # Generate based on intent with better context preservation
        if intent == QuestionIntent.DEFINITION:
            return f"What is {primary_concept}?"
        elif intent == QuestionIntent.EXPLANATION:
            if "error" in primary_concept or "timeout" in primary_concept:
                return f"How does {primary_concept} handling work?"
            return f"How does {primary_concept} work?"
        elif intent == QuestionIntent.SPECIFICATION:
            return f"What are the specifications for {primary_concept}?"
        elif intent == QuestionIntent.LIST:
            return f"List all {primary_concept} types"
        elif intent == QuestionIntent.TROUBLESHOOTING:
            return f"How to troubleshoot {primary_concept} issues?"
        elif intent == QuestionIntent.COMPARISON:
            if len(concepts) > 1:
                return f"Compare {concepts[0]} and {concepts[1]}"
            return f"What are the types of {primary_concept}?"
        else:
            # For complex technical questions, preserve more context
            if len(concepts) > 2:
                return f"Explain {primary_concept} in PCIe"
            return f"Explain {primary_concept}"
    
    def get_question_variations(self, normalized_question: str) -> List[str]:
        """Get known variations for a normalized question"""
        for template in self.question_templates:
            if template.canonical_form.lower() == normalized_question.lower():
                return template.variations
        return []
    
    def suggest_related_questions(self, concepts: List[str]) -> List[str]:
        """Suggest related questions based on concepts"""
        suggestions = []
        
        for template in self.question_templates:
            # Check if template shares concepts
            shared_concepts = set(concepts) & set(template.key_concepts)
            if shared_concepts:
                suggestions.append(template.canonical_form)
        
        return suggestions[:5]  # Limit to 5 suggestions