"""
Answer Verification and Confidence Scoring System
Validates RAG answers and provides confidence metrics
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
from difflib import SequenceMatcher

class VerificationMethod(Enum):
    """Answer verification methods"""
    EXACT_MATCH = "exact_match"
    KEYWORD_MATCH = "keyword_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FACT_VERIFICATION = "fact_verification"
    CONTEXT_MATCH = "context_match"

@dataclass
class AnswerVerification:
    """Result of answer verification"""
    confidence: float
    verification_method: VerificationMethod
    matched_keywords: List[str]
    missing_keywords: List[str]
    fact_accuracy: float
    explanation: str
    metadata: Dict[str, any]

@dataclass
class VerifiedAnswer:
    """Answer with verification information"""
    content: str
    acceptable_forms: List[str]
    required_keywords: List[str]
    required_facts: List[str]
    category: str
    confidence_threshold: float

class AnswerVerifier:
    """Verifies RAG answers against known correct answers and quality metrics"""
    
    def __init__(self):
        self.verified_answers = self._build_verified_answer_database()
        self.fact_patterns = {
            'register': r'(0x[0-9A-Fa-f]+)',
            'bit_field': r'\[(\d+):(\d+)\]|\[(\d+)\]',
            'timeout': r'(\d+(?:\.\d+)?)\s*(ms|μs|us|seconds?)',
            'error_type': r'(\w+\s*Error|Error\s+\w+)',
            'state': r'([A-Z][a-z]+(?:\.[A-Z][a-z]+)*)\s*state',
            'tlp_type': r'(Memory|IO|Configuration|Message)\s*(Read|Write)?'
        }
    
    def verify_answer(self, 
                     question: str, 
                     answer: str, 
                     retrieved_context: List[Dict],
                     expected_keywords: List[str] = None) -> AnswerVerification:
        """Comprehensive answer verification"""
        
        # Find best matching verified answer
        verified_match = self._find_verified_answer(question)
        
        if verified_match:
            return self._verify_against_known_answer(answer, verified_match)
        
        # Fallback to heuristic verification
        return self._heuristic_verification(answer, question, retrieved_context, expected_keywords)
    
    def _build_verified_answer_database(self) -> List[VerifiedAnswer]:
        """Build database of verified correct answers"""
        return [
            # 3DW Memory Request TLP Header
            VerifiedAnswer(
                content="3DW Memory Request Header: DW0 [31:29] Fmt=000 (3DW no data) or 010 (3DW with data), [28:24] Type=00000 (MRd) or 00001 (MWr), [23] T=0, [22:20] TC (Traffic Class), [19] T=0, [18:17] Attr[1:0], [16] LN=0, [15] TH=0, [14] TD (TLP Digest Present), [13] EP (Error Poisoned), [12:11] Attr[3:2], [10:9] AT (Address Type), [8:0] Length in DW. DW1 [31:16] Requester ID, [15:8] Tag, [7:4] Last DW BE, [3:0] First DW BE. DW2 [31:2] Address[31:2], [1:0] Reserved=00.",
                acceptable_forms=[
                    "3dw memory request header",
                    "3dw memory tlp header",
                    "memory request tlp format",
                    "memory request header structure"
                ],
                required_keywords=["3dw", "memory", "request", "header", "dw0", "dw1", "dw2", "fmt", "type"],
                required_facts=["[31:29]", "[28:24]", "[31:16]", "[15:8]", "[31:2]"],
                category="tlp",
                confidence_threshold=0.8
            ),
            
            # LTSSM States
            VerifiedAnswer(
                content="LTSSM States and Timeouts: Detect.Quiet (indefinite, Tx disabled) → Detect.Active (12ms timeout, receiver detection using load test) → Polling.Active (24ms timeout, send TS1, achieve bit lock and symbol lock) → Polling.Configuration (32ms timeout, send/receive TS2, confirm symbol lock) → Configuration.Linkwidth.Start (32ms timeout, negotiate link width using TS1) → Configuration.Complete (2ms timeout) → L0 (normal operation). Recovery (24ms timeout) for error recovery.",
                acceptable_forms=[
                    "ltssm states",
                    "link training states",
                    "ltssm state machine",
                    "pcie link states"
                ],
                required_keywords=["ltssm", "detect", "polling", "configuration", "l0", "timeout", "training"],
                required_facts=["12ms", "24ms", "32ms", "2ms", "Detect.Active", "Polling.Active"],
                category="ltssm",
                confidence_threshold=0.7
            ),
            
            # Power States
            VerifiedAnswer(
                content="PCIe Power States: Device Power States: D0 (Fully Functional), D1 (optional, context preserved), D2 (optional, greater power reduction), D3hot (wake-up capable, configuration space accessible), D3cold (no power, not wake-up capable). Link Power States: L0 (Active), L0s (Standby, <1μs), L1 (Sleep, <10μs), L2 (Deep Sleep), L3 (Off). ASPM L0s: immediate entry. ASPM L1: requires handshaking.",
                acceptable_forms=[
                    "pcie power states",
                    "device power states",
                    "link power states",
                    "d-states and l-states"
                ],
                required_keywords=["power", "states", "d0", "d1", "d2", "d3", "l0", "l1", "l2", "aspm"],
                required_facts=["D0", "D3hot", "D3cold", "L0s", "L1", "<1μs", "<10μs"],
                category="power_management",
                confidence_threshold=0.7
            ),
            
            # Flow Control
            VerifiedAnswer(
                content="PCIe uses credit-based flow control with 6 credit types: Posted Header (PH), Posted Data (PD), Non-Posted Header (NPH), Non-Posted Data (NPD), Completion Header (CplH), Completion Data (CplD). Credits are advertised during link initialization via FC DLLPs. Transmitter tracks available credits, decrements on TLP transmission, increments on UpdateFC DLLP receipt.",
                acceptable_forms=[
                    "pcie flow control",
                    "credit based flow control",
                    "flow control mechanism",
                    "pcie credits"
                ],
                required_keywords=["flow", "control", "credit", "posted", "non-posted", "completion", "dllp"],
                required_facts=["PH", "PD", "NPH", "NPD", "CplH", "CplD", "UpdateFC"],
                category="flow_control",
                confidence_threshold=0.8
            )
        ]
    
    def _find_verified_answer(self, question: str) -> Optional[VerifiedAnswer]:
        """Find matching verified answer for question"""
        question_lower = question.lower()
        
        best_match = None
        best_score = 0.0
        
        for verified in self.verified_answers:
            for acceptable_form in verified.acceptable_forms:
                # Calculate similarity
                similarity = SequenceMatcher(None, question_lower, acceptable_form.lower()).ratio()
                
                # Check for keyword overlap
                question_words = set(re.findall(r'\w+', question_lower))
                form_words = set(re.findall(r'\w+', acceptable_form.lower()))
                overlap = len(question_words & form_words)
                total = len(question_words | form_words)
                keyword_score = overlap / total if total > 0 else 0
                
                # Combined score
                combined_score = (similarity * 0.6) + (keyword_score * 0.4)
                
                if combined_score > best_score and combined_score >= 0.3:
                    best_score = combined_score
                    best_match = verified
        
        return best_match
    
    def _verify_against_known_answer(self, answer: str, verified: VerifiedAnswer) -> AnswerVerification:
        """Verify answer against known correct answer"""
        answer_lower = answer.lower()
        verified_lower = verified.content.lower()
        
        # Exact match check
        if answer.strip() == verified.content.strip():
            return AnswerVerification(
                confidence=1.0,
                verification_method=VerificationMethod.EXACT_MATCH,
                matched_keywords=verified.required_keywords,
                missing_keywords=[],
                fact_accuracy=1.0,
                explanation="Perfect exact match with verified answer",
                metadata={"verified_answer_id": verified.category}
            )
        
        # Keyword verification
        matched_keywords = []
        missing_keywords = []
        
        for keyword in verified.required_keywords:
            if keyword.lower() in answer_lower:
                matched_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        keyword_score = len(matched_keywords) / len(verified.required_keywords)
        
        # Fact verification
        fact_matches = 0
        for fact in verified.required_facts:
            if fact in answer:
                fact_matches += 1
        
        fact_accuracy = fact_matches / len(verified.required_facts) if verified.required_facts else 1.0
        
        # Overall confidence
        confidence = (keyword_score * 0.6) + (fact_accuracy * 0.4)
        
        # Determine verification method
        if fact_accuracy > 0.8:
            method = VerificationMethod.FACT_VERIFICATION
        elif keyword_score > 0.7:
            method = VerificationMethod.KEYWORD_MATCH
        else:
            method = VerificationMethod.CONTEXT_MATCH
        
        explanation = f"Keyword match: {keyword_score:.1%}, Fact accuracy: {fact_accuracy:.1%}"
        
        return AnswerVerification(
            confidence=confidence,
            verification_method=method,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            fact_accuracy=fact_accuracy,
            explanation=explanation,
            metadata={
                "verified_answer_id": verified.category,
                "keyword_score": keyword_score,
                "fact_score": fact_accuracy
            }
        )
    
    def _heuristic_verification(self, 
                               answer: str, 
                               question: str, 
                               context: List[Dict],
                               expected_keywords: List[str] = None) -> AnswerVerification:
        """Fallback heuristic verification when no verified answer exists"""
        
        confidence_factors = []
        
        # 1. Length appropriateness
        word_count = len(answer.split())
        if 10 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 10:
            length_score = word_count / 10  # Too short
        else:
            length_score = max(0.5, 200 / word_count)  # Too long
        confidence_factors.append(("length", length_score, 0.1))
        
        # 2. Technical content detection
        tech_score = self._assess_technical_content(answer)
        confidence_factors.append(("technical", tech_score, 0.2))
        
        # 3. Fact consistency
        fact_score = self._verify_facts(answer)
        confidence_factors.append(("facts", fact_score, 0.3))
        
        # 4. Context relevance
        context_score = self._assess_context_relevance(answer, context)
        confidence_factors.append(("context", context_score, 0.2))
        
        # 5. Expected keywords
        keyword_score = 1.0
        matched_keywords = []
        missing_keywords = []
        
        if expected_keywords:
            answer_lower = answer.lower()
            for keyword in expected_keywords:
                if keyword.lower() in answer_lower:
                    matched_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            keyword_score = len(matched_keywords) / len(expected_keywords)
        
        confidence_factors.append(("keywords", keyword_score, 0.2))
        
        # Calculate weighted confidence
        total_confidence = sum(score * weight for _, score, weight in confidence_factors)
        
        # Generate explanation
        factor_explanations = [f"{name}: {score:.1%}" for name, score, _ in confidence_factors]
        explanation = f"Heuristic verification - {', '.join(factor_explanations)}"
        
        return AnswerVerification(
            confidence=total_confidence,
            verification_method=VerificationMethod.SEMANTIC_SIMILARITY,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            fact_accuracy=fact_score,
            explanation=explanation,
            metadata={
                "confidence_factors": dict((name, score) for name, score, _ in confidence_factors)
            }
        )
    
    def _assess_technical_content(self, answer: str) -> float:
        """Assess technical content quality"""
        score = 0.0
        
        # Check for technical patterns
        if re.search(r'0x[0-9A-Fa-f]+', answer):  # Hex values
            score += 0.2
        if re.search(r'\[\d+:\d+\]|\[\d+\]', answer):  # Bit fields
            score += 0.3
        if re.search(r'\b[A-Z]{2,}\b', answer):  # Acronyms
            score += 0.2
        if re.search(r'\d+\s*(ms|μs|us|seconds?)', answer):  # Timeouts
            score += 0.2
        if re.search(r'\w+\s*(Register|TLP|DLLP|Error)', answer, re.IGNORECASE):  # Technical terms
            score += 0.1
        
        return min(score, 1.0)
    
    def _verify_facts(self, answer: str) -> float:
        """Verify factual consistency"""
        facts_found = 0
        total_checks = 0
        
        for pattern_name, pattern in self.fact_patterns.items():
            matches = re.findall(pattern, answer, re.IGNORECASE)
            if matches:
                facts_found += 1
            total_checks += 1
        
        # Additional consistency checks
        consistency_score = 1.0
        
        # Check for contradictory information
        if "impossible" in answer.lower() or "cannot" in answer.lower():
            consistency_score *= 0.8
        
        if total_checks > 0:
            fact_density = facts_found / total_checks
        else:
            fact_density = 0.5  # Neutral if no technical patterns
        
        return fact_density * consistency_score
    
    def _assess_context_relevance(self, answer: str, context: List[Dict]) -> float:
        """Assess how well answer matches retrieved context"""
        if not context:
            return 0.5  # Neutral if no context
        
        answer_words = set(re.findall(r'\w+', answer.lower()))
        context_words = set()
        
        for ctx in context:
            content = ctx.get('content', '')
            ctx_words = set(re.findall(r'\w+', content.lower()))
            context_words.update(ctx_words)
        
        if not context_words:
            return 0.5
        
        overlap = len(answer_words & context_words)
        total_answer_words = len(answer_words)
        
        relevance = overlap / total_answer_words if total_answer_words > 0 else 0
        return min(relevance, 1.0)
    
    def get_confidence_explanation(self, verification: AnswerVerification) -> str:
        """Get human-readable confidence explanation"""
        if verification.confidence >= 0.9:
            level = "Very High"
        elif verification.confidence >= 0.7:
            level = "High"
        elif verification.confidence >= 0.5:
            level = "Medium"
        elif verification.confidence >= 0.3:
            level = "Low"
        else:
            level = "Very Low"
        
        return f"{level} confidence ({verification.confidence:.1%}) - {verification.explanation}"