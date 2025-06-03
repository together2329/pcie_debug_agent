#!/usr/bin/env python3
"""
PCIe Compliance Intelligence Engine

Provides instant compliance violation detection, specification reference lookup,
and compliance-specific analysis for PCIe debugging scenarios.
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

@dataclass
class ComplianceViolation:
    """Detected compliance violation"""
    violation_id: str
    violation_type: str
    description: str
    severity: str  # HIGH, MEDIUM, LOW
    spec_reference: str
    detected_text: str
    confidence: float
    remediation: str
    detection_timestamp: datetime

@dataclass
class SpecificationReference:
    """PCIe specification reference"""
    section: str
    title: str
    chapter: str
    version: str
    page: Optional[int] = None
    excerpt: Optional[str] = None

class ViolationType(Enum):
    """Types of PCIe compliance violations"""
    FLR_VIOLATION = "flr_violation"
    CRS_VIOLATION = "crs_violation"
    TIMEOUT_VIOLATION = "timeout_violation"
    PROTOCOL_VIOLATION = "protocol_violation"
    POWER_VIOLATION = "power_violation"
    ORDERING_VIOLATION = "ordering_violation"
    FLOW_CONTROL_VIOLATION = "flow_control_violation"
    ERROR_HANDLING_VIOLATION = "error_handling_violation"
    CONFIGURATION_VIOLATION = "configuration_violation"

class ComplianceIntelligence:
    """PCIe compliance intelligence and violation detection engine"""
    
    def __init__(self):
        self.violation_patterns = self._initialize_violation_patterns()
        self.spec_references = self._initialize_spec_references()
        self.compliance_rules = self._initialize_compliance_rules()
        
    def _initialize_violation_patterns(self) -> Dict[ViolationType, List[Dict]]:
        """Initialize comprehensive violation detection patterns"""
        return {
            ViolationType.FLR_VIOLATION: [
                {
                    'pattern': r'(?:successful\s+)?completion.*?(?:during|while|in).*?(?:flr|function\s+level\s+reset)',
                    'id': 'FLR_001',
                    'description': 'Device sending completion during Function Level Reset',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 6.6.2',
                    'remediation': 'Device must stop processing requests and not send completions during FLR',
                    'confidence_base': 0.9
                },
                {
                    'pattern': r'(?:flr|function\s+level\s+reset).*?(?:timeout|exceed.*?100ms|takes.*?long)',
                    'id': 'FLR_002', 
                    'description': 'FLR exceeding 100ms timeout requirement',
                    'severity': 'MEDIUM',
                    'spec_ref': 'PCIe Base Spec 6.6.2',
                    'remediation': 'FLR must complete within 100ms; check device implementation',
                    'confidence_base': 0.85
                },
                {
                    'pattern': r'(?:memory|io|config).*?request.*?(?:during|while).*?(?:flr|function\s+level\s+reset)',
                    'id': 'FLR_003',
                    'description': 'Device responding to requests during FLR',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 6.6.2',
                    'remediation': 'Device must not respond to memory, I/O, or config requests during FLR',
                    'confidence_base': 0.88
                },
                {
                    'pattern': r'(?:interrupt|msi).*?(?:during|while).*?(?:flr|function\s+level\s+reset)',
                    'id': 'FLR_004',
                    'description': 'Device generating interrupts during FLR',
                    'severity': 'MEDIUM',
                    'spec_ref': 'PCIe Base Spec 6.6.2',
                    'remediation': 'Device must not generate interrupts during FLR',
                    'confidence_base': 0.8
                }
            ],
            
            ViolationType.CRS_VIOLATION: [
                {
                    'pattern': r'(?:crs|configuration\s+request\s+retry).*?(?:excessive|too\s+many|loop|stuck)',
                    'id': 'CRS_001',
                    'description': 'Excessive CRS responses indicating potential compliance issue',
                    'severity': 'MEDIUM',
                    'spec_ref': 'PCIe Base Spec 2.3.2',
                    'remediation': 'Check device readiness; CRS should be temporary during initialization',
                    'confidence_base': 0.75
                },
                {
                    'pattern': r'(?:crs|configuration\s+request\s+retry).*?(?:after|beyond).*?(?:enumeration|initialization)',
                    'id': 'CRS_002',
                    'description': 'CRS responses continuing after device initialization',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 2.3.2',
                    'remediation': 'Device should complete initialization and stop returning CRS',
                    'confidence_base': 0.85
                }
            ],
            
            ViolationType.TIMEOUT_VIOLATION: [
                {
                    'pattern': r'completion\s+timeout.*?(?:less\s+than|below|under)\s*(?:50|25).*?(?:μs|us|microsecond)',
                    'id': 'CTO_001',
                    'description': 'Completion timeout value below minimum specification',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 2.9.2',
                    'remediation': 'Completion timeout must be at least 50μs',
                    'confidence_base': 0.9
                },
                {
                    'pattern': r'completion\s+timeout.*?(?:exceed|above|over)\s*64.*?(?:s|second)',
                    'id': 'CTO_002',
                    'description': 'Completion timeout value exceeds maximum specification',
                    'severity': 'MEDIUM',
                    'spec_ref': 'PCIe Base Spec 2.9.2',
                    'remediation': 'Completion timeout should not exceed 64 seconds',
                    'confidence_base': 0.85
                }
            ],
            
            ViolationType.PROTOCOL_VIOLATION: [
                {
                    'pattern': r'(?:tlp|transaction).*?(?:malformed|invalid\s+format|bad\s+header)',
                    'id': 'PROTO_001',
                    'description': 'Malformed TLP detected',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 2.2',
                    'remediation': 'Check TLP format compliance and header construction',
                    'confidence_base': 0.9
                },
                {
                    'pattern': r'(?:posted|non-posted|completion).*?(?:ordering|reorder).*?violation',
                    'id': 'PROTO_002',
                    'description': 'Transaction ordering violation',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 2.4',
                    'remediation': 'Review transaction ordering rules and implementation',
                    'confidence_base': 0.85
                }
            ],
            
            ViolationType.POWER_VIOLATION: [
                {
                    'pattern': r'(?:l0s|l1).*?(?:exit|entry).*?(?:timeout|exceed|violation)',
                    'id': 'PWR_001',
                    'description': 'ASPM state transition timeout violation',
                    'severity': 'MEDIUM',
                    'spec_ref': 'PCIe Base Spec 5.2',
                    'remediation': 'Check ASPM timing parameters and device capability',
                    'confidence_base': 0.8
                }
            ],
            
            ViolationType.FLOW_CONTROL_VIOLATION: [
                {
                    'pattern': r'(?:flow\s+control|credit).*?(?:violation|overflow|underflow)',
                    'id': 'FC_001',
                    'description': 'Flow control credit violation',
                    'severity': 'HIGH',
                    'spec_ref': 'PCIe Base Spec 3.6',
                    'remediation': 'Review credit management and DLLP handling',
                    'confidence_base': 0.9
                }
            ]
        }
    
    def _initialize_spec_references(self) -> Dict[str, List[SpecificationReference]]:
        """Initialize comprehensive specification references"""
        return {
            'flr': [
                SpecificationReference(
                    section='6.6.2',
                    title='Function Level Reset (FLR)',
                    chapter='Conventional PCI Compatibility',
                    version='6.2',
                    excerpt='Function Level Reset provides a way for software to reset a specific Function without affecting other Functions in the device'
                ),
                SpecificationReference(
                    section='6.6.2.1',
                    title='FLR Mechanism',
                    chapter='Conventional PCI Compatibility', 
                    version='6.2',
                    excerpt='The FLR mechanism allows software to reset an individual Function'
                )
            ],
            'crs': [
                SpecificationReference(
                    section='2.3.2',
                    title='Configuration Request Retry Status (CRS)',
                    chapter='Transaction Layer',
                    version='6.2',
                    excerpt='CRS enables a Function to signal that it is not ready to respond to Configuration Requests'
                )
            ],
            'completion_timeout': [
                SpecificationReference(
                    section='2.9.2',
                    title='Completion Timeout',
                    chapter='Transaction Layer',
                    version='6.2',
                    excerpt='Requesters must implement a timeout mechanism for Non-Posted Requests'
                )
            ],
            'tlp_format': [
                SpecificationReference(
                    section='2.2',
                    title='Transaction Layer Packet (TLP) Format',
                    chapter='Transaction Layer',
                    version='6.2',
                    excerpt='TLPs are used to communicate transactions between components'
                )
            ],
            'flow_control': [
                SpecificationReference(
                    section='3.6',
                    title='Flow Control',
                    chapter='Data Link Layer',
                    version='6.2',
                    excerpt='Flow Control ensures that TLPs are not dropped due to buffer overflow'
                )
            ],
            'ltssm': [
                SpecificationReference(
                    section='4.2',
                    title='Link Training and Status State Machine (LTSSM)',
                    chapter='Physical Layer',
                    version='6.2',
                    excerpt='The LTSSM controls the Physical Layer initialization, training, and operational states'
                )
            ],
            'power_management': [
                SpecificationReference(
                    section='5.2',
                    title='Link Power Management',
                    chapter='Power Management',
                    version='6.2',
                    excerpt='Link Power Management allows the Link to enter low power states'
                )
            ]
        }
    
    def _initialize_compliance_rules(self) -> List[Dict]:
        """Initialize compliance rules for automated checking"""
        return [
            {
                'rule_id': 'RULE_001',
                'description': 'FLR must complete within 100ms',
                'category': 'timing',
                'severity': 'HIGH',
                'spec_ref': 'PCIe Base Spec 6.6.2'
            },
            {
                'rule_id': 'RULE_002', 
                'description': 'No completions during FLR',
                'category': 'protocol',
                'severity': 'HIGH',
                'spec_ref': 'PCIe Base Spec 6.6.2'
            },
            {
                'rule_id': 'RULE_003',
                'description': 'CRS only during initialization',
                'category': 'protocol',
                'severity': 'MEDIUM',
                'spec_ref': 'PCIe Base Spec 2.3.2'
            },
            {
                'rule_id': 'RULE_004',
                'description': 'Completion timeout minimum 50μs',
                'category': 'timing',
                'severity': 'HIGH',
                'spec_ref': 'PCIe Base Spec 2.9.2'
            }
        ]
    
    def detect_violations(self, content: str, context: str = "") -> List[ComplianceViolation]:
        """Detect compliance violations in content"""
        violations = []
        combined_content = f"{content} {context}".lower()
        
        for violation_type, patterns in self.violation_patterns.items():
            for pattern_info in patterns:
                matches = list(re.finditer(pattern_info['pattern'], combined_content, re.IGNORECASE))
                
                for match in matches:
                    confidence = self._calculate_violation_confidence(
                        pattern_info, match, combined_content
                    )
                    
                    violation = ComplianceViolation(
                        violation_id=pattern_info['id'],
                        violation_type=violation_type.value,
                        description=pattern_info['description'],
                        severity=pattern_info['severity'],
                        spec_reference=pattern_info['spec_ref'],
                        detected_text=match.group(0),
                        confidence=confidence,
                        remediation=pattern_info['remediation'],
                        detection_timestamp=datetime.now()
                    )
                    violations.append(violation)
        
        # Remove duplicates and sort by severity/confidence
        violations = self._deduplicate_violations(violations)
        violations.sort(key=lambda v: (
            {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[v.severity],
            v.confidence
        ), reverse=True)
        
        return violations
    
    def _calculate_violation_confidence(self, pattern_info: Dict, match, content: str) -> float:
        """Calculate confidence for detected violation"""
        base_confidence = pattern_info['confidence_base']
        
        # Adjust based on context
        match_text = match.group(0)
        
        # Longer matches are generally more reliable
        if len(match_text) > 50:
            base_confidence += 0.05
        elif len(match_text) < 20:
            base_confidence -= 0.05
        
        # Check for supporting context
        window_start = max(0, match.start() - 100)
        window_end = min(len(content), match.end() + 100)
        context_window = content[window_start:window_end]
        
        # Look for supporting technical terms
        supporting_terms = ['error', 'timeout', 'failure', 'violation', 'specification', 'compliance']
        support_count = sum(1 for term in supporting_terms if term in context_window)
        
        if support_count >= 2:
            base_confidence += 0.1
        elif support_count == 0:
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.1), 1.0)
    
    def _deduplicate_violations(self, violations: List[ComplianceViolation]) -> List[ComplianceViolation]:
        """Remove duplicate violations"""
        seen = set()
        unique_violations = []
        
        for violation in violations:
            key = (violation.violation_id, violation.detected_text[:50])
            if key not in seen:
                seen.add(key)
                unique_violations.append(violation)
        
        return unique_violations
    
    def get_specification_reference(self, topic: str) -> List[SpecificationReference]:
        """Get specification references for a topic"""
        topic_lower = topic.lower()
        
        # Direct topic lookup
        if topic_lower in self.spec_references:
            return self.spec_references[topic_lower]
        
        # Fuzzy matching for common terms
        topic_mappings = {
            'function level reset': 'flr',
            'configuration request retry': 'crs',
            'transaction layer packet': 'tlp_format',
            'link training': 'ltssm',
            'power management': 'power_management',
            'flow control': 'flow_control'
        }
        
        for key_phrase, topic_key in topic_mappings.items():
            if key_phrase in topic_lower:
                return self.spec_references.get(topic_key, [])
        
        # Return empty list if no match found
        return []
    
    def instant_compliance_check(self, query: str, content: str) -> Dict[str, any]:
        """Perform instant compliance check on query and content"""
        # Detect violations
        violations = self.detect_violations(content, query)
        
        # Get relevant spec references
        spec_refs = []
        for topic in ['flr', 'crs', 'completion_timeout', 'tlp_format']:
            if topic in query.lower() or topic.replace('_', ' ') in query.lower():
                spec_refs.extend(self.get_specification_reference(topic))
        
        # Determine overall compliance status
        high_violations = [v for v in violations if v.severity == 'HIGH']
        medium_violations = [v for v in violations if v.severity == 'MEDIUM']
        
        if high_violations:
            status = 'NON_COMPLIANT'
            risk_level = 'HIGH'
        elif medium_violations:
            status = 'POTENTIAL_ISSUES'
            risk_level = 'MEDIUM'
        else:
            status = 'COMPLIANT'
            risk_level = 'LOW'
        
        return {
            'compliance_status': status,
            'risk_level': risk_level,
            'violations': violations,
            'specification_references': spec_refs,
            'recommendations': self._generate_recommendations(violations),
            'summary': self._generate_compliance_summary(violations, status)
        }
    
    def _generate_recommendations(self, violations: List[ComplianceViolation]) -> List[str]:
        """Generate actionable recommendations based on violations"""
        recommendations = []
        
        if not violations:
            recommendations.append("No compliance violations detected. Continue monitoring for potential issues.")
            return recommendations
        
        violation_types = set(v.violation_type for v in violations)
        
        if 'flr_violation' in violation_types:
            recommendations.append("Review FLR implementation: ensure 100ms timeout and no activity during reset")
            
        if 'crs_violation' in violation_types:
            recommendations.append("Check device initialization: CRS should only occur during startup")
            
        if 'timeout_violation' in violation_types:
            recommendations.append("Verify timeout parameters: ensure compliance with specification ranges")
            
        if 'protocol_violation' in violation_types:
            recommendations.append("Review protocol implementation: check TLP format and transaction ordering")
        
        # Add general recommendations
        high_severity_count = sum(1 for v in violations if v.severity == 'HIGH')
        if high_severity_count > 0:
            recommendations.append(f"Priority: Address {high_severity_count} high-severity violations first")
        
        recommendations.append("Test compliance with official PCIe compliance test suites")
        
        return recommendations
    
    def _generate_compliance_summary(self, violations: List[ComplianceViolation], status: str) -> str:
        """Generate human-readable compliance summary"""
        if not violations:
            return "✅ No compliance violations detected in the analyzed content."
        
        high_count = sum(1 for v in violations if v.severity == 'HIGH')
        medium_count = sum(1 for v in violations if v.severity == 'MEDIUM')
        low_count = sum(1 for v in violations if v.severity == 'LOW')
        
        summary = f"🔍 Compliance Analysis: {status}\n"
        
        if high_count > 0:
            summary += f"❌ {high_count} high-severity violation(s) detected\n"
        if medium_count > 0:
            summary += f"⚠️ {medium_count} medium-severity violation(s) detected\n"
        if low_count > 0:
            summary += f"ℹ️ {low_count} low-severity violation(s) detected\n"
        
        summary += f"\nMost critical: {violations[0].description}" if violations else ""
        
        return summary
    
    def get_violation_statistics(self) -> Dict[str, int]:
        """Get statistics about violation detection capabilities"""
        total_patterns = sum(len(patterns) for patterns in self.violation_patterns.values())
        
        return {
            'violation_types': len(self.violation_patterns),
            'total_patterns': total_patterns,
            'spec_references': sum(len(refs) for refs in self.spec_references.values()),
            'compliance_rules': len(self.compliance_rules)
        }