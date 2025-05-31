#!/usr/bin/env python3
"""
Enhanced AI-Only PCIe Error Analyzer
Significantly improved AI analysis without waveform data
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ErrorType(Enum):
    MALFORMED_TLP = "MALFORMED_TLP"
    CRC_ERROR = "CRC_ERROR"
    TIMEOUT = "TIMEOUT"
    ECRC_ERROR = "ECRC_ERROR"
    COMPLETION_TIMEOUT = "COMPLETION_TIMEOUT"
    UNSUPPORTED_REQUEST = "UNSUPPORTED_REQUEST"
    POISONED_TLP = "POISONED_TLP"


class LTSSMState(Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    RECOVERY = "RECOVERY"
    DETECT = "DETECT"
    POLLING = "POLLING"
    CONFIG = "CONFIG"
    DISABLED = "DISABLED"


@dataclass
class PCIeEvent:
    timestamp: Optional[float]
    event_type: str
    description: str
    severity: str
    context: Dict[str, Any]
    inferred_cause: Optional[str] = None


@dataclass
class ErrorPattern:
    pattern_name: str
    confidence: float
    evidence: List[str]
    likely_cause: str
    recommended_action: str


class EnhancedAIAnalyzer:
    """Enhanced AI-only analyzer with sophisticated PCIe protocol knowledge"""
    
    def __init__(self):
        self.events = []
        self.error_patterns = []
        self.ltssm_states = []
        self.recovery_events = []
        self.protocol_knowledge = self._load_protocol_knowledge()
        self.timing_patterns = []
        
    def _load_protocol_knowledge(self) -> Dict[str, Any]:
        """Load comprehensive PCIe protocol knowledge base"""
        return {
            'error_signatures': {
                'malformed_tlp': {
                    'keywords': ['malformed', 'invalid tlp', 'bad format', 'tlp error', 'header error'],
                    'typical_causes': ['protocol violation', 'software bug', 'encoding error'],
                    'recovery_pattern': 'immediate_retry',
                    'severity': 'HIGH'
                },
                'crc_error': {
                    'keywords': ['crc', 'checksum', 'lcrc', 'ecrc'],
                    'typical_causes': ['signal integrity', 'noise', 'cable issue'],
                    'recovery_pattern': 'retry_with_nak',
                    'severity': 'MEDIUM'
                },
                'timeout': {
                    'keywords': ['timeout', 'no response', 'completion timeout'],
                    'typical_causes': ['device hang', 'routing issue', 'clock problem'],
                    'recovery_pattern': 'retry_or_abort',
                    'severity': 'HIGH'
                }
            },
            'recovery_indicators': {
                'patterns': ['entering recovery', 'ltssm recovery', 'link retrain', 'recovery complete'],
                'typical_duration': 10000,  # nanoseconds
                'max_acceptable': 5,  # cycles before concern
                'severity_thresholds': {
                    'low': 2,
                    'medium': 5,
                    'high': 10
                }
            },
            'protocol_violations': {
                'tlp_formation': {
                    'symptoms': ['invalid type', 'bad length', 'header mismatch'],
                    'root_causes': ['driver bug', 'firmware issue', 'configuration error']
                },
                'flow_control': {
                    'symptoms': ['credit overflow', 'unauthorized transaction'],
                    'root_causes': ['buffer management', 'device misconfiguration']
                }
            }
        }
    
    def analyze_logs(self, log_data: List[str]) -> Dict[str, Any]:
        """Enhanced log analysis with intelligent pattern recognition"""
        print("🔍 Enhanced AI Analysis: Processing logs with advanced pattern recognition...")
        
        # Parse logs into structured events
        self._parse_log_entries(log_data)
        
        # Apply intelligent pattern recognition
        self._detect_error_patterns()
        
        # Infer timing relationships
        self._infer_timing_relationships()
        
        # Analyze LTSSM state transitions
        self._analyze_ltssm_patterns()
        
        # Perform root cause analysis
        root_causes = self._perform_root_cause_analysis()
        
        # Generate comprehensive results
        results = {
            'analysis_method': 'Enhanced AI-Only',
            'total_events': len(self.events),
            'errors_detected': len([e for e in self.events if 'error' in e.event_type.lower()]),
            'recovery_cycles': len(self.recovery_events),
            'error_patterns': self.error_patterns,
            'root_causes': root_causes,
            'recommendations': self._generate_recommendations(),
            'timing_analysis': self._generate_timing_analysis(),
            'confidence_scores': self._calculate_confidence_scores(),
            'severity_assessment': self._assess_severity()
        }
        
        return results
    
    def _parse_log_entries(self, log_data: List[str]):
        """Parse log entries with enhanced pattern recognition"""
        timestamp_patterns = [
            r'(\d+)\s*ns',  # nanosecond timestamps
            r'(\d+\.\d+)\s*ms',  # millisecond timestamps
            r'\[(\d+)\]',  # bracketed timestamps
        ]
        
        for i, log_entry in enumerate(log_data):
            # Extract timestamp if present
            timestamp = None
            for pattern in timestamp_patterns:
                match = re.search(pattern, log_entry)
                if match:
                    timestamp = float(match.group(1))
                    break
            
            # Classify event type and extract context
            event = self._classify_log_event(log_entry, timestamp or i * 1000)
            if event:
                self.events.append(event)
    
    def _classify_log_event(self, log_entry: str, timestamp: float) -> Optional[PCIeEvent]:
        """Classify log event with sophisticated pattern matching"""
        log_lower = log_entry.lower()
        
        # Error detection with enhanced patterns
        if any(keyword in log_lower for keyword in ['error', 'fail', 'invalid', 'bad']):
            error_type = self._identify_error_type(log_entry)
            return PCIeEvent(
                timestamp=timestamp,
                event_type=f"ERROR_{error_type}",
                description=log_entry,
                severity=self._assess_event_severity(log_entry),
                context={'raw_log': log_entry, 'error_type': error_type}
            )
        
        # Recovery detection
        elif any(keyword in log_lower for keyword in ['recovery', 'retrain', 'retry']):
            self.recovery_events.append(timestamp)
            return PCIeEvent(
                timestamp=timestamp,
                event_type="LTSSM_RECOVERY",
                description=log_entry,
                severity="MEDIUM",
                context={'recovery_trigger': self._identify_recovery_trigger(log_entry)}
            )
        
        # Transaction events
        elif any(keyword in log_lower for keyword in ['tlp', 'transaction', 'request', 'completion']):
            return PCIeEvent(
                timestamp=timestamp,
                event_type="TRANSACTION",
                description=log_entry,
                severity="INFO",
                context={'transaction_type': self._identify_transaction_type(log_entry)}
            )
        
        return None
    
    def _identify_error_type(self, log_entry: str) -> str:
        """Identify specific error type from log entry"""
        log_lower = log_entry.lower()
        
        for error_type, info in self.protocol_knowledge['error_signatures'].items():
            if any(keyword in log_lower for keyword in info['keywords']):
                return error_type.upper()
        
        return "UNKNOWN"
    
    def _detect_error_patterns(self):
        """Detect sophisticated error patterns in event sequence"""
        error_events = [e for e in self.events if 'ERROR' in e.event_type]
        
        # Pattern 1: Repeated malformed TLP errors
        malformed_errors = [e for e in error_events if 'MALFORMED' in e.event_type]
        if len(malformed_errors) >= 2:
            self.error_patterns.append(ErrorPattern(
                pattern_name="repeated_malformed_tlp",
                confidence=0.9,
                evidence=[f"Detected {len(malformed_errors)} malformed TLP errors"],
                likely_cause="Protocol violation in TLP formation logic",
                recommended_action="Review TLP generation code and validate header fields"
            ))
        
        # Pattern 2: Recovery cycles after errors
        error_recovery_correlation = 0
        for error in error_events:
            nearby_recoveries = [r for r in self.recovery_events 
                               if abs(r - error.timestamp) < 50000]  # 50ms window
            error_recovery_correlation += len(nearby_recoveries)
        
        if error_recovery_correlation > 0:
            self.error_patterns.append(ErrorPattern(
                pattern_name="error_induced_recovery",
                confidence=0.8,
                evidence=[f"Found {error_recovery_correlation} recovery events correlated with errors"],
                likely_cause="Errors triggering link recovery mechanisms",
                recommended_action="Investigate error root causes to reduce recovery frequency"
            ))
        
        # Pattern 3: Excessive recovery cycles
        if len(self.recovery_events) > self.protocol_knowledge['recovery_indicators']['max_acceptable']:
            severity = "HIGH" if len(self.recovery_events) > 10 else "MEDIUM"
            self.error_patterns.append(ErrorPattern(
                pattern_name="excessive_recovery_cycles",
                confidence=0.95,
                evidence=[f"Detected {len(self.recovery_events)} recovery cycles (threshold: {self.protocol_knowledge['recovery_indicators']['max_acceptable']})"],
                likely_cause="Signal integrity issues or systematic protocol violations",
                recommended_action="Check physical layer and signal quality"
            ))
    
    def _infer_timing_relationships(self):
        """Infer timing relationships without explicit waveform data"""
        # Calculate event intervals
        event_intervals = []
        for i in range(1, len(self.events)):
            interval = self.events[i].timestamp - self.events[i-1].timestamp
            event_intervals.append(interval)
        
        # Detect timing patterns
        if event_intervals:
            avg_interval = sum(event_intervals) / len(event_intervals)
            self.timing_patterns.append({
                'average_event_interval': avg_interval,
                'total_analysis_window': self.events[-1].timestamp - self.events[0].timestamp,
                'event_density': len(self.events) / (self.events[-1].timestamp - self.events[0].timestamp) * 1000000,  # events per second
                'potential_timing_issues': any(interval > 100000 for interval in event_intervals)  # >100ms gaps
            })
    
    def _analyze_ltssm_patterns(self):
        """Analyze LTSSM state transition patterns"""
        recovery_events = [e for e in self.events if 'RECOVERY' in e.event_type]
        
        if recovery_events:
            # Calculate recovery frequency
            if len(recovery_events) > 1:
                recovery_intervals = []
                for i in range(1, len(recovery_events)):
                    interval = recovery_events[i].timestamp - recovery_events[i-1].timestamp
                    recovery_intervals.append(interval)
                
                avg_recovery_interval = sum(recovery_intervals) / len(recovery_intervals)
                
                # Assess if recovery frequency indicates problems
                if avg_recovery_interval < 100000:  # <100ms between recoveries
                    self.ltssm_states.append({
                        'pattern': 'frequent_recovery',
                        'severity': 'HIGH',
                        'evidence': f'Recovery every {avg_recovery_interval/1000:.1f}ms on average',
                        'implication': 'Unstable link condition'
                    })
    
    def _perform_root_cause_analysis(self) -> Dict[str, Any]:
        """Perform sophisticated root cause analysis"""
        root_causes = {}
        
        # Analyze error patterns for root causes
        for pattern in self.error_patterns:
            if 'malformed' in pattern.pattern_name:
                root_causes['protocol_violation'] = {
                    'confidence': pattern.confidence,
                    'evidence': pattern.evidence,
                    'category': 'Software/Firmware Issue',
                    'specific_area': 'TLP Formation Logic'
                }
            elif 'recovery' in pattern.pattern_name:
                root_causes['signal_integrity'] = {
                    'confidence': pattern.confidence,
                    'evidence': pattern.evidence,
                    'category': 'Physical Layer Issue',
                    'specific_area': 'Link Stability'
                }
        
        # Infer additional causes from event patterns
        error_count = len([e for e in self.events if 'ERROR' in e.event_type])
        recovery_count = len(self.recovery_events)
        
        if recovery_count > error_count * 2:  # Many recoveries per error
            root_causes['systematic_instability'] = {
                'confidence': 0.8,
                'evidence': [f'Recovery/Error ratio: {recovery_count}/{error_count}'],
                'category': 'System Integration Issue',
                'specific_area': 'Link Training or Power Management'
            }
        
        return root_causes
    
    def _generate_recommendations(self) -> List[Dict[str, str]]:
        """Generate targeted recommendations based on analysis"""
        recommendations = []
        
        for pattern in self.error_patterns:
            recommendations.append({
                'area': pattern.pattern_name,
                'severity': 'HIGH' if pattern.confidence > 0.8 else 'MEDIUM',
                'action': pattern.recommended_action,
                'rationale': f"Pattern detected with {pattern.confidence*100:.0f}% confidence"
            })
        
        # Add general recommendations based on event analysis
        if len(self.recovery_events) > 5:
            recommendations.append({
                'area': 'link_stability',
                'severity': 'HIGH',
                'action': 'Perform comprehensive signal integrity analysis',
                'rationale': f'Excessive recovery cycles detected ({len(self.recovery_events)})'
            })
        
        return recommendations
    
    def _generate_timing_analysis(self) -> Dict[str, Any]:
        """Generate timing analysis without waveform data"""
        if not self.timing_patterns:
            return {'available': False, 'reason': 'Insufficient timing data in logs'}
        
        timing = self.timing_patterns[0]
        return {
            'available': True,
            'event_distribution': 'Analyzed from log timestamps',
            'potential_issues': timing['potential_timing_issues'],
            'recommendations': [
                'Enable high-resolution timestamps in logging',
                'Add timing markers for critical events',
                'Consider VCD analysis for precise timing'
            ] if timing['potential_timing_issues'] else []
        }
    
    def _calculate_confidence_scores(self) -> Dict[str, float]:
        """Calculate confidence scores for different analysis aspects"""
        scores = {
            'error_detection': 0.8,  # Good at finding errors in logs
            'error_classification': 0.7,  # Decent classification from keywords
            'recovery_detection': 0.3,  # Poor - only sees logged recoveries
            'root_cause_identification': 0.6,  # Improved with pattern analysis
            'timing_analysis': 0.2,  # Very limited without waveforms
            'overall': 0.0
        }
        
        # Calculate overall confidence
        scores['overall'] = sum(scores.values()) / (len(scores) - 1)
        return scores
    
    def _assess_severity(self) -> str:
        """Assess overall severity of detected issues"""
        high_severity_indicators = 0
        
        # Check for high-severity patterns
        for pattern in self.error_patterns:
            if pattern.confidence > 0.8:
                high_severity_indicators += 1
        
        # Check for excessive recovery
        if len(self.recovery_events) > 10:
            high_severity_indicators += 1
        
        # Check for multiple error types
        error_types = set(e.context.get('error_type', 'unknown') for e in self.events if 'ERROR' in e.event_type)
        if len(error_types) > 2:
            high_severity_indicators += 1
        
        if high_severity_indicators >= 2:
            return "HIGH"
        elif high_severity_indicators >= 1:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_event_severity(self, log_entry: str) -> str:
        """Assess severity of individual log event"""
        log_lower = log_entry.lower()
        
        if any(word in log_lower for word in ['critical', 'fatal', 'severe']):
            return "CRITICAL"
        elif any(word in log_lower for word in ['error', 'fail', 'invalid']):
            return "HIGH"
        elif any(word in log_lower for word in ['warning', 'retry', 'recovery']):
            return "MEDIUM"
        else:
            return "LOW"
    
    def _identify_recovery_trigger(self, log_entry: str) -> str:
        """Identify what triggered the recovery"""
        log_lower = log_entry.lower()
        
        if 'error' in log_lower:
            return "error_induced"
        elif 'timeout' in log_lower:
            return "timeout_induced"
        elif 'training' in log_lower:
            return "training_failure"
        else:
            return "unknown"
    
    def _identify_transaction_type(self, log_entry: str) -> str:
        """Identify PCIe transaction type"""
        log_lower = log_entry.lower()
        
        if 'read' in log_lower:
            return "memory_read"
        elif 'write' in log_lower:
            return "memory_write"
        elif 'completion' in log_lower:
            return "completion"
        elif 'config' in log_lower:
            return "configuration"
        else:
            return "unknown"
    
    def query(self, question: str) -> str:
        """Answer questions about the analysis"""
        question_lower = question.lower()
        
        if 'error' in question_lower and 'type' in question_lower:
            error_events = [e for e in self.events if 'ERROR' in e.event_type]
            if error_events:
                error_types = [e.context.get('error_type', 'unknown') for e in error_events]
                return f"Enhanced AI Analysis: Detected {len(error_events)} errors of types: {list(set(error_types))}. Confidence in classification: 70%"
            else:
                return "Enhanced AI Analysis: No errors detected in available log data."
        
        elif 'recovery' in question_lower:
            if self.recovery_events:
                severity = "HIGH" if len(self.recovery_events) > 10 else "MEDIUM" if len(self.recovery_events) > 5 else "LOW"
                return f"Enhanced AI Analysis: Detected {len(self.recovery_events)} recovery cycles (severity: {severity}). Pattern analysis suggests systematic instability."
            else:
                return "Enhanced AI Analysis: No recovery cycles detected in log data."
        
        elif 'root cause' in question_lower:
            if self.error_patterns:
                main_cause = self.error_patterns[0].likely_cause
                confidence = self.error_patterns[0].confidence
                return f"Enhanced AI Analysis: Primary root cause identified as '{main_cause}' with {confidence*100:.0f}% confidence based on pattern analysis."
            else:
                return "Enhanced AI Analysis: Insufficient data for confident root cause identification. Recommend waveform analysis."
        
        elif 'timing' in question_lower:
            if self.timing_patterns:
                issues = self.timing_patterns[0]['potential_timing_issues']
                if issues:
                    return "Enhanced AI Analysis: Potential timing irregularities detected in log timestamps. High-resolution waveform analysis recommended."
                else:
                    return "Enhanced AI Analysis: No obvious timing issues detected in available log timestamps."
            else:
                return "Enhanced AI Analysis: Limited timing analysis possible with log data only. VCD analysis recommended for precise timing."
        
        else:
            return f"Enhanced AI Analysis: Processed {len(self.events)} events with {len(self.error_patterns)} patterns identified. Confidence scores available for detailed analysis."


def create_enhanced_log_dataset() -> List[str]:
    """Create enhanced log dataset with more realistic PCIe error logs"""
    return [
        "2920000ns: PCIe ERROR - Malformed TLP detected, Type=7 invalid for memory request",
        "2920050ns: PCIe ERROR - TLP header validation failed, dropping packet",
        "2920100ns: PCIe WARNING - Sending Unsupported Request completion", 
        "2921000ns: PCIe INFO - LTSSM entering Recovery.RcvrLock state",
        "2922000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2923000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2925000ns: PCIe ERROR - Malformed TLP detected, Type=7 invalid for memory request",
        "2925050ns: PCIe ERROR - TLP header validation failed, dropping packet", 
        "2925100ns: PCIe WARNING - Sending Unsupported Request completion",
        "2926000ns: PCIe INFO - LTSSM entering Recovery.RcvrLock state",
        "2927000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2928000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2930000ns: PCIe INFO - Memory read request, Tag=1, Address=0x1000",
        "2930500ns: PCIe INFO - Completion received, Tag=1, Status=SUCCESS",
        "2935000ns: PCIe INFO - Memory write request, Tag=2, Address=0x2000",
        "2935500ns: PCIe INFO - Write completion received, Tag=2",
        "2940000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "2941000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link", 
        "2942000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2950000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "2951000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2952000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2960000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "2961000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2962000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2970000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "2971000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2972000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2980000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "2981000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2982000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "2990000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "2991000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "2992000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "3000000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "3001000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "3002000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "3010000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "3011000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link", 
        "3012000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "3020000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state",
        "3021000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "3022000ns: PCIe INFO - LTSSM Recovery completed, returning to L0",
        "3030000ns: PCIe WARNING - LTSSM entering Recovery.RcvrLock state", 
        "3031000ns: PCIe INFO - LTSSM Recovery.RcvrCfg - retraining link",
        "3032000ns: PCIe INFO - LTSSM Recovery completed, returning to L0"
    ]


if __name__ == "__main__":
    print("🚀 Enhanced AI-Only PCIe Error Analysis")
    print("=" * 60)
    
    # Create enhanced analyzer
    analyzer = EnhancedAIAnalyzer()
    
    # Generate realistic log data
    log_data = create_enhanced_log_dataset()
    print(f"📊 Processing {len(log_data)} enhanced log entries...")
    
    # Perform analysis
    results = analyzer.analyze_logs(log_data)
    
    # Display results
    print(f"\n📈 Enhanced AI Analysis Results:")
    print(f"  - Total Events: {results['total_events']}")
    print(f"  - Errors Detected: {results['errors_detected']}")
    print(f"  - Recovery Cycles: {results['recovery_cycles']}")
    print(f"  - Error Patterns: {len(results['error_patterns'])}")
    print(f"  - Root Causes: {len(results['root_causes'])}")
    print(f"  - Overall Severity: {results['severity_assessment']}")
    
    print(f"\n🎯 Confidence Scores:")
    for aspect, score in results['confidence_scores'].items():
        print(f"  - {aspect.replace('_', ' ').title()}: {score*100:.1f}%")
    
    print(f"\n🔍 Detected Error Patterns:")
    for pattern in results['error_patterns']:
        print(f"  - {pattern.pattern_name}: {pattern.confidence*100:.0f}% confidence")
        print(f"    Cause: {pattern.likely_cause}")
    
    print(f"\n🎪 Query Testing:")
    test_queries = [
        "What type of errors occurred?",
        "How many recovery cycles happened?", 
        "What is the root cause?",
        "Are there timing issues?"
    ]
    
    for query in test_queries:
        response = analyzer.query(query)
        print(f"  Q: {query}")
        print(f"  A: {response}\n")