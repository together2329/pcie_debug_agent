#!/usr/bin/env python3
"""
VCD-Based PCIe Error Analysis System
Integrates waveform analysis with RAG/LLM for intelligent debugging
"""

import re
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
# import numpy as np  # Optional dependency

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulators.vcd_rag_analyzer import PCIeVCDAnalyzer, VCDEvent


@dataclass
class ErrorContext:
    """Context information for an error event"""
    error_time: int
    error_type: str
    preceding_events: List[VCDEvent]
    following_events: List[VCDEvent]
    signal_states: Dict[str, Any]
    ltssm_history: List[Tuple[int, str]]
    transaction_history: List[VCDEvent]
    recovery_attempts: int
    related_errors: List[VCDEvent]


@dataclass
class TimingMetrics:
    """Timing analysis metrics"""
    transaction_latencies: List[int]
    recovery_durations: List[int]
    error_intervals: List[int]
    state_durations: Dict[str, List[int]]
    timeout_events: List[Tuple[int, str]]


class PCIeVCDErrorAnalyzer:
    """Advanced VCD-based error analysis for PCIe"""
    
    def __init__(self, vcd_file: str):
        self.vcd_file = vcd_file
        self.analyzer = PCIeVCDAnalyzer(vcd_file)
        self.error_contexts = []
        self.timing_metrics = None
        self.root_causes = {}
        self.recommendations = []
        
    def analyze_errors(self) -> Dict[str, Any]:
        """Perform comprehensive error analysis"""
        # Parse VCD file
        print("Parsing VCD file...")
        vcd_data = self.analyzer.parse_vcd()
        
        # Extract error contexts
        print("Extracting error contexts...")
        self._extract_error_contexts()
        
        # Perform timing analysis
        print("Analyzing timing...")
        self.timing_metrics = self._analyze_timing()
        
        # Correlate errors
        print("Correlating errors...")
        correlations = self._correlate_errors()
        
        # Identify root causes
        print("Identifying root causes...")
        self._identify_root_causes()
        
        # Generate recommendations
        print("Generating recommendations...")
        self._generate_recommendations()
        
        return {
            'error_contexts': self.error_contexts,
            'timing_metrics': self.timing_metrics,
            'correlations': correlations,
            'root_causes': self.root_causes,
            'recommendations': self.recommendations
        }
        
    def _extract_error_contexts(self):
        """Extract context for each error event"""
        errors = self.analyzer.errors
        all_events = sorted(self.analyzer.events, key=lambda e: e.time)
        
        for error in errors:
            # Find surrounding events (±1ms window)
            window_start = error.time - 1000000  # 1ms before
            window_end = error.time + 1000000    # 1ms after
            
            preceding = [e for e in all_events 
                        if window_start <= e.time < error.time][-10:]  # Last 10 events
            following = [e for e in all_events 
                        if error.time < e.time <= window_end][:10]    # Next 10 events
            
            # Extract LTSSM history
            ltssm_history = [(e.time, e.details['new_state']) 
                            for e in self.analyzer.state_changes 
                            if e.time <= error.time][-5:]  # Last 5 states
            
            # Extract transaction history
            tx_history = [e for e in self.analyzer.transactions 
                         if e.time <= error.time][-5:]  # Last 5 transactions
            
            # Count recovery attempts
            recovery_count = sum(1 for e in preceding 
                               if e.event_type == "LTSSM State Change" 
                               and e.details.get('new_state') == 'RECOVERY')
            
            # Find related errors
            related = [e for e in errors 
                      if e != error and abs(e.time - error.time) < 1000000]
            
            context = ErrorContext(
                error_time=error.time,
                error_type=error.details.get('error_type', 'UNKNOWN'),
                preceding_events=preceding,
                following_events=following,
                signal_states=self._get_signal_states_at_time(error.time),
                ltssm_history=ltssm_history,
                transaction_history=tx_history,
                recovery_attempts=recovery_count,
                related_errors=related
            )
            
            self.error_contexts.append(context)
            
    def _get_signal_states_at_time(self, time: int) -> Dict[str, Any]:
        """Get all signal states at a specific time"""
        states = {}
        
        for symbol, signal_info in self.analyzer.signals.items():
            # Find the last change before or at this time
            last_value = 'x'
            for change_time, value in signal_info.get('changes', []):
                if change_time <= time:
                    last_value = value
                else:
                    break
            states[signal_info['name']] = last_value
            
        return states
        
    def _analyze_timing(self) -> TimingMetrics:
        """Perform detailed timing analysis"""
        metrics = TimingMetrics(
            transaction_latencies=[],
            recovery_durations=[],
            error_intervals=[],
            state_durations=defaultdict(list),
            timeout_events=[]
        )
        
        # Analyze transaction latencies
        transactions = sorted(self.analyzer.transactions, key=lambda t: t.time)
        for i, tx in enumerate(transactions):
            if tx.details.get('type') == 'Memory Read':
                # Look for completion
                completion_time = self._find_completion_time(tx, transactions[i+1:])
                if completion_time:
                    latency = completion_time - tx.time
                    metrics.transaction_latencies.append(latency)
                    
                    # Check for timeout
                    if latency > 1000000:  # 1ms timeout threshold
                        metrics.timeout_events.append((tx.time, f"Tag {tx.details.get('tag')}"))
                        
        # Analyze recovery durations
        state_changes = self.analyzer.state_changes
        for i, change in enumerate(state_changes):
            if change.details['new_state'] == 'RECOVERY':
                # Find when it exits recovery
                for j in range(i+1, len(state_changes)):
                    if state_changes[j].details['previous_state'] == 'RECOVERY':
                        duration = state_changes[j].time - change.time
                        metrics.recovery_durations.append(duration)
                        break
                        
        # Analyze error intervals
        if len(self.analyzer.errors) > 1:
            errors = sorted(self.analyzer.errors, key=lambda e: e.time)
            for i in range(1, len(errors)):
                interval = errors[i].time - errors[i-1].time
                metrics.error_intervals.append(interval)
                
        # Analyze state durations
        for i in range(1, len(state_changes)):
            state = state_changes[i-1].details['new_state']
            duration = state_changes[i].time - state_changes[i-1].time
            metrics.state_durations[state].append(duration)
            
        return metrics
        
    def _find_completion_time(self, read_tx: VCDEvent, 
                            following_txs: List[VCDEvent]) -> Optional[int]:
        """Find completion time for a read transaction"""
        read_tag = read_tx.details.get('tag')
        
        for tx in following_txs:
            if (tx.details.get('type') == 'Completion' and 
                tx.details.get('tag') == read_tag):
                return tx.time
                
        return None
        
    def _correlate_errors(self) -> Dict[str, Any]:
        """Correlate errors with system events"""
        correlations = {
            'error_to_recovery': [],
            'error_clusters': [],
            'error_to_timeout': [],
            'error_to_state': defaultdict(list)
        }
        
        # Correlate errors with recovery events
        for context in self.error_contexts:
            # Check if recovery follows error
            recovery_after = any(e.event_type == "LTSSM State Change" and 
                               e.details.get('new_state') == 'RECOVERY'
                               for e in context.following_events)
            
            if recovery_after:
                correlations['error_to_recovery'].append({
                    'error_time': context.error_time,
                    'error_type': context.error_type,
                    'caused_recovery': True
                })
                
        # Find error clusters
        if len(self.analyzer.errors) > 1:
            cluster_threshold = 1000000  # 1ms
            current_cluster = [self.analyzer.errors[0]]
            
            for error in self.analyzer.errors[1:]:
                if error.time - current_cluster[-1].time <= cluster_threshold:
                    current_cluster.append(error)
                else:
                    if len(current_cluster) > 1:
                        correlations['error_clusters'].append({
                            'start_time': current_cluster[0].time,
                            'end_time': current_cluster[-1].time,
                            'error_count': len(current_cluster),
                            'error_types': [e.details.get('error_type') for e in current_cluster]
                        })
                    current_cluster = [error]
                    
        # Correlate with timeouts
        if self.timing_metrics:
            for timeout_time, tag in self.timing_metrics.timeout_events:
                # Find errors near timeout
                nearby_errors = [e for e in self.analyzer.errors
                               if abs(e.time - timeout_time) < 2000000]  # 2ms window
                
                if nearby_errors:
                    correlations['error_to_timeout'].append({
                        'timeout_time': timeout_time,
                        'tag': tag,
                        'nearby_errors': [(e.time, e.details.get('error_type')) 
                                        for e in nearby_errors]
                    })
                    
        # Correlate errors with LTSSM states
        for context in self.error_contexts:
            if context.ltssm_history:
                current_state = context.ltssm_history[-1][1]
                correlations['error_to_state'][current_state].append(context.error_type)
                
        return correlations
        
    def _identify_root_causes(self):
        """Identify root causes based on patterns"""
        # Pattern-based root cause analysis
        patterns = {
            'signal_integrity': {
                'indicators': ['excessive_recovery', 'crc_errors', 'link_flapping'],
                'check': lambda: len([c for c in self.error_contexts 
                                    if c.recovery_attempts > 3]) > 0
            },
            'timeout_issue': {
                'indicators': ['completion_timeout', 'missing_completion'],
                'check': lambda: len(self.timing_metrics.timeout_events) > 0
            },
            'protocol_violation': {
                'indicators': ['malformed_tlp', 'unsupported_request'],
                'check': lambda: any(c.error_type == 'MALFORMED_TLP' 
                                   for c in self.error_contexts)
            },
            'power_delivery': {
                'indicators': ['link_down_events', 'repeated_training'],
                'check': lambda: self._check_link_stability()
            }
        }
        
        for cause, pattern in patterns.items():
            if pattern['check']():
                self.root_causes[cause] = {
                    'detected': True,
                    'confidence': self._calculate_confidence(pattern['indicators']),
                    'evidence': self._gather_evidence(pattern['indicators'])
                }
                
    def _check_link_stability(self) -> bool:
        """Check for link stability issues"""
        # Count link down events
        down_events = sum(1 for e in self.analyzer.state_changes
                         if e.details['new_state'] in ['DETECT', 'POLLING'])
        
        # Check for repeated training
        training_events = sum(1 for e in self.analyzer.state_changes
                            if e.details['new_state'] == 'CONFIG')
        
        return down_events > 2 or training_events > 2
        
    def _calculate_confidence(self, indicators: List[str]) -> float:
        """Calculate confidence score for root cause"""
        # Simple scoring based on number of indicators present
        score = 0.0
        
        if 'excessive_recovery' in indicators and len(self.timing_metrics.recovery_durations) > 5:
            score += 0.3
            
        if 'crc_errors' in indicators and any(c.error_type == 'CRC_ERROR' for c in self.error_contexts):
            score += 0.3
            
        if 'completion_timeout' in indicators and len(self.timing_metrics.timeout_events) > 0:
            score += 0.4
            
        return min(score, 1.0)
        
    def _gather_evidence(self, indicators: List[str]) -> List[str]:
        """Gather evidence for root cause"""
        evidence = []
        
        if 'excessive_recovery' in indicators:
            recovery_count = len(self.timing_metrics.recovery_durations)
            avg_duration = sum(self.timing_metrics.recovery_durations) / len(self.timing_metrics.recovery_durations) if recovery_count > 0 else 0
            evidence.append(f"Recovery events: {recovery_count}, avg duration: {avg_duration:.0f}ns")
            
        if 'completion_timeout' in indicators:
            timeout_count = len(self.timing_metrics.timeout_events)
            evidence.append(f"Completion timeouts: {timeout_count}")
            
        return evidence
        
    def _generate_recommendations(self):
        """Generate actionable recommendations"""
        for cause, details in self.root_causes.items():
            if not details['detected']:
                continue
                
            if cause == 'signal_integrity':
                self.recommendations.append({
                    'issue': 'Signal Integrity Problems',
                    'severity': 'HIGH',
                    'actions': [
                        'Check PCIe trace routing for impedance discontinuities',
                        'Verify differential pair matching and spacing',
                        'Measure eye diagram at receiver',
                        'Check for EMI sources near PCIe lanes',
                        'Verify power supply noise levels',
                        'Consider reducing link speed for testing'
                    ],
                    'evidence': details['evidence']
                })
                
            elif cause == 'timeout_issue':
                self.recommendations.append({
                    'issue': 'Completion Timeout Problems',
                    'severity': 'CRITICAL',
                    'actions': [
                        'Increase completion timeout value in device configuration',
                        'Check credit flow between devices',
                        'Verify device power management settings',
                        'Check for deadlock conditions in transaction ordering',
                        'Monitor buffer occupancy levels',
                        'Verify BAR configuration and address decoding'
                    ],
                    'evidence': details['evidence']
                })
                
            elif cause == 'protocol_violation':
                self.recommendations.append({
                    'issue': 'PCIe Protocol Violations',
                    'severity': 'HIGH',
                    'actions': [
                        'Review TLP formation logic in transmitter',
                        'Verify TLP header field values',
                        'Check for correct message routing',
                        'Validate request/completion pairing',
                        'Ensure proper credit management',
                        'Review PCIe specification compliance'
                    ],
                    'evidence': details['evidence']
                })
                
    def generate_rag_enhanced_report(self) -> str:
        """Generate report with RAG-ready sections"""
        report = f"""
PCIe VCD Error Analysis Report
==============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {self.vcd_file}

## Executive Summary
Total Errors Analyzed: {len(self.error_contexts)}
Root Causes Identified: {len([c for c, d in self.root_causes.items() if d['detected']])}
Critical Issues: {len([r for r in self.recommendations if r['severity'] == 'CRITICAL'])}

## Error Timeline Analysis
"""
        
        # Add error timeline
        for i, context in enumerate(self.error_contexts[:5], 1):
            report += f"\n### Error {i}: {context.error_type} at {context.error_time}ns\n"
            report += f"LTSSM State: {context.ltssm_history[-1][1] if context.ltssm_history else 'UNKNOWN'}\n"
            report += f"Recovery Attempts Before Error: {context.recovery_attempts}\n"
            
            if context.preceding_events:
                report += "\nPreceding Events:\n"
                for event in context.preceding_events[-3:]:
                    report += f"  - [{event.time}ns] {event.event_type}\n"
                    
        # Add timing analysis
        if self.timing_metrics:
            report += f"\n## Timing Analysis\n"
            avg_latency = sum(self.timing_metrics.transaction_latencies) / len(self.timing_metrics.transaction_latencies) if self.timing_metrics.transaction_latencies else 0
            max_latency = max(self.timing_metrics.transaction_latencies) if self.timing_metrics.transaction_latencies else 0
            report += f"Average Transaction Latency: {avg_latency:.0f}ns\n"
            report += f"Maximum Transaction Latency: {max_latency:.0f}ns\n"
            report += f"Timeout Events: {len(self.timing_metrics.timeout_events)}\n"
            
            if self.timing_metrics.recovery_durations:
                avg_recovery = sum(self.timing_metrics.recovery_durations) / len(self.timing_metrics.recovery_durations)
                report += f"Average Recovery Duration: {avg_recovery:.0f}ns\n"
                
        # Add root cause analysis
        report += "\n## Root Cause Analysis\n"
        for cause, details in self.root_causes.items():
            if details['detected']:
                report += f"\n### {cause.replace('_', ' ').title()}\n"
                report += f"Confidence: {details['confidence']:.1%}\n"
                report += "Evidence:\n"
                for evidence in details['evidence']:
                    report += f"  - {evidence}\n"
                    
        # Add recommendations
        report += "\n## Recommendations\n"
        for rec in self.recommendations:
            report += f"\n### {rec['issue']} (Severity: {rec['severity']})\n"
            report += "Actions:\n"
            for action in rec['actions']:
                report += f"  - {action}\n"
                
        # Add RAG-optimized sections
        report += "\n## RAG-Optimized Error Descriptions\n"
        for context in self.error_contexts:
            report += f"\n[Time: {context.error_time}ns] PCIe Error Event\n"
            report += f"Type: {context.error_type}\n"
            report += f"Context: Error occurred in LTSSM state {context.ltssm_history[-1][1] if context.ltssm_history else 'UNKNOWN'}\n"
            report += f"Impact: {context.recovery_attempts} recovery cycles detected\n"
            
        return report


def analyze_pcie_errors_with_vcd(vcd_file: str, user_queries: List[str] = None):
    """Main function to analyze PCIe errors using VCD data"""
    print(f"\nAnalyzing PCIe errors from VCD: {vcd_file}")
    print("=" * 60)
    
    # Create analyzer
    analyzer = PCIeVCDErrorAnalyzer(vcd_file)
    
    # Perform analysis
    results = analyzer.analyze_errors()
    
    # Generate report
    report = analyzer.generate_rag_enhanced_report()
    
    # Save report
    report_file = f"pcie_error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"\nAnalysis complete. Report saved to: {report_file}")
    
    # Show summary
    print("\nAnalysis Summary:")
    print(f"  - Errors Found: {len(analyzer.error_contexts)}")
    print(f"  - Root Causes: {list(analyzer.root_causes.keys())}")
    print(f"  - Recommendations: {len(analyzer.recommendations)}")
    
    # Process user queries if provided
    if user_queries:
        print("\n" + "="*60)
        print("Query Results")
        print("="*60)
        
        for query in user_queries:
            print(f"\nQ: {query}")
            
            # This is where RAG integration would process the query
            # For now, we'll show what information is available
            if "root cause" in query.lower():
                print("A: Based on VCD analysis, the root causes are:")
                for cause, details in analyzer.root_causes.items():
                    if details['detected']:
                        print(f"   - {cause}: {details['confidence']:.1%} confidence")
                        
            elif "recommendation" in query.lower():
                print("A: Recommended actions based on waveform analysis:")
                for rec in analyzer.recommendations[:2]:
                    print(f"   - {rec['issue']}: {rec['actions'][0]}")
                    
            elif "timing" in query.lower():
                if analyzer.timing_metrics:
                    print(f"A: Timing analysis shows:")
                    avg_lat = sum(analyzer.timing_metrics.transaction_latencies) / len(analyzer.timing_metrics.transaction_latencies) if analyzer.timing_metrics.transaction_latencies else 0
                    print(f"   - Avg latency: {avg_lat:.0f}ns")
                    print(f"   - Timeouts: {len(analyzer.timing_metrics.timeout_events)}")
                    
    return analyzer


# Example usage
if __name__ == "__main__":
    # Check if VCD file exists
    vcd_file = "pcie_waveform.vcd"
    
    if not Path(vcd_file).exists():
        print(f"VCD file '{vcd_file}' not found.")
        print("Please run ./generate_vcd.sh first.")
        sys.exit(1)
        
    # Example queries that would be processed by RAG
    example_queries = [
        "What is the root cause of the PCIe errors?",
        "What recommendations do you have for fixing the issues?",
        "Are there any timing problems in the system?"
    ]
    
    # Run analysis
    analyzer = analyze_pcie_errors_with_vcd(vcd_file, example_queries)