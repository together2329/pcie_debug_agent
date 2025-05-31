#!/usr/bin/env python3
"""
Complete VCD-RAG Integration for PCIe Error Analysis
Connects waveform analysis with the existing RAG/LLM system
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulators.vcd_error_analyzer import PCIeVCDErrorAnalyzer
from src.simulators.vcd_rag_analyzer import PCIeVCDAnalyzer
from src.simulators.rag_integration import SimulatorRAGBridge, create_rag_bridge


class VCDRAGIntegration:
    """Integrates VCD analysis with RAG engine for intelligent debugging"""
    
    def __init__(self, rag_bridge: Optional[SimulatorRAGBridge] = None):
        self.rag_bridge = rag_bridge
        self.vcd_analyzer = None
        self.error_analyzer = None
        self.analysis_results = None
        
    def analyze_vcd_file(self, vcd_file: str) -> Dict[str, Any]:
        """Analyze VCD file and prepare for RAG ingestion"""
        print(f"Analyzing VCD file: {vcd_file}")
        
        # Create analyzers
        self.vcd_analyzer = PCIeVCDAnalyzer(vcd_file)
        self.error_analyzer = PCIeVCDErrorAnalyzer(vcd_file)
        
        # Perform analysis
        vcd_data = self.vcd_analyzer.parse_vcd()
        self.analysis_results = self.error_analyzer.analyze_errors()
        
        # Generate text chunks for RAG
        text_chunks = self._generate_rag_chunks()
        
        # Ingest into RAG if bridge available
        if self.rag_bridge:
            self._ingest_to_rag(text_chunks)
            
        return {
            'vcd_data': vcd_data,
            'error_analysis': self.analysis_results,
            'text_chunks': text_chunks
        }
        
    def _generate_rag_chunks(self) -> List[str]:
        """Generate comprehensive text chunks for RAG ingestion"""
        chunks = []
        
        # 1. Summary chunk
        summary = self._create_summary_chunk()
        chunks.append(summary)
        
        # 2. Error context chunks
        for context in self.error_analyzer.error_contexts:
            chunk = self._create_error_context_chunk(context)
            chunks.append(chunk)
            
        # 3. Timing analysis chunks
        if self.error_analyzer.timing_metrics:
            timing_chunk = self._create_timing_chunk(self.error_analyzer.timing_metrics)
            chunks.append(timing_chunk)
            
        # 4. Root cause chunks
        for cause, details in self.error_analyzer.root_causes.items():
            if details['detected']:
                chunk = self._create_root_cause_chunk(cause, details)
                chunks.append(chunk)
                
        # 5. Recommendation chunks
        for rec in self.error_analyzer.recommendations:
            chunk = self._create_recommendation_chunk(rec)
            chunks.append(chunk)
            
        # 6. Pattern analysis chunks
        patterns = self._extract_patterns()
        for pattern in patterns:
            chunks.append(pattern)
            
        return chunks
        
    def _create_summary_chunk(self) -> str:
        """Create summary chunk for RAG"""
        return f"""PCIe Waveform Analysis Summary
Source: {self.error_analyzer.vcd_file}
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Key Findings:
- Total Errors: {len(self.error_analyzer.error_contexts)}
- Error Types: {', '.join(set(c.error_type for c in self.error_analyzer.error_contexts))}
- Root Causes Identified: {len([c for c, d in self.error_analyzer.root_causes.items() if d['detected']])}
- Critical Issues: {len([r for r in self.error_analyzer.recommendations if r['severity'] == 'CRITICAL'])}

Signal Statistics:
- Total Events: {len(self.vcd_analyzer.events)}
- Transactions: {len(self.vcd_analyzer.transactions)}
- State Changes: {len(self.vcd_analyzer.state_changes)}
- Recovery Cycles: {len([e for e in self.vcd_analyzer.state_changes if e.details['new_state'] == 'RECOVERY'])}
"""
        
    def _create_error_context_chunk(self, context) -> str:
        """Create detailed error context chunk"""
        chunk = f"""PCIe Error Analysis: {context.error_type} at {context.error_time}ns

Error Details:
- Type: {context.error_type}
- Time: {context.error_time}ns
- LTSSM State: {context.ltssm_history[-1][1] if context.ltssm_history else 'UNKNOWN'}
- Recovery Attempts: {context.recovery_attempts}

Context Analysis:
- This error occurred during {context.ltssm_history[-1][1] if context.ltssm_history else 'unknown'} state
- {len(context.preceding_events)} events preceded this error
- {len(context.related_errors)} related errors detected nearby

Signal States at Error:
"""
        # Add key signal states
        important_signals = ['link_up', 'ltssm_state', 'tlp_valid', 'error_valid']
        for sig in important_signals:
            if sig in context.signal_states:
                chunk += f"- {sig}: {context.signal_states[sig]}\n"
                
        # Add preceding events summary
        if context.preceding_events:
            chunk += "\nPreceding Events:\n"
            for event in context.preceding_events[-3:]:
                chunk += f"- [{event.time}ns] {event.event_type}: {event.details}\n"
                
        return chunk
        
    def _create_timing_chunk(self, metrics) -> str:
        """Create timing analysis chunk"""
        # import numpy as np  # Using built-in functions instead
        
        chunk = f"""PCIe Timing Analysis Results

Transaction Performance:
- Average Latency: {sum(metrics.transaction_latencies) / len(metrics.transaction_latencies) if metrics.transaction_latencies else 0:.0f}ns
- Maximum Latency: {max(metrics.transaction_latencies) if metrics.transaction_latencies else 0:.0f}ns
- Minimum Latency: {min(metrics.transaction_latencies) if metrics.transaction_latencies else 0:.0f}ns
- Timeout Events: {len(metrics.timeout_events)}

Recovery Analysis:
- Recovery Events: {len(metrics.recovery_durations)}
- Average Recovery Time: {sum(metrics.recovery_durations) / len(metrics.recovery_durations) if metrics.recovery_durations else 0:.0f}ns
- Maximum Recovery Time: {max(metrics.recovery_durations) if metrics.recovery_durations else 0:.0f}ns

Error Timing:
- Error Clustering: {'Yes' if metrics.error_intervals and min(metrics.error_intervals) < 1000000 else 'No'}
- Average Error Interval: {sum(metrics.error_intervals) / len(metrics.error_intervals) if metrics.error_intervals else 0:.0f}ns
"""
        
        # Add timeout details if any
        if metrics.timeout_events:
            chunk += "\nTimeout Details:\n"
            for time, tag in metrics.timeout_events[:5]:
                chunk += f"- [{time}ns] Transaction timeout for {tag}\n"
                
        return chunk
        
    def _create_root_cause_chunk(self, cause: str, details: Dict) -> str:
        """Create root cause analysis chunk"""
        chunk = f"""Root Cause Analysis: {cause.replace('_', ' ').title()}

Confidence Level: {details['confidence']:.1%}
Detection: CONFIRMED

Evidence:
"""
        for evidence in details['evidence']:
            chunk += f"- {evidence}\n"
            
        # Add specific details based on cause
        if cause == 'signal_integrity':
            chunk += """
Analysis: Signal integrity issues detected based on excessive recovery cycles and CRC errors.
This indicates problems with the physical layer, possibly due to:
- Poor PCB routing or impedance mismatches
- Electromagnetic interference (EMI)
- Inadequate power delivery
- Clock jitter or timing issues
"""
        elif cause == 'timeout_issue':
            chunk += """
Analysis: Completion timeout issues detected, indicating:
- Device not responding to requests
- Credit flow problems preventing completions
- Possible deadlock in transaction ordering
- Power management causing device unavailability
"""
        elif cause == 'protocol_violation':
            chunk += """
Analysis: Protocol violations detected in TLP formation:
- Malformed TLP headers
- Invalid field combinations
- Incorrect message routing
- Non-compliant request formatting
"""
            
        return chunk
        
    def _create_recommendation_chunk(self, rec: Dict) -> str:
        """Create recommendation chunk"""
        chunk = f"""PCIe Debug Recommendation: {rec['issue']}

Severity: {rec['severity']}

Recommended Actions:
"""
        for i, action in enumerate(rec['actions'], 1):
            chunk += f"{i}. {action}\n"
            
        if rec.get('evidence'):
            chunk += "\nSupporting Evidence:\n"
            for evidence in rec['evidence']:
                chunk += f"- {evidence}\n"
                
        return chunk
        
    def _extract_patterns(self) -> List[str]:
        """Extract additional patterns from analysis"""
        patterns = []
        
        # Pattern 1: Error clustering
        if self.error_analyzer.timing_metrics and self.error_analyzer.timing_metrics.error_intervals:
            if any(interval < 1000000 for interval in self.error_analyzer.timing_metrics.error_intervals):
                patterns.append("""Pattern: Error Clustering Detected
Multiple errors occurring within 1ms window indicates:
- Systematic issue rather than random failures
- Possible cascading failure scenario
- Need to address root cause to prevent error propagation
""")
                
        # Pattern 2: Recovery inefficiency
        recovery_events = [e for e in self.vcd_analyzer.state_changes 
                          if e.details['new_state'] == 'RECOVERY']
        if len(recovery_events) > 5:
            patterns.append(f"""Pattern: Excessive Recovery Cycles
Recovery entered {len(recovery_events)} times, indicating:
- Link stability issues requiring frequent retraining
- Possible signal integrity problems
- Need to investigate physical layer health
- Consider reducing link speed for stability
""")
            
        return patterns
        
    def _ingest_to_rag(self, text_chunks: List[str]):
        """Ingest text chunks into RAG engine"""
        if not self.rag_bridge:
            return
            
        print(f"Ingesting {len(text_chunks)} chunks into RAG engine...")
        
        # Process each chunk
        for chunk in text_chunks:
            # Simulate log processing (RAG bridge expects log format)
            self.rag_bridge.log_capture.process_log(f"[VCD Analysis] {chunk}")
            
        # Process any remaining batch
        self.rag_bridge._process_log_batch()
        
        print("VCD analysis ingested into RAG engine")
        
    def query_vcd_analysis(self, query: str) -> Dict[str, Any]:
        """Query the VCD analysis using RAG"""
        if not self.rag_bridge:
            return {
                'answer': 'RAG engine not available. Please initialize with create_rag_bridge()',
                'confidence': 0.0
            }
            
        # Enhance query with VCD context
        enhanced_query = f"""
Based on the PCIe waveform analysis from VCD file:
- Total Errors: {len(self.error_analyzer.error_contexts)}
- Root Causes: {list(self.error_analyzer.root_causes.keys())}
- Timing Issues: {len(self.error_analyzer.timing_metrics.timeout_events) if self.error_analyzer.timing_metrics else 0}

User Query: {query}
"""
        
        # Get analysis from RAG
        result = self.rag_bridge.analyze_simulation(enhanced_query)
        
        return result
        
    def generate_intelligent_report(self) -> str:
        """Generate report with AI insights"""
        # Get base report
        report = self.error_analyzer.generate_rag_enhanced_report()
        
        # Add AI analysis section if RAG available
        if self.rag_bridge:
            report += "\n## AI-Powered Insights\n"
            
            # Query for insights
            queries = [
                "What is the most likely root cause of these PCIe errors?",
                "What is the recommended debugging approach?",
                "Are these errors related to hardware or firmware issues?"
            ]
            
            for query in queries:
                result = self.query_vcd_analysis(query)
                report += f"\n### {query}\n"
                report += f"{result.answer}\n"
                if hasattr(result, 'confidence'):
                    report += f"(Confidence: {result.confidence:.2f})\n"
                    
        return report


def demonstrate_vcd_rag_integration():
    """Demonstrate complete VCD-RAG integration"""
    print("PCIe VCD-RAG Integration Demo")
    print("=" * 60)
    
    # Check for VCD file
    vcd_file = "pcie_waveform.vcd"
    if not Path(vcd_file).exists():
        print(f"VCD file '{vcd_file}' not found.")
        print("Please run ./generate_vcd.sh first.")
        return
        
    # Try to create RAG bridge
    rag_bridge = None
    try:
        print("\nInitializing RAG engine...")
        rag_bridge = create_rag_bridge()
        print("✓ RAG engine initialized")
    except Exception as e:
        print(f"⚠ RAG engine not available: {e}")
        print("Continuing with VCD analysis only...")
        
    # Create integration
    integration = VCDRAGIntegration(rag_bridge)
    
    # Analyze VCD
    print(f"\nAnalyzing VCD file...")
    results = integration.analyze_vcd_file(vcd_file)
    
    print(f"\nAnalysis complete:")
    print(f"  - Events extracted: {len(results['vcd_data']['events'])}")
    print(f"  - Errors found: {len(integration.error_analyzer.error_contexts)}")
    print(f"  - Text chunks created: {len(results['text_chunks'])}")
    
    # Generate report
    print("\nGenerating intelligent report...")
    report = integration.generate_intelligent_report()
    
    # Save report
    report_file = f"vcd_rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"Report saved to: {report_file}")
    
    # Demonstrate queries
    if rag_bridge:
        print("\n" + "="*60)
        print("Interactive Query Demo")
        print("="*60)
        
        example_queries = [
            "Why did the PCIe link enter recovery state multiple times?",
            "What hardware changes would fix these errors?",
            "Is this a signal integrity or protocol issue?"
        ]
        
        for query in example_queries:
            print(f"\nQ: {query}")
            result = integration.query_vcd_analysis(query)
            print(f"A: {result.answer}")
            
    else:
        print("\n(RAG queries unavailable - install dependencies for full functionality)")
        
    # Show sample from report
    print("\n" + "="*60)
    print("Report Preview")
    print("="*60)
    print(report[:1000] + "...\n")


if __name__ == "__main__":
    demonstrate_vcd_rag_integration()