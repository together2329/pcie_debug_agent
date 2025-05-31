#!/usr/bin/env python3
"""
Standalone VCD Analysis Demo
Demonstrates VCD error analysis without full RAG dependencies
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import our VCD analyzers
from vcd_error_analyzer import PCIeVCDErrorAnalyzer
from vcd_rag_analyzer import PCIeVCDAnalyzer


class MockRAGEngine:
    """Mock RAG engine for demonstration"""
    
    def __init__(self):
        self.knowledge_base = []
        
    def add_knowledge(self, chunks: List[str]):
        """Add knowledge chunks"""
        self.knowledge_base.extend(chunks)
        print(f"Added {len(chunks)} knowledge chunks to RAG engine")
        
    def query(self, question: str) -> str:
        """Mock intelligent query processing"""
        question_lower = question.lower()
        
        # Simple pattern matching for demo
        if "root cause" in question_lower:
            return """Based on waveform analysis, the primary root cause appears to be protocol violations. 
The system detected malformed TLP packets which indicates issues with:
1. TLP header formation logic
2. Field validation in the transmitter  
3. Possible firmware bugs in packet generation
4. Protocol compliance verification gaps

Recommendation: Review the TLP generation code and validate against PCIe specification."""

        elif "fix" in question_lower or "recommendation" in question_lower:
            return """To fix the detected issues:

1. Immediate Actions:
   - Review TLP formation logic in transmitter
   - Validate TLP header field values
   - Check protocol compliance test coverage

2. Investigation:
   - Capture and analyze the malformed TLP details
   - Review recent firmware changes
   - Run protocol compliance tests

3. Long-term:
   - Implement stricter TLP validation
   - Add real-time protocol monitoring
   - Enhance error reporting mechanisms"""

        elif "signal integrity" in question_lower:
            return """The VCD analysis shows multiple recovery cycles indicating potential signal integrity issues:

Physical Layer Analysis:
- 12 recovery events detected (threshold: >5 = HIGH severity)
- Recovery cycles suggest link instability
- No direct CRC errors but pattern indicates SI problems

Recommendations:
- Check PCIe trace impedance and routing
- Measure eye diagrams at receiver
- Verify power delivery to PCIe components
- Check for EMI sources near PCIe lanes"""

        elif "timing" in question_lower:
            return """Timing analysis from waveform data:

Transaction Timing:
- Most transactions complete normally
- No significant latency issues detected
- No completion timeouts in analyzed window

State Machine Timing:
- LTSSM transitions occur at normal intervals
- Recovery cycles complete within spec
- Link training timing appears normal

The timing itself is not the primary issue - focus on protocol compliance."""

        else:
            return f"""Based on the VCD analysis, I can help with:
- Error root cause analysis
- Signal integrity assessment  
- Timing analysis
- Protocol compliance issues
- Debugging recommendations

Your question: "{question}"

The waveform shows {len([k for k in self.knowledge_base if 'Error' in k])} error events and provides detailed timing information for analysis."""


def run_comprehensive_vcd_analysis():
    """Run comprehensive VCD analysis with mock AI"""
    print("=" * 70)
    print("PCIe VCD Error Analysis with AI Integration Demo")
    print("=" * 70)
    
    # Check for VCD file
    vcd_file = "pcie_waveform.vcd"
    if not Path(vcd_file).exists():
        print(f"VCD file '{vcd_file}' not found.")
        print("Please run ./generate_vcd.sh first.")
        return
        
    print(f"\n1. Analyzing VCD file: {vcd_file}")
    print("-" * 50)
    
    # Run basic VCD analysis
    vcd_analyzer = PCIeVCDAnalyzer(vcd_file)
    vcd_data = vcd_analyzer.parse_vcd()
    
    print(f"✓ Basic analysis complete:")
    print(f"  - Events: {len(vcd_data['events'])}")
    print(f"  - Transactions: {len(vcd_data['transactions'])}")
    print(f"  - Errors: {len(vcd_data['errors'])}")
    print(f"  - State changes: {len(vcd_data['state_changes'])}")
    
    # Run advanced error analysis
    print(f"\n2. Running advanced error analysis...")
    print("-" * 50)
    
    error_analyzer = PCIeVCDErrorAnalyzer(vcd_file)
    error_results = error_analyzer.analyze_errors()
    
    print(f"✓ Error analysis complete:")
    print(f"  - Error contexts: {len(error_results['error_contexts'])}")
    print(f"  - Root causes: {list(error_results['root_causes'].keys())}")
    print(f"  - Recommendations: {len(error_results['recommendations'])}")
    
    # Generate knowledge for AI
    print(f"\n3. Generating AI knowledge base...")
    print("-" * 50)
    
    # Create mock RAG engine
    ai_engine = MockRAGEngine()
    
    # Generate knowledge chunks
    knowledge_chunks = []
    
    # Add summary chunk
    summary = f"""PCIe VCD Analysis Summary:
- Source: {vcd_file}
- Total events: {len(vcd_data['events'])}
- Errors found: {len(vcd_data['errors'])}
- Root causes: {list(error_results['root_causes'].keys())}
- Critical issues: {len([r for r in error_results['recommendations'] if r.get('severity') == 'CRITICAL'])}
"""
    knowledge_chunks.append(summary)
    
    # Add error details
    for i, context in enumerate(error_results['error_contexts']):
        chunk = f"""Error {i+1}: {context.error_type} at {context.error_time}ns
- LTSSM state: {context.ltssm_history[-1][1] if context.ltssm_history else 'UNKNOWN'}
- Recovery attempts: {context.recovery_attempts}
- Related errors: {len(context.related_errors)}
"""
        knowledge_chunks.append(chunk)
        
    # Add recommendations
    for rec in error_results['recommendations']:
        chunk = f"""Recommendation for {rec['issue']} (Severity: {rec['severity']}):
Actions: {'; '.join(rec['actions'][:3])}
"""
        knowledge_chunks.append(chunk)
        
    # Add to AI knowledge base
    ai_engine.add_knowledge(knowledge_chunks)
    
    # Generate comprehensive report
    print(f"\n4. Generating comprehensive report...")
    print("-" * 50)
    
    report = error_analyzer.generate_rag_enhanced_report()
    
    # Add AI insights to report
    ai_queries = [
        "What is the root cause of these PCIe errors?",
        "What are the recommended fixes?",
        "Are these signal integrity issues?"
    ]
    
    ai_insights = "\n## AI-Powered Insights\n"
    for query in ai_queries:
        response = ai_engine.query(query)
        ai_insights += f"\n### Q: {query}\n"
        ai_insights += f"{response}\n"
        
    report += ai_insights
    
    # Save report
    report_file = f"comprehensive_vcd_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"✓ Report saved to: {report_file}")
    
    # Interactive Q&A demo
    print(f"\n5. Interactive AI Q&A Demo")
    print("-" * 50)
    
    demo_questions = [
        "What is the most likely root cause?",
        "How can I fix these protocol violations?",
        "Are there signal integrity problems?",
        "What timing issues were found?"
    ]
    
    for question in demo_questions:
        print(f"\n🤔 Question: {question}")
        print("🤖 AI Response:")
        response = ai_engine.query(question)
        # Format response nicely
        lines = response.split('\n')
        for line in lines:
            if line.strip():
                print(f"   {line.strip()}")
        print()
        
    # Show key findings
    print("=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Waveform Statistics:")
    print(f"   • Simulation duration: ~4.4ms")
    print(f"   • Total events captured: {len(vcd_data['events'])}")
    print(f"   • PCIe transactions: {len(vcd_data['transactions'])}")
    print(f"   • LTSSM state changes: {len(vcd_data['state_changes'])}")
    
    print(f"\n🚨 Error Analysis:")
    error_types = {}
    for context in error_results['error_contexts']:
        error_types[context.error_type] = error_types.get(context.error_type, 0) + 1
    
    for error_type, count in error_types.items():
        print(f"   • {error_type}: {count} occurrences")
        
    print(f"\n🔍 Root Causes Identified:")
    for cause, details in error_results['root_causes'].items():
        if details['detected']:
            print(f"   • {cause.replace('_', ' ').title()}: {details['confidence']:.1%} confidence")
            
    print(f"\n🛠️  Actionable Recommendations:")
    for i, rec in enumerate(error_results['recommendations'][:3], 1):
        print(f"   {i}. {rec['issue']} - {rec['actions'][0]}")
        
    print(f"\n💡 Next Steps:")
    print(f"   1. Review TLP formation logic (protocol violations detected)")
    print(f"   2. Check signal integrity (multiple recovery cycles)")
    print(f"   3. Validate against PCIe specification")
    print(f"   4. Implement enhanced error monitoring")
    
    print(f"\n📋 Files Generated:")
    print(f"   • {report_file} - Comprehensive analysis report")
    print(f"   • pcie_waveform.vcd - Waveform data for GTKWave")
    print(f"   • vcd_analysis_report.txt - Basic VCD analysis")
    
    print(f"\n🎯 This demo shows how VCD waveform analysis can be enhanced")
    print(f"   with AI to provide intelligent debugging insights!")


if __name__ == "__main__":
    run_comprehensive_vcd_analysis()