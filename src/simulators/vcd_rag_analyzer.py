#!/usr/bin/env python3
"""
VCD Waveform Analysis with RAG Integration
Demonstrates how to extract insights from VCD files and use AI for analysis
"""

import re
import json
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class VCDEvent:
    """Represents a significant event extracted from VCD"""
    def __init__(self, time: int, event_type: str, details: Dict[str, Any]):
        self.time = time
        self.event_type = event_type
        self.details = details
        
    def to_text(self) -> str:
        """Convert event to RAG-friendly text format"""
        text = f"[Time: {self.time}ns] {self.event_type}\n"
        for key, value in self.details.items():
            text += f"  - {key}: {value}\n"
        return text


class PCIeVCDAnalyzer:
    """Analyzes VCD files for PCIe-specific patterns and events"""
    
    def __init__(self, vcd_file: str):
        self.vcd_file = vcd_file
        self.signals = {}
        self.events = []
        self.transactions = []
        self.errors = []
        self.state_changes = []
        
    def parse_vcd(self) -> Dict[str, Any]:
        """Parse VCD file and extract PCIe events"""
        with open(self.vcd_file, 'r') as f:
            content = f.read()
            
        # Parse header for signal definitions
        self._parse_header(content)
        
        # Extract events from value changes
        self._extract_events(content)
        
        # Analyze patterns
        self._analyze_patterns()
        
        return {
            'signals': self.signals,
            'events': self.events,
            'transactions': self.transactions,
            'errors': self.errors,
            'state_changes': self.state_changes
        }
        
    def _parse_header(self, content: str):
        """Parse VCD header for signal information"""
        var_pattern = r'\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[\d+:\d+\])?\s+\$end'
        for match in re.finditer(var_pattern, content):
            width = int(match.group(1))
            symbol = match.group(2)
            name = match.group(3)
            self.signals[symbol] = {
                'name': name,
                'width': width,
                'changes': []
            }
            
    def _extract_events(self, content: str):
        """Extract PCIe-relevant events from VCD"""
        # Find value change section
        changes_start = content.find('$dumpvars')
        if changes_start == -1:
            return
            
        changes_section = content[changes_start:]
        current_time = 0
        
        # Track signal states
        signal_states = {}
        
        for line in changes_section.split('\n'):
            line = line.strip()
            
            # Time change
            if line.startswith('#'):
                new_time = int(line[1:])
                
                # Check for events at this time
                self._check_for_events(current_time, new_time, signal_states)
                current_time = new_time
                
            # Value change
            elif len(line) >= 2:
                if line[0] in '01xzXZ':
                    value = line[0]
                    symbol = line[1:]
                    if symbol in self.signals:
                        signal_states[symbol] = value
                        self.signals[symbol]['changes'].append((current_time, value))
                elif line.startswith('b'):
                    parts = line[1:].split()
                    if len(parts) == 2:
                        value = parts[0]
                        symbol = parts[1]
                        if symbol in self.signals:
                            signal_states[symbol] = value
                            self.signals[symbol]['changes'].append((current_time, value))
                            
    def _check_for_events(self, start_time: int, end_time: int, states: Dict[str, str]):
        """Check for PCIe events based on signal states"""
        # Find relevant signals
        tlp_valid = self._get_signal_by_name('tlp_valid')
        error_valid = self._get_signal_by_name('error_valid')
        ltssm_state = self._get_signal_by_name('ltssm_state')
        
        # Check for transactions
        if tlp_valid and tlp_valid in states and states[tlp_valid] == '1':
            self._extract_transaction(end_time, states)
            
        # Check for errors
        if error_valid and error_valid in states and states[error_valid] == '1':
            self._extract_error(end_time, states)
            
        # Check for state changes
        if ltssm_state and ltssm_state in states:
            self._check_state_change(end_time, states)
            
    def _get_signal_by_name(self, name: str) -> str:
        """Get signal symbol by name"""
        for symbol, info in self.signals.items():
            if info['name'] == name:
                return symbol
        return None
        
    def _extract_transaction(self, time: int, states: Dict[str, str]):
        """Extract transaction details"""
        details = {
            'type': self._decode_tlp_type(states),
            'address': self._decode_address(states),
            'tag': self._decode_tag(states)
        }
        
        event = VCDEvent(time, "PCIe Transaction", details)
        self.events.append(event)
        self.transactions.append(event)
        
    def _extract_error(self, time: int, states: Dict[str, str]):
        """Extract error details"""
        error_type_symbol = self._get_signal_by_name('error_type')
        error_type = 'UNKNOWN'
        
        if error_type_symbol and error_type_symbol in states:
            error_code = int(states[error_type_symbol], 2) if states[error_type_symbol] not in ['x', 'z'] else 0
            error_map = {1: 'CRC_ERROR', 2: 'TIMEOUT', 3: 'ECRC_ERROR', 4: 'MALFORMED_TLP'}
            error_type = error_map.get(error_code, f'ERROR_{error_code}')
            
        details = {
            'error_type': error_type,
            'ltssm_state': self._decode_ltssm_state(states)
        }
        
        event = VCDEvent(time, "PCIe Error", details)
        self.events.append(event)
        self.errors.append(event)
        
    def _check_state_change(self, time: int, states: Dict[str, str]):
        """Check for LTSSM state changes"""
        ltssm_symbol = self._get_signal_by_name('ltssm_state')
        if ltssm_symbol and ltssm_symbol in states:
            state = self._decode_ltssm_state(states)
            
            # Check if this is a new state
            if not self.state_changes or self.state_changes[-1].details['new_state'] != state:
                details = {
                    'new_state': state,
                    'previous_state': self.state_changes[-1].details['new_state'] if self.state_changes else 'UNKNOWN'
                }
                
                event = VCDEvent(time, "LTSSM State Change", details)
                self.events.append(event)
                self.state_changes.append(event)
                
    def _decode_tlp_type(self, states: Dict[str, str]) -> str:
        """Decode TLP type from signal states"""
        tlp_type_symbol = self._get_signal_by_name('tlp_type')
        if tlp_type_symbol and tlp_type_symbol in states:
            value = states[tlp_type_symbol]
            if value not in ['x', 'z']:
                type_code = int(value, 2) if len(value) > 1 else int(value)
                type_map = {0: 'Memory Read', 1: 'Memory Write', 2: 'Completion'}
                return type_map.get(type_code, f'Type_{type_code}')
        return 'UNKNOWN'
        
    def _decode_address(self, states: Dict[str, str]) -> str:
        """Decode address from signal states"""
        addr_symbol = self._get_signal_by_name('tlp_address')
        if addr_symbol and addr_symbol in states:
            value = states[addr_symbol]
            if value not in ['x', 'z'] and value.isdigit():
                return f"0x{int(value, 2):08x}"
        return '0x????????'
        
    def _decode_tag(self, states: Dict[str, str]) -> int:
        """Decode tag from signal states"""
        tag_symbol = self._get_signal_by_name('tlp_tag')
        if tag_symbol and tag_symbol in states:
            value = states[tag_symbol]
            if value not in ['x', 'z'] and value.isdigit():
                return int(value, 2)
        return 0
        
    def _decode_ltssm_state(self, states: Dict[str, str]) -> str:
        """Decode LTSSM state"""
        ltssm_symbol = self._get_signal_by_name('ltssm_state')
        if ltssm_symbol and ltssm_symbol in states:
            value = states[ltssm_symbol]
            if value not in ['x', 'z']:
                state_code = int(value, 2) if len(value) > 1 else int(value)
                state_map = {0: 'DETECT', 1: 'POLLING', 2: 'CONFIG', 3: 'L0', 4: 'RECOVERY'}
                return state_map.get(state_code, f'STATE_{state_code}')
        return 'UNKNOWN'
        
    def _analyze_patterns(self):
        """Analyze events for patterns and anomalies"""
        # Detect excessive recovery cycles
        recovery_count = sum(1 for e in self.state_changes 
                           if e.details['new_state'] == 'RECOVERY')
        
        if recovery_count > 5:
            event = VCDEvent(
                0,  # Summary event
                "Pattern: Excessive Recovery",
                {
                    'recovery_count': recovery_count,
                    'severity': 'HIGH',
                    'recommendation': 'Check signal integrity and power delivery'
                }
            )
            self.events.append(event)
            
        # Detect timeout patterns
        timeout_errors = [e for e in self.errors if 'TIMEOUT' in e.details.get('error_type', '')]
        if timeout_errors:
            event = VCDEvent(
                0,
                "Pattern: Completion Timeouts",
                {
                    'timeout_count': len(timeout_errors),
                    'severity': 'CRITICAL',
                    'recommendation': 'Verify device responsiveness and credit flow'
                }
            )
            self.events.append(event)
            
    def generate_rag_text(self) -> List[str]:
        """Generate text chunks for RAG ingestion"""
        chunks = []
        
        # Summary chunk
        summary = f"""PCIe Waveform Analysis Summary
Source: {self.vcd_file}
Total Events: {len(self.events)}
Transactions: {len(self.transactions)}
Errors: {len(self.errors)}
State Changes: {len(self.state_changes)}
"""
        chunks.append(summary)
        
        # Event chunks
        for event in self.events:
            chunks.append(event.to_text())
            
        # Pattern analysis chunks
        patterns = [e for e in self.events if e.event_type.startswith("Pattern:")]
        if patterns:
            pattern_text = "Detected Patterns and Anomalies:\n"
            for pattern in patterns:
                pattern_text += f"\n{pattern.to_text()}"
            chunks.append(pattern_text)
            
        return chunks
        
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
PCIe VCD Analysis Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source: {self.vcd_file}

## Summary Statistics
- Total Events: {len(self.events)}
- Transactions: {len(self.transactions)}
- Errors Detected: {len(self.errors)}
- LTSSM State Changes: {len(self.state_changes)}

## Error Analysis
"""
        
        if self.errors:
            error_types = defaultdict(int)
            for error in self.errors:
                error_types[error.details.get('error_type', 'UNKNOWN')] += 1
                
            report += "Error Distribution:\n"
            for error_type, count in error_types.items():
                report += f"  - {error_type}: {count} occurrences\n"
                
            report += "\nError Timeline:\n"
            for error in self.errors[:10]:  # First 10 errors
                report += f"  {error.to_text()}\n"
        else:
            report += "No errors detected.\n"
            
        report += "\n## Transaction Analysis\n"
        if self.transactions:
            tx_types = defaultdict(int)
            for tx in self.transactions:
                tx_types[tx.details.get('type', 'UNKNOWN')] += 1
                
            report += "Transaction Types:\n"
            for tx_type, count in tx_types.items():
                report += f"  - {tx_type}: {count}\n"
                
        report += "\n## State Machine Analysis\n"
        if self.state_changes:
            report += "LTSSM State Progression:\n"
            for i, change in enumerate(self.state_changes[:10]):
                report += f"  {i+1}. [{change.time}ns] {change.details['previous_state']} → {change.details['new_state']}\n"
                
        # Add pattern detection results
        patterns = [e for e in self.events if e.event_type.startswith("Pattern:")]
        if patterns:
            report += "\n## Detected Patterns\n"
            for pattern in patterns:
                report += f"\n{pattern.to_text()}"
                
        return report


def demonstrate_vcd_analysis():
    """Demonstrate VCD analysis capabilities"""
    print("PCIe VCD Waveform Analysis Demo")
    print("=" * 50)
    
    # Check if VCD file exists
    vcd_file = "pcie_waveform.vcd"
    if not Path(vcd_file).exists():
        print(f"VCD file '{vcd_file}' not found.")
        print("Run ./generate_vcd.sh first to create a waveform.")
        return
        
    # Analyze VCD
    print(f"\nAnalyzing {vcd_file}...")
    analyzer = PCIeVCDAnalyzer(vcd_file)
    analysis = analyzer.parse_vcd()
    
    # Generate report
    report = analyzer.generate_analysis_report()
    print(report)
    
    # Generate RAG text chunks
    print("\n" + "="*50)
    print("RAG-Ready Text Chunks")
    print("="*50)
    
    chunks = analyzer.generate_rag_text()
    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
        print(f"\nChunk {i+1}:")
        print("-" * 30)
        print(chunk)
        
    print(f"\n... and {len(chunks)-5} more chunks")
    
    # Show how to query
    print("\n" + "="*50)
    print("Example AI Queries")
    print("="*50)
    
    example_queries = [
        "What errors occurred during the PCIe simulation?",
        "Why did the link enter recovery state multiple times?",
        "What is the root cause of the CRC errors?",
        "How can I fix the timeout issues?"
    ]
    
    for query in example_queries:
        print(f"\nQ: {query}")
        print("A: [AI would analyze the VCD data and provide recommendations]")
        
    # Save analysis
    output_file = "vcd_analysis_report.txt"
    with open(output_file, 'w') as f:
        f.write(report)
        f.write("\n\nRAG Text Chunks:\n")
        f.write("="*50 + "\n")
        for chunk in chunks:
            f.write(f"\n{chunk}\n")
            
    print(f"\n\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    demonstrate_vcd_analysis()