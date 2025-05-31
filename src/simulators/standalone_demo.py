#!/usr/bin/env python3
"""
Standalone PCIe Error Simulation Demo
Demonstrates the integration concept without external dependencies
"""

import time
import random
from datetime import datetime
from typing import List, Dict, Any


class PCIeErrorSimulator:
    """Simulates PCIe operations and error injection"""
    
    def __init__(self):
        self.time_ns = 0
        self.logs = []
        self.errors = []
        self.transactions = []
        
    def log(self, message: str, is_error: bool = False):
        """Add a log entry"""
        entry = {
            "timestamp": self.time_ns,
            "message": message,
            "is_error": is_error
        }
        self.logs.append(entry)
        if is_error:
            self.errors.append(entry)
        print(f"[{self.time_ns:6d}ns] {message}")
        
    def advance_time(self, ns: int):
        """Advance simulation time"""
        self.time_ns += ns
        
    def simulate_link_training(self):
        """Simulate PCIe link training"""
        print("\n--- Link Training Phase ---")
        self.log("PCIe: LTSSM state DETECT")
        self.advance_time(100)
        
        self.log("PCIe: LTSSM entering POLLING")
        self.advance_time(200)
        
        self.log("PCIe: LTSSM entering CONFIG")
        self.advance_time(300)
        
        self.log("PCIe: Link UP - Speed Gen3 Width x16")
        self.advance_time(50)
        
    def send_tlp(self, tlp_type: str, address: int, tag: int):
        """Send a TLP transaction"""
        transaction = {
            "time": self.time_ns,
            "type": tlp_type,
            "address": address,
            "tag": tag
        }
        self.transactions.append(transaction)
        
        self.log(f"PCIe: TLP {tlp_type} - Addr=0x{address:08x} Tag={tag}")
        self.advance_time(10)
        
        # Generate completion for reads
        if tlp_type == "MRd":
            self.advance_time(100)
            self.log(f"PCIe: Completion - Tag={tag} Status=SC")
            
    def inject_crc_error(self):
        """Inject CRC error"""
        print("\n--- Injecting CRC Error ---")
        self.log("PCIe: ERROR - CRC error detected on TLP", is_error=True)
        self.advance_time(50)
        self.log("PCIe: ERROR - Link entering RECOVERY state", is_error=True)
        self.advance_time(500)
        self.log("PCIe: RECOVERY complete, link restored to L0")
        
    def inject_timeout_error(self):
        """Inject timeout error"""
        print("\n--- Injecting Timeout Error ---")
        self.log("PCIe: ERROR - Completion timeout for Tag=99", is_error=True)
        self.advance_time(1000)
        self.log("PCIe: ERROR - Device not responding", is_error=True)
        
    def inject_ecrc_error(self):
        """Inject ECRC error"""
        print("\n--- Injecting ECRC Error ---")
        self.log("PCIe: ERROR - ECRC mismatch in TLP", is_error=True)
        self.advance_time(50)
        self.log("PCIe: ERROR - TLP discarded due to ECRC failure", is_error=True)
        
    def inject_malformed_tlp(self):
        """Inject malformed TLP error"""
        print("\n--- Injecting Malformed TLP ---")
        self.log("PCIe: ERROR - Malformed TLP detected", is_error=True)
        self.advance_time(20)
        self.log("PCIe: ERROR - Invalid TLP format field", is_error=True)
        self.log("PCIe: ERROR - TLP dropped, sending UR completion", is_error=True)


class SimpleAnalyzer:
    """Simple analyzer that mimics RAG/LLM analysis"""
    
    def __init__(self):
        self.error_patterns = {
            "CRC": {
                "cause": "Signal integrity issues, EMI, or physical layer problems",
                "impact": "Corrupted data transmission, link recovery required",
                "fix": "Check signal routing, add shielding, verify termination"
            },
            "timeout": {
                "cause": "Device not responding, completion lost, or deadlock",
                "impact": "Transaction failure, potential system hang",
                "fix": "Increase timeout values, check device power state, verify routing"
            },
            "ECRC": {
                "cause": "End-to-end data corruption, memory errors",
                "impact": "Data integrity compromise, transaction retry needed",
                "fix": "Enable ECC memory, check data path integrity"
            },
            "Malformed": {
                "cause": "Protocol violation, incorrect TLP formatting",
                "impact": "Transaction rejected, UR status returned",
                "fix": "Fix TLP generation logic, verify protocol compliance"
            }
        }
        
    def analyze_errors(self, errors: List[Dict]):
        """Analyze errors and provide insights"""
        if not errors:
            return "No errors detected in simulation."
            
        analysis = []
        error_types = {}
        
        # Categorize errors
        for error in errors:
            msg = error["message"]
            for error_type in ["CRC", "timeout", "ECRC", "Malformed"]:
                if error_type in msg:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                    
        # Generate analysis
        analysis.append(f"Total errors detected: {len(errors)}")
        analysis.append("\nError breakdown:")
        
        for error_type, count in error_types.items():
            analysis.append(f"  - {error_type}: {count} occurrences")
            
        analysis.append("\nDetailed Analysis:")
        
        for error_type, count in error_types.items():
            if error_type in self.error_patterns:
                pattern = self.error_patterns[error_type]
                analysis.append(f"\n{error_type} Errors ({count} occurrences):")
                analysis.append(f"  Likely Cause: {pattern['cause']}")
                analysis.append(f"  Impact: {pattern['impact']}")
                analysis.append(f"  Recommended Fix: {pattern['fix']}")
                
        return "\n".join(analysis)
        
    def generate_recommendations(self, errors: List[Dict], transactions: List[Dict]):
        """Generate recommendations based on simulation results"""
        recommendations = []
        
        if len(errors) > 10:
            recommendations.append("- High error rate detected. Check physical layer integrity.")
            
        # Check for patterns
        if any("RECOVERY" in e["message"] for e in errors):
            recommendations.append("- Multiple link recovery events. Consider reducing link speed for stability.")
            
        if any("timeout" in e["message"] for e in errors):
            recommendations.append("- Timeout errors present. Review completion credit flow and buffer sizes.")
            
        if len(transactions) > 0 and len(errors) / len(transactions) > 0.1:
            recommendations.append("- Error rate >10%. System reliability compromised.")
            
        return recommendations


def run_comprehensive_test():
    """Run comprehensive PCIe error simulation"""
    print("="*70)
    print("PCIe Error Simulation and Analysis Demo")
    print("="*70)
    
    simulator = PCIeErrorSimulator()
    analyzer = SimpleAnalyzer()
    
    # Phase 1: Link Training
    simulator.simulate_link_training()
    
    # Phase 2: Normal Operations
    print("\n--- Normal Transaction Phase ---")
    for i in range(5):
        simulator.send_tlp("MRd", 0x1000 + i*0x100, i)
        simulator.advance_time(50)
        
    # Phase 3: Error Injection Tests
    error_scenarios = [
        ("CRC Error Test", simulator.inject_crc_error),
        ("Timeout Error Test", simulator.inject_timeout_error),
        ("ECRC Error Test", simulator.inject_ecrc_error),
        ("Malformed TLP Test", simulator.inject_malformed_tlp)
    ]
    
    for test_name, error_func in error_scenarios:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        # Normal transactions before error
        for i in range(3):
            simulator.send_tlp("MWr", 0x2000 + i*4, 50+i)
            simulator.advance_time(20)
            
        # Inject error
        error_func()
        
        # Recovery transactions
        for i in range(2):
            simulator.send_tlp("MRd", 0x3000 + i*4, 100+i)
            simulator.advance_time(30)
            
    # Phase 4: Analysis
    print("\n" + "="*70)
    print("SIMULATION ANALYSIS")
    print("="*70)
    
    # Basic statistics
    print(f"\nSimulation Statistics:")
    print(f"  Total simulation time: {simulator.time_ns}ns")
    print(f"  Total log entries: {len(simulator.logs)}")
    print(f"  Total transactions: {len(simulator.transactions)}")
    print(f"  Total errors: {len(simulator.errors)}")
    
    # AI-style analysis
    print("\n--- Error Analysis ---")
    analysis = analyzer.analyze_errors(simulator.errors)
    print(analysis)
    
    # Recommendations
    print("\n--- Recommendations ---")
    recommendations = analyzer.generate_recommendations(simulator.errors, simulator.transactions)
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("No specific recommendations. System operating within normal parameters.")
        
    # Generate report
    print("\n--- Generating Report ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"pcie_simulation_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PCIe Error Simulation Report\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("Simulation Summary:\n")
        f.write(f"  Duration: {simulator.time_ns}ns\n")
        f.write(f"  Transactions: {len(simulator.transactions)}\n")
        f.write(f"  Errors: {len(simulator.errors)}\n\n")
        
        f.write("Error Log:\n")
        f.write("-"*50 + "\n")
        for error in simulator.errors:
            f.write(f"[{error['timestamp']}ns] {error['message']}\n")
            
        f.write("\n" + analysis)
        
        if recommendations:
            f.write("\n\nRecommendations:\n")
            for rec in recommendations:
                f.write(f"{rec}\n")
                
    print(f"Report saved to: {report_file}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    
    # Show how this would integrate with real RTL
    print("\nIntegration with RTL Simulation:")
    print("1. Replace simulated logs with real Verilog $display outputs")
    print("2. Use cocotb to drive actual PCIe RTL signals")
    print("3. pyuvm provides full UVM methodology in Python")
    print("4. RAG/LLM analyzes logs in real-time for intelligent insights")
    
    print("\nTo run with actual RTL simulation:")
    print("  1. Install cocotb: pip install cocotb pyuvm")
    print("  2. cd src/simulators")
    print("  3. make test_comprehensive")


if __name__ == "__main__":
    run_comprehensive_test()