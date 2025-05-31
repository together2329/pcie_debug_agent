#!/usr/bin/env python3
"""
Simple PCIe Simulation Test without cocotb dependencies
Demonstrates error injection and RAG analysis
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulators.rag_integration import SimulatorRAGBridge, create_rag_bridge


class SimplePCIeSimulator:
    """Simple PCIe simulator that generates logs without RTL"""
    
    def __init__(self):
        self.time_ns = 0
        self.link_up = False
        self.ltssm_state = "DETECT"
        self.error_injected = False
        
    def advance_time(self, ns):
        self.time_ns += ns
        
    def generate_log(self, message):
        return f"[{self.time_ns}] {message}"
        
    def simulate_link_training(self):
        """Simulate PCIe link training sequence"""
        logs = []
        
        # LTSSM state progression
        logs.append(self.generate_log("PCIe: LTSSM entering POLLING"))
        self.advance_time(100)
        
        logs.append(self.generate_log("PCIe: LTSSM entering CONFIG"))
        self.advance_time(200)
        
        self.link_up = True
        logs.append(self.generate_log("PCIe: Link UP - Speed Gen3 Width x16"))
        self.advance_time(50)
        
        return logs
        
    def simulate_transaction(self, tlp_type="MRd", address=0x1000, tag=0):
        """Simulate a PCIe transaction"""
        logs = []
        
        # TLP reception
        if tlp_type == "MRd":
            type_code = 0
        elif tlp_type == "MWr":
            type_code = 1
        else:
            type_code = 2
            
        logs.append(self.generate_log(
            f"PCIe: TLP Received - Type={type_code} Addr=0x{address:08x} "
            f"Data=0x00000000 Tag={tag}"
        ))
        self.advance_time(10)
        
        # Completion for reads
        if tlp_type == "MRd":
            self.advance_time(100)
            logs.append(self.generate_log(
                f"PCIe: Completion generated - Tag={tag} Status=SC"
            ))
            
        return logs
        
    def inject_error(self, error_type):
        """Inject various error types"""
        logs = []
        self.error_injected = True
        
        if error_type == "crc":
            logs.append(self.generate_log("PCIe: ERROR - CRC error injected"))
            self.advance_time(50)
            logs.append(self.generate_log("PCIe: ERROR - Entering RECOVERY due to errors"))
            self.advance_time(200)
            logs.append(self.generate_log("PCIe: RECOVERY complete, back to L0"))
            
        elif error_type == "timeout":
            logs.append(self.generate_log("PCIe: ERROR - Link training timeout in POLLING"))
            self.advance_time(1000)
            logs.append(self.generate_log("PCIe: ERROR - Completion timeout for tag 99"))
            
        elif error_type == "ecrc":
            logs.append(self.generate_log("PCIe: ERROR - ECRC error injected"))
            self.advance_time(50)
            
        elif error_type == "malformed":
            logs.append(self.generate_log("PCIe: ERROR - Malformed TLP detected"))
            self.advance_time(20)
            logs.append(self.generate_log("PCIe: ERROR - Unsupported TLP type 0x7"))
            
        return logs


def run_error_scenario(error_type, rag_bridge):
    """Run a specific error scenario"""
    print(f"\n{'='*60}")
    print(f"Running {error_type.upper()} Error Scenario")
    print(f"{'='*60}\n")
    
    # Create simulator
    sim = SimplePCIeSimulator()
    
    # Generate simulation logs
    all_logs = []
    
    # Link training
    all_logs.extend(sim.simulate_link_training())
    
    # Normal transactions
    for i in range(3):
        all_logs.extend(sim.simulate_transaction("MRd", 0x1000 + i*4, i))
        sim.advance_time(50)
        
    # Inject error
    all_logs.extend(sim.inject_error(error_type))
    
    # More transactions after error
    for i in range(3):
        all_logs.extend(sim.simulate_transaction("MWr", 0x2000 + i*4, 100+i))
        sim.advance_time(50)
    
    # Process logs through RAG bridge
    print("Simulation Logs:")
    print("-" * 40)
    for log in all_logs:
        print(f"SIM: {log}")
        rag_bridge.log_capture.process_log(log)
        
    print("\n" + "-" * 40)
    
    # Analyze results
    print("\nAnalyzing with AI...")
    
    # Get error summary
    error_summary = rag_bridge.log_capture.get_error_summary()
    print(f"\nDetected {error_summary['total_errors']} errors")
    print("Error breakdown:")
    for err_type, count in error_summary['error_counts'].items():
        print(f"  - {err_type}: {count}")
        
    # AI Analysis
    queries = [
        f"What caused the {error_type} error and how can it be fixed?",
        "What is the impact of this error on PCIe operation?",
        "What recovery mechanisms were triggered?"
    ]
    
    for query in queries:
        print(f"\n{'─'*50}")
        print(f"Q: {query}")
        print(f"{'─'*50}")
        
        try:
            result = rag_bridge.analyze_simulation(query)
            print(f"A: {result.answer}")
            
            if hasattr(result, 'confidence') and result.confidence:
                print(f"\nConfidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"Analysis error: {e}")


def main():
    """Run PCIe error simulation and analysis demo"""
    print("PCIe Error Simulation and Analysis Demo")
    print("=" * 60)
    print("(Running without cocotb - using simulated logs)")
    
    # Try to initialize RAG bridge
    print("\nInitializing AI analysis engine...")
    try:
        rag_bridge = create_rag_bridge()
        print("✓ AI engine ready")
    except Exception as e:
        print(f"✗ Failed to initialize AI engine: {e}")
        print("\nNote: The RAG engine requires the local LLM to be properly configured.")
        print("Continuing with simulation only...")
        
        # Create a mock bridge for demonstration
        class MockBridge:
            def __init__(self):
                self.log_capture = SimulatorRAGBridge(None).log_capture
                
            def analyze_simulation(self, query):
                class MockResult:
                    answer = "AI analysis not available - LLM not configured"
                    confidence = 0.0
                return MockResult()
                
        rag_bridge = MockBridge()
    
    # Run different error scenarios
    error_types = ["crc", "timeout", "ecrc", "malformed"]
    
    for error_type in error_types:
        run_error_scenario(error_type, rag_bridge)
        time.sleep(1)  # Brief pause between scenarios
        
    # Generate summary report
    print(f"\n{'='*60}")
    print("Simulation Summary")
    print(f"{'='*60}")
    
    error_summary = rag_bridge.log_capture.get_error_summary()
    print(f"\nTotal errors detected: {error_summary['total_errors']}")
    print("\nError type distribution:")
    for err_type, count in error_summary['error_counts'].items():
        print(f"  - {err_type}: {count} occurrences")
        
    print(f"\nTotal transactions processed: {len(rag_bridge.log_capture.transaction_buffer)}")
    print(f"Total log entries: {len(rag_bridge.log_capture.log_buffer)}")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"pcie_sim_report_{timestamp}.txt"
    
    report_content = f"""PCIe Error Simulation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Total Errors: {error_summary['total_errors']}
- Error Types: {error_summary['error_counts']}
- Total Transactions: {len(rag_bridge.log_capture.transaction_buffer)}
- Total Log Entries: {len(rag_bridge.log_capture.log_buffer)}

Sample Error Logs:
"""
    
    for error in rag_bridge.log_capture.error_buffer[:10]:
        report_content += f"\n[{error['timestamp']}ns] {error.get('error_type', 'UNKNOWN')}: {error['raw']}"
        
    with open(report_file, 'w') as f:
        f.write(report_content)
        
    print(f"\nReport saved to: {report_file}")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()