#!/usr/bin/env python3
"""
PCIe Error Simulation and Analysis Demo

This script demonstrates:
1. Running PCIe UVM testbench with error injection
2. Capturing simulator logs in real-time
3. Analyzing errors using RAG/LLM
4. Generating comprehensive error reports
"""

import subprocess
import sys
import time
import threading
import queue
from pathlib import Path
import re
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.simulators.rag_integration import SimulatorRAGBridge, create_rag_bridge


class PCIeErrorDemo:
    """Demo runner for PCIe error simulation and analysis"""
    
    def __init__(self):
        self.log_queue = queue.Queue()
        self.simulation_complete = False
        self.rag_bridge = None
        
    def run_simulation(self, test_name: str):
        """Run a specific test and capture output"""
        print(f"\n{'='*60}")
        print(f"Running PCIe Test: {test_name}")
        print(f"{'='*60}\n")
        
        # Run make command
        cmd = ["make", "-C", str(Path(__file__).parent), f"TESTCASE={test_name}"]
        
        try:
            # Start the simulation process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Capture output line by line
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.rstrip()
                    print(f"SIM: {line}")
                    
                    # Add to queue for processing
                    self.log_queue.put(line)
                    
                    # Process PCIe specific logs
                    if "PCIe:" in line or "ERROR" in line:
                        if self.rag_bridge:
                            self.rag_bridge.log_capture.process_log(line)
                            
            process.wait()
            
            if process.returncode == 0:
                print("\n✓ Simulation completed successfully")
            else:
                print(f"\n✗ Simulation failed with return code: {process.returncode}")
                
        except Exception as e:
            print(f"Error running simulation: {e}")
            
    def analyze_results(self):
        """Analyze simulation results using RAG/LLM"""
        if not self.rag_bridge:
            print("RAG bridge not initialized")
            return
            
        print(f"\n{'='*60}")
        print("Analyzing Simulation Results with AI")
        print(f"{'='*60}\n")
        
        # Get error summary
        error_summary = self.rag_bridge.log_capture.get_error_summary()
        
        print(f"Detected {error_summary['total_errors']} errors")
        print("\nError Types:")
        for error_type, count in error_summary['error_counts'].items():
            print(f"  - {error_type}: {count}")
            
        # Perform various analyses
        queries = [
            "What PCIe errors occurred and what is their root cause?",
            "What is the sequence of events that led to the errors?",
            "How can these PCIe errors be fixed in the design?",
            "What are the potential impacts of these errors on system operation?"
        ]
        
        for query in queries:
            print(f"\n{'─'*50}")
            print(f"Q: {query}")
            print(f"{'─'*50}")
            
            result = self.rag_bridge.analyze_simulation(query)
            print(f"A: {result.answer}")
            
            if result.confidence:
                print(f"\nConfidence: {result.confidence:.2f}")
                
    def generate_report(self, output_file: str = "pcie_error_report.txt"):
        """Generate comprehensive error analysis report"""
        if not self.rag_bridge:
            print("RAG bridge not initialized")
            return
            
        print(f"\n{'='*60}")
        print("Generating Comprehensive Report")
        print(f"{'='*60}\n")
        
        report = self.rag_bridge.get_simulation_report()
        
        # Add timestamp and test information
        full_report = f"""
PCIe Error Simulation and Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{report}
"""
        
        # Save to file
        output_path = Path(__file__).parent / output_file
        with open(output_path, 'w') as f:
            f.write(full_report)
            
        print(f"Report saved to: {output_path}")
        
        # Also print summary
        print("\nReport Summary:")
        print("─" * 50)
        lines = report.split('\n')
        for line in lines[:20]:  # First 20 lines
            print(line)
        print("...")
        print(f"\n(Full report saved to {output_path})")
        
    def run_demo(self):
        """Run the complete demo"""
        print("PCIe Error Simulation and Analysis Demo")
        print("=" * 60)
        
        # Initialize RAG bridge
        print("\nInitializing AI analysis engine...")
        try:
            self.rag_bridge = create_rag_bridge()
            print("✓ AI engine ready")
        except Exception as e:
            print(f"✗ Failed to initialize AI engine: {e}")
            print("Continuing with simulation only...")
            
        # Run different error scenarios
        test_scenarios = [
            ("test_pcie_crc_errors", "CRC Error Test"),
            ("test_pcie_timeout", "Timeout Error Test"),
            ("test_pcie_malformed_tlp", "Malformed TLP Test"),
        ]
        
        for test_name, description in test_scenarios:
            print(f"\n{'='*60}")
            print(f"Scenario: {description}")
            print(f"{'='*60}")
            
            # Run simulation
            self.run_simulation(test_name)
            
            # Analyze if RAG is available
            if self.rag_bridge:
                self.analyze_results()
                
            # Brief pause between tests
            time.sleep(2)
            
        # Generate final report
        if self.rag_bridge:
            self.generate_report()
            
        print("\n" + "="*60)
        print("Demo Complete!")
        print("="*60)
        
    def run_interactive(self):
        """Run in interactive mode"""
        print("PCIe Error Analysis - Interactive Mode")
        print("=" * 60)
        
        # Initialize RAG bridge
        print("\nInitializing AI analysis engine...")
        try:
            self.rag_bridge = create_rag_bridge()
            print("✓ AI engine ready")
        except Exception as e:
            print(f"✗ Failed to initialize AI engine: {e}")
            return
            
        # Show available tests
        tests = {
            "1": ("test_pcie_crc_errors", "CRC Errors"),
            "2": ("test_pcie_timeout", "Timeout Errors"),
            "3": ("test_pcie_ecrc_errors", "ECRC Errors"),
            "4": ("test_pcie_malformed_tlp", "Malformed TLP"),
            "5": ("test_pcie_stress", "Stress Test"),
            "6": ("test_pcie_comprehensive", "All Error Types")
        }
        
        while True:
            print("\nAvailable Tests:")
            for key, (_, desc) in tests.items():
                print(f"  {key}. {desc}")
            print("  q. Quit")
            
            choice = input("\nSelect test to run: ").strip().lower()
            
            if choice == 'q':
                break
                
            if choice in tests:
                test_name, desc = tests[choice]
                
                # Run simulation
                self.run_simulation(test_name)
                
                # Analyze results
                self.analyze_results()
                
                # Ask if user wants to query
                while True:
                    query = input("\nEnter analysis query (or 'done'): ").strip()
                    if query.lower() == 'done':
                        break
                        
                    result = self.rag_bridge.analyze_simulation(query)
                    print(f"\nAnalysis: {result.answer}")
                    
                # Generate report option
                if input("\nGenerate report? (y/n): ").lower() == 'y':
                    self.generate_report(f"report_{test_name}_{int(time.time())}.txt")
                    
            else:
                print("Invalid choice")
                
        print("\nThank you for using PCIe Error Analysis!")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PCIe Error Simulation and Analysis Demo"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--test", "-t",
        choices=["crc", "timeout", "ecrc", "malformed", "stress", "all"],
        help="Run specific test"
    )
    
    args = parser.parse_args()
    
    demo = PCIeErrorDemo()
    
    if args.interactive:
        demo.run_interactive()
    elif args.test:
        # Map test names to make targets
        test_map = {
            "crc": "test_pcie_crc_errors",
            "timeout": "test_pcie_timeout",
            "ecrc": "test_pcie_ecrc_errors",
            "malformed": "test_pcie_malformed_tlp",
            "stress": "test_pcie_stress",
            "all": "test_pcie_comprehensive"
        }
        
        test_name = test_map[args.test]
        
        # Initialize RAG
        try:
            demo.rag_bridge = create_rag_bridge()
        except Exception as e:
            print(f"Warning: Could not initialize AI engine: {e}")
            
        # Run test
        demo.run_simulation(test_name)
        
        if demo.rag_bridge:
            demo.analyze_results()
            demo.generate_report(f"report_{args.test}_{int(time.time())}.txt")
    else:
        # Run full demo
        demo.run_demo()


if __name__ == "__main__":
    main()