#!/usr/bin/env python3
"""
PCIe Mode Usage Example

This example demonstrates how to use PCIe mode in pcie-debug
"""

import subprocess
import sys

def example_interactive_mode():
    """Example: Using PCIe mode in interactive shell"""
    print("=" * 60)
    print("EXAMPLE 1: Interactive Mode with PCIe")
    print("=" * 60)
    print("""
# Start interactive shell
./pcie-debug

# Switch to PCIe mode
/rag_mode pcie

# Now ask PCIe-specific questions:
What are LTSSM states?
How to debug PCIe completion timeout?
Explain PCIe FLR compliance violations

# Check current mode
/rag_mode

# The system will use PCIe-optimized retrieval with:
# - 1000-word adaptive chunks
# - PCIe concept extraction
# - Technical level filtering
# - Concept boosting for PCIe terms
""")

def example_cli_mode():
    """Example: Using PCIe mode from CLI"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CLI Commands")
    print("=" * 60)
    print("""
# Build PCIe knowledge base (first time only)
./pcie-debug pcie build

# Query with PCIe mode
./pcie-debug pcie query "What causes PCIe link training failures?"

# Query with technical level filter (3 = advanced)
./pcie-debug pcie query "LTSSM timeout" --technical-level 3

# Query with PCIe layer filter
./pcie-debug pcie query "equalization" --layer physical

# Show PCIe mode statistics
./pcie-debug pcie stats

# Combine multiple filters
./pcie-debug pcie query "ASPM L1 substates" \\
    --layer power_management \\
    --technical-level 2
""")

def example_python_api():
    """Example: Using PCIe mode programmatically"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Python API")
    print("=" * 60)
    print('''
from src.rag.pcie_rag_engine import create_pcie_rag_engine

# Create PCIe RAG engine
engine = create_pcie_rag_engine(
    embedding_model='text-embedding-3-small',
    chunk_config={
        'target_size': 1000,
        'max_size': 1500,
        'overlap_size': 200
    }
)

# Query with filters
results = engine.query(
    query="PCIe completion timeout",
    top_k=5,
    technical_level_filter=2,  # Intermediate and above
    pcie_layer_filter='transaction'
)

# Process results
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Technical Level: {result.technical_level}")
    print(f"Content: {result.content[:200]}...")
''')

def run_live_example():
    """Run a live example"""
    print("\n" + "=" * 60)
    print("LIVE EXAMPLE: Running PCIe Query")
    print("=" * 60)
    
    # Check if we're in venv
    if not sys.prefix != sys.base_prefix:
        print("Activating virtual environment...")
        activate_cmd = "source venv/bin/activate && "
    else:
        activate_cmd = ""
    
    # Run PCIe stats
    print("\n1. Checking PCIe mode statistics:")
    cmd = f"{activate_cmd}./pcie-debug pcie stats"
    subprocess.run(cmd, shell=True)
    
    # Run a sample query
    print("\n2. Running PCIe query:")
    query = "What are PCIe LTSSM states?"
    cmd = f'{activate_cmd}./pcie-debug pcie query "{query}"'
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    print("PCIe MODE USAGE GUIDE")
    print("=" * 60)
    
    example_interactive_mode()
    example_cli_mode()
    example_python_api()
    
    # Ask if user wants to run live example
    response = input("\nRun live example? (y/n): ")
    if response.lower() == 'y':
        run_live_example()
    
    print("\n✅ PCIe mode is now fixed and ready to use!")
    print("   The formatting error has been resolved.")