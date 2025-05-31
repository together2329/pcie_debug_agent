# PCIe UVM Simulator Integration

This module provides a complete open-source UVM testbench environment for PCIe error simulation and analysis using Icarus Verilog, cocotb, and pyuvm.

## Features

- **100% Open Source**: Uses Icarus Verilog, cocotb, and pyuvm
- **UVM Methodology**: Full UVM testbench with drivers, monitors, sequences, and scoreboard
- **Error Injection**: Comprehensive error injection capabilities:
  - CRC errors
  - Timeout errors
  - ECRC errors
  - Malformed TLPs
  - Link training failures
- **Real-time Analysis**: Integration with RAG/LLM for intelligent error analysis
- **Automated Reports**: Generate comprehensive error analysis reports

## Directory Structure

```
src/simulators/
├── rtl/                    # Verilog RTL files
│   └── pcie_lite.v        # Simplified PCIe module with error injection
├── testbench/             # pyuvm testbench components
│   ├── pcie_base_test.py  # Base test classes and UVM components
│   └── pcie_sequences.py  # Error injection sequences
├── tests/                 # Test cases
│   └── test_pcie_errors.py # PCIe error test scenarios
├── Makefile              # Build and run tests
├── rag_integration.py    # RAG/LLM integration for analysis
├── run_pcie_error_demo.py # Demo script
└── setup_simulator.sh    # Setup script
```

## Setup

1. **Install Icarus Verilog**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install iverilog
   
   # MacOS
   brew install icarus-verilog
   
   # Windows
   # Download from http://bleyer.org/icarus/
   ```

2. **Run Setup Script**:
   ```bash
   cd src/simulators
   ./setup_simulator.sh
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

### Individual Tests

```bash
# Test CRC errors
make test_crc

# Test timeout errors
make test_timeout

# Test ECRC errors
make test_ecrc

# Test malformed TLPs
make test_malformed

# Run stress test
make test_stress

# Run all tests
make test_comprehensive
```

### With Waveforms

```bash
# Generate VCD waveform
make test_crc WAVES=1
```

### Demo Script

The demo script runs simulations and analyzes results using AI:

```bash
# Run full demo
python run_pcie_error_demo.py

# Interactive mode
python run_pcie_error_demo.py --interactive

# Run specific test
python run_pcie_error_demo.py --test crc
```

## Example Output

```
Running PCIe Test: test_pcie_crc_errors
========================================================

SIM: [1000] PCIe: TLP Received - Type=0 Addr=0x00001000 Data=0x00000000 Tag=0
SIM: [1500] PCIe: ERROR - CRC error injected
SIM: [2000] PCIe: ERROR - Entering RECOVERY due to errors

Analyzing Simulation Results with AI
========================================================

Detected 2 errors

Error Types:
  - CRC_ERROR: 2

Q: What PCIe errors occurred and what is their root cause?
──────────────────────────────────────────────────────
A: The simulation detected 2 CRC (Cyclic Redundancy Check) errors. These errors 
occurred during PCIe transaction layer packet (TLP) transmission. The root causes 
could be:
1. Signal integrity issues on the PCIe link
2. Clock domain crossing problems
3. Electromagnetic interference
4. Faulty SerDes implementation

The errors triggered the LTSSM to enter RECOVERY state for link retraining.
```

## UVM Testbench Architecture

### Components

1. **PCIeTransaction**: UVM sequence item representing a PCIe TLP
2. **PCIeDriver**: Drives transactions to DUT
3. **PCIeMonitor**: Monitors DUT interfaces for transactions and errors
4. **PCIeScoreboard**: Tracks and validates transactions
5. **PCIeAgent**: Contains driver, monitor, and sequencer
6. **PCIeEnv**: Test environment containing agent and scoreboard

### Error Injection Sequences

- `PCIeErrorInjectionSequence`: Coordinated error injection
- `PCIeTargetedErrorSequence`: Specific error scenarios
- `PCIeStressSequence`: High-traffic stress testing

## Integration with RAG/LLM

The simulator integrates with the PCIe Debug Agent's RAG engine to provide intelligent analysis:

1. **Real-time Log Capture**: Simulator logs are captured and parsed
2. **Vector Embedding**: Logs are embedded and stored in the vector database
3. **AI Analysis**: LLM analyzes errors and provides recommendations
4. **Report Generation**: Comprehensive reports with AI insights

## Extending the Testbench

### Adding New Error Types

1. Add error injection signal to RTL:
   ```verilog
   input inject_new_error,
   ```

2. Create new sequence in `pcie_sequences.py`:
   ```python
   class PCIeNewErrorSequence(PCIeBaseSequence):
       async def body(self):
           # Inject new error type
   ```

3. Add test case in `test_pcie_errors.py`:
   ```python
   class PCIeNewErrorTest(PCIeBaseTest):
       async def run_sequences(self):
           # Run new error sequence
   ```

### Adding New Analysis Queries

Edit `run_pcie_error_demo.py` to add new analysis queries:
```python
queries = [
    "What is the impact of this error on PCIe performance?",
    "How can the firmware detect and recover from this error?",
]
```

## Troubleshooting

1. **Icarus Verilog not found**: Install using system package manager
2. **cocotb import error**: Ensure Python dependencies are installed
3. **Simulation hangs**: Check for infinite loops in sequences
4. **No waveforms**: Add `WAVES=1` to make command

## Next Steps

1. Add more complex PCIe protocol features (e.g., power management, hot-plug)
2. Implement coverage collection
3. Add SystemVerilog assertions
4. Create GUI for test selection and analysis
5. Integrate with CI/CD pipeline