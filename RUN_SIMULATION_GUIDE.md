# Complete Guide: How to Run PCIe RTL Simulation

## Quick Start (3 Options)

### Option 1: Standalone Demo (No Setup Required) ✅
```bash
cd pcie_debug_agent/src/simulators
python3 standalone_demo.py
```
This works immediately without any dependencies!

### Option 2: Simple Verilog Simulation (Minimal Setup) ✅
```bash
# Install Icarus Verilog first
brew install icarus-verilog    # macOS
# OR
sudo apt-get install iverilog  # Ubuntu/Linux

# Run simulation and generate waveform
cd pcie_debug_agent/src/simulators
./generate_vcd.sh

# View waveform in text format
python3 vcd_viewer.py
```

### Option 3: Full UVM Testbench (Complete Setup) 🚀
```bash
# One-time setup
cd pcie_debug_agent
python3 -m venv venv
source venv/bin/activate
pip install cocotb pyuvm pytest colorama tabulate

# Run UVM tests
cd src/simulators
make test_comprehensive
```

## Detailed Instructions

### Step 1: Install Prerequisites

#### macOS:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Icarus Verilog
brew install icarus-verilog

# (Optional) Install GTKWave for graphical waveform viewing
brew install --cask gtkwave
```

#### Ubuntu/Debian:
```bash
# Update package list
sudo apt-get update

# Install Icarus Verilog
sudo apt-get install iverilog

# (Optional) Install GTKWave
sudo apt-get install gtkwave
```

#### Windows:
1. Download Icarus Verilog from: http://bleyer.org/icarus/
2. Run the installer
3. Add to PATH during installation

### Step 2: Clone or Navigate to Project
```bash
# If you haven't cloned yet
git clone https://github.com/together2329/pcie_debug_agent.git

# Navigate to project
cd pcie_debug_agent
```

### Step 3: Choose Your Simulation Method

## Method A: Simple Standalone Demo (Fastest)

```bash
cd src/simulators
python3 standalone_demo.py
```

**What it does:**
- Simulates PCIe transactions without actual RTL
- Injects all error types (CRC, timeout, ECRC, malformed)
- Provides AI-style analysis
- Generates report

**Expected output:**
```
======================================================================
PCIe Error Simulation and Analysis Demo
======================================================================
[     0ns] PCIe: LTSSM state DETECT
[   100ns] PCIe: LTSSM entering POLLING
...
Total errors detected: 9
Error breakdown:
  - CRC: 3 occurrences
  - timeout: 1 occurrences
...
```

## Method B: Verilog Simulation with Waveform

```bash
cd src/simulators

# Generate VCD waveform
./generate_vcd.sh

# View waveform in terminal
python3 vcd_viewer.py

# OR view graphically (if GTKWave installed)
gtkwave pcie_waveform.vcd
```

**What it does:**
- Runs actual Verilog RTL simulation
- Generates VCD waveform file
- Shows PCIe protocol in action

**Expected output:**
```
Generating VCD waveform file...
Compiling Verilog...
Running simulation...
[105000] PCIe: LTSSM entering POLLING
[125000] PCIe: Link UP - Speed Gen3 Width x16
...
Success! VCD file generated: pcie_waveform.vcd
```

## Method C: Full UVM Testbench (Most Comprehensive)

### First-time setup:
```bash
# From project root
cd pcie_debug_agent

# Create Python virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python packages
pip install cocotb pyuvm pytest colorama tabulate
```

### Run UVM tests:
```bash
cd src/simulators

# Quick test (30 seconds)
make test_crc

# All error types (3 minutes)
make test_comprehensive

# With waveform
make test_crc WAVES=1

# Interactive AI analysis
python3 run_pcie_error_demo.py --interactive
```

**Available test targets:**
| Command | Tests | Duration |
|---------|-------|----------|
| `make test_crc` | CRC errors | ~30s |
| `make test_timeout` | Completion timeouts | ~45s |
| `make test_ecrc` | End-to-end CRC errors | ~30s |
| `make test_malformed` | Malformed TLPs | ~30s |
| `make test_stress` | High-volume stress test | ~60s |
| `make test_comprehensive` | All error scenarios | ~3min |

## Understanding the Output

### 1. Console Output
```
[1000] PCIe: TLP Received - Type=0 Addr=0x00001000 Tag=0
[1500] PCIe: ERROR - CRC error injected
[2000] PCIe: ERROR - Entering RECOVERY due to errors
[2500] PCIe: RECOVERY complete, back to L0
```

### 2. Error Summary
```
PCIe Test Summary:
  Total Transactions: 25
  Total Errors: 9
  
Error Types:
  - CRC_ERROR: 3
  - TIMEOUT: 1
  - ECRC_ERROR: 2
  - MALFORMED_TLP: 1
```

### 3. AI Analysis (when using demo script)
```
Q: What caused the CRC error and how can it be fixed?
A: The CRC errors indicate signal integrity issues. Root causes:
   1. EMI interference on PCIe lanes
   2. Poor PCB routing or termination
   3. Clock jitter
   
   Fix: Check signal routing, add shielding, verify termination resistors
```

## File Locations

After running simulations, you'll find:
- `pcie_waveform.vcd` - Waveform file
- `pcie_simulation_report_*.txt` - Analysis reports
- `sim_build/` - cocotb simulation artifacts
- `logs/` - Detailed logs

## Troubleshooting

### "command not found: iverilog"
```bash
# Install Icarus Verilog
brew install icarus-verilog  # macOS
sudo apt-get install iverilog  # Linux
```

### "No module named 'cocotb'"
```bash
# Make sure you're in virtual environment
source venv/bin/activate
pip install cocotb pyuvm
```

### "Permission denied"
```bash
# Make scripts executable
chmod +x generate_vcd.sh
chmod +x run_pcie_error_demo.py
```

### Just want to see it work?
```bash
# This always works - no dependencies!
python3 src/simulators/standalone_demo.py
```

## Interactive Analysis Mode

For the best experience with AI-powered analysis:

```bash
cd src/simulators
python3 run_pcie_error_demo.py --interactive

# Menu options:
1. CRC Errors
2. Timeout Errors  
3. ECRC Errors
4. Malformed TLP
5. Stress Test
6. All Error Types
q. Quit

Select test to run: 1
```

Then ask questions like:
- "What is the root cause of these errors?"
- "How can I fix the timeout issues?"
- "What's the impact on system performance?"

## Summary

The easiest way to start:
1. `python3 src/simulators/standalone_demo.py` - Works immediately
2. `./src/simulators/generate_vcd.sh` - Generates waveform (needs Icarus)
3. Full UVM requires Python package setup but provides most features

Choose based on your needs:
- **Quick demo**: Use standalone_demo.py
- **See waveforms**: Use generate_vcd.sh
- **Full testing**: Set up cocotb/pyuvm environment