# Quick Commands - PCIe RTL Simulation

## Fastest Way to See It Working

### Option 1: Standalone Demo (No Setup Required)
```bash
cd src/simulators
python3 standalone_demo.py
```
This runs immediately without any dependencies!

### Option 2: Full RTL Simulation (Requires Setup)

#### One-Time Setup:
```bash
# 1. Install Icarus Verilog
brew install icarus-verilog    # macOS
# OR
sudo apt-get install iverilog  # Ubuntu

# 2. Create Python environment
python3 -m venv venv
source venv/bin/activate

# 3. Install packages
pip install cocotb pyuvm pytest colorama tabulate
```

#### Run Simulation:
```bash
cd src/simulators

# Quick test (30 seconds)
make test_crc

# Full test suite (2-3 minutes)
make test_comprehensive

# With waveforms
make test_crc WAVES=1
gtkwave dump.vcd
```

## Interactive AI Analysis

```bash
cd src/simulators

# Option 1: Standalone (works immediately)
python3 standalone_demo.py

# Option 2: With RTL integration (after setup)
python3 run_pcie_error_demo.py --interactive
```

## What Each Test Does

| Command | What It Tests | Duration |
|---------|---------------|----------|
| `make test_crc` | CRC error injection and recovery | ~30s |
| `make test_timeout` | Completion timeout handling | ~45s |
| `make test_ecrc` | End-to-end CRC errors | ~30s |
| `make test_malformed` | Protocol violations | ~30s |
| `make test_stress` | High-volume stress test | ~60s |
| `make test_comprehensive` | All error types | ~3min |

## Example Output

```bash
$ make test_crc

[1000] PCIe: TLP Received - Type=0 Addr=0x00001000 Tag=0
[1500] PCIe: ERROR - CRC error injected
[2000] PCIe: ERROR - Entering RECOVERY due to errors
[2500] PCIe: RECOVERY complete, back to L0

PCIe Test Summary:
  Total Transactions: 10
  Total Errors: 2
  
✓ Test Passed
```

## Troubleshooting

**"command not found: iverilog"**
```bash
# Install Icarus Verilog first
brew install icarus-verilog
```

**"No module named 'cocotb'"**
```bash
# Activate virtual environment
source venv/bin/activate
pip install cocotb pyuvm
```

**Just want to see it work?**
```bash
# Run the standalone demo - no dependencies!
python3 src/simulators/standalone_demo.py
```