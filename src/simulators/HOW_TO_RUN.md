# How to Run PCIe RTL Simulation with UVM Testbench

This guide explains how to set up and run the PCIe RTL simulation with the pyuvm testbench.

## Prerequisites

### 1. Install Icarus Verilog

**macOS:**
```bash
brew install icarus-verilog
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install iverilog
```

**Fedora/RHEL:**
```bash
sudo dnf install iverilog
```

**Windows:**
Download installer from: http://bleyer.org/icarus/

### 2. Install GTKWave (Optional - for viewing waveforms)

**macOS:**
```bash
brew install --cask gtkwave
```

**Linux:**
```bash
sudo apt-get install gtkwave
```

## Setup Instructions

### Step 1: Create Python Virtual Environment

```bash
# Navigate to project root
cd /path/to/pcie_debug_agent

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install Python Dependencies

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install simulator-specific dependencies
pip install -r src/simulators/requirements.txt
```

### Step 3: Verify Installation

```bash
cd src/simulators

# Run setup script
./setup_simulator.sh

# Or manually verify:
python3 -c "import cocotb; print(f'cocotb {cocotb.__version__} installed')"
python3 -c "import pyuvm; print('pyuvm installed')"
iverilog -V
```

## Running Simulations

### Quick Start - Run All Tests

```bash
cd src/simulators
make test_all
```

### Individual Test Scenarios

#### 1. CRC Error Test
```bash
make test_crc
```
This test:
- Sends normal PCIe transactions
- Injects CRC errors
- Verifies link recovery mechanism
- Tests system stability after recovery

#### 2. Timeout Error Test
```bash
make test_timeout
```
This test:
- Generates read requests
- Simulates completion timeouts
- Verifies timeout detection
- Tests error handling

#### 3. ECRC Error Test
```bash
make test_ecrc
```
This test:
- Tests End-to-End CRC errors
- Verifies data integrity checks
- Tests error reporting

#### 4. Malformed TLP Test
```bash
make test_malformed
```
This test:
- Injects malformed Transaction Layer Packets
- Tests protocol violation detection
- Verifies error responses

#### 5. Stress Test
```bash
make test_stress
```
This test:
- Runs high-volume transactions
- Injects multiple error types
- Tests system under load

#### 6. Comprehensive Test
```bash
make test_comprehensive
```
This test:
- Runs all error scenarios sequentially
- Provides complete coverage
- Generates detailed reports

### Running with Waveform Generation

To generate VCD waveforms for debugging:

```bash
# Run any test with WAVES=1
make test_crc WAVES=1

# View waveform
gtkwave dump.vcd
```

### Running with Different Simulators

```bash
# Default (Icarus Verilog)
make test_crc

# With Verilator (limited UVM support)
make test_crc SIM=verilator
```

## Running the AI-Powered Analysis Demo

### Interactive Mode
```bash
cd src/simulators
python run_pcie_error_demo.py --interactive
```

This will:
1. Show available test options
2. Run selected simulation
3. Analyze results with AI
4. Allow custom queries
5. Generate reports

### Automatic Demo Mode
```bash
python run_pcie_error_demo.py
```

This runs all error scenarios automatically and generates a comprehensive report.

### Run Specific Test with Analysis
```bash
python run_pcie_error_demo.py --test crc
python run_pcie_error_demo.py --test timeout
python run_pcie_error_demo.py --test ecrc
python run_pcie_error_demo.py --test malformed
```

## Understanding the Output

### Console Output Example
```
[1000] PCIe: TLP Received - Type=0 Addr=0x00001000 Data=0x00000000 Tag=0
[1500] PCIe: ERROR - CRC error injected
[2000] PCIe: ERROR - Entering RECOVERY due to errors
[2500] PCIe: RECOVERY complete, back to L0

PCIe Test Summary:
  Total Transactions: 25
  Total Errors: 3
```

### Log Files
- `sim_build/cocotb.log` - Detailed simulation log
- `pcie_error_report_*.txt` - AI analysis reports
- `dump.vcd` - Waveform file (if WAVES=1)

## Customizing Tests

### Adding New Error Types

1. Edit `src/simulators/rtl/pcie_lite.v`:
```verilog
// Add new error injection input
input inject_new_error,

// Add error logic
if (inject_new_error) begin
    error_valid <= 1'b1;
    error_type <= ERR_NEW_TYPE;
end
```

2. Create new sequence in `src/simulators/testbench/pcie_sequences.py`:
```python
class PCIeNewErrorSequence(PCIeBaseSequence):
    async def body(self):
        dut.inject_new_error.value = 1
        # Send transaction
        await self.send_transaction()
        dut.inject_new_error.value = 0
```

3. Add test case in `src/simulators/tests/test_pcie_errors.py`:
```python
@cocotb.test()
async def test_new_error(dut):
    await uvm_root().run_test(PCIeNewErrorTest)
```

### Modifying Transaction Patterns

Edit sequences in `pcie_sequences.py`:
```python
# Change number of transactions
seq.num_transactions = 20

# Change address patterns
tr.address = 0x4000 + (i * 0x100)

# Change data patterns
tr.data = 0xDEADBEEF
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'cocotb'**
   - Solution: Activate virtual environment and install dependencies
   ```bash
   source venv/bin/activate
   pip install cocotb pyuvm
   ```

2. **Icarus Verilog not found**
   - Solution: Install Icarus Verilog for your OS (see Prerequisites)

3. **Simulation hangs**
   - Check for infinite loops in sequences
   - Add timeout to tests
   - Use Ctrl+C to interrupt

4. **No waveform generated**
   - Add `WAVES=1` to make command
   - Check for `dump.vcd` in current directory

5. **AI analysis not working**
   - Ensure main project dependencies are installed
   - Check if local LLM is properly configured
   - Verify embeddings model is downloaded

### Debug Mode

For detailed debug output:
```bash
# Set cocotb log level
export COCOTB_LOG_LEVEL=DEBUG
make test_crc

# Enable Python debugging
python -m pdb run_pcie_error_demo.py
```

## Advanced Usage

### Batch Testing
```bash
# Run all tests and save results
for test in crc timeout ecrc malformed; do
    make test_$test > results_$test.log 2>&1
done
```

### Continuous Integration
```bash
# Add to CI pipeline
cd src/simulators
make clean
make test_all
python run_pcie_error_demo.py --test all
```

### Custom Analysis Queries

When running interactive mode, you can ask specific questions:
- "What is the root cause of the CRC errors?"
- "How can I prevent timeout errors in my design?"
- "What is the impact of ECRC errors on system performance?"
- "Generate Verilog code to fix the malformed TLP issue"

## Next Steps

1. **Explore the RTL**: Look at `rtl/pcie_lite.v` to understand the design
2. **Study UVM Components**: Review `testbench/pcie_base_test.py`
3. **Customize Sequences**: Modify `testbench/pcie_sequences.py`
4. **Enhance Analysis**: Add queries to `run_pcie_error_demo.py`
5. **Add Coverage**: Implement functional coverage in testbench

## Resources

- [cocotb Documentation](https://docs.cocotb.org/)
- [pyuvm Documentation](https://pyuvm.github.io/)
- [Icarus Verilog Manual](http://iverilog.icarus.com/)
- [PCIe Specification](https://pcisig.com/specifications)

---

For questions or issues, please check the main project README or open an issue on GitHub.