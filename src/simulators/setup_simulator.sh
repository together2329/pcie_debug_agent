#!/bin/bash

# Setup script for PCIe UVM Simulator Environment

echo "PCIe Debug Agent - Simulator Setup"
echo "=================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"

# Check Python version
echo -n "Checking Python version... "
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "Found Python $PYTHON_VERSION"
    
    # Check if version is 3.8 or higher
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
        echo "✓ Python version is compatible"
    else
        echo "✗ Python 3.8 or higher is required"
        exit 1
    fi
else
    echo "✗ Python 3 not found"
    exit 1
fi

# Check for Icarus Verilog
echo -n "Checking for Icarus Verilog... "
if command_exists iverilog; then
    IVERILOG_VERSION=$(iverilog -V 2>&1 | head -n 1)
    echo "Found $IVERILOG_VERSION"
else
    echo "Not found"
    echo ""
    echo "Icarus Verilog is required for simulation. Install it using:"
    
    case $OS in
        linux)
            echo "  Ubuntu/Debian: sudo apt-get install iverilog"
            echo "  Fedora/RHEL: sudo dnf install iverilog"
            echo "  Arch: sudo pacman -S iverilog"
            ;;
        macos)
            echo "  MacOS: brew install icarus-verilog"
            ;;
        windows)
            echo "  Windows: Download from http://bleyer.org/icarus/"
            ;;
    esac
    
    read -p "Do you want to continue without Icarus Verilog? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: Not in a virtual environment. It's recommended to use venv."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Create a virtual environment with: python3 -m venv venv"
        echo "Then activate it and run this script again."
        exit 1
    fi
fi

# Install simulator requirements
pip install -r requirements.txt

# Install main project requirements if needed
if [ -f "../../requirements.txt" ]; then
    echo "Installing main project dependencies..."
    pip install -r ../../requirements.txt
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p logs
mkdir -p results
mkdir -p waveforms

# Test cocotb installation
echo ""
echo "Testing cocotb installation..."
python3 -c "import cocotb; print(f'✓ cocotb version {cocotb.__version__} installed')"

# Test pyuvm installation
echo "Testing pyuvm installation..."
python3 -c "import pyuvm; print('✓ pyuvm installed successfully')"

# Run a simple test to verify everything works
echo ""
echo "Running verification test..."

# Create a simple test
cat > test_setup.py << 'EOF'
import cocotb
from cocotb.triggers import Timer

@cocotb.test()
async def test_setup(dut):
    """Simple test to verify setup"""
    dut._log.info("Setup verification test started")
    await Timer(1, units='ns')
    dut._log.info("Setup verification test passed")
EOF

# Create a minimal Verilog file
cat > test_module.v << 'EOF'
module test_module(input clk);
    initial begin
        $display("Simulator is working!");
    end
endmodule
EOF

# Create test Makefile
cat > Makefile.test << 'EOF'
SIM = icarus
TOPLEVEL_LANG = verilog
VERILOG_SOURCES = test_module.v
TOPLEVEL = test_module
MODULE = test_setup
include $(shell cocotb-config --makefiles)/Makefile.sim
EOF

# Run the test
echo "Running setup verification..."
make -f Makefile.test > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Setup verification passed!"
else
    echo "✗ Setup verification failed. Check the installation."
fi

# Cleanup test files
rm -f test_setup.py test_module.v Makefile.test
rm -rf sim_build

echo ""
echo "Setup complete!"
echo ""
echo "To run the PCIe tests:"
echo "  cd src/simulators"
echo "  make test_comprehensive"
echo ""
echo "Available test targets:"
echo "  make test_crc      - Test CRC errors"
echo "  make test_timeout  - Test timeouts"
echo "  make test_stress   - Run stress test"
echo "  make test_all      - Run all tests"
echo ""
echo "For help: make help"