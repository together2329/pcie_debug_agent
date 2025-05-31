#!/bin/bash

# Quick Start Script for PCIe RTL Simulation
# This script automates the setup and runs a demo simulation

set -e  # Exit on error

echo "================================================"
echo "PCIe RTL Simulation Quick Start"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    print_status "Python3 found: $(python3 --version)"
else
    print_error "Python3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Check Icarus Verilog
if command -v iverilog &> /dev/null; then
    print_status "Icarus Verilog found: $(iverilog -V 2>&1 | head -n 1)"
else
    print_warning "Icarus Verilog not found."
    echo "    Please install it:"
    echo "    - macOS: brew install icarus-verilog"
    echo "    - Ubuntu: sudo apt-get install iverilog"
    echo "    - Fedora: sudo dnf install iverilog"
    exit 1
fi

# Step 2: Set up Python environment
echo ""
echo "Step 2: Setting up Python environment..."

# Check if we're in project root
if [ ! -f "requirements.txt" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if venv exists
if [ -d "venv" ]; then
    print_status "Virtual environment found"
else
    print_warning "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate venv
print_status "Activating virtual environment..."
source venv/bin/activate

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing dependencies..."

# Check if packages are already installed
if python -c "import cocotb, pyuvm" 2>/dev/null; then
    print_status "Simulator packages already installed"
else
    print_warning "Installing Python packages..."
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    pip install --quiet -r src/simulators/requirements.txt
    print_status "All packages installed"
fi

# Step 4: Run verification
echo ""
echo "Step 4: Verifying installation..."

cd src/simulators

# Verify cocotb
if python -c "import cocotb; print(f'  cocotb {cocotb.__version__}')" 2>/dev/null; then
    print_status "cocotb verified"
else
    print_error "cocotb verification failed"
    exit 1
fi

# Verify pyuvm
if python -c "import pyuvm; print('  pyuvm installed')" 2>/dev/null; then
    print_status "pyuvm verified"
else
    print_error "pyuvm verification failed"
    exit 1
fi

# Step 5: Run demo
echo ""
echo "Step 5: Running demonstration..."
echo ""
echo "Choose a demo option:"
echo "1) Quick CRC error test (fastest)"
echo "2) Comprehensive test suite"
echo "3) Interactive analysis mode"
echo "4) Standalone demo (no cocotb required)"
echo ""
read -p "Select option (1-4): " choice

case $choice in
    1)
        echo ""
        print_status "Running CRC error test..."
        make test_crc
        ;;
    2)
        echo ""
        print_status "Running comprehensive test suite..."
        make test_comprehensive
        ;;
    3)
        echo ""
        print_status "Starting interactive analysis mode..."
        python run_pcie_error_demo.py --interactive
        ;;
    4)
        echo ""
        print_status "Running standalone demo..."
        python standalone_demo.py
        ;;
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Demo Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "- View simulation logs: cat sim_build/cocotb.log"
echo "- Read the guide: cat HOW_TO_RUN.md"
echo "- Explore test files in: src/simulators/tests/"
echo "- Modify RTL in: src/simulators/rtl/"
echo ""
echo "To run more tests:"
echo "  cd src/simulators"
echo "  make help  # Show all available targets"
echo ""