#!/bin/bash

# One-click PCIe simulation runner
# This script provides the easiest way to run PCIe simulations

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔═══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║        PCIe Debug Agent - Simulation Runner       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════╝${NC}"
echo ""

# Check current directory
if [ ! -f "README.md" ] || [ ! -d "src/simulators" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    echo "Usage: ./run_pcie_sim.sh"
    exit 1
fi

# Menu
echo "Choose simulation type:"
echo ""
echo "  1) 🚀 Quick Demo (No setup required)"
echo "  2) 📊 Generate Waveform (Requires Icarus Verilog)"  
echo "  3) 🧪 Full UVM Tests (Requires Python packages)"
echo "  4) 📋 View Last Report"
echo "  5) ℹ️  Setup Instructions"
echo ""
read -p "Select option (1-5): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running Quick Demo...${NC}\n"
        cd src/simulators
        python3 standalone_demo.py
        ;;
        
    2)
        echo -e "\n${GREEN}Generating Waveform...${NC}\n"
        # Check for iverilog
        if ! command -v iverilog &> /dev/null; then
            echo -e "${YELLOW}Icarus Verilog not found!${NC}"
            echo "Install with:"
            echo "  macOS: brew install icarus-verilog"
            echo "  Linux: sudo apt-get install iverilog"
            exit 1
        fi
        
        cd src/simulators
        ./generate_vcd.sh
        
        echo -e "\n${GREEN}Waveform generated!${NC}"
        echo "View with:"
        echo "  python3 vcd_viewer.py        # Text viewer"
        echo "  gtkwave pcie_waveform.vcd   # GUI viewer"
        ;;
        
    3)
        echo -e "\n${GREEN}Running Full UVM Tests...${NC}\n"
        
        # Check for virtual environment
        if [ ! -d "venv" ]; then
            echo -e "${YELLOW}Creating virtual environment...${NC}"
            python3 -m venv venv
        fi
        
        # Activate venv
        source venv/bin/activate
        
        # Check for cocotb
        if ! python -c "import cocotb" 2>/dev/null; then
            echo -e "${YELLOW}Installing required packages...${NC}"
            pip install -q cocotb pyuvm pytest colorama tabulate
        fi
        
        cd src/simulators
        
        echo "Select test:"
        echo "  1) CRC Errors"
        echo "  2) Timeout Errors"
        echo "  3) All Tests"
        read -p "Choice (1-3): " test_choice
        
        case $test_choice in
            1) make test_crc ;;
            2) make test_timeout ;;
            3) make test_comprehensive ;;
            *) echo "Invalid choice"; exit 1 ;;
        esac
        ;;
        
    4)
        echo -e "\n${GREEN}Viewing Last Report...${NC}\n"
        # Find most recent report
        latest_report=$(ls -t pcie_simulation_report_*.txt 2>/dev/null | head -1)
        
        if [ -z "$latest_report" ]; then
            # Try in simulators directory
            latest_report=$(ls -t src/simulators/pcie_simulation_report_*.txt 2>/dev/null | head -1)
        fi
        
        if [ -n "$latest_report" ]; then
            cat "$latest_report"
        else
            echo -e "${RED}No report found. Run a simulation first!${NC}"
        fi
        ;;
        
    5)
        echo -e "\n${GREEN}Setup Instructions${NC}"
        echo "═══════════════════"
        echo ""
        echo "1. For Quick Demo - No setup needed!"
        echo ""
        echo "2. For Waveform Generation:"
        echo "   macOS:  brew install icarus-verilog"
        echo "   Linux:  sudo apt-get install iverilog"
        echo ""
        echo "3. For Full UVM Tests:"
        echo "   python3 -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install cocotb pyuvm"
        echo ""
        echo "4. For Waveform Viewing:"
        echo "   macOS:  brew install --cask gtkwave"
        echo "   Linux:  sudo apt-get install gtkwave"
        ;;
        
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"