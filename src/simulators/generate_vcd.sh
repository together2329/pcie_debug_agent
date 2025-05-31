#!/bin/bash

# Script to generate VCD waveform file using Icarus Verilog

echo "Generating VCD waveform file..."

# Compile the Verilog files
echo "Compiling Verilog..."
iverilog -o pcie_sim simple_tb.v rtl/pcie_lite.v

if [ $? -ne 0 ]; then
    echo "Error: Compilation failed"
    exit 1
fi

# Run the simulation
echo "Running simulation..."
vvp pcie_sim

if [ $? -ne 0 ]; then
    echo "Error: Simulation failed"
    exit 1
fi

# Check if VCD file was generated
if [ -f "pcie_waveform.vcd" ]; then
    echo ""
    echo "Success! VCD file generated: pcie_waveform.vcd"
    echo "File size: $(ls -lh pcie_waveform.vcd | awk '{print $5}')"
    echo ""
    echo "To view the waveform:"
    echo "  1. With GTKWave (recommended):"
    echo "     gtkwave pcie_waveform.vcd"
    echo ""
    echo "  2. With WaveDrom (online):"
    echo "     Upload pcie_waveform.vcd to https://wavedrom.com/"
    echo ""
    echo "  3. Convert to other formats:"
    echo "     vcd2wavedrom < pcie_waveform.vcd > waveform.json"
else
    echo "Error: VCD file not generated"
    exit 1
fi

# Clean up
rm -f pcie_sim