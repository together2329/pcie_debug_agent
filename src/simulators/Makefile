# Makefile for PCIe UVM testbench with cocotb and Icarus Verilog

# Simulator
SIM ?= icarus
TOPLEVEL_LANG ?= verilog

# Source files
VERILOG_SOURCES = $(PWD)/rtl/pcie_lite.v

# TOPLEVEL is the name of the toplevel module in your Verilog
TOPLEVEL = pcie_lite

# MODULE is the basename of the Python test file
MODULE = test_pcie_errors

# Test directory
export PYTHONPATH := $(PWD)/testbench:$(PYTHONPATH)

# cocotb variables
export COCOTB_REDUCED_LOG_FMT=1
export COCOTB_RESOLVE_X=ZEROS

# Icarus Verilog specific
COMPILE_ARGS += -g2012
COMPILE_ARGS += -Wall
COMPILE_ARGS += -Wno-timescale

# Include cocotb's make rules
include $(shell cocotb-config --makefiles)/Makefile.sim

# Custom targets
.PHONY: all clean test_crc test_timeout test_comprehensive view_logs test_vcd

all: test_comprehensive

# VCD generation test
test_vcd:
	@echo "Running VCD generation test..."
	$(MAKE) TESTCASE=test_vcd_generation MODULE=test_simple_vcd WAVES=1

# Individual test targets
test_crc:
	@echo "Running CRC error test..."
	$(MAKE) TESTCASE=test_pcie_crc_errors

test_timeout:
	@echo "Running timeout test..."
	$(MAKE) TESTCASE=test_pcie_timeout

test_ecrc:
	@echo "Running ECRC error test..."
	$(MAKE) TESTCASE=test_pcie_ecrc_errors

test_malformed:
	@echo "Running malformed TLP test..."
	$(MAKE) TESTCASE=test_pcie_malformed_tlp

test_stress:
	@echo "Running stress test..."
	$(MAKE) TESTCASE=test_pcie_stress

test_comprehensive:
	@echo "Running comprehensive test suite..."
	$(MAKE) TESTCASE=test_pcie_comprehensive

# Run all tests
test_all:
	@echo "Running all PCIe tests..."
	$(MAKE)

# View simulation logs
view_logs:
	@if [ -f sim_build/cocotb.log ]; then \
		tail -f sim_build/cocotb.log; \
	else \
		echo "No log file found. Run a test first."; \
	fi

# Clean simulation files
clean::
	@rm -rf sim_build
	@rm -rf __pycache__
	@rm -rf .pytest_cache
	@rm -rf results.xml
	@rm -f *.vcd
	@find . -name "*.pyc" -delete

# Help
help:
	@echo "PCIe UVM Testbench Makefile"
	@echo "==========================="
	@echo ""
	@echo "Targets:"
	@echo "  make               - Run comprehensive test"
	@echo "  make test_crc      - Run CRC error test"
	@echo "  make test_timeout  - Run timeout test"
	@echo "  make test_ecrc     - Run ECRC error test"
	@echo "  make test_malformed - Run malformed TLP test"
	@echo "  make test_stress   - Run stress test"
	@echo "  make test_all      - Run all tests"
	@echo "  make clean         - Clean build files"
	@echo "  make view_logs     - View simulation logs"
	@echo ""
	@echo "Examples:"
	@echo "  make test_crc WAVES=1  - Run with waveform generation"
	@echo "  make SIM=verilator    - Use Verilator (limited UVM support)"