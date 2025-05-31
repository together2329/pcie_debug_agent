"""
PCIe UVM Base Test Components using pyuvm
"""
import cocotb
from cocotb.triggers import RisingEdge, Timer, FallingEdge
from cocotb.clock import Clock
from pyuvm import *
import random
from dataclasses import dataclass
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PCIeTransaction(uvm_sequence_item):
    """PCIe Transaction Item"""
    tlp_type: int = 0  # 0: MRd, 1: MWr, 2: CplD
    address: int = 0
    data: int = 0
    tag: int = 0
    length: int = 1  # DW length
    
    def __str__(self):
        tlp_names = {0: "MRd", 1: "MWr", 2: "CplD"}
        return f"PCIe_TLP[{tlp_names.get(self.tlp_type, 'Unknown')}] " \
               f"addr=0x{self.address:08x} data=0x{self.data:08x} tag={self.tag}"


class PCIeDriver(uvm_driver):
    """PCIe Transaction Driver"""
    
    def build_phase(self):
        self.ap = uvm_analysis_port("ap", self)
        
    def start_of_simulation_phase(self):
        self.dut = cocotb.top
        
    async def run_phase(self):
        await RisingEdge(self.dut.rst_n)
        
        while True:
            # Get next transaction
            tr = await self.seq_item_port.get_next_item()
            
            # Drive transaction
            await self.drive_transaction(tr)
            
            # Send to analysis port
            self.ap.write(tr)
            
            self.seq_item_port.item_done()
    
    async def drive_transaction(self, tr: PCIeTransaction):
        """Drive a single transaction onto the interface"""
        # Wait for ready
        while self.dut.tlp_ready.value == 0:
            await RisingEdge(self.dut.clk)
            
        # Drive the transaction
        self.dut.tlp_valid.value = 1
        self.dut.tlp_type.value = tr.tlp_type
        self.dut.tlp_address.value = tr.address
        self.dut.tlp_data.value = tr.data
        self.dut.tlp_tag.value = tr.tag
        self.dut.tlp_length.value = tr.length
        
        await RisingEdge(self.dut.clk)
        
        # Deassert valid
        self.dut.tlp_valid.value = 0
        
        # Log transaction
        logger.info(f"Driver sent: {tr}")


class PCIeMonitor(uvm_component):
    """PCIe Interface Monitor"""
    
    def build_phase(self):
        self.ap = uvm_analysis_port("ap", self)
        self.error_ap = uvm_analysis_port("error_ap", self)
        
    def start_of_simulation_phase(self):
        self.dut = cocotb.top
        
    async def run_phase(self):
        await RisingEdge(self.dut.rst_n)
        
        # Fork monitoring tasks
        await cocotb.start(self.monitor_transactions())
        await cocotb.start(self.monitor_errors())
        await cocotb.start(self.monitor_completions())
        
    async def monitor_transactions(self):
        """Monitor TLP transactions"""
        while True:
            await RisingEdge(self.dut.clk)
            
            if self.dut.tlp_valid.value and self.dut.tlp_ready.value:
                tr = PCIeTransaction()
                tr.tlp_type = int(self.dut.tlp_type.value)
                tr.address = int(self.dut.tlp_address.value)
                tr.data = int(self.dut.tlp_data.value)
                tr.tag = int(self.dut.tlp_tag.value)
                tr.length = int(self.dut.tlp_length.value)
                
                self.ap.write(tr)
                logger.info(f"Monitor captured: {tr}")
                
    async def monitor_errors(self):
        """Monitor error signals"""
        while True:
            await RisingEdge(self.dut.clk)
            
            if self.dut.error_valid.value:
                error_type = int(self.dut.error_type.value)
                error_names = {
                    1: "CRC_ERROR",
                    2: "TIMEOUT",
                    3: "ECRC_ERROR",
                    4: "MALFORMED_TLP",
                    5: "UNSUPPORTED_REQUEST"
                }
                error_name = error_names.get(error_type, f"UNKNOWN_{error_type}")
                
                error_info = {
                    "type": error_name,
                    "header": int(self.dut.error_header.value),
                    "time": cocotb.utils.get_sim_time()
                }
                
                self.error_ap.write(error_info)
                logger.error(f"PCIe Error Detected: {error_name}")
                
    async def monitor_completions(self):
        """Monitor completion interface"""
        while True:
            await RisingEdge(self.dut.clk)
            
            if self.dut.cpl_valid.value and self.dut.cpl_ready.value:
                cpl_status = int(self.dut.cpl_status.value)
                status_names = {0: "SC", 1: "UR", 2: "CA"}
                
                logger.info(f"Completion: tag={int(self.dut.cpl_tag.value)} "
                           f"status={status_names.get(cpl_status, 'Unknown')}")


class PCIeScoreboard(uvm_component):
    """PCIe Transaction Scoreboard"""
    
    def build_phase(self):
        self.expected_queue = []
        self.error_count = 0
        self.transaction_count = 0
        
    def write(self, tr):
        """Receive transactions from monitor"""
        self.transaction_count += 1
        logger.info(f"Scoreboard received transaction #{self.transaction_count}: {tr}")
        
    def write_error(self, error_info):
        """Receive error notifications"""
        self.error_count += 1
        logger.error(f"Scoreboard logged error #{self.error_count}: {error_info}")
        
    def report_phase(self):
        logger.info(f"\n{'='*60}")
        logger.info(f"PCIe Test Summary:")
        logger.info(f"  Total Transactions: {self.transaction_count}")
        logger.info(f"  Total Errors: {self.error_count}")
        logger.info(f"{'='*60}\n")


class PCIeAgent(uvm_agent):
    """PCIe Agent containing driver, monitor, sequencer"""
    
    def build_phase(self):
        self.sequencer = uvm_sequencer("sequencer", self)
        self.driver = PCIeDriver("driver", self)
        self.monitor = PCIeMonitor("monitor", self)
        
    def connect_phase(self):
        self.driver.seq_item_port.connect(self.sequencer.seq_item_export)


class PCIeEnv(uvm_env):
    """PCIe Test Environment"""
    
    def build_phase(self):
        self.agent = PCIeAgent("agent", self)
        self.scoreboard = PCIeScoreboard("scoreboard", self)
        
    def connect_phase(self):
        self.agent.monitor.ap.connect(self.scoreboard.write)
        self.agent.monitor.error_ap.connect(self.scoreboard.write_error)
        self.agent.driver.ap.connect(self.scoreboard.write)


class PCIeBaseTest(uvm_test):
    """Base test class for PCIe tests"""
    
    def build_phase(self):
        self.env = PCIeEnv("env", self)
        
    def start_of_simulation_phase(self):
        self.dut = cocotb.top
        
    async def configure_dut(self):
        """Configure DUT settings"""
        # Set up clocks
        cocotb.start_soon(Clock(self.dut.clk, 10, units="ns").start())
        
        # Reset
        self.dut.rst_n.value = 0
        self.dut.tlp_valid.value = 0
        self.dut.inject_crc_error.value = 0
        self.dut.inject_timeout.value = 0
        self.dut.inject_ecrc_error.value = 0
        self.dut.inject_malformed_tlp.value = 0
        self.dut.cpl_ready.value = 1
        
        await Timer(100, units="ns")
        self.dut.rst_n.value = 1
        await Timer(100, units="ns")
        
        # Wait for link up
        while not self.dut.link_up.value:
            await RisingEdge(self.dut.clk)
            
        logger.info("PCIe link is up!")
        
    async def run_phase(self):
        await self.configure_dut()
        
        # Run sequences
        await self.run_sequences()
        
        # Wait for completion
        await Timer(1000, units="ns")
        
    async def run_sequences(self):
        """Override in derived tests to run specific sequences"""
        pass