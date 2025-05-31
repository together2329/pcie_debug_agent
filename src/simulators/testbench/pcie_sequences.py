"""
PCIe Test Sequences for Error Injection and Testing
"""
from pyuvm import *
from pcie_base_test import PCIeTransaction
import random
import logging

logger = logging.getLogger(__name__)


class PCIeBaseSequence(uvm_sequence):
    """Base sequence class for PCIe transactions"""
    
    def __init__(self, name="PCIeBaseSequence"):
        super().__init__(name)
        self.num_transactions = 10
        

class PCIeReadSequence(PCIeBaseSequence):
    """Generate PCIe memory read transactions"""
    
    async def body(self):
        for i in range(self.num_transactions):
            tr = PCIeTransaction()
            tr.tlp_type = 0  # Memory Read
            tr.address = random.randint(0x1000, 0x2000) & ~0x3  # 4-byte aligned
            tr.tag = i % 256
            tr.length = random.choice([1, 2, 4, 8])  # DW length
            
            await self.start_item(tr)
            await self.finish_item(tr)
            
            logger.info(f"Read sequence sent: {tr}")


class PCIeWriteSequence(PCIeBaseSequence):
    """Generate PCIe memory write transactions"""
    
    async def body(self):
        for i in range(self.num_transactions):
            tr = PCIeTransaction()
            tr.tlp_type = 1  # Memory Write
            tr.address = random.randint(0x1000, 0x2000) & ~0x3
            tr.data = random.randint(0, 0xFFFFFFFF)
            tr.tag = i % 256
            tr.length = 1
            
            await self.start_item(tr)
            await self.finish_item(tr)
            
            logger.info(f"Write sequence sent: {tr}")


class PCIeMixedTrafficSequence(PCIeBaseSequence):
    """Generate mixed read/write traffic"""
    
    async def body(self):
        for i in range(self.num_transactions):
            tr = PCIeTransaction()
            
            # Randomly choose read or write
            if random.random() < 0.5:
                tr.tlp_type = 0  # Read
                tr.length = random.choice([1, 2, 4])
            else:
                tr.tlp_type = 1  # Write
                tr.data = random.randint(0, 0xFFFFFFFF)
                tr.length = 1
                
            tr.address = random.randint(0x1000, 0x2000) & ~0x3
            tr.tag = i % 256
            
            await self.start_item(tr)
            await self.finish_item(tr)


class PCIeErrorInjectionSequence(PCIeBaseSequence):
    """Sequence with coordinated error injection"""
    
    def __init__(self, name="PCIeErrorInjectionSequence", error_type="crc"):
        super().__init__(name)
        self.error_type = error_type
        self.sequencer = None
        
    async def body(self):
        # Get DUT handle
        dut = cocotb.top
        
        logger.info(f"Starting error injection sequence: {self.error_type}")
        
        # Send some normal transactions first
        for i in range(3):
            tr = PCIeTransaction()
            tr.tlp_type = 0  # Read
            tr.address = 0x1000 + (i * 4)
            tr.tag = i
            
            await self.start_item(tr)
            await self.finish_item(tr)
            
        # Inject error based on type
        if self.error_type == "crc":
            logger.warning("Injecting CRC error")
            dut.inject_crc_error.value = 1
            
        elif self.error_type == "timeout":
            logger.warning("Injecting timeout error")
            dut.inject_timeout.value = 1
            
        elif self.error_type == "ecrc":
            logger.warning("Injecting ECRC error")
            dut.inject_ecrc_error.value = 1
            
        elif self.error_type == "malformed":
            logger.warning("Injecting malformed TLP")
            dut.inject_malformed_tlp.value = 1
            
        # Send transaction that will trigger error
        tr = PCIeTransaction()
        tr.tlp_type = 0  # Read
        tr.address = 0x1500
        tr.tag = 99
        
        await self.start_item(tr)
        await self.finish_item(tr)
        
        # Wait a bit
        await cocotb.triggers.Timer(100, units="ns")
        
        # Clear error injection
        dut.inject_crc_error.value = 0
        dut.inject_timeout.value = 0
        dut.inject_ecrc_error.value = 0
        dut.inject_malformed_tlp.value = 0
        
        # Send more normal transactions
        for i in range(3):
            tr = PCIeTransaction()
            tr.tlp_type = 1  # Write
            tr.address = 0x2000 + (i * 4)
            tr.data = 0xDEADBEEF + i
            tr.tag = 100 + i
            
            await self.start_item(tr)
            await self.finish_item(tr)


class PCIeStressSequence(PCIeBaseSequence):
    """High-stress test sequence"""
    
    def __init__(self, name="PCIeStressSequence"):
        super().__init__(name)
        self.num_transactions = 100
        
    async def body(self):
        # Rapid-fire transactions
        for i in range(self.num_transactions):
            tr = PCIeTransaction()
            
            # Mix of transaction types
            tr.tlp_type = random.choice([0, 1])
            tr.address = random.randint(0x0, 0xFFFF) & ~0x3
            tr.data = random.randint(0, 0xFFFFFFFF)
            tr.tag = i % 256
            tr.length = random.choice([1, 2, 4, 8, 16])
            
            await self.start_item(tr)
            await self.finish_item(tr)
            
            # Occasionally no delay between transactions
            if random.random() > 0.8:
                await cocotb.triggers.Timer(1, units="ns")


class PCIeTargetedErrorSequence(PCIeBaseSequence):
    """Sequence that targets specific error conditions"""
    
    async def body(self):
        dut = cocotb.top
        
        # Test 1: Back-to-back errors
        logger.info("Test 1: Back-to-back CRC errors")
        for i in range(3):
            dut.inject_crc_error.value = 1
            
            tr = PCIeTransaction()
            tr.tlp_type = 0
            tr.address = 0x3000 + (i * 4)
            tr.tag = 50 + i
            
            await self.start_item(tr)
            await self.finish_item(tr)
            
            dut.inject_crc_error.value = 0
            await cocotb.triggers.Timer(50, units="ns")
            
        # Test 2: Error during completion
        logger.info("Test 2: Error during completion")
        tr = PCIeTransaction()
        tr.tlp_type = 0  # Read - will generate completion
        tr.address = 0x4000
        tr.tag = 200
        
        await self.start_item(tr)
        await self.finish_item(tr)
        
        # Inject error while completion is pending
        await cocotb.triggers.Timer(10, units="ns")
        dut.inject_ecrc_error.value = 1
        await cocotb.triggers.Timer(100, units="ns")
        dut.inject_ecrc_error.value = 0
        
        # Test 3: Multiple simultaneous errors
        logger.info("Test 3: Multiple simultaneous errors")
        dut.inject_crc_error.value = 1
        dut.inject_ecrc_error.value = 1
        
        tr = PCIeTransaction()
        tr.tlp_type = 1  # Write
        tr.address = 0x5000
        tr.data = 0xBADC0DE
        tr.tag = 255
        
        await self.start_item(tr)
        await self.finish_item(tr)
        
        await cocotb.triggers.Timer(100, units="ns")
        dut.inject_crc_error.value = 0
        dut.inject_ecrc_error.value = 0