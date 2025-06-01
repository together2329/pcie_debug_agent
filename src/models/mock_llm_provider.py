"""
Mock LLM Provider for testing without actual models
"""

import random
import time
from typing import Optional, Dict, Any

class MockLLMProvider:
    """Mock LLM that provides realistic-looking responses for testing"""
    
    def __init__(self, model_path: str = "", **kwargs):
        self.model_path = model_path
        self.model_loaded = False
        self.verbose = kwargs.get('verbose', False)
        
    def load_model(self) -> bool:
        """Simulate model loading"""
        if self.verbose:
            print("Loading mock model...")
        time.sleep(0.1)  # Simulate loading time
        self.model_loaded = True
        return True
    
    def generate_completion(self, prompt: str, temperature: float = 0.7, 
                          max_tokens: int = 1000, **kwargs) -> str:
        """Generate mock completion based on the prompt"""
        
        # Simulate thinking time
        time.sleep(0.5)
        
        # Extract key information from prompt
        prompt_lower = prompt.lower()
        
        # PCIe-specific responses based on common queries
        if "completion timeout" in prompt_lower:
            return self._completion_timeout_response()
        elif "link training" in prompt_lower:
            return self._link_training_response()
        elif "aer" in prompt_lower or "advanced error" in prompt_lower:
            return self._aer_response()
        elif "power management" in prompt_lower or "aspm" in prompt_lower:
            return self._power_management_response()
        elif "signal integrity" in prompt_lower:
            return self._signal_integrity_response()
        else:
            return self._generic_pcie_response(prompt)
    
    def _completion_timeout_response(self) -> str:
        return """Based on the analysis, PCIe completion timeout errors typically occur due to:

1. **Device Response Issues**
   - Device not responding within the specified timeout period (typically 50ms-60s)
   - Faulty endpoint device or intermediate switches
   - Device in unexpected power state (D3cold)

2. **Configuration Problems**
   - Incorrect completion timeout values in Device Control 2 Register
   - Mismatched timeout settings between root complex and endpoint
   - BIOS/firmware bugs affecting timeout handling

3. **Hardware/Signal Issues**
   - Poor signal integrity causing packet corruption
   - Inadequate power delivery to the device
   - Physical connection problems (loose cards, damaged connectors)

**Recommended Actions:**
- Check device power state: `lspci -vv | grep -i power`
- Verify completion timeout settings: `setpci -s <device> CAP_EXP+0x28.w`
- Test with different PCIe slots if possible
- Update device firmware and system BIOS
- Monitor AER logs for additional error patterns"""
    
    def _link_training_response(self) -> str:
        return """PCIe link training failures can be diagnosed through these steps:

1. **Check Link Status**
   ```bash
   lspci -vv | grep -E "LnkCap|LnkSta"
   ```
   
2. **Common Causes:**
   - Incompatible link speed settings
   - Reference clock issues (100MHz +/-300ppm)
   - Equalization failures at Gen3/Gen4 speeds
   - Power sequencing problems

3. **Debugging Steps:**
   - Force lower link speed: `setpci -s <device> CAP_EXP+0x30.w=0x0001`
   - Check for electrical idle exits
   - Verify PERST# timing requirements
   - Review platform error logs

4. **Resolution:**
   - Update firmware/BIOS
   - Clean PCIe connector contacts
   - Verify power supply stability
   - Try different PCIe slots"""
    
    def _aer_response(self) -> str:
        return """Advanced Error Reporting (AER) provides detailed PCIe error information:

**Common AER Errors:**
1. **Correctable Errors**
   - Bad TLP (Transaction Layer Packet)
   - Bad DLLP (Data Link Layer Packet)
   - Replay Timer Timeout
   - Receiver Error

2. **Uncorrectable Errors**
   - Data Link Protocol Error
   - Poisoned TLP
   - Completion Timeout
   - Unexpected Completion

**Analysis Commands:**
```bash
# Enable AER
setpci -s <device> CAP_EXP+0x08.w=0x0f

# Check AER capability
lspci -vv | grep -A20 "Advanced Error"

# Monitor AER events
dmesg | grep -i aer
```

**Mitigation:**
- Enable ECRC (End-to-End CRC)
- Adjust replay timer values
- Update device drivers
- Check signal integrity"""
    
    def _power_management_response(self) -> str:
        return """PCIe power management issues and ASPM (Active State Power Management):

**Diagnosis:**
1. Check current ASPM state:
   ```bash
   lspci -vv | grep ASPM
   cat /sys/module/pcie_aspm/parameters/policy
   ```

2. **Common Issues:**
   - L0s/L1 exit latency exceeding device tolerance
   - Clock request (CLKREQ#) signaling problems
   - Incompatible ASPM settings between endpoints

3. **Solutions:**
   - Disable ASPM: `pcie_aspm=off` (kernel parameter)
   - Force specific policy: `pcie_aspm.policy=performance`
   - Adjust L1 substate settings in BIOS

4. **Testing:**
   - Monitor power state transitions
   - Verify link recovery after idle periods
   - Check for increased error rates with ASPM enabled"""
    
    def _signal_integrity_response(self) -> str:
        return """PCIe signal integrity analysis and troubleshooting:

**Key Measurements:**
1. **Eye Diagram Analysis**
   - Eye height/width margins
   - Jitter components (DJ, RJ, TJ)
   - Rise/fall times

2. **Common Issues:**
   - Excessive insertion loss
   - Impedance discontinuities
   - Crosstalk between differential pairs
   - Power supply noise coupling

3. **Diagnostic Approach:**
   - Use PCIe protocol analyzer if available
   - Check for retransmissions: `ethtool -S <interface>`
   - Monitor error counters
   - Verify termination and equalization settings

4. **Improvements:**
   - Ensure proper PCB design (100Ω differential impedance)
   - Use quality cables for external connections
   - Verify power delivery network (PDN) design
   - Consider re-drivers or retimers for long traces"""
    
    def _generic_pcie_response(self, prompt: str) -> str:
        keywords = ["error", "debug", "issue", "problem", "fail", "timeout", "link"]
        relevant_keyword = next((kw for kw in keywords if kw in prompt.lower()), "analysis")
        
        return f"""Based on your query about PCIe {relevant_keyword}, here's a general debugging approach:

1. **System Information**
   ```bash
   lspci -tv  # Tree view of devices
   lspci -vv  # Detailed device info
   dmesg | grep -i pcie  # Kernel messages
   ```

2. **Performance Monitoring**
   - Check link speed and width
   - Monitor error counters
   - Verify BAR allocations
   - Review interrupt assignments

3. **Common Tools**
   - `pcieport` - PCIe port driver info
   - `setpci` - Direct configuration access
   - `lspci` - List and query devices
   - `/sys/bus/pci/` - Sysfs interface

4. **Next Steps**
   - Collect comprehensive logs
   - Identify error patterns
   - Test with minimal configuration
   - Consult hardware specifications

For specific issues, please provide error messages or symptoms for targeted analysis."""
    
    def get_info(self) -> dict:
        """Get provider information"""
        return {
            "name": "Mock LLM Provider",
            "model": "mock-llm",
            "type": "built-in",
            "status": "ready",
            "description": "Built-in mock model for PCIe debugging"
        }
    
    def __del__(self):
        """Cleanup"""
        pass


# Make it available as LocalLLMProvider for compatibility
LocalLLMProvider = MockLLMProvider
