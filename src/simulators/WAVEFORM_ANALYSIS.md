# PCIe Simulation Waveform Analysis

## VCD File Generated Successfully

The simulation generated a VCD (Value Change Dump) file: `pcie_waveform.vcd` (22KB)

## Waveform Summary

### 1. Link Training Sequence (0-300ns)
```
Time     | LTSSM State | Link Status | Description
---------|-------------|-------------|------------------
0-100ns  | DETECT      | Down        | Initial detection
105ns    | POLLING     | Down        | Polling for link partner
115ns    | CONFIG      | Down        | Configuration phase
125ns    | L0          | UP          | Link operational
```

### 2. Normal Transactions (250-500ns)
- **250ns**: Memory Read to address 0x1000 (Tag=1)
- **465ns**: Memory Write to address 0x2000, Data=0xDEADBEEF (Tag=2)

### 3. Error Injection Tests

#### CRC Error (675-785ns)
```
Time  | Event
------|-----------------------------------------------
675ns | CRC error injected → LTSSM enters RECOVERY
685ns | Recovery complete → Back to L0
695ns | Multiple recovery cycles due to persistent CRC errors
735ns | Read transaction during recovery (Tag=3)
```

#### Timeout Error (1485ns)
- Completion timeout injected for Tag=4
- Transaction accepted but completion delayed

#### ECRC Error (2595-2705ns)
- Similar to CRC error pattern
- Multiple recovery cycles
- Write transaction with data 0xCAFEBABE

#### Malformed TLP (2905-2925ns)
```
Time   | Signal        | Value | Description
-------|---------------|-------|------------------
2905ns | tlp_type      | 7     | Invalid TLP type
2915ns | error_valid   | 1     | Error detected
2915ns | error_type    | 4     | MALFORMED_TLP
```

## Key Observations

1. **LTSSM State Machine**:
   - Proper progression: DETECT → POLLING → CONFIG → L0
   - Recovery mechanism working correctly for errors

2. **Error Handling**:
   - All error types properly detected
   - Recovery cycles initiated for physical layer errors (CRC/ECRC)
   - Malformed TLP flagged but doesn't trigger recovery

3. **Transaction Flow**:
   - Normal transactions complete successfully
   - Transactions continue after error recovery

## Viewing Options

### 1. Open Source GUI Viewers

**GTKWave** (Recommended):
```bash
# Install
brew install --cask gtkwave  # macOS
sudo apt-get install gtkwave # Linux

# View
gtkwave pcie_waveform.vcd
```

**Surfer**:
```bash
# Web-based viewer
# Upload pcie_waveform.vcd to https://surfer-project.org/
```

### 2. Text-Based Viewer
```bash
python3 vcd_viewer.py pcie_waveform.vcd
```

### 3. Convert to Other Formats

**To JSON (WaveDrom format)**:
```bash
pip install vcdvcd
vcdcat pcie_waveform.vcd --format json > waveform.json
```

**To SVG/PNG**:
```bash
pip install wavedrom
wavedrom -i waveform.json -s waveform.svg
```

## Signal Description

| Signal | Width | Description |
|--------|-------|-------------|
| clk | 1 | System clock (100MHz) |
| rst_n | 1 | Active-low reset |
| ltssm_state | 4 | Link Training State Machine |
| link_up | 1 | Link operational status |
| tlp_valid | 1 | TLP valid indicator |
| tlp_type | 3 | Transaction type (0=MRd, 1=MWr) |
| error_valid | 1 | Error detected flag |
| error_type | 4 | Error type code |

## Error Type Codes
- 0: No error
- 1: CRC error
- 2: Timeout
- 3: ECRC error
- 4: Malformed TLP

## Next Steps

1. **Install GTKWave** for graphical waveform viewing
2. **Analyze specific signals** by zooming into error injection points
3. **Correlate with log files** to understand system behavior
4. **Modify testbench** to test additional scenarios

The VCD file provides complete visibility into the PCIe simulation, allowing detailed debugging of protocol violations and error conditions.