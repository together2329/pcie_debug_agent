#!/usr/bin/env python3
"""
Simple VCD viewer - displays waveform in text format
"""

import re
from collections import defaultdict
import sys

class VCDParser:
    def __init__(self, filename):
        self.filename = filename
        self.signals = {}
        self.timescale = "1ns"
        self.changes = defaultdict(list)
        
    def parse(self):
        """Parse VCD file"""
        with open(self.filename, 'r') as f:
            content = f.read()
            
        # Parse header
        self._parse_header(content)
        
        # Parse value changes
        self._parse_changes(content)
        
    def _parse_header(self, content):
        """Parse VCD header for signal definitions"""
        # Find timescale
        ts_match = re.search(r'\$timescale\s+(\S+)\s+\$end', content)
        if ts_match:
            self.timescale = ts_match.group(1)
            
        # Find signal definitions
        var_pattern = r'\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[\d+:\d+\])?\s+\$end'
        for match in re.finditer(var_pattern, content):
            width = int(match.group(1))
            symbol = match.group(2)
            name = match.group(3)
            self.signals[symbol] = {
                'name': name,
                'width': width,
                'values': []
            }
            
    def _parse_changes(self, content):
        """Parse value changes"""
        # Find dumpvars section end
        dumpvars_end = content.find('$end', content.find('$dumpvars'))
        if dumpvars_end == -1:
            return
            
        changes_section = content[dumpvars_end+4:]
        
        current_time = 0
        for line in changes_section.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Time change
            if line.startswith('#'):
                current_time = int(line[1:])
                
            # Value change
            elif len(line) >= 2:
                if line[0] in '01xzXZ':
                    # Single bit signal
                    value = line[0]
                    symbol = line[1:]
                    if symbol in self.signals:
                        self.changes[current_time].append((symbol, value))
                elif line[0] == 'b':
                    # Multi-bit signal
                    parts = line[1:].split()
                    if len(parts) == 2:
                        value = parts[0]
                        symbol = parts[1]
                        if symbol in self.signals:
                            self.changes[current_time].append((symbol, value))
                            
    def print_waveform(self, signal_names=None, start_time=0, end_time=None):
        """Print waveform in text format"""
        if not signal_names:
            # Show most interesting signals
            signal_names = ['clk', 'rst_n', 'tlp_valid', 'tlp_type', 
                          'error_valid', 'error_type', 'ltssm_state', 'link_up']
            
        # Find matching signals
        display_signals = []
        for symbol, info in self.signals.items():
            if info['name'] in signal_names:
                display_signals.append((symbol, info))
                
        if not display_signals:
            print("No matching signals found")
            return
            
        # Get time range
        times = sorted(self.changes.keys())
        if not times:
            print("No signal changes found")
            return
            
        if end_time is None:
            end_time = times[-1]
            
        # Build value history for each signal
        signal_values = {}
        for symbol, info in display_signals:
            signal_values[symbol] = []
            current_value = 'x'
            
            for t in times:
                if t > end_time:
                    break
                    
                # Check if signal changed at this time
                for changed_symbol, new_value in self.changes[t]:
                    if changed_symbol == symbol:
                        current_value = new_value
                        
                if t >= start_time:
                    signal_values[symbol].append((t, current_value))
                    
        # Print waveform
        print(f"\nVCD Waveform Viewer - {self.filename}")
        print(f"Timescale: {self.timescale}")
        print("=" * 80)
        
        # Print header
        print(f"{'Time':>10} | ", end='')
        for symbol, info in display_signals:
            name = info['name']
            width = max(len(name), 8)
            print(f"{name:^{width}} | ", end='')
        print()
        print("-" * 80)
        
        # Print values at key times
        sample_times = []
        for t in times:
            if t >= start_time and t <= end_time:
                # Include times where interesting signals change
                for symbol, value in self.changes[t]:
                    if symbol in [s[0] for s in display_signals]:
                        if symbol in [s for s, i in display_signals if i['name'] in 
                                    ['error_valid', 'tlp_valid', 'ltssm_state']]:
                            sample_times.append(t)
                            break
                            
        # Limit number of rows
        if len(sample_times) > 50:
            step = len(sample_times) // 50
            sample_times = sample_times[::step]
            
        # Print values
        for t in sample_times:
            print(f"{t:>10} | ", end='')
            
            for symbol, info in display_signals:
                # Find value at this time
                value = 'x'
                for change_time, val in signal_values[symbol]:
                    if change_time <= t:
                        value = val
                    else:
                        break
                        
                # Format value
                width = max(len(info['name']), 8)
                if info['width'] == 1:
                    print(f"{value:^{width}} | ", end='')
                else:
                    # Multi-bit value
                    if value.startswith('b'):
                        value = value
                    int_val = int(value, 2) if value not in ['x', 'z'] else 0
                    if info['name'] == 'ltssm_state':
                        state_names = {0: 'DETECT', 1: 'POLL', 2: 'CONFIG', 3: 'L0', 4: 'RECOV'}
                        state = state_names.get(int_val, f'ST{int_val}')
                        print(f"{state:^{width}} | ", end='')
                    elif info['name'] == 'error_type':
                        err_names = {0: 'NONE', 1: 'CRC', 2: 'TMOUT', 3: 'ECRC', 4: 'MLFRM'}
                        err = err_names.get(int_val, f'ERR{int_val}')
                        print(f"{err:^{width}} | ", end='')
                    else:
                        print(f"{int_val:^{width}} | ", end='')
            print()
            
        print("=" * 80)
        
    def print_summary(self):
        """Print summary of VCD file"""
        print(f"\nVCD File Summary:")
        print(f"  Timescale: {self.timescale}")
        print(f"  Total signals: {len(self.signals)}")
        print(f"  Total time points: {len(self.changes)}")
        
        if self.changes:
            times = sorted(self.changes.keys())
            print(f"  Time range: {times[0]} - {times[-1]}")
            
        print(f"\nKey signals found:")
        interesting = ['clk', 'rst_n', 'error_valid', 'error_type', 
                      'tlp_valid', 'link_up', 'ltssm_state']
        for symbol, info in self.signals.items():
            if info['name'] in interesting:
                print(f"  {info['name']}: {info['width']} bit(s)")


def main():
    if len(sys.argv) < 2:
        vcd_file = "pcie_waveform.vcd"
    else:
        vcd_file = sys.argv[1]
        
    print(f"Parsing {vcd_file}...")
    
    parser = VCDParser(vcd_file)
    parser.parse()
    
    # Print summary
    parser.print_summary()
    
    # Print waveform
    print("\n" + "="*80)
    print("WAVEFORM DISPLAY")
    print("="*80)
    
    # Show different time ranges
    print("\n1. Link Training Phase (0-300000):")
    parser.print_waveform(start_time=0, end_time=300000)
    
    print("\n2. Error Injection Phase (600000-1000000):")
    parser.print_waveform(start_time=600000, end_time=1000000)
    
    print("\n3. Full Simulation Overview:")
    parser.print_waveform()
    
    print("\nNote: To view full waveform graphically, install GTKWave:")
    print("  macOS: brew install --cask gtkwave")
    print("  Linux: sudo apt-get install gtkwave")
    print(f"  Then run: gtkwave {vcd_file}")


if __name__ == "__main__":
    main()