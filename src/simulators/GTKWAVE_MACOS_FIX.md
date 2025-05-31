# GTKWave macOS Security Fix

## Issue
macOS shows warning: "Apple cannot check GTKWave for malicious software"

## Solutions

### Method 1: Allow in Security Settings (Recommended)
```bash
# 1. Try to open GTKWave
gtkwave pcie_waveform.vcd

# 2. When blocked, go to:
# System Settings → Privacy & Security → Security

# 3. You'll see: "gtkwave was blocked from use"
# Click "Open Anyway"

# 4. Try again:
gtkwave pcie_waveform.vcd

# 5. Click "Open" in the popup
```

### Method 2: Remove Quarantine Attribute
```bash
# Remove the quarantine flag
sudo xattr -r -d com.apple.quarantine /Applications/gtkwave.app

# Or if installed via Homebrew:
sudo xattr -r -d com.apple.quarantine /opt/homebrew/Caskroom/gtkwave/
```

### Method 3: Command Line Override
```bash
# Allow from command line
sudo spctl --add /Applications/gtkwave.app
sudo spctl --enable --label "gtkwave"
```

### Method 4: Right-Click Method
1. Find GTKWave in Applications
2. Right-click (or Control-click) on gtkwave.app
3. Select "Open" from the menu
4. Click "Open" in the dialog

## Alternative: Web-Based Viewers (No Installation)

### 1. Use the Text-Based Viewer
```bash
python3 vcd_viewer.py pcie_waveform.vcd
```

### 2. WaveJSON Converter
```bash
# Install converter
pip install vcdvcd

# Convert to JSON
vcdcat pcie_waveform.vcd --format json > waveform.json

# View at: https://wavedrom.com/editor.html
```

### 3. Online VCD Viewers
- Upload to: https://vc.drom.io/
- Or use: https://www.edaplayground.com/

## Prevention for Future Installs

### Use Homebrew with --no-quarantine
```bash
# Reinstall without quarantine
brew uninstall --cask gtkwave
brew install --cask --no-quarantine gtkwave
```

## Why This Happens
- macOS Gatekeeper blocks unsigned apps
- GTKWave is open source and not notarized by Apple
- This is a security feature, not malware

## Quick Terminal Viewer Alternative
```bash
# Our custom viewer works without any warnings
python3 vcd_viewer.py

# Shows waveforms in terminal:
# Time | clk | rst_n | tlp_valid | error_valid | ltssm_state
# -----|-----|-------|-----------|-------------|------------
# 100  |  1  |   1   |     0     |      0      |   DETECT
```