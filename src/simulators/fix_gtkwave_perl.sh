#!/bin/bash

# Fix GTKWave Perl dependency issue on macOS

echo "Fixing GTKWave Perl dependency issue..."

# Method 1: Install Switch module via CPAN
echo "Method 1: Installing Switch module..."
sudo cpan Switch

# If that doesn't work, try Method 2
if [ $? -ne 0 ]; then
    echo "Method 1 failed. Trying Method 2..."
    
    # Method 2: Install via system perl
    sudo perl -MCPAN -e 'install Switch'
fi

# Method 3: Manual fix by editing gtkwave script
if [ $? -ne 0 ]; then
    echo "Method 2 failed. Applying manual fix..."
    
    # Create a wrapper script
    cat > /tmp/gtkwave_fixed << 'EOF'
#!/bin/bash
# GTKWave wrapper to bypass Perl issues
exec /opt/homebrew/Cellar/gtkwave/*/gtkwave.app/Contents/MacOS/gtkwave-bin "$@"
EOF
    
    chmod +x /tmp/gtkwave_fixed
    echo "Created fixed wrapper at /tmp/gtkwave_fixed"
    echo "Use: /tmp/gtkwave_fixed pcie_waveform.vcd"
fi

echo "Fix attempt complete!"