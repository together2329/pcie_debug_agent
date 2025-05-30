#!/usr/bin/env python3
"""
Script to run the enhanced UI version of PCIe Debug Agent
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run the enhanced UI"""
    # Get the path to the enhanced app
    app_path = Path(__file__).parent / "src" / "ui" / "app_refactored.py"
    
    if not app_path.exists():
        print(f"Error: Enhanced app not found at {app_path}")
        return 1
    
    print("🚀 Starting PCIe Debug Agent Enhanced UI...")
    print(f"Running: {app_path}")
    
    # Run streamlit with the enhanced app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n✅ PCIe Debug Agent stopped.")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())