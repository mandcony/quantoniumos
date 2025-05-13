#!/usr/bin/env python
import os
import sys
import subprocess

# Define the path to the resonance analyzer script
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
ANALYZER_SCRIPT = os.path.join(SCRIPT_DIR, "q_resonance_analyzer.py")

def main():
    """Launch the Quantum Resonance Analyzer application."""
    print("🔹 Launching Quantum Resonance Analyzer...")
    
    # Check if the script exists
    if not os.path.exists(ANALYZER_SCRIPT):
        print(f"❌ Error: Analyzer script not found at {ANALYZER_SCRIPT}")
        sys.exit(1)
    
    # Launch the application with Python
    try:
        subprocess.Popen([sys.executable, ANALYZER_SCRIPT])
        print("✅ Quantum Resonance Analyzer launched successfully")
    except Exception as e:
        print(f"❌ Error launching Resonance Analyzer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()