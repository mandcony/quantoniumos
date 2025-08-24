#!/usr/bin/env python3
"""
Quantum Crypto App Launcher with Unified Design
"""
import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from quantonium_app_wrapper import launch_app
    from quantum_crypto import QuantumCrypto

    def main():
        return launch_app(QuantumCrypto, "Quantum Crypto", "Quantum Cryptography")

    if __name__ == "__main__":
        sys.exit(main())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure PyQt5 and required packages are installed")

    # Fallback main function for validation
    def main():
        print("Quantum Crypto launcher (fallback mode)")
        return 0
