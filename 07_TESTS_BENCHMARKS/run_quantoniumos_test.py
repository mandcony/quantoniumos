#!/usr/bin/env python3
"""
QuantoniumOS Test Script

This script demonstrates the main features and components of QuantoniumOS
in a command-line interface since we can't run the GUI in this environment.
"""

import os
import sys
import time

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import QuantoniumOS components
import core.quantoniumos as quantoniumosfrom core.quantonium_os_unified
import QuantoniumOSUnified
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
import paper_compliant_rft_fixed as paper_compliant_rft_fixed
def print_header(text):
    """Print a header with the given text"""
    print("\n" + "=" * 80)
    print(f" {text} ".center(80, "="))
    print("=" * 80)

def print_section(text):
    """Print a section header with the given text"""
    print("\n" + "-" * 80)
    print(f" {text} ".center(80, "-"))
    print("-" * 80)

def main():
    """Main test function"""
    print_header("QuantoniumOS Test Script")
    
    # Step 1: Initialize QuantoniumOS
    print_section("Initializing QuantoniumOS")
    os_instance = quantoniumos.QuantoniumOS()
    status = os_instance.start()
    print(f"QuantoniumOS {status['version']} - Status: {status['status']}")
    
    # Step 2: Initialize unified system
    print_section("Initializing Unified System")
    unified = QuantoniumOSUnified()
    result = unified.initialize()
    print(f"Unified System Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    # Step 3: Run unified system
    print_section("Running Unified System")
    result = unified.run()
    print(f"Unified System Run Status: {result['status']}")
    print(f"Message: {result['message']}")
    
    # Step 4: Test RFT Component
    print_section("Testing RFT Component")
    print("Initializing Paper-Compliant RFT...")
    rft = paper_compliant_rft_fixed.PaperCompliantRFT(size=64, num_resonators=8)
    
    # Create test data
    import numpy as np
    test_data = np.random.random(64)
    
    # Apply transform
    print("Applying RFT transform...")
    result = rft.transform(test_data)
    print(f"RFT Transform Status: {result['status']}")
    print(f"Transform Size: {len(result['transformed'])}")
    
    # Apply inverse transform
    print("Applying inverse RFT transform...")
    inverse = rft.inverse_transform(result['transformed'])
    max_error = np.max(np.abs(test_data - inverse['signal']))
    print(f"Roundtrip Error: {max_error:.6e}")
    
    # Step 5: Test Encryption
    print_section("Testing RFT Encryption")
    crypto = paper_compliant_rft_fixed.FixedRFTCryptoBindings()
    crypto.init_engine()
    
    message = b"This is a test message for the QuantoniumOS RFT encryption algorithm"
    key = b"SecretKey123"
    
    print(f"Original message: {message}")
    print("Encrypting message...")
    encrypted = crypto.encrypt_block(message, key)
    print(f"Encrypted size: {len(encrypted)} bytes")
    
    print("Decrypting message...")
    decrypted = crypto.decrypt_block(encrypted, key)
    print(f"Decrypted message: {decrypted}")
    print(f"Roundtrip successful: {message == decrypted}")
    
    # Step 6: Test Key Avalanche
    print_section("Testing Key Avalanche Effect")
    results = crypto.validate_compliance()
    print(f"Key Avalanche Effect: {results['reported_avalanche']:.4f}")
    print(f"Target Avalanche (Paper): {results['target_avalanche']:.4f}")
    print(f"Avalanche Compliance: {results['avalanche_compliance']}")
    
    # Step 7: Shut down QuantoniumOS
    print_section("Shutting Down QuantoniumOS")
    shutdown = os_instance.stop()
    print(f"QuantoniumOS Status: {shutdown['status']}")
    
    print_header("QuantoniumOS Test Completed Successfully")

if __name__ == "__main__":
    main()
