#!/usr/bin/env python3
"""
Cipher validation test for unified RFT crypto implementation
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Any

# Add this directory to path for imports
sys.path.append(os.path.dirname(__file__))

# Import unified interface
from unified_crypto import UnifiedRFTCrypto

def main():
    """Main function to run cipher validation with implementation choice"""
    parser = argparse.ArgumentParser(description="Enhanced RFT Crypto validation")
    parser.add_argument("--use-assembly", action="store_true", 
                        help="Use assembly implementation if available")
    parser.add_argument("--force-python", action="store_true",
                        help="Force pure Python implementation even if assembly is available")
    args = parser.parse_args()
    
    # Determine which implementation to use
    use_assembly = args.use_assembly and not args.force_python
    
    # Check available implementations
    implementations = UnifiedRFTCrypto.available_implementations()
    print("\nAvailable implementations:")
    print(f"  Pure Python: {'✓ Available' if implementations['python'] else '✗ Not available'}")
    print(f"  Assembly:    {'✓ Available' if implementations['assembly'] else '✗ Not available'}")
    
    # Expected throughput
    expected = UnifiedRFTCrypto.expected_throughput()
    print("\nExpected throughput:")
    print(f"  Pure Python: {expected['python']:.3f} MB/s")
    print(f"  Assembly:    {expected['assembly']:.3f} MB/s")
    
    # Try to create unified interface
    try:
        # Use a random key for testing
        import secrets
        key = secrets.token_bytes(32)
        
        # Create unified interface
        crypto = UnifiedRFTCrypto(key, use_assembly=use_assembly)
        impl_name = crypto.get_implementation_name()
        is_assembly = crypto.is_assembly_implementation()
        
        print(f"\nUsing implementation: {impl_name}")
        
        # Run simple test
        message = b"This is a test message for crypto validation"
        encrypted = crypto.encrypt(message)
        decrypted = crypto.decrypt(encrypted)
        
        if message == decrypted:
            print("✓ Encryption/decryption test passed")
        else:
            print("✗ Encryption/decryption test failed")
            return 1
        
        # Performance test
        sizes = [1024, 4096, 16384, 65536]
        print("\n⚡ Performance Test:")
        
        for size in sizes:
            test_data = secrets.token_bytes(size)
            
            # Test encryption
            start_time = time.perf_counter()
            encrypted = crypto.encrypt(test_data)
            encrypt_time = time.perf_counter() - start_time
            
            # Test decryption
            start_time = time.perf_counter()
            decrypted = crypto.decrypt(encrypted)
            decrypt_time = time.perf_counter() - start_time
            
            # Calculate throughput
            encrypt_mbps = (size / encrypt_time) / (1024 * 1024) if encrypt_time > 0 else 0
            decrypt_mbps = (size / decrypt_time) / (1024 * 1024) if decrypt_time > 0 else 0
            
            print(f"  {size:6d} bytes: {encrypt_mbps:.3f} MB/s encrypt, {decrypt_mbps:.3f} MB/s decrypt")
        
        # Compare to expected throughput
        target = expected['assembly'] if is_assembly else expected['python']
        print(f"\n  Expected throughput for this implementation: {target:.3f} MB/s")
        
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
