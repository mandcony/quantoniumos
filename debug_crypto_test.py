"""
QuantoniumOS - Debug Enhanced Cryptography Test
Reduced test with debug output to identify hanging points
"""

import os
import sys
import traceback
import time

# Configure more verbose error reporting
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add the project root to Python's module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def debug_test_encryption():
    """Test encryption with debug output"""
    print("Testing enhanced encryption...")
    
    try:
        from core.encryption.resonance_encrypt import resonance_encrypt
        
        # Basic test with timing
        print("  Step 1: Import successful")
        
        # Create small test data
        input_data = b"Hello QuantoniumOS! This is a test."
        key = "test_key_123"
        
        print(f"  Step 2: Encrypting small test data (len={len(input_data)})")
        start_time = time.time()
        ciphertext = resonance_encrypt(input_data, key)
        elapsed = time.time() - start_time
        print(f"  Step 3: Encryption completed in {elapsed:.2f} seconds")
        print(f"  Result: Ciphertext length: {len(ciphertext)}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

def debug_test_hash():
    """Test hash function with debug output"""
    print("\nTesting enhanced hash function...")
    
    try:
        from core.encryption.geometric_waveform_hash import GeometricWaveformHash
        
        # Basic test with timing
        print("  Step 1: Import successful")
        
        # Create small test waveform
        waveform = [0.1 * i for i in range(100)]  # Simple ramp function
        
        print(f"  Step 2: Creating hash object")
        start_time = time.time()
        hash_function = GeometricWaveformHash(waveform=waveform)
        elapsed = time.time() - start_time
        print(f"  Step 3: Object created in {elapsed:.2f} seconds")
        
        print("  Step 4: Generating hash")
        start_time = time.time()
        hash_value = hash_function.generate_hash()
        elapsed = time.time() - start_time
        print(f"  Step 5: Hash generated in {elapsed:.2f} seconds")
        print(f"  Result: Hash length: {len(hash_value)}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

def debug_test_single_avalanche_encryption():
    """Test a single avalanche iteration"""
    print("\nTesting single avalanche encryption iteration...")
    
    try:
        import random
        import os
        from core.encryption.resonance_encrypt import resonance_encrypt
        
        # Single test
        print("  Step 1: Generating random data")
        input_data = os.urandom(32)  # Small data for quick test
        key = "test_key_123"
        
        print("  Step 2: First encryption")
        start_time = time.time()
        ciphertext1 = resonance_encrypt(input_data, key)
        elapsed = time.time() - start_time
        print(f"  Step 3: First encryption completed in {elapsed:.2f} seconds")
        
        # Modify a single bit
        print("  Step 4: Modifying input")
        modified_input = bytearray(input_data)
        byte_index = 0
        bit_index = 0
        modified_input[byte_index] ^= (1 << bit_index)
        
        print("  Step 5: Second encryption")
        start_time = time.time()
        ciphertext2 = resonance_encrypt(bytes(modified_input), key)
        elapsed = time.time() - start_time
        print(f"  Step 6: Second encryption completed in {elapsed:.2f} seconds")
        
        # Compare results
        print("  Step 7: Comparing results")
        actual_cipher1 = ciphertext1[40:]  # Skip signature and token
        actual_cipher2 = ciphertext2[40:]
        
        # Count bit differences
        diff_bits = 0
        for b1, b2 in zip(actual_cipher1, actual_cipher2):
            xor = b1 ^ b2
            for bit in range(8):
                if (xor >> bit) & 1:
                    diff_bits += 1
        
        total_bits = min(len(actual_cipher1), len(actual_cipher2)) * 8
        change_percentage = (diff_bits / total_bits) * 100
        
        print(f"  Result: {diff_bits} bits changed out of {total_bits} ({change_percentage:.2f}%)")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

def debug_test_single_avalanche_hash():
    """Test a single avalanche iteration for hash"""
    print("\nTesting single avalanche hash iteration...")
    
    try:
        import random
        from core.encryption.geometric_waveform_hash import GeometricWaveformHash
        
        # Single test
        print("  Step 1: Generating waveform")
        waveform1 = [0.1 * i for i in range(100)]  # Simple ramp function
        
        print("  Step 2: Creating first hash object")
        start_time = time.time()
        hash_function = GeometricWaveformHash(waveform=waveform1)
        elapsed = time.time() - start_time
        print(f"  Step 3: First object created in {elapsed:.2f} seconds")
        
        print("  Step 4: Generating first hash")
        start_time = time.time()
        hash1 = hash_function.generate_hash().encode('utf-8')
        elapsed = time.time() - start_time
        print(f"  Step 5: First hash generated in {elapsed:.2f} seconds")
        
        # Modify a single value
        print("  Step 6: Modifying waveform")
        waveform2 = waveform1.copy()
        waveform2[0] += 0.00001
        
        print("  Step 7: Creating second hash object")
        start_time = time.time()
        hash_function = GeometricWaveformHash(waveform=waveform2)
        elapsed = time.time() - start_time
        print(f"  Step 8: Second object created in {elapsed:.2f} seconds")
        
        print("  Step 9: Generating second hash")
        start_time = time.time()
        hash2 = hash_function.generate_hash().encode('utf-8')
        elapsed = time.time() - start_time
        print(f"  Step 10: Second hash generated in {elapsed:.2f} seconds")
        
        # Compare results
        print("  Step 11: Comparing results")
        # Count bit differences
        diff_bits = 0
        for b1, b2 in zip(hash1, hash2):
            xor = b1 ^ b2
            for bit in range(8):
                if (xor >> bit) & 1:
                    diff_bits += 1
        
        total_bits = min(len(hash1), len(hash2)) * 8
        change_percentage = (diff_bits / total_bits) * 100
        
        print(f"  Result: {diff_bits} bits changed out of {total_bits} ({change_percentage:.2f}%)")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("QuantoniumOS Debug Crypto Tester")
        print("=" * 50)
        
        logging.info("Starting debug tests")
        
        # Check if core modules are accessible
        print("Checking module paths...")
        module_paths = [
            os.path.join(current_dir, 'core'),
            os.path.join(current_dir, 'core', 'encryption')
        ]
        for path in module_paths:
            exists = os.path.exists(path)
            print(f"Path {path}: {'EXISTS' if exists else 'MISSING'}")
            if exists:
                print(f"  Contents: {os.listdir(path)}")
        
        # Basic encryption test
        print("\nAttempting encryption test...")
        debug_test_encryption()
        
        # Basic hash test
        print("\nAttempting hash test...")
        debug_test_hash()
        
        # Single avalanche tests
        print("\nAttempting avalanche encryption test...")
        debug_test_single_avalanche_encryption()
        
        print("\nAttempting avalanche hash test...")
        debug_test_single_avalanche_hash()
        
        print("\nDebug testing complete.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
