#!/usr/bin/env python3
"""
Test script for verifying the true_rft_engine_bindings.py implementation.

This script tests the TrueRFTEngine class to ensure it correctly uses the
TrueResonanceFourierTransform from true_rft_exact.py.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add the RFT_ALGORITHMS directory to the path
rft_dir = Path("/workspaces/quantoniumos/04_RFT_ALGORITHMS")
if str(rft_dir) not in sys.path:
    sys.path.append(str(rft_dir))

# Import the necessary modules
try:
    from true_rft_exact import TrueResonanceFourierTransform
    print("Successfully imported TrueResonanceFourierTransform")
except ImportError as e:
    print(f"Failed to import TrueResonanceFourierTransform: {e}")
    sys.exit(1)

try:
    from true_rft_engine_bindings import TrueRFTEngine, get_global_engine
    print("Successfully imported TrueRFTEngine")
except ImportError as e:
    print(f"Failed to import TrueRFTEngine: {e}")
    sys.exit(1)

def test_engine_initialization():
    """Test that the engine initializes correctly"""
    print("\nTesting engine initialization...")
    engine = TrueRFTEngine(size=16)
    result = engine.init()
    print(f"Initialization result: {result}")
    assert result["status"] == "SUCCESS", "Engine initialization failed"
    print("Engine initialization test passed!")

def test_transform_and_inverse():
    """Test that the transform and inverse transform work correctly"""
    print("\nTesting transform and inverse transform...")
    engine = TrueRFTEngine(size=16)
    
    # Create some test data
    test_data = np.random.rand(16) + 1j * np.random.rand(16)
    
    # Apply the transform
    transformed = engine.compute_rft(test_data)["result"]
    
    # Apply the inverse transform
    reconstructed = engine.compute_inverse_rft(transformed)["result"]
    
    # Verify that the reconstructed data matches the original
    error = np.mean(np.abs(test_data - reconstructed))
    print(f"Reconstruction error: {error}")
    assert error < 1e-10, f"Reconstruction error too high: {error}"
    print("Transform and inverse transform test passed!")

def test_energy_conservation():
    """Test that the transform conserves energy"""
    print("\nTesting energy conservation...")
    engine = TrueRFTEngine(size=16)
    
    # Create some test data
    test_data = np.random.rand(16) + 1j * np.random.rand(16)
    
    # Calculate the energy of the original data
    original_energy = np.sum(np.abs(test_data)**2)
    
    # Apply the transform
    transformed = engine.compute_rft(test_data)["result"]
    
    # Calculate the energy of the transformed data
    transformed_energy = np.sum(np.abs(transformed)**2)
    
    # Verify that energy is conserved
    energy_ratio = transformed_energy / original_energy
    print(f"Energy before transform: {original_energy}")
    print(f"Energy after transform: {transformed_energy}")
    print(f"Energy ratio: {energy_ratio}")
    assert abs(energy_ratio - 1.0) < 1e-10, f"Energy not conserved, ratio: {energy_ratio}"
    print("Energy conservation test passed!")

def test_cryptographic_functions():
    """Test the cryptographic functions"""
    print("\nTesting cryptographic functions...")
    engine = TrueRFTEngine(size=16)
    
    # Test key generation
    key = engine.generate_key(input_data="test data", salt="test salt")
    print(f"Generated key length: {len(key)}")
    assert len(key) == 32, f"Key length incorrect: {len(key)}"
    
    # Test encryption and decryption
    test_data = "This is a test message for encryption and decryption."
    encrypted = engine.encrypt(test_data, key)
    decrypted = engine.decrypt(encrypted, key)
    
    # Convert decrypted bytes back to string
    decrypted_str = decrypted.decode('utf-8', errors='ignore').rstrip('\x00')
    
    print(f"Original: {test_data}")
    print(f"Decrypted: {decrypted_str}")
    assert test_data in decrypted_str, "Decryption failed"
    print("Cryptographic functions test passed!")

def test_global_engine():
    """Test the global engine instance"""
    print("\nTesting global engine...")
    engine1 = get_global_engine()
    engine2 = get_global_engine()
    
    # Verify that both instances are the same object
    assert engine1 is engine2, "Global engine instances are not the same"
    print("Global engine test passed!")

if __name__ == "__main__":
    print("Starting tests for TrueRFTEngine...")
    
    try:
        test_engine_initialization()
        test_transform_and_inverse()
        test_energy_conservation()
        test_cryptographic_functions()
        test_global_engine()
        
        print("\nAll tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
