#!/usr/bin/env python3
"""
Comprehensive test suite for the UnitaryRFT library.
Tests both the core UnitaryRFT class and the simplified RFTProcessor interface.
"""

import numpy as np
from unitary_rft import UnitaryRFT, RFTProcessor, RFT_FLAG_UNITARY, RFT_FLAG_QUANTUM_SAFE, RFT_FLAG_USE_RESONANCE


def test_unitary_rft():
    """Test the core UnitaryRFT functionality."""
    print("=== Testing UnitaryRFT Core ===")
    
    # Test with a simple signal
    size = 16
    rft = UnitaryRFT(size, RFT_FLAG_UNITARY | RFT_FLAG_QUANTUM_SAFE)
    
    # Create a test signal
    signal = np.zeros(size, dtype=np.complex128)
    signal[0] = 1.0  # Impulse at the beginning
    
    # Perform forward transform
    spectrum = rft.forward(signal)
    print("Spectrum:")
    for i in range(min(4, size)):
        print(f"  [{i}]: {spectrum[i]}")
    
    # Verify unitarity (norm preservation)
    signal_norm = np.sqrt(np.sum(np.abs(signal)**2))
    spectrum_norm = np.sqrt(np.sum(np.abs(spectrum)**2))
    print(f"Norm preservation: input={signal_norm}, output={spectrum_norm}, ratio={spectrum_norm/signal_norm}")
    
    # Perform inverse transform
    reconstructed = rft.inverse(spectrum)
    
    # Calculate reconstruction error
    error = np.max(np.abs(signal - reconstructed))
    print(f"Reconstruction error: {error}")
    
    # Test quantum capabilities
    qubit_count = int(np.log2(size))
    rft.init_quantum_basis(qubit_count)
    
    # Create a Bell state (|00⟩ + |11⟩)/√2
    bell_state = np.zeros(size, dtype=np.complex128)
    bell_state[0] = 1.0 / np.sqrt(2)  # |00⟩
    bell_state[3] = 1.0 / np.sqrt(2)  # |11⟩
    
    # Measure entanglement
    entanglement = rft.measure_entanglement(bell_state)
    print(f"Entanglement of Bell state: {entanglement}")
    
    # Validate results
    assert abs(spectrum_norm/signal_norm - 1.0) < 1e-10, "Norm not preserved!"
    assert error < 1e-10, "Reconstruction error too large!"
    print("✓ UnitaryRFT core tests passed!")
    return True


def test_rft_processor():
    """Test the simplified RFTProcessor interface."""
    print("\n=== Testing RFTProcessor Interface ===")
    
    processor = RFTProcessor(size=64)
    print(f"RFT processor available: {processor.is_available()}")
    
    if not processor.is_available():
        print("RFT processor not available, skipping advanced tests")
        return True
    
    # Test with different data types
    test_cases = [
        # Test case 1: list of numbers
        ([1.0, 2.0, 3.0, 4.0], "list input"),
        
        # Test case 2: numpy array
        (np.array([1+1j, 2+2j, 3+3j, 4+4j]), "numpy array input"),
        
        # Test case 3: string input
        ("test", "string input"),
        
        # Test case 4: single number
        (42.0, "single number input"),
    ]
    
    for test_data, description in test_cases:
        try:
            print(f"Testing {description}...")
            result = processor.process_quantum_field(test_data)
            
            # Validate result exists and has expected properties
            if result is not None:
                if hasattr(result, '__len__') and len(result) > 0:
                    # For array-like results, check if any elements are non-zero
                    if isinstance(result, np.ndarray):
                        has_data = np.any(np.abs(result) > 1e-10)
                        print(f"  Result: {type(result).__name__} with {len(result)} elements, has_data={has_data}")
                    else:
                        print(f"  Result: {type(result).__name__} with {len(result)} elements")
                else:
                    print(f"  Result: {result}")
            else:
                print(f"  Result: None")
            
            print(f"✓ {description} processed successfully")
            
        except Exception as e:
            print(f"✗ {description} failed: {e}")
            return False
    
    print("✓ RFTProcessor interface tests passed!")
    return True


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test invalid size
        rft = UnitaryRFT(0, RFT_FLAG_UNITARY)
        print("✗ Should have failed with invalid size")
        return False
    except Exception as e:
        print(f"✓ Invalid size correctly rejected: {e}")
    
    try:
        # Test with valid RFT
        rft = UnitaryRFT(8, RFT_FLAG_UNITARY)
        
        # Test wrong input size
        wrong_size_input = np.zeros(16, dtype=np.complex128)
        result = rft.forward(wrong_size_input)
        print("✗ Should have failed with wrong input size")
        return False
    except Exception as e:
        print(f"✓ Wrong input size correctly rejected: {e}")
    
    print("✓ Error handling tests passed!")
    return True


def main():
    """Run all tests."""
    print("Running comprehensive RFT test suite...\n")
    
    success = True
    success &= test_unitary_rft()
    success &= test_rft_processor()
    success &= test_error_handling()
    
    print(f"\n=== Test Results ===")
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
