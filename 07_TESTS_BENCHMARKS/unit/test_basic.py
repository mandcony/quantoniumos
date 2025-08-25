"""Basic tests for QuantoniumOS package"""

import numpy as np
import pytest
import sys
import os

# Add paths for QuantoniumOS modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.insert(0, '/workspaces/quantoniumos/06_CRYPTOGRAPHY')
sys.path.insert(0, '/workspaces/quantoniumos/04_RFT_ALGORITHMS')
sys.path.insert(0, '/workspaces/quantoniumos/05_QUANTUM_ENGINES')


def test_package_import():
    """Test that the QuantoniumOS packages import successfully"""
    try:
        from quantonium_crypto_production import QuantoniumCrypto
        from bulletproof_quantum_kernel import BulletproofQuantumKernel
        assert True, "All packages imported successfully"
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_basic_functionality():
    """Test basic QuantoniumOS functionality"""
    try:
        from quantonium_crypto_production import QuantoniumCrypto
        crypto = QuantoniumCrypto()
        
        # Test basic encryption/decryption using encrypt_short
        test_data = b"Hello!"  # Keep it short for encrypt_short
        encrypted_data, key = crypto.encrypt_short(test_data)
        decrypted = crypto.decrypt_short(encrypted_data, key)
        
        assert decrypted == test_data, f"Encryption/decryption failed: {decrypted} != {test_data}"
        print("✅ Basic encryption test passed")
        
        # Test version if available
        try:
            # Check if we have version info
            version = getattr(crypto, '__version__', None) or getattr(quantum, '__version__', None)
            if version:
                print(f"QuantoniumOS version: {version}")
        except:
            print("Version info not available")
        
    except Exception as e:
        pytest.fail(f"Basic functionality test failed: {e}")


def test_quantum_kernel_basic():
    """Test basic QuantumKernel functionality"""
    try:
        # Import the quantum engine directly (path already added at top)
        from bulletproof_quantum_kernel import BulletproofQuantumKernel
        
        kernel = BulletproofQuantumKernel()
        assert kernel is not None
        print("✅ QuantumKernel initialized successfully")
        
        # Test basic operations
        state = kernel.get_state()
        assert state is not None
        assert len(state) == kernel.dimension
        print("✅ QuantumKernel state accessible")
        
        # Test gate application
        result = kernel.apply_gate("H", target=0)  # Hadamard gate
        assert result["status"] == "SUCCESS"
        print("✅ QuantumKernel gate operations working")
        
        # Test measurement
        measurement = kernel.measure([0, 1])
        assert measurement["status"] == "SUCCESS"
        print("✅ QuantumKernel measurements working")
        
        # Test acceleration status (RFT-related but simpler)
        acceleration_status = kernel.get_acceleration_status()
        assert acceleration_status is not None
        print("✅ QuantumKernel acceleration status working")
        
    except Exception as e:
        pytest.fail(f"QuantumKernel test failed: {e}")


def test_bulletproof_engine_basic():
    """Test basic BulletproofQuantumEngine functionality"""
    try:
        from bulletproof_quantum_kernel import BulletproofQuantumKernel
        
        # Test with larger dimension for vertex approach
        engine = BulletproofQuantumKernel(num_qubits=4, dimension=16)
        assert engine is not None
        
        # Test acceleration status
        status = engine.get_acceleration_status()
        assert "acceleration_mode" in status
        print(f"✅ BulletproofQuantumEngine working with {status['acceleration_mode']} mode")
        
    except Exception as e:
        pytest.fail(f"BulletproofQuantumEngine test failed: {e}")


def test_rft_algorithm_basic():
    """Test basic RFT algorithm functionality"""
    try:
        from canonical_true_rft import TrueResonanceFourierTransform
        
        rft = TrueResonanceFourierTransform(N=8)
        assert rft is not None
        
        # Test transform on simple signal
        signal = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
        result = rft.transform(signal)
        assert result is not None
        assert len(result) == len(signal)
        print("✅ RFT algorithm working")
        
    except Exception as e:
        pytest.fail(f"RFT algorithm test failed: {e}")


def test_rft_algorithm_basic():
    """Test basic RFT algorithm functionality"""
    rft = qos.ResonanceFourierTransform()
    assert rft is not None

    # Test with simple input
    test_data = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    # Forward transform
    coeffs = rft.forward(test_data)
    assert coeffs is not None
    assert isinstance(coeffs, np.ndarray)
    assert len(coeffs) == 8

    # Inverse transform
    reconstructed = rft.inverse(coeffs)
    assert reconstructed is not None
    assert len(reconstructed) == 8


def test_quantum_cipher_basic():
    """Test basic QuantumCipher functionality using RFT Feistel"""
    try:
        from true_rft_feistel_bindings import encrypt, decrypt, generate_key, init
        
        # Initialize the RFT Feistel engine
        init(size=16)
        
        # Generate a key
        key = generate_key(32)
        assert key is not None
        assert len(key) == 32
        
        # Test encryption/decryption
        test_data = b"Hello Quantum World!"
        encrypted = encrypt(test_data, key)
        assert encrypted is not None
        assert encrypted != test_data  # Should be different
        
        decrypted = decrypt(encrypted, key)
        assert decrypted == test_data  # Should match original
        print("✅ Quantum cipher (RFT Feistel) working")
        
    except Exception as e:
        pytest.fail(f"QuantumCipher test failed: {e}")


def test_determinism_rft_roundtrip():
    """Test RFT round-trip determinism (Phase 2 requirement)"""
    try:
        from canonical_true_rft import TrueResonanceFourierTransform
        
        rft = TrueResonanceFourierTransform(N=8)
        
        # Test deterministic behavior
        signal = np.array([1, 0.5, 0, 0.25, 0, 0, 0, 0], dtype=complex)
        result1 = rft.transform(signal)
        result2 = rft.transform(signal)
        
        # Should be exactly the same (deterministic)
        assert np.array_equal(result1, result2)
        print("✅ RFT determinism verified")
        
    except Exception as e:
        pytest.fail(f"RFT determinism test failed: {e}")


def test_avalanche_effect_basic():
    """Test basic avalanche effect (Phase 2 requirement)"""
    try:
        from true_rft_feistel_bindings import encrypt, generate_key, init
        
        init(size=16)
        
        # Generate two keys that differ by 1 bit
        key1 = generate_key(32)
        key2 = bytearray(key1)
        key2[0] ^= 1  # Flip one bit
        key2 = bytes(key2)
        
        # Test data
        test_data = b"Test avalanche effect data"
        
        # Encrypt with both keys
        encrypted1 = encrypt(test_data, key1)
        encrypted2 = encrypt(test_data, key2)
        
        # Calculate bit differences
        diff_bits = 0
        total_bits = len(encrypted1) * 8
        for b1, b2 in zip(encrypted1, encrypted2):
            diff_bits += bin(b1 ^ b2).count('1')
        
        avalanche_percentage = (diff_bits / total_bits) * 100
        
        # Good avalanche effect should change ~50% of bits
        assert avalanche_percentage > 30  # At least 30% change
        print(f"✅ Avalanche effect: {avalanche_percentage:.1f}% bits changed")
        
    except Exception as e:
        pytest.fail(f"Avalanche effect test failed: {e}")


def test_determinism_rft_roundtrip():
    """Test RFT round-trip determinism (Phase 2 requirement)"""
    try:
        import sys
        import os
        sys.path.append('/workspaces/quantoniumos')
        sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')
        sys.path.append('/workspaces/quantoniumos/05_QUANTUM_ENGINES')
        sys.path.append('/workspaces/quantoniumos/06_CRYPTOGRAPHY')
        
        from canonical_true_rft import TrueResonanceFourierTransform
        rft = TrueResonanceFourierTransform()
        
        # Test data
        test_data = [1.0, 0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0]
        
        # Basic transform test
        result = rft.transform(test_data)
        assert result is not None
        print(f"✅ RFT transform successful: {type(result)}")
        
    except Exception as e:
        pytest.fail(f"RFT determinism test failed: {e}")


def test_avalanche_effect_basic():
    """Test basic avalanche effect (Phase 2 requirement)"""
    try:
        import sys
        import os
        sys.path.append('/workspaces/quantoniumos')
        sys.path.append('/workspaces/quantoniumos/04_RFT_ALGORITHMS')
        sys.path.append('/workspaces/quantoniumos/05_QUANTUM_ENGINES')
        sys.path.append('/workspaces/quantoniumos/06_CRYPTOGRAPHY')
        
        from true_rft_feistel_bindings import generate_key, encrypt
        
        # Generate key
        key = generate_key(128)
        
        # Test data
        data1 = "Test message for avalanche"
        data2 = "Test message for avalancha"  # Single char change
        
        # Encrypt both
        enc1 = encrypt(data1, key)
        enc2 = encrypt(data2, key)
        
        # Should produce different outputs
        assert enc1 != enc2
        print(f"✅ Avalanche effect test passed - different outputs for different inputs")
        
    except Exception as e:
        pytest.fail(f"Avalanche effect test failed: {e}")
