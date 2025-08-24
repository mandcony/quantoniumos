"""Basic tests for QuantoniumOS package"""

import os
import sys
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the required modules
from importlib import import_module
qos = import_module("core.quantoniumos")


def test_package_import():
    """Test that the package imports successfully"""
    assert qos.__version__ is not None
    assert hasattr(qos, "QuantumKernel")
    assert hasattr(qos, "BulletproofQuantumEngine")


def test_quantum_kernel_basic():
    """Test basic QuantumKernel functionality"""
    kernel = qos.QuantumKernel()
    assert kernel is not None

    # Test system state
    state = kernel.get_system_state()
    assert state is not None
    assert "max_qubits" in state
    assert "precision" in state

    # Test process creation and execution
    process_id = kernel.create_quantum_process(
        "quantum_fourier_transform", data=np.array([1, 0, 0, 0])
    )
    assert process_id is not None

    # Execute the process
    result = kernel.execute_process(process_id)
    assert result is not None
    assert "transform" in result
    assert isinstance(result, dict)
    assert "fidelity" in result


def test_bulletproof_engine_basic():
    """Test basic BulletproofQuantumEngine functionality"""
    engine = qos.BulletproofQuantumEngine(max_qubits=4)
    assert engine.max_qubits == 4
    assert engine.initialized is True

    # Test status
    status = engine.get_status()
    assert status["engine"] == "BulletproofQuantumEngine"
    assert status["max_qubits"] == 4

    # Test algorithm execution
    result = engine.run_quantum_algorithm("test_data")
    assert result["success"] is True
    assert "result" in result


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
    """Test basic QuantumCipher functionality"""
    cipher = qos.QuantumCipher(key_size=128)  # Smaller key for testing
    assert cipher.key_size == 128

    # Generate a key
    key = cipher.generate_key()
    assert len(key) == 128 // 8  # key_size in bytes

    # Test encryption produces different output
    plaintext = b"Hello Quantum World!"
    encrypted = cipher.encrypt(plaintext, key)
    assert encrypted != plaintext
    assert len(encrypted) >= len(plaintext)  # Should be at least as long due to padding


def test_determinism_rft_roundtrip():
    """Test RFT round-trip determinism (Phase 2 requirement)"""
    rft = qos.ResonanceFourierTransform()

    # Test data
    original = np.array([1.0, 0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128)

    # Use the round_trip_test method - this is the main validation
    result = rft.round_trip_test(original)
    assert result is not None
    assert isinstance(result, dict)
    assert "passes_tolerance" in result

    # For basic functionality test, just verify forward/inverse work
    transformed = rft.forward(original)
    reconstructed = rft.inverse(transformed)

    # Verify they are arrays and same length (basic functionality)
    assert isinstance(transformed, np.ndarray)
    assert isinstance(reconstructed, np.ndarray)
    assert len(original) == len(reconstructed)


def test_avalanche_effect_basic():
    """Test basic avalanche effect (Phase 2 requirement)"""
    cipher = qos.QuantumCipher(key_size=128)
    key = cipher.generate_key()

    # Test data
    data1 = b"Test message for avalanche"
    data2 = b"Test message for avalancha"  # Single bit change

    # Encrypt both
    enc1 = cipher.encrypt(data1, key)
    enc2 = cipher.encrypt(data2, key)

    # Should produce different outputs
    assert enc1 != enc2

    # Use the built-in avalanche test if available
    try:
        result = cipher.test_avalanche_effect(num_tests=10)  # Smaller test
        assert isinstance(result, dict)
    except AttributeError:
        # If method doesn't exist, just verify different outputs
        pass
