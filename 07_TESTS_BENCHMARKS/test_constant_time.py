"""
Constant-time and side-channel resistance tests for QuantoniumOS cryptographic primitives.

This module implements micro-benchmarks to assert <=5ns variance across
key-dependent branches, ensuring constant-time execution.
"""

import secrets
import statistics
import time
from typing import Any, Callable, List

import pytest

# Maximum allowed timing variance in nanoseconds for constant-time operations
MAX_TIMING_VARIANCE_NS = 5.0


def measure_timing_variance(
    func: Callable, test_inputs: List[Any], iterations: int = 1000
) -> float:
    """
    Measure timing variance across different inputs.

    Args:
        func: Function to test for constant-time behavior
        test_inputs: List of different inputs to test
        iterations: Number of iterations per input

    Returns:
        Timing variance in nanoseconds
    """
    all_timings = []

    # Warm up to minimize JIT/cache effects
    for _ in range(100):
        func(test_inputs[0])

    # Measure timing for each input
    for input_data in test_inputs:
        input_timings = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            func(input_data)
            end = time.perf_counter_ns()
            input_timings.append(end - start)
        all_timings.extend(input_timings)

    return statistics.variance(all_timings)


def generate_crypto_test_inputs(size: int = 32) -> List[bytes]:
    """Generate cryptographic test inputs with different bit patterns."""
    return [
        b"\x00" * size,  # All zeros
        b"\xff" * size,  # All ones
        secrets.token_bytes(size),  # Random
        b"\xaa" * size,  # Alternating 10101010
        b"\x55" * size,  # Alternating 01010101
        b"\xf0" * size,  # Alternating nibbles 11110000
        b"\x0f" * size,  # Alternating nibbles 00001111
    ]


class ConstantTimeTestSuite:
    """Test suite for constant-time cryptographic operations."""

    def test_constant_time_property(self, func: Callable, test_name: str) -> bool:
        """
        Test that a function has constant-time properties.

        Args:
            func: Function to test
            test_name: Name for logging

        Returns:
            True if timing variance is within acceptable bounds
        """
        test_inputs = generate_crypto_test_inputs()
        variance = measure_timing_variance(func, test_inputs)

        print(
            f"{test_name}: Timing variance = {variance:.2f}ns (max allowed: {MAX_TIMING_VARIANCE_NS}ns)"
        )

        return variance <= MAX_TIMING_VARIANCE_NS

    def test_key_independent_timing(self, encryption_func: Callable) -> bool:
        """
        Test that encryption timing is independent of key values.

        Args:
            encryption_func: Encryption function taking (plaintext, key)

        Returns:
            True if key-independent timing verified
        """
        plaintext = b"\x00" * 32  # Fixed plaintext
        keys = generate_crypto_test_inputs(32)  # Different key patterns

        timings = []
        for key in keys:
            # Warm up
            for _ in range(50):
                encryption_func(plaintext, key)

            # Measure
            key_timings = []
            for _ in range(500):
                start = time.perf_counter_ns()
                encryption_func(plaintext, key)
                end = time.perf_counter_ns()
                key_timings.append(end - start)

            timings.append(statistics.mean(key_timings))

        variance = statistics.variance(timings)
        print(f"Key-independent timing: variance = {variance:.2f}ns")

        return variance <= MAX_TIMING_VARIANCE_NS

    def test_data_independent_timing(self, encryption_func: Callable) -> bool:
        """
        Test that encryption timing is independent of plaintext values.

        Args:
            encryption_func: Encryption function taking (plaintext, key)

        Returns:
            True if data-independent timing verified
        """
        key = secrets.token_bytes(32)  # Fixed key
        plaintexts = generate_crypto_test_inputs(32)  # Different plaintext patterns

        timings = []
        for plaintext in plaintexts:
            # Warm up
            for _ in range(50):
                encryption_func(plaintext, key)

            # Measure
            plaintext_timings = []
            for _ in range(500):
                start = time.perf_counter_ns()
                encryption_func(plaintext, key)
                end = time.perf_counter_ns()
                plaintext_timings.append(end - start)

            timings.append(statistics.mean(plaintext_timings))

        variance = statistics.variance(timings)
        print(f"Data-independent timing: variance = {variance:.2f}ns")

        return variance <= MAX_TIMING_VARIANCE_NS


def test_geometric_waveform_hash_constant_time():
    """Test constant-time properties of geometric waveform hash."""
    try:
        from core.encryption.geometric_waveform_hash import \
            GeometricWaveformHash

        def hash_func(data: bytes) -> str:
            hasher = GeometricWaveformHash()
            if hasattr(hasher, "waveform"):
                # Convert bytes to waveform
                hasher.waveform = [float(b) / 255.0 for b in data[:100]]
                if len(hasher.waveform) < 100:
                    hasher.waveform.extend([0.0] * (100 - len(hasher.waveform)))
                hasher.calculate_geometric_properties()
                return hasher.generate_hash()
            return ""

        suite = ConstantTimeTestSuite()
        assert suite.test_constant_time_property(hash_func, "GeometricWaveformHash")

    except ImportError:
        pytest.skip("GeometricWaveformHash not available")


def test_symbolic_xor_constant_time():
    """Test constant-time properties of symbolic XOR encryption."""
    try:
        from quantoniumos.core.encryption import encrypt_symbolic

        def encrypt_func(plaintext: bytes, key: bytes) -> str:
            pt_hex = plaintext[:16].hex()  # Take first 16 bytes
            key_hex = key[:16].hex()
            result = encrypt_symbolic(pt_hex, key_hex)
            return result.get("ciphertext", "")

        suite = ConstantTimeTestSuite()
        assert suite.test_key_independent_timing(encrypt_func)
        assert suite.test_data_independent_timing(encrypt_func)

    except ImportError:
        pytest.skip("encrypt_symbolic not available")


if __name__ == "__main__":
    print("Running constant-time verification tests...")

    # Create test suite
    suite = ConstantTimeTestSuite()

    # Test example function (replace with actual crypto functions)
    def example_crypto_func(data: bytes) -> bytes:
        """Example function for testing - replace with real crypto implementations."""
        return data  # Placeholder

    # Run tests
    result = suite.test_constant_time_property(example_crypto_func, "ExampleCrypto")
    print(f"Example test passed: {result}")

    # Run pytest tests
    print("\nRunning pytest suite...")
    pytest.main([__file__, "-v"])
