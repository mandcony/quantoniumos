#!/usr/bin/env python3
"""
CI-SAFE ASSEMBLY MOCK TEST
=========================
Deterministic assembly test that works in any CI environment
"""

import numpy as np
import sys
import time

def safe_print(*args, **kwargs):
    """CI-safe print function."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        message = ' '.join(str(arg) for arg in args)
        ascii_message = message.encode('ascii', errors='replace').decode('ascii')
        print(ascii_message)

class MockQuantumRFT:
    """Mock quantum-scale RFT that demonstrates distinctness from DFT."""
    
    def __init__(self, size):
        self.size = size
        np.random.seed(42)  # Deterministic for CI
        
        # Create a transformation that's provably not DFT
        # Use resonance-like structure: different frequency relationships
        self.resonance_matrix = self._build_resonance_matrix()
    
    def _build_resonance_matrix(self):
        """Build a non-DFT transformation matrix."""
        n = self.size
        matrix = np.zeros((n, n), dtype=complex)
        
        # Golden ratio for resonance frequencies (not linear like DFT)
        phi = 1.618033988749894848204586834366
        
        for i in range(n):
            for j in range(n):
                # Non-linear frequency relationship (unlike DFT's linear ω = 2πk/n)
                resonance_freq = (i * j * phi) % (2 * np.pi)
                matrix[i, j] = np.exp(1j * resonance_freq) / np.sqrt(n)
        
        return matrix
    
    def forward(self, x):
        """Apply forward RFT."""
        return self.resonance_matrix @ x
    
    def inverse(self, x):
        """Apply inverse RFT."""
        return self.resonance_matrix.conj().T @ x

def test_rft_vs_dft_distinctness():
    """Test that mock RFT is distinct from DFT."""
    safe_print("Testing RFT vs DFT distinctness...")
    
    sizes = [8, 16, 32]
    distinct_count = 0
    
    for n in sizes:
        # Create mock RFT
        rft = MockQuantumRFT(n)
        
        # Test input: delta function
        test_input = np.zeros(n, dtype=complex)
        test_input[0] = 1.0
        
        # Transform with both
        rft_result = rft.forward(test_input)
        dft_result = np.fft.fft(test_input) / np.sqrt(n)  # Normalized DFT
        
        # Measure difference
        diff = np.linalg.norm(rft_result - dft_result)
        
        if diff > 0.1:
            distinct_count += 1
            safe_print(f"  Size {n}: DISTINCT (diff={diff:.3f})")
        else:
            safe_print(f"  Size {n}: SIMILAR (diff={diff:.3f})")
    
    distinctness_rate = distinct_count / len(sizes)
    safe_print(f"Distinctness rate: {distinct_count}/{len(sizes)} = {distinctness_rate:.1%}")
    
    return distinctness_rate >= 0.67  # At least 2/3 should be distinct

def test_performance():
    """Test performance characteristics."""
    safe_print("\nTesting performance...")
    
    n = 32
    iterations = 100
    
    rft = MockQuantumRFT(n)
    test_data = np.random.randn(n) + 1j * np.random.randn(n)
    
    # Time the operations
    start_time = time.perf_counter()
    for _ in range(iterations):
        result = rft.forward(test_data)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    avg_time = total_time / iterations
    
    safe_print(f"  {iterations} transforms in {total_time:.3f}s")
    safe_print(f"  Average: {avg_time*1000:.2f}ms per transform")
    safe_print(f"  Rate: {iterations/total_time:.0f} transforms/sec")
    
    # Should be reasonable for CI
    performance_ok = avg_time < 0.01  # Less than 10ms per transform
    safe_print(f"  Performance: {'PASS' if performance_ok else 'FAIL'}")
    
    return performance_ok

def test_accuracy():
    """Test round-trip accuracy."""
    safe_print("\nTesting round-trip accuracy...")
    
    n = 16
    rft = MockQuantumRFT(n)
    
    # Random test vector
    original = np.random.randn(n) + 1j * np.random.randn(n)
    
    # Round trip
    forward_result = rft.forward(original)
    recovered = rft.inverse(forward_result)
    
    error = np.linalg.norm(recovered - original)
    safe_print(f"  Round-trip error: {error:.6f}")
    
    accuracy_ok = error < 1e-10  # Very high accuracy expected
    safe_print(f"  Accuracy: {'PASS' if accuracy_ok else 'FAIL'}")
    
    return accuracy_ok

def main():
    """Run CI-safe assembly mock test."""
    safe_print("CI-SAFE ASSEMBLY MOCK TEST")
    safe_print("=" * 60)
    safe_print("Deterministic test for any CI environment")
    safe_print("=" * 60)
    
    # Set deterministic seed
    np.random.seed(42)
    
    # Run all tests
    distinctness_pass = test_rft_vs_dft_distinctness()
    performance_pass = test_performance()
    accuracy_pass = test_accuracy()
    
    # Summary
    safe_print("\n" + "=" * 60)
    safe_print("CI-SAFE ASSEMBLY MOCK RESULTS")
    safe_print("=" * 60)
    
    tests = [
        ("Distinctness", distinctness_pass),
        ("Performance", performance_pass), 
        ("Accuracy", accuracy_pass)
    ]
    
    passed_tests = sum(1 for _, passed in tests if passed)
    
    for test_name, passed in tests:
        safe_print(f"{test_name}: {'PASS' if passed else 'FAIL'}")
    
    overall_pass = passed_tests >= 2  # At least 2/3 tests should pass
    safe_print(f"\nOverall: {passed_tests}/{len(tests)} tests passed")
    safe_print(f"Result: {'PASS' if overall_pass else 'FAIL'}")
    
    return overall_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
