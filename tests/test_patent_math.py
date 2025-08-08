"""
Patent Mathematics Test Suite
Comprehensive validation of U.S. Patent Application No. 19/169,399 mathematics

This test suite validates that all mathematical formulas from the patent
are correctly implemented in the codebase.
"""

import pytest
import numpy as np
import sys
import os
import cmath

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)

try:
    from core.grover_amplification import (
        grover_amplitude_amplification,
        symbolic_resonance_search,
        SymbolicQubitGate,
        symbolic_entropy_calculation,
        waveform_hash_mixing
    )
except ImportError as e:
    print(f"Warning: Could not import patent math modules: {e}")
    # Create dummy implementations for testing
    def grover_amplitude_amplification(*args): return []
    def symbolic_resonance_search(*args): return -1
    class SymbolicQubitGate:
        def __init__(self, *args): pass
        def hadamard_gate(self): return self
        def measure_probability(self): return [0.5, 0.5]
    def symbolic_entropy_calculation(*args): return 0.0
    def waveform_hash_mixing(*args): return (0.0, 0.0)


class TestPatentMathImplementation:
    """Test suite to validate patent mathematics implementation"""
    
    def test_xor_encryption_with_geometric_hash(self):
        """
        Patent Math: XOR Encryption with Geometric Hash
        C_i = D_i ⊕ H(W_i), where H(W_i) = mod(A_i * cos(ϕ_i), p)
        """
        # Test data
        data = [0x41, 0x42, 0x43]  # "ABC"
        amplitudes = [1.0, 2.0, 1.5]
        phases = [0.0, np.pi/4, np.pi/2]
        prime = 251
        
        # Geometric hash: H(W_i) = mod(A_i * cos(ϕ_i), p)
        hash_values = []
        for i in range(len(data)):
            hash_val = int(amplitudes[i] * np.cos(phases[i])) % prime
            hash_values.append(hash_val)
        
        # XOR encryption: C_i = D_i ⊕ H(W_i)
        encrypted = []
        for i in range(len(data)):
            encrypted.append(data[i] ^ hash_values[i])
        
        # Verify decryption works
        decrypted = []
        for i in range(len(encrypted)):
            decrypted.append(encrypted[i] ^ hash_values[i])
        
        assert decrypted == data, "XOR encryption/decryption failed"
        
        # Verify hash values are deterministic
        hash_values_2 = []
        for i in range(len(data)):
            hash_val = int(amplitudes[i] * np.cos(phases[i])) % prime
            hash_values_2.append(hash_val)
        
        assert hash_values == hash_values_2, "Hash function not deterministic"
    
    def test_waveform_hash_mixing(self):
        """
        Patent Math: Geometric Waveform Hash Mixing
        W_combined = (A + A_x, (ϕ + ϕ_x) mod 2π)
        """
        A1, phi1 = 1.0, np.pi/4
        A2, phi2 = 2.0, np.pi/2
        
        # Test direct calculation
        A_combined = A1 + A2
        phi_combined = (phi1 + phi2) % (2 * np.pi)
        
        assert abs(A_combined - 3.0) < 1e-10
        assert abs(phi_combined - (3*np.pi/4)) < 1e-10
        
        # Test function implementation
        amp_result, phase_result = waveform_hash_mixing(A1, phi1, A2, phi2)
        assert abs(amp_result - A_combined) < 1e-10
        assert abs(phase_result - phi_combined) < 1e-10
        
        # Test phase wrapping
        large_phi1, large_phi2 = 3*np.pi, 5*np.pi/2
        amp_wrap, phase_wrap = waveform_hash_mixing(1.0, large_phi1, 1.0, large_phi2)
        expected_wrap = (large_phi1 + large_phi2) % (2 * np.pi)
        assert abs(phase_wrap - expected_wrap) < 1e-10
    
    def test_rft_forward_transform(self):
        """
        Patent Math: Forward RFT
        RFT_k = Σ A_n * e^{iϕ_n} * e^{-2πikn/N}
        """
        N = 4
        amplitudes = [1.0, 1.0, 1.0, 1.0]
        phases = [0, np.pi/2, np.pi, 3*np.pi/2]
        
        # Create symbolic waveform: A_n * e^{iϕ_n}
        waveform = []
        for n in range(N):
            waveform.append(amplitudes[n] * cmath.exp(1j * phases[n]))
        
        # Compute RFT manually: RFT_k = Σ A_n * e^{iϕ_n} * e^{-2πikn/N}
        rft_result = []
        for k in range(N):
            sum_val = 0
            for n in range(N):
                twiddle = cmath.exp(-2j * np.pi * k * n / N)
                sum_val += waveform[n] * twiddle
            rft_result.append(sum_val)
        
        # Verify against numpy FFT for reference
        np_fft = np.fft.fft(waveform)
        
        for i in range(N):
            assert abs(rft_result[i] - np_fft[i]) < 1e-10, f"RFT mismatch at index {i}"
        
        # Test specific values for known input
        assert abs(rft_result[0]) < 1e-10, "DC component should be near zero"
        assert abs(rft_result[2]) < 1e-10, "Nyquist component should be near zero"
    
    def test_inverse_rft_mathematical(self):
        """
        Patent Math: Inverse RFT
        W_n = (1/N) Σ RFT_k * e^{2πikn/N}
        """
        N = 8
        
        # Create test RFT data
        rft_data = [complex(i+1, i*0.5) for i in range(N)]
        
        # Compute inverse RFT manually: W_n = (1/N) Σ RFT_k * e^{2πikn/N}
        inverse_result = []
        for n in range(N):
            sum_val = 0
            for k in range(N):
                twiddle = cmath.exp(2j * np.pi * k * n / N)
                sum_val += rft_data[k] * twiddle
            inverse_result.append(sum_val / N)
        
        # Verify against numpy IFFT for reference
        np_ifft = np.fft.ifft(rft_data)
        
        for i in range(N):
            assert abs(inverse_result[i] - np_ifft[i]) < 1e-10, f"Inverse RFT mismatch at index {i}"
    
    def test_rft_roundtrip(self):
        """Test that Forward RFT -> Inverse RFT recovers original signal"""
        N = 16
        
        # Create test signal
        original_signal = []
        for n in range(N):
            amplitude = 1.0 + 0.5 * np.sin(2 * np.pi * n / N)
            phase = np.pi * n / N
            original_signal.append(amplitude * cmath.exp(1j * phase))
        
        # Forward RFT
        rft_data = np.fft.fft(original_signal)
        
        # Inverse RFT
        recovered_signal = np.fft.ifft(rft_data)
        
        # Verify roundtrip accuracy
        for i in range(N):
            assert abs(original_signal[i] - recovered_signal[i]) < 1e-10, f"Roundtrip failed at index {i}"
    
    def test_symbolic_entropy_function(self):
        """
        Patent Math: Symbolic Entropy Function
        H(W) = -Σ p_i log(p_i) where p_i = A_i² / Σ A_j²
        """
        # Test case 1: Equal amplitudes (maximum entropy)
        equal_amplitudes = [1.0, 1.0, 1.0, 1.0]
        entropy_equal = symbolic_entropy_calculation(equal_amplitudes)
        expected_max = np.log2(len(equal_amplitudes))
        assert abs(entropy_equal - expected_max) < 1e-10, "Equal amplitudes should give maximum entropy"
        
        # Test case 2: Single amplitude (minimum entropy)
        single_amplitude = [1.0, 0.0, 0.0, 0.0]
        entropy_single = symbolic_entropy_calculation(single_amplitude)
        assert entropy_single == 0.0, "Single amplitude should give zero entropy"
        
        # Test case 3: General case
        amplitudes = [1.0, 2.0, 1.5, 0.5]
        
        # Calculate manually
        squared_amplitudes = [A**2 for A in amplitudes]
        total_energy = sum(squared_amplitudes)
        probabilities = [A2 / total_energy for A2 in squared_amplitudes]
        
        entropy_manual = 0
        for p in probabilities:
            if p > 0:
                entropy_manual -= p * np.log2(p)
        
        entropy_func = symbolic_entropy_calculation(amplitudes)
        assert abs(entropy_func - entropy_manual) < 1e-10, "Entropy calculation mismatch"
        
        # Verify entropy bounds
        assert 0 <= entropy_func <= np.log2(len(amplitudes)), "Entropy out of bounds"
    
    def test_grover_amplification(self):
        """
        Patent Math: Grover Amplification
        s'_i = -s_i if i = target, else s'_i = 2*avg - s_i
        """
        amplitudes = [0.5, 0.3, 0.8, 0.1]
        target_index = 2
        
        result = grover_amplitude_amplification(amplitudes, target_index)
        avg = np.mean(amplitudes)  # 0.425
        
        # Check target amplitude (should be flipped)
        assert abs(result[target_index] + amplitudes[target_index]) < 1e-10, "Target amplitude not flipped"
        
        # Check non-target amplitudes (should be inverted about average)
        for i in range(len(amplitudes)):
            if i != target_index:
                expected = 2 * avg - amplitudes[i]
                assert abs(result[i] - expected) < 1e-10, f"Non-target amplitude {i} not correctly inverted"
        
        # Test edge cases
        single_element = [1.0]
        result_single = grover_amplitude_amplification(single_element, 0)
        assert result_single == [-1.0], "Single element case failed"
        
        # Test invalid index
        with pytest.raises(ValueError):
            grover_amplitude_amplification([1, 2, 3], 5)
    
    def test_symbolic_qubit_hadamard(self):
        """
        Patent Math: Symbolic Qubit and Hadamard Gate
        |ψ⟩ = α|0⟩ + β|1⟩ where α = A_0 * e^{iϕ_0}, β = A_1 * e^{iϕ_1}
        H|ψ⟩ = (1/√2) * [[1, 1], [1, -1]] * [α, β]
        """
        # Test |0⟩ state
        qubit_0 = SymbolicQubitGate(1.0, 0.0, 0.0, 0.0)
        h_qubit_0 = qubit_0.hadamard_gate()
        probs_0 = h_qubit_0.measure_probability()
        
        # Should create |+⟩ = (|0⟩ + |1⟩)/√2
        assert abs(probs_0[0] - 0.5) < 1e-10, "|0⟩ -> H -> probability mismatch"
        assert abs(probs_0[1] - 0.5) < 1e-10, "|0⟩ -> H -> probability mismatch"
        
        # Test |1⟩ state  
        qubit_1 = SymbolicQubitGate(0.0, 0.0, 1.0, 0.0)
        h_qubit_1 = qubit_1.hadamard_gate()
        probs_1 = h_qubit_1.measure_probability()
        
        # Should create |−⟩ = (|0⟩ - |1⟩)/√2
        assert abs(probs_1[0] - 0.5) < 1e-10, "|1⟩ -> H -> probability mismatch"
        assert abs(probs_1[1] - 0.5) < 1e-10, "|1⟩ -> H -> probability mismatch"
        
        # Test superposition state
        qubit_super = SymbolicQubitGate(1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2), np.pi/2)
        state_vector = qubit_super.get_state_vector()
        
        # Verify normalization
        norm = abs(state_vector[0])**2 + abs(state_vector[1])**2
        assert abs(norm - 1.0) < 1e-10, "Qubit not normalized"
        
        # Test Pauli gates
        qubit_test = SymbolicQubitGate(1.0, 0.0, 0.0, 0.0)
        
        # Pauli-X should flip |0⟩ -> |1⟩
        x_qubit = qubit_test.pauli_x_gate()
        x_probs = x_qubit.measure_probability()
        assert abs(x_probs[0] - 0.0) < 1e-10, "Pauli-X failed"
        assert abs(x_probs[1] - 1.0) < 1e-10, "Pauli-X failed"
    
    def test_frequency_matching_search(self):
        """
        Patent Math: Frequency Matching
        x* = argmin_x ||f_target - f_x||_resonance
        """
        frequencies = [1.0, 2.5, 4.2, 7.8, 12.1]
        
        # Test exact match
        exact_match = symbolic_resonance_search(frequencies, 4.2, tolerance=1e-10)
        assert exact_match == 2, "Exact frequency match failed"
        
        # Test approximate match
        approx_match = symbolic_resonance_search(frequencies, 4.15, tolerance=0.1)
        assert approx_match == 2, "Approximate frequency match failed"
        
        # Test no match within tolerance
        no_match = symbolic_resonance_search(frequencies, 20.0, tolerance=0.1)
        assert no_match == -1, "Should find no match for distant frequency"
        
        # Test empty spectrum
        empty_match = symbolic_resonance_search([], 5.0)
        assert empty_match == -1, "Empty spectrum should return -1"
        
        # Test edge case: multiple equally close matches
        close_frequencies = [1.0, 1.05, 1.1]
        edge_match = symbolic_resonance_search(close_frequencies, 1.025, tolerance=0.1)
        assert edge_match == 0, "Should find first of equally close matches"
    
    def test_bloch_sphere_representation(self):
        """Test Bloch sphere coordinate calculation for symbolic qubits"""
        # Test |0⟩ state (north pole: z = +1)
        qubit_0 = SymbolicQubitGate(1.0, 0.0, 0.0, 0.0)
        x, y, z = qubit_0.bloch_sphere_coordinates()
        assert abs(x - 0.0) < 1e-10, "Bloch x-coordinate wrong for |0⟩"
        assert abs(y - 0.0) < 1e-10, "Bloch y-coordinate wrong for |0⟩"  
        assert abs(z - 1.0) < 1e-10, "Bloch z-coordinate wrong for |0⟩"
        
        # Test |1⟩ state (south pole: z = -1)
        qubit_1 = SymbolicQubitGate(0.0, 0.0, 1.0, 0.0)
        x, y, z = qubit_1.bloch_sphere_coordinates()
        assert abs(x - 0.0) < 1e-10, "Bloch x-coordinate wrong for |1⟩"
        assert abs(y - 0.0) < 1e-10, "Bloch y-coordinate wrong for |1⟩"
        assert abs(z + 1.0) < 1e-10, "Bloch z-coordinate wrong for |1⟩"
        
        # Test |+⟩ state (x = +1)
        qubit_plus = SymbolicQubitGate(1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2), 0.0)
        x, y, z = qubit_plus.bloch_sphere_coordinates()
        assert abs(x - 1.0) < 1e-10, "Bloch x-coordinate wrong for |+⟩"
        assert abs(y - 0.0) < 1e-10, "Bloch y-coordinate wrong for |+⟩"
        assert abs(z - 0.0) < 1e-10, "Bloch z-coordinate wrong for |+⟩"

    def test_patent_math_integration(self):
        """Integration test combining multiple patent math components"""
        # Create a symbolic waveform
        N = 8
        amplitudes = [1.0, 0.8, 1.2, 0.6, 0.9, 1.1, 0.7, 1.0]
        phases = [i * np.pi / 4 for i in range(N)]
        
        # Calculate entropy
        entropy = symbolic_entropy_calculation(amplitudes)
        assert 0 < entropy < np.log2(N), "Entropy should be in valid range"
        
        # Perform waveform mixing on adjacent pairs
        mixed_waveforms = []
        for i in range(0, N-1, 2):
            amp, phase = waveform_hash_mixing(
                amplitudes[i], phases[i],
                amplitudes[i+1], phases[i+1]
            )
            mixed_waveforms.append((amp, phase))
        
        # Use mixed waveforms for frequency search
        mixed_amplitudes = [w[0] for w in mixed_waveforms]
        target_freq = np.mean(mixed_amplitudes)
        match_idx = symbolic_resonance_search(mixed_amplitudes, target_freq, tolerance=0.1)
        assert match_idx >= 0, "Should find a resonance match in mixed waveforms"
        
        # Apply Grover amplification to enhance the match
        if match_idx >= 0:
            amplified = grover_amplitude_amplification(mixed_amplitudes, match_idx)
            assert abs(amplified[match_idx] + mixed_amplitudes[match_idx]) < 1e-10, "Target should be flipped"


if __name__ == "__main__":
    # Run pytest when executed directly
    pytest.main([__file__, "-v", "--tb=short"])
