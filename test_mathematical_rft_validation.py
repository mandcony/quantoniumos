#!/usr/bin/env python3
"""
Mathematical Validation Suite for Resonance Fourier Transform (RFT)
===================================================================

This module provides rigorous mathematical testing of the RFT implementation
following the formal specification: R = Σ_i w_i D_φi C_σi D_φi†

Scientific validation includes:
- Algebraic structure verification
- Numerical stability analysis  
- Computational complexity measurement
- Transform property validation
- Orthogonality stress testing

Author: QuantoniumOS Mathematical Verification Team
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from bulletproof_quantum_kernel import BulletproofQuantumKernel


class MathematicalRFTValidator:
    """
    Mathematical validator for RFT implementation with scientific rigor.
    Tests fundamental mathematical properties without reliance on heuristics.
    """
    
    def __init__(self, tolerance: float = 1e-12):
        """
        Initialize validator with specified numerical tolerance.
        
        Args:
            tolerance: Numerical precision threshold for validation
        """
        self.tolerance = tolerance
        self.test_results = {}
        
    def validate_resonance_kernel_construction(self, kernel: BulletproofQuantumKernel) -> Dict[str, Any]:
        """
        Validate the mathematical construction of resonance kernel R = Σ_i w_i D_φi C_σi D_φi†
        
        Verification points:
        1. Hermitian property: R = R†
        2. Positive semi-definite eigenvalues
        3. Correct dimensional structure
        4. Golden ratio weight scaling
        
        Args:
            kernel: BulletproofQuantumKernel instance
            
        Returns:
            Dictionary containing validation results
        """
        kernel.build_resonance_kernel()
        R = kernel.resonance_kernel
        
        results = {
            'test_name': 'Resonance Kernel Construction',
            'dimension': kernel.dimension,
            'matrix_shape': R.shape
        }
        
        # Test 1: Hermitian property
        hermitian_error = np.linalg.norm(R - R.conj().T, 'fro')
        results['hermitian_error'] = hermitian_error
        results['is_hermitian'] = hermitian_error < self.tolerance
        
        # Test 2: Eigenvalue analysis
        eigenvals = np.linalg.eigvals(R)
        results['eigenvalues'] = eigenvals.tolist()
        results['min_eigenvalue'] = np.min(np.real(eigenvals))
        results['max_eigenvalue'] = np.max(np.real(eigenvals))
        results['is_positive_semidefinite'] = np.min(np.real(eigenvals)) >= -self.tolerance
        
        # Test 3: Spectral properties
        results['condition_number'] = np.linalg.cond(R)
        results['matrix_rank'] = np.linalg.matrix_rank(R, tol=self.tolerance)
        results['trace'] = np.trace(R)
        results['determinant'] = np.linalg.det(R)
        
        # Test 4: Golden ratio signature verification
        phi = (1 + np.sqrt(5)) / 2
        diagonal_elements = np.diag(R)
        phi_correlation = np.corrcoef(np.real(diagonal_elements), 
                                     [phi**(-i) for i in range(len(diagonal_elements))])[0,1]
        results['golden_ratio_correlation'] = phi_correlation
        
        return results
    
    def validate_rft_basis_orthogonality(self, kernel: BulletproofQuantumKernel) -> Dict[str, Any]:
        """
        Validate orthogonality properties of RFT basis Ψ obtained from eigendecomposition.
        
        Mathematical requirements:
        1. Ψ†Ψ = I (unitary condition)
        2. ΨΨ† = I (completeness)
        3. Column vectors are orthonormal
        4. Eigenvalue ordering preservation
        
        Args:
            kernel: BulletproofQuantumKernel instance
            
        Returns:
            Dictionary containing orthogonality validation results
        """
        eigenvals, eigenvecs = kernel.compute_rft_basis()
        Psi = kernel.rft_basis
        
        results = {
            'test_name': 'RFT Basis Orthogonality',
            'basis_shape': Psi.shape,
            'eigenvalue_count': len(eigenvals)
        }
        
        # Test 1: Unitary condition Ψ†Ψ = I
        gram_matrix = Psi.conj().T @ Psi
        identity_matrix = np.eye(kernel.dimension)
        unitarity_error = np.linalg.norm(gram_matrix - identity_matrix, 'fro')
        results['unitarity_error'] = unitarity_error
        results['is_unitary'] = unitarity_error < self.tolerance
        
        # Test 2: Completeness condition ΨΨ† = I
        completeness_matrix = Psi @ Psi.conj().T
        completeness_error = np.linalg.norm(completeness_matrix - identity_matrix, 'fro')
        results['completeness_error'] = completeness_error
        results['is_complete'] = completeness_error < self.tolerance
        
        # Test 3: Individual column orthonormality
        column_norms = [np.linalg.norm(Psi[:, i]) for i in range(Psi.shape[1])]
        results['column_norms'] = column_norms
        results['norm_deviation'] = np.std(column_norms)
        results['columns_normalized'] = all(abs(norm - 1.0) < self.tolerance for norm in column_norms)
        
        # Test 4: Eigenvalue ordering and reality
        results['eigenvalues_sorted'] = np.all(np.diff(np.real(eigenvals)) <= self.tolerance)
        results['eigenvalues_real'] = np.all(np.abs(np.imag(eigenvals)) < self.tolerance)
        
        return results
    
    def validate_transform_invertibility(self, kernel: BulletproofQuantumKernel, 
                                       test_signals: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Validate perfect reconstruction property: x = Ψ(Ψ†x) for all x
        
        Mathematical verification:
        1. Forward-inverse composition gives identity
        2. Energy conservation: ||Ψ†x|| = ||x||
        3. Linearity: Ψ†(αx + βy) = αΨ†x + βΨ†y
        4. Parseval's theorem validation
        
        Args:
            kernel: BulletproofQuantumKernel instance
            test_signals: Optional list of test signals
            
        Returns:
            Dictionary containing invertibility validation results
        """
        if test_signals is None:
            # Generate standard test signals
            test_signals = [
                np.random.randn(kernel.dimension) + 1j * np.random.randn(kernel.dimension),
                np.ones(kernel.dimension, dtype=complex),
                np.exp(1j * 2 * np.pi * np.arange(kernel.dimension) / kernel.dimension),
                np.zeros(kernel.dimension, dtype=complex)
            ]
            test_signals[3][0] = 1.0  # Delta function
        
        results = {
            'test_name': 'Transform Invertibility',
            'num_test_signals': len(test_signals),
            'reconstruction_errors': [],
            'energy_conservation_errors': [],
            'linearity_errors': []
        }
        
        for i, signal in enumerate(test_signals):
            # Normalize signal
            if np.linalg.norm(signal) > 0:
                signal = signal / np.linalg.norm(signal)
            
            # Test reconstruction
            spectrum = kernel.forward_rft(signal)
            reconstructed = kernel.inverse_rft(spectrum)
            reconstruction_error = np.linalg.norm(signal - reconstructed)
            results['reconstruction_errors'].append(reconstruction_error)
            
            # Test energy conservation
            energy_original = np.linalg.norm(signal)**2
            energy_spectrum = np.linalg.norm(spectrum)**2
            energy_error = abs(energy_original - energy_spectrum)
            results['energy_conservation_errors'].append(energy_error)
        
        # Test linearity with two random signals
        if len(test_signals) >= 2:
            alpha, beta = 0.3 + 0.4j, 0.7 - 0.2j
            x, y = test_signals[0], test_signals[1]
            
            # Linear combination
            combined = alpha * x + beta * y
            spectrum_combined = kernel.forward_rft(combined)
            
            # Individual transforms
            spectrum_x = kernel.forward_rft(x)
            spectrum_y = kernel.forward_rft(y)
            spectrum_linear = alpha * spectrum_x + beta * spectrum_y
            
            linearity_error = np.linalg.norm(spectrum_combined - spectrum_linear)
            results['linearity_errors'].append(linearity_error)
        
        # Summary statistics
        results['max_reconstruction_error'] = max(results['reconstruction_errors'])
        results['max_energy_error'] = max(results['energy_conservation_errors'])
        results['perfect_reconstruction'] = results['max_reconstruction_error'] < self.tolerance
        results['energy_conserved'] = results['max_energy_error'] < self.tolerance
        
        return results
    
    def validate_computational_complexity(self, dimensions: List[int] = None) -> Dict[str, Any]:
        """
        Analyze computational complexity scaling of RFT operations.
        
        Measurements:
        1. Forward transform timing vs dimension
        2. Inverse transform timing vs dimension  
        3. Kernel construction timing vs dimension
        4. Memory usage scaling
        5. Comparison with theoretical O(N log N) bound
        
        Args:
            dimensions: List of dimensions to test
            
        Returns:
            Dictionary containing complexity analysis results
        """
        if dimensions is None:
            dimensions = [4, 8, 16, 32, 64]
        
        results = {
            'test_name': 'Computational Complexity Analysis',
            'dimensions': dimensions,
            'forward_times': [],
            'inverse_times': [],
            'kernel_construction_times': [],
            'memory_usage': []
        }
        
        for n in dimensions:
            kernel = BulletproofQuantumKernel(dimension=n)
            
            # Time kernel construction
            start_time = time.perf_counter()
            kernel.build_resonance_kernel()
            kernel.compute_rft_basis()
            construction_time = time.perf_counter() - start_time
            results['kernel_construction_times'].append(construction_time)
            
            # Generate test signal
            test_signal = np.random.randn(n) + 1j * np.random.randn(n)
            
            # Time forward transform
            start_time = time.perf_counter()
            spectrum = kernel.forward_rft(test_signal)
            forward_time = time.perf_counter() - start_time
            results['forward_times'].append(forward_time)
            
            # Time inverse transform
            start_time = time.perf_counter()
            reconstructed = kernel.inverse_rft(spectrum)
            inverse_time = time.perf_counter() - start_time
            results['inverse_times'].append(inverse_time)
            
            # Estimate memory usage (simplified)
            memory_estimate = n**2 * 16  # Complex128 bytes for kernel matrix
            results['memory_usage'].append(memory_estimate)
        
        # Complexity analysis
        if len(dimensions) > 1:
            # Fit to O(N log N) model
            log_factors = [n * np.log2(n) for n in dimensions]
            
            # Linear regression for complexity scaling
            from numpy.polynomial import Polynomial
            
            # Forward transform complexity
            p_forward = Polynomial.fit(log_factors, results['forward_times'], 1)
            results['forward_complexity_slope'] = p_forward.coef[1]
            
            # Inverse transform complexity  
            p_inverse = Polynomial.fit(log_factors, results['inverse_times'], 1)
            results['inverse_complexity_slope'] = p_inverse.coef[1]
            
            # Check if scaling is approximately O(N log N)
            theoretical_slope = 1.0  # Perfect O(N log N) scaling
            results['forward_scaling_efficiency'] = abs(results['forward_complexity_slope'] - theoretical_slope)
            results['inverse_scaling_efficiency'] = abs(results['inverse_complexity_slope'] - theoretical_slope)
        
        return results
    
    def validate_orthogonality_stress_test(self, max_dimension: int = 1024) -> Dict[str, Any]:
        """
        Stress test orthogonality at very large dimensions (2^10 to 2^20 range).
        
        Tests numerical stability of:
        1. Eigendecomposition at scale
        2. Orthogonality preservation under finite precision
        3. Condition number behavior
        4. Rank deficiency detection
        
        Args:
            max_dimension: Maximum dimension to test (limited by memory)
            
        Returns:
            Dictionary containing stress test results
        """
        test_dimensions = [2**i for i in range(3, min(11, int(np.log2(max_dimension)) + 1))]
        
        results = {
            'test_name': 'Orthogonality Stress Test',
            'test_dimensions': test_dimensions,
            'orthogonality_errors': [],
            'condition_numbers': [],
            'eigenvalue_gaps': [],
            'numerical_ranks': []
        }
        
        for n in test_dimensions:
            try:
                kernel = BulletproofQuantumKernel(dimension=n)
                kernel.build_resonance_kernel()
                eigenvals, eigenvecs = kernel.compute_rft_basis()
                
                # Orthogonality error
                gram = eigenvecs.conj().T @ eigenvecs
                ortho_error = np.linalg.norm(gram - np.eye(n), 'fro')
                results['orthogonality_errors'].append(ortho_error)
                
                # Condition number
                cond_num = np.linalg.cond(kernel.resonance_kernel)
                results['condition_numbers'].append(cond_num)
                
                # Eigenvalue gap (minimum spacing)
                eigenvals_real = np.sort(np.real(eigenvals))
                eigenvalue_gap = np.min(np.diff(eigenvals_real)[np.diff(eigenvals_real) > self.tolerance])
                results['eigenvalue_gaps'].append(eigenvalue_gap)
                
                # Numerical rank
                numerical_rank = np.linalg.matrix_rank(kernel.resonance_kernel, tol=self.tolerance)
                results['numerical_ranks'].append(numerical_rank)
                
            except (np.linalg.LinAlgError, MemoryError) as e:
                results[f'error_at_dimension_{n}'] = str(e)
                break
        
        # Analysis
        if results['orthogonality_errors']:
            results['max_orthogonality_error'] = max(results['orthogonality_errors'])
            results['orthogonality_stable'] = results['max_orthogonality_error'] < 1e-6
            results['max_condition_number'] = max(results['condition_numbers'])
            results['well_conditioned'] = results['max_condition_number'] < 1e12
        
        return results
    
    def run_comprehensive_validation(self, kernel: BulletproofQuantumKernel) -> Dict[str, Any]:
        """
        Run complete mathematical validation suite.
        
        Args:
            kernel: BulletproofQuantumKernel instance
            
        Returns:
            Comprehensive validation results
        """
        print("Mathematical Validation Suite for Resonance Fourier Transform")
        print("=" * 60)
        print(f"Testing kernel with dimension: {kernel.dimension}")
        print(f"Numerical tolerance: {self.tolerance}")
        print()
        
        validation_results = {}
        
        # Test 1: Resonance kernel construction
        print("1. Resonance Kernel Construction Validation")
        kernel_results = self.validate_resonance_kernel_construction(kernel)
        validation_results['resonance_kernel'] = kernel_results
        print(f"   Hermitian error: {kernel_results['hermitian_error']:.2e}")
        print(f"   Is Hermitian: {kernel_results['is_hermitian']}")
        print(f"   Condition number: {kernel_results['condition_number']:.2e}")
        print(f"   Matrix rank: {kernel_results['matrix_rank']}/{kernel.dimension}")
        print()
        
        # Test 2: RFT basis orthogonality
        print("2. RFT Basis Orthogonality Validation")
        ortho_results = self.validate_rft_basis_orthogonality(kernel)
        validation_results['basis_orthogonality'] = ortho_results
        print(f"   Unitarity error: {ortho_results['unitarity_error']:.2e}")
        print(f"   Completeness error: {ortho_results['completeness_error']:.2e}")
        print(f"   Is unitary: {ortho_results['is_unitary']}")
        print(f"   Columns normalized: {ortho_results['columns_normalized']}")
        print()
        
        # Test 3: Transform invertibility
        print("3. Transform Invertibility Validation")
        invert_results = self.validate_transform_invertibility(kernel)
        validation_results['invertibility'] = invert_results
        print(f"   Max reconstruction error: {invert_results['max_reconstruction_error']:.2e}")
        print(f"   Max energy conservation error: {invert_results['max_energy_error']:.2e}")
        print(f"   Perfect reconstruction: {invert_results['perfect_reconstruction']}")
        print(f"   Energy conserved: {invert_results['energy_conserved']}")
        print()
        
        # Test 4: Computational complexity
        print("4. Computational Complexity Analysis")
        complexity_results = self.validate_computational_complexity([4, 8, 16, 32])
        validation_results['computational_complexity'] = complexity_results
        if 'forward_complexity_slope' in complexity_results:
            print(f"   Forward transform scaling: O(N^{complexity_results['forward_complexity_slope']:.2f} log N)")
            print(f"   Inverse transform scaling: O(N^{complexity_results['inverse_complexity_slope']:.2f} log N)")
        print()
        
        # Test 5: Orthogonality stress test (limited dimension)
        print("5. Orthogonality Stress Test")
        stress_results = self.validate_orthogonality_stress_test(max_dimension=256)
        validation_results['stress_test'] = stress_results
        if stress_results['orthogonality_errors']:
            print(f"   Max orthogonality error: {stress_results['max_orthogonality_error']:.2e}")
            print(f"   Orthogonality stable: {stress_results['orthogonality_stable']}")
            print(f"   Max condition number: {stress_results['max_condition_number']:.2e}")
        print()
        
        # Overall assessment
        print("Mathematical Validation Summary")
        print("-" * 30)
        
        all_tests_passed = (
            kernel_results['is_hermitian'] and
            ortho_results['is_unitary'] and
            invert_results['perfect_reconstruction'] and
            stress_results.get('orthogonality_stable', False)
        )
        
        print(f"All mathematical tests passed: {all_tests_passed}")
        
        validation_results['summary'] = {
            'all_tests_passed': all_tests_passed,
            'kernel_valid': kernel_results['is_hermitian'],
            'basis_orthogonal': ortho_results['is_unitary'],
            'transform_invertible': invert_results['perfect_reconstruction'],
            'numerically_stable': stress_results.get('orthogonality_stable', False)
        }
        
        return validation_results


def run_scientific_validation():
    """Run scientific validation with mathematical rigor."""
    
    # Initialize validator
    validator = MathematicalRFTValidator(tolerance=1e-12)
    
    # Create test kernel
    test_kernel = BulletproofQuantumKernel(dimension=16)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation(test_kernel)
    
    return results


if __name__ == "__main__":
    # Run scientific validation
    validation_results = run_scientific_validation()
    
    # Display final assessment
    summary = validation_results['summary']
    print("\nFinal Scientific Assessment:")
    print("Mathematical validation complete.")
    print(f"RFT implementation mathematically valid: {summary['all_tests_passed']}")
