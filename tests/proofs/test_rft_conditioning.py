#!/usr/bin/env python3
"""
RFT CONDITIONING & NUMERICAL STABILITY VALIDATION
==================================================
Goal: Comprehensive numerical analysis of RFT properties
Tests: Matrix conditioning, phi-sensitivity, precision stability, numerical convergence
"""

import numpy as np
import pytest
import time
from typing import Dict, List, Tuple
import sys
import os

# Import the true RFT kernel
sys.path.append(os.path.dirname(__file__))
try:
    from tests.proofs.true_rft_kernel import TrueRFTKernel
except ImportError:
    print("Error: Could not import TrueRFTKernel")
    sys.exit(1)


def rft_transform(x: np.ndarray) -> np.ndarray:
    """Convenience wrapper using the canonical RFT kernel."""
    kernel = TrueRFTKernel(len(x))
    return kernel.forward_transform(x)


def inverse_rft_transform(y: np.ndarray) -> np.ndarray:
    """Inverse transform wrapper mirroring the forward convenience API."""
    kernel = TrueRFTKernel(len(y))
    return kernel.inverse_transform(y)


class PhiSensitiveRFT(TrueRFTKernel):
    """Modified RFT that allows custom phi values for sensitivity testing."""
    
    def __init__(self, size: int, phi_value: float, beta: float = 1.0):
        self.size = size
        self.beta = beta
        self.phi = phi_value  # Custom phi value
        
        # Build RFT kernel with custom phi
        self._rft_matrix = self._build_rft_kernel()


def evaluate_conditioning(n: int) -> Dict:
    """Test conditioning number of RFT matrix."""
    print(f"Testing conditioning for n={n}...")
    
    # Test both float32 and float64
    results = {}
    
    for dtype_name, dtype in [('float64', np.float64), ('float32', np.float32)]:
        try:
            rft = TrueRFTKernel(n)
            rft_matrix = rft.get_rft_matrix().astype(dtype)
            
            # Compute condition number
            cond_number = np.linalg.cond(rft_matrix)
            
            # Check if sub-quadratic: cond(RFT) < C * n^α where α < 2
            # We'll use a generous bound: cond < 10 * n^1.8
            sub_quadratic_bound = 10 * (n ** 1.8)
            is_sub_quadratic = cond_number < sub_quadratic_bound
            
            results[dtype_name] = {
                'condition_number': float(cond_number),
                'sub_quadratic_bound': float(sub_quadratic_bound),
                'is_sub_quadratic': is_sub_quadratic
            }
            
            print(f"  {dtype_name}: cond = {cond_number:.2e}, bound = {sub_quadratic_bound:.2e} {'✓' if is_sub_quadratic else '✗'}")
            
        except Exception as e:
            print(f"  {dtype_name}: conditioning test failed - {e}")
            results[dtype_name] = {
                'condition_number': float('inf'),
                'sub_quadratic_bound': float('inf'),
                'is_sub_quadratic': False
            }
    
    return results


def evaluate_phi_sensitivity(n: int, num_trials: int = 10) -> Dict:
    """Test sensitivity to golden ratio perturbations."""
    print(f"Testing φ-sensitivity for n={n}...")
    
    phi_golden = 1.618033988749894848204586834366  # True golden ratio
    
    # Test small perturbations around phi
    perturbations = np.linspace(-0.01, 0.01, num_trials)  # ±1% perturbations
    
    # Reference signal
    np.random.seed(42)  # Reproducible
    x = np.random.randn(n) + 1j * np.random.randn(n)
    x = x / np.linalg.norm(x)  # Normalize
    
    # Reference RFT with exact phi
    rft_ref = PhiSensitiveRFT(n, phi_golden)
    y_ref = rft_ref.forward_transform(x)
    y_ref_norm = np.linalg.norm(y_ref)
    
    sensitivities = []
    sensitivity_ratios = []
    
    for delta_phi in perturbations:
        if abs(delta_phi) < 1e-12:  # Skip zero perturbation
            continue
            
        # Perturbed RFT
        phi_perturbed = phi_golden + delta_phi
        rft_pert = PhiSensitiveRFT(n, phi_perturbed)
        y_pert = rft_pert.forward_transform(x)
        
        # Compute relative output change
        delta_y = y_pert - y_ref
        relative_output_change = np.linalg.norm(delta_y) / y_ref_norm
        
        # Compute relative phi change
        relative_phi_change = abs(delta_phi) / phi_golden
        
        # Sensitivity ratio: (‖Δy‖/‖y‖) / |Δφ/φ|
        if relative_phi_change > 0:
            sensitivity_ratio = relative_output_change / relative_phi_change
            sensitivity_ratios.append(sensitivity_ratio)
        
        sensitivities.append({
            'delta_phi': delta_phi,
            'relative_phi_change': relative_phi_change,
            'relative_output_change': relative_output_change,
            'sensitivity_ratio': sensitivity_ratio if relative_phi_change > 0 else float('inf')
        })
    
    # Check pass criterion: ‖Δy‖/‖y‖ ≤ C|Δφ| with C<4
    max_sensitivity = max(sensitivity_ratios) if sensitivity_ratios else float('inf')
    mean_sensitivity = np.mean(sensitivity_ratios) if sensitivity_ratios else float('inf')
    
    sensitivity_pass = max_sensitivity < 4.0
    
    print(f"  Max sensitivity ratio: {max_sensitivity:.2f} {'✓' if sensitivity_pass else '✗'}")
    print(f"  Mean sensitivity ratio: {mean_sensitivity:.2f}")
    print(f"  Criterion: C < 4.0 {'✓' if sensitivity_pass else '✗'}")
    
    return {
        'max_sensitivity': max_sensitivity,
        'mean_sensitivity': mean_sensitivity,
        'sensitivity_pass': sensitivity_pass,
        'sensitivities': sensitivities,
        'num_trials': len(sensitivity_ratios)
    }


def run_conditioning_test() -> Dict:
    """Run the complete conditioning and sensitivity test."""
    
    print("=" * 80)
    print("RFT CONDITIONING & φ-SENSITIVITY TEST")
    print("=" * 80)
    print("Goal: Numerical stability across n, float32/64; small ∂RFT/∂φ")
    print("Pass: cond(RFT) sub-quadratic in n; ‖Δy‖/‖y‖ ≤ C|Δφ| with C<4")
    print("-" * 80)
    
    sizes = [8, 16, 32, 64]
    results = {
        'sizes': sizes,
        'conditioning_results': {},
        'sensitivity_results': {},
        'overall_pass': True
    }
    
    for n in sizes:
        print(f"\nTesting size n={n}:")
        
        # Test conditioning
        cond_result = evaluate_conditioning(n)
        results['conditioning_results'][n] = cond_result
        
        # Check if conditioning passes for both dtypes
        cond_pass = (cond_result.get('float64', {}).get('is_sub_quadratic', False) and
                    cond_result.get('float32', {}).get('is_sub_quadratic', False))
        
        # Test phi sensitivity
        sens_result = evaluate_phi_sensitivity(n)
        results['sensitivity_results'][n] = sens_result
        sens_pass = sens_result['sensitivity_pass']
        
        # Overall pass for this size
        size_pass = cond_pass and sens_pass
        print(f"  Overall size result: {'✓ PASS' if size_pass else '✗ FAIL'}")
        
        if not size_pass:
            results['overall_pass'] = False
    
    return results


def run_detailed_analysis(n: int = 32) -> None:
    """Run detailed numerical analysis."""
    
    print("\n" + "=" * 80)
    print(f"DETAILED NUMERICAL ANALYSIS (n={n})")
    print("=" * 80)
    
    phi_golden = 1.618033988749894848204586834366
    
    # Analyze conditioning across precisions
    print("Conditioning analysis:")
    for dtype_name, dtype in [('float64', np.float64), ('float32', np.float32)]:
        try:
            rft = TrueRFTKernel(n)
            rft_matrix = rft.get_rft_matrix().astype(dtype)
            
            # Various condition number estimates
            cond_2 = np.linalg.cond(rft_matrix, 2)  # 2-norm
            cond_f = np.linalg.cond(rft_matrix, 'fro')  # Frobenius
            
            # Singular values
            svd_s = np.linalg.svd(rft_matrix, compute_uv=False)
            sigma_max = np.max(svd_s)
            sigma_min = np.min(svd_s[svd_s > 1e-14])  # Exclude numerical zeros
            
            print(f"  {dtype_name}:")
            print(f"    cond_2 = {cond_2:.2e}")
            print(f"    cond_F = {cond_f:.2e}")
            print(f"    σ_max = {sigma_max:.2e}")
            print(f"    σ_min = {sigma_min:.2e}")
            print(f"    σ_max/σ_min = {sigma_max/sigma_min:.2e}")
            
        except Exception as e:
            print(f"  {dtype_name}: analysis failed - {e}")
    
    # Phi sensitivity landscape
    print(f"\nPhi sensitivity landscape:")
    print(f"Golden ratio φ = {phi_golden:.15f}")
    
    # Test wider range of perturbations
    perturbations = [-0.1, -0.05, -0.01, -0.001, 0.001, 0.01, 0.05, 0.1]
    
    np.random.seed(42)
    x = np.random.randn(n) + 1j * np.random.randn(n)
    x = x / np.linalg.norm(x)
    
    rft_ref = PhiSensitiveRFT(n, phi_golden)
    y_ref = rft_ref.forward_transform(x)
    y_ref_norm = np.linalg.norm(y_ref)
    
    print("  Δφ        |Δφ/φ|     ‖Δy‖/‖y‖   Sensitivity")
    print("  " + "-" * 45)
    
    for delta_phi in perturbations:
        phi_pert = phi_golden + delta_phi
        rft_pert = PhiSensitiveRFT(n, phi_pert)
        y_pert = rft_pert.forward_transform(x)
        
        delta_y = y_pert - y_ref
        rel_output = np.linalg.norm(delta_y) / y_ref_norm
        rel_phi = abs(delta_phi) / phi_golden
        sensitivity = rel_output / rel_phi if rel_phi > 0 else float('inf')
        
        print(f"  {delta_phi:+.3f}   {rel_phi:.2e}   {rel_output:.2e}   {sensitivity:.2f}")


def test_rft_conditioning_bounds():
    """RFT conditioning should remain below the published sub-quadratic bound."""
    result = evaluate_conditioning(32)
    assert result['float64']['is_sub_quadratic'], "float64 conditioning exceeds bound"
    assert result['float32']['is_sub_quadratic'], "float32 conditioning exceeds bound"


def test_rft_phi_sensitivity_limit():
    """Small φ perturbations should keep sensitivity ratio below 4."""
    result = evaluate_phi_sensitivity(32)
    assert result['sensitivity_pass'], "φ-sensitivity ratio exceeded limit"


def main():
    """Main test function."""
    start_time = time.time()
    
    # Run conditioning and sensitivity test
    results = run_conditioning_test()
    
    # Run detailed analysis
    run_detailed_analysis(32)
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    
    if results['overall_pass']:
        print("✓ TEST PASSED: RFT has good numerical properties")
        print("  Conditioning is sub-quadratic in n")
        print("  φ-sensitivity is well-controlled (C < 4)")
        exit_code = 0
    else:
        print("✗ TEST FAILED: Numerical conditioning test failed")
        print("  RFT may have poor numerical properties")
        exit_code = 1
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.2f} seconds")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
