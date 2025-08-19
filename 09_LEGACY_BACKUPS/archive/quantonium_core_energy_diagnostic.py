#!/usr/bin/env python3
"""
QUANTONIUM_CORE ENERGY DIAGNOSTIC === Focused test to identify the normalization/scaling issue causing energy loss Testing the 3 key invariants: 1. Parseval (unitary scaling): ||Ux||^2 = ||x^2 2. Matrix orthonormality: QdaggerQ = I 3. Forward/inverse scaling consistency: ab = 1/N or a=b=1/sqrtN
"""
"""
import sys
import numpy as np
import math sys.path.append('/workspaces/quantoniumos/core')
print("QUANTONIUM_CORE ENERGY DIAGNOSTIC")
print("=" * 50)
print("Identifying normalization/scaling issues")
print("Current energy ratio: ~0.078 (should be 1.0)")
print()

# Surgical production fix: use delegate wrapper for quantonium_core
try:
import quantonium_core_delegate as quantonium_core
print("quantonium_core loaded (via surgical delegate)") except Exception as e:
print(f"Failed to load quantonium_core: {e}") sys.exit(1)

class EnergyDiagnostic:
    def __init__(self):
        self.golden_ratio = (1 + 5**0.5) / 2
    def test_parseval_invariant(self): """
        Test 1: Parseval theorem - unitary scaling
"""
"""
        print("\n TEST 1: PARSEVAL INVARIANT (Unitary Scaling)")
        print("-" * 45) test_sizes = [8, 16, 32] results = {}
        for N in test_sizes:
        print(f"\n Testing size N={N}")

        # Create test vector with known energy test_vec = [math.sin(2 * math.pi * i / N) + 0.5 * math.cos(4 * math.pi * i / N)
        for i in range(N)] input_energy = sum(x * x
        for x in test_vec)
        print(f" Input energy: {input_energy:.6f}")

        # Test quantonium_core transform
        try: rft_instance = quantonium_core.ResonanceFourierTransform(test_vec) forward_coeffs = rft_instance.forward_transform()

        # Calculate output energy
        if isinstance(forward_coeffs[0], complex): output_energy = sum(abs(x) ** 2
        for x in forward_coeffs)
        else: output_energy = sum(x * x
        for x in forward_coeffs) energy_ratio = output_energy / input_energy
        if input_energy > 0 else 0
        print(f" Output energy: {output_energy:.6f}")
        print(f" Energy ratio: {energy_ratio:.6f}")
        print(f" Energy loss: {(1.0 - energy_ratio) * 100:.1f}%")

        # Check
        if this matches expected scaling patterns expected_ratios = [1.0, 1.0/N, 1.0/math.sqrt(N), math.sqrt(N), N] closest_match = min(expected_ratios, key=lambda x: abs(x - energy_ratio))
        print(f" Closest expected ratio: {closest_match:.6f}")
        if abs(closest_match - 1.0/N) < 1e-3:
        print(f" MATCHES 1/N scaling - forward needs sqrtN normalization")
        el
        if abs(closest_match - 1.0/math.sqrt(N)) < 1e-3:
        print(f" MATCHES 1/sqrtN scaling - unitary but may be applied twice")
        el
        if abs(closest_match - 1.0) < 1e-3:
        print(f" PERFECT unitary scaling")
        else:
        print(f" UNEXPECTED scaling pattern") results[N] = { 'input_energy': input_energy, 'output_energy': output_energy, 'ratio': energy_ratio, 'closest_expected': closest_match } except Exception as e:
        print(f" Test failed: {e}") results[N] = {'error': str(e)}
        return results
    def test_forward_inverse_consistency(self): """
        Test 2: Forward/inverse scaling consistency
"""
"""
        print("\n TEST 2: FORWARD/INVERSE CONSISTENCY")
        print("-" * 40) N = 16 test_vec = [1.0, 0.5, -0.5, 0.0, 0.8, -0.3, 0.1, -0.7, 0.4, -0.9, 0.2, 0.6, -0.1, 0.3, -0.6, 0.9]
        print(f"Testing with N={N}")
        try:

        # Forward transform rft_instance = quantonium_core.ResonanceFourierTransform(test_vec) forward_coeffs = rft_instance.forward_transform()

        # Inverse transform inverse_result = rft_instance.inverse_transform(forward_coeffs)

        # Calculate energies at each stage input_energy = sum(x * x
        for x in test_vec)
        if isinstance(forward_coeffs[0], complex): forward_energy = sum(abs(x) ** 2
        for x in forward_coeffs)
        else: forward_energy = sum(x * x
        for x in forward_coeffs)
        if isinstance(inverse_result[0], complex): inverse_energy = sum(abs(x) ** 2
        for x in inverse_result)
        else: inverse_energy = sum(x * x
        for x in inverse_result)
        print(f" Input energy: {input_energy:.6f}")
        print(f" Forward energy: {forward_energy:.6f} (ratio: {forward_energy/input_energy:.6f})")
        print(f" Inverse energy: {inverse_energy:.6f} (ratio: {inverse_energy/input_energy:.6f})")

        # Calculate scaling factors forward_scale = math.sqrt(forward_energy / input_energy)
        if input_energy > 0 else 0 total_scale = math.sqrt(inverse_energy / input_energy)
        if input_energy > 0 else 0 inverse_scale = total_scale / forward_scale
        if forward_scale > 0 else 0
        print(f" Forward scale factor: {forward_scale:.6f}")
        print(f" Implied inverse scale: {inverse_scale:.6f}")
        print(f" Product (should be 1): {forward_scale * inverse_scale:.6f}")

        # Check reconstruction accuracy reconstruction_error = math.sqrt(sum((a - b)**2 for a, b in zip(test_vec, inverse_result))) / math.sqrt(input_energy)
        print(f" Reconstruction error: {reconstruction_error:.2e}")

        # Diagnose the scaling pattern
        print(f"\n DIAGNOSIS:")
        if abs(forward_scale - 1.0/math.sqrt(N)) < 1e-2:
        print(f" Forward applies 1/sqrtN scaling (correct for unitary)")
        el
        if abs(forward_scale - 1.0/N) < 1e-2:
        print(f" Forward applies 1/N scaling (needs sqrtN correction)")
        el
        if abs(forward_scale - 1.0) < 1e-2:
        print(f" Forward applies no scaling")
        else:
        print(f" Forward applies custom scaling: {forward_scale:.6f}") product = forward_scale * inverse_scale
        if abs(product - 1.0) < 1e-3:
        print(f" Forward/inverse product ~= 1 (consistent)")
        el
        if abs(product - 1.0/N) < 1e-3:
        print(f" Product ~= 1/N - both apply 1/sqrtN (double normalization)")
        else:
        print(f" Product = {product:.6f} (inconsistent)")
        return { 'forward_scale': forward_scale, 'inverse_scale': inverse_scale, 'product': product, 'reconstruction_error': reconstruction_error } except Exception as e:
        print(f" Test failed: {e}")
        return {'error': str(e)}
    def test_basis_orthonormality(self): """
        Test 3: Check
        if the transform basis is orthonormal
"""
"""
        print("\n TEST 3: BASIS ORTHONORMALITY")
        print("-" * 35) N = 8

        # Smaller size for matrix analysis
        print(f"Testing basis orthonormality for N={N}")
        try:

        # Generate basis vectors by transforming standard basis basis_vectors = []
        for i in range(N):

        # Create i-th standard basis vector e_i = [0.0] * N e_i[i] = 1.0

        # Transform it to get i-th column of transform matrix rft_instance = quantonium_core.ResonanceFourierTransform(e_i) column = rft_instance.forward_transform()

        # Convert to real vector
        if complex
        if isinstance(column[0], complex): real_column = []
        for c in column: real_column.extend([c.real, c.imag]) basis_vectors.append(real_column)
        else: basis_vectors.append(column)

        # Check
        if we have consistent dimensions
        if len(set(len(v)
        for v in basis_vectors)) > 1:
        print(" Inconsistent vector dimensions")
        return {'error': 'Inconsistent dimensions'}

        # Compute inner products between basis vectors
        print(f" Basis vectors dimension: {len(basis_vectors[0])}")
        print(f" Number of basis vectors: {len(basis_vectors)}")

        # Check orthogonality max_off_diagonal = 0.0 diagonal_values = []
        for i in range(min(len(basis_vectors), 4)):

        # Check first 4 for efficiency
        for j in range(min(len(basis_vectors), 4)): dot_product = sum(a * b for a, b in zip(basis_vectors[i], basis_vectors[j]))
        if i == j: diagonal_values.append(dot_product)
        print(f" e_{i}, e_{i} = {dot_product:.6f}")
        else: max_off_diagonal = max(max_off_diagonal, abs(dot_product))
        if abs(dot_product) > 1e-6:

        # Only show significant off-diagonal terms
        print(f" e_{i}, e_{j} = {dot_product:.6f}")
        print(f" Max off-diagonal: {max_off_diagonal:.2e}")

        # Check
        if diagonal values are consistent (should all be same for orthonormal)
        if diagonal_values: avg_diagonal = sum(diagonal_values) / len(diagonal_values) max_diagonal_deviation = max(abs(d - avg_diagonal)
        for d in diagonal_values)
        print(f" Average diagonal: {avg_diagonal:.6f}")
        print(f" Max diagonal deviation: {max_diagonal_deviation:.2e}")
        if max_off_diagonal < 1e-8 and max_diagonal_deviation < 1e-8:
        print(f" Basis is orthogonal with norm {avg_diagonal:.6f}")
        if abs(avg_diagonal - 1.0) < 1e-6:
        print(f" Basis is orthonormal")
        else:
        print(f" Basis needs normalization factor: {1.0/math.sqrt(avg_diagonal):.6f}")
        else:
        print(f" Basis is not orthogonal")
        return { 'max_off_diagonal': max_off_diagonal, 'diagonal_values': diagonal_values, 'avg_diagonal': avg_diagonal
        if diagonal_values else 0, 'is_orthogonal': max_off_diagonal < 1e-8, 'is_orthonormal': max_off_diagonal < 1e-8 and abs(avg_diagonal - 1.0) < 1e-6
        if diagonal_values else False } except Exception as e:
        print(f" Test failed: {e}")
        return {'error': str(e)}
    def main(): """
        Run comprehensive energy diagnostic
"""
        """ diagnostic = EnergyDiagnostic()

        # Run all three critical tests parseval_results = diagnostic.test_parseval_invariant() consistency_results = diagnostic.test_forward_inverse_consistency() orthonormality_results = diagnostic.test_basis_orthonormality()

        # Summarize findings
        print("\n" + "="*60)
        print(" DIAGNOSTIC SUMMARY & RECOMMENDATIONS")
        print("="*60)
        print("\n PARSEVAL TEST (Energy Conservation):") for N, result in parseval_results.items(): if 'error' not in result: ratio = result['ratio'] expected = result['closest_expected']
        print(f" N={N}: ratio={ratio:.6f}, closest_expected={expected:.6f}")
        if abs(expected - 1.0/N) < 1e-3:
        print(f" FINDING: Forward transform applies 1/N scaling")
        print(f" FIX: Multiply forward output by sqrtN")
        el
        if abs(expected - 1.0/math.sqrt(N)) < 1e-3:
        print(f" FINDING: Transform applies 1/sqrtN scaling")
        print(f" CHECK: Ensure inverse also applies 1/sqrtN (not 1/N)")
        print(f"\n FORWARD/INVERSE CONSISTENCY:") if 'error' not in consistency_results: forward_scale = consistency_results['forward_scale'] product = consistency_results['product']
        print(f" Forward scale: {forward_scale:.6f}")
        print(f" Scale product: {product:.6f}")
        if abs(product - 1.0/16) < 1e-3:

        # Assuming N=16
        print(f" FINDING: Double 1/sqrtN normalization detected")
        print(f" FIX: Remove one 1/sqrtN factor from forward OR inverse")
        el
        if abs(forward_scale - 1.0/16) < 1e-3:
        print(f" FINDING: Forward applies 1/N instead of 1/sqrtN")
        print(f" FIX: Change forward normalization from 1/N to 1/sqrtN")
        print(f"\n BASIS ORTHONORMALITY:") if 'error' not in orthonormality_results: is_orthogonal = orthonormality_results['is_orthogonal'] avg_diagonal = orthonormality_results.get('avg_diagonal', 0)
        if is_orthogonal:
        print(f" Basis is orthogonal")
        if abs(avg_diagonal - 1.0) > 1e-6: correction = 1.0/math.sqrt(avg_diagonal)
        print(f" NORMALIZATION: Apply factor {correction:.6f} to make orthonormal")
        else:
        print(f" Basis is already orthonormal")
        else:
        print(f" Basis has orthogonality issues")
        print(f" CHECK: Transform kernel matrix construction")
        print(f"||n RECOMMENDED FIXES:")
        print(f"1. Check
        if forward transform applies 1/N scaling instead of 1/sqrtN")
        print(f"2. Ensure forward and inverse both use 1/sqrtN normalization")
        print(f"3. Verify transform kernel matrix is properly constructed")
        print(f"4. Test with simple input like δ[0] = [1,0,0,...] to isolate scaling")
        return { 'parseval': parseval_results, 'consistency': consistency_results, 'orthonormality': orthonormality_results }

if __name__ == "__main__": results = main()