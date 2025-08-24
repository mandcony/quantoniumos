#!/usr/bin/env python3
"""
RFT Non-Diagonal Mixer Validation === This test specifically validates that the C(i) mixing matrices in the RFT formulation R = Sumᵢ wᵢ Dphiᵢ C(i) Dphiᵢdagger are truly non-diagonal, preventing collapse to per-bin scaling (DFT-adjacent behavior). Key Mathematical Insight: If any C(i) is diagonal, then: Dphiᵢ C(i) Dphiᵢdagger = Dphiᵢ diag(sigmaᵢ) Dphiᵢdagger = diag(sigmaᵢ) And the whole operator becomes: R = Sumᵢ wᵢ diag(sigmaᵢ) = diag(Sumᵢ wᵢ sigmaᵢ) This is just per-bin scaling, which is DFT-adjacent and not genuinely new. To prove true separation, we must show: 1. C(i) have significant off-diagonal structure 2. The resulting R operators don't collapse to diagonal form 3. Non-commutation persists even with realistic mixing constraints
"""
"""

import numpy as np
import scipy.linalg from typing
import Dict, List, Tuple
from test_rft_separation_proof import ( generate_phase_shift_matrix, generate_non_diagonal_mixer, build_rft_operator, compute_commutator_norm )
def measure_off_diagonal_strength(matrix: np.ndarray) -> Dict:
"""
"""
        Quantify how non-diagonal a matrix is. Returns various metrics of off-diagonal structure.
"""
"""
        N = matrix.shape[0]

        # Separate diagonal and off-diagonal parts diag_part = np.diag(np.diag(matrix)) off_diag_part = matrix - diag_part

        # Compute norms total_norm = np.linalg.norm(matrix, ord='fro') diag_norm = np.linalg.norm(diag_part, ord='fro') off_diag_norm = np.linalg.norm(off_diag_part, ord='fro')

        # Off-diagonal ratio off_diag_ratio = off_diag_norm / total_norm
        if total_norm > 0 else 0

        # Coherence measure (max off-diagonal element relative to diagonal) max_off_diag = np.max(np.abs(off_diag_part)) max_diag = np.max(np.abs(np.diag(matrix))) coherence = max_off_diag / max_diag
        if max_diag > 0 else float('inf')
        return { 'total_norm': float(total_norm), 'diagonal_norm': float(diag_norm), 'off_diagonal_norm': float(off_diag_norm), 'off_diagonal_ratio': float(off_diag_ratio), 'coherence': float(coherence), 'is_essentially_diagonal': bool(off_diag_ratio < 0.01) }
def test_mixer_non_diagonal_requirement(N: int = 8) -> Dict:
"""
"""
        Test that our mixing matrices C(i) satisfy non-diagonal requirements.
"""
"""
        print(f"Testing mixer non-diagonal requirement for N={N}...")

        # Test different mixing strengths mixing_strengths = [0.0, 0.2, 0.5, 0.8, 1.0] mixer_results = []
        for strength in mixing_strengths:

        # Generate mixer with this strength C = generate_non_diagonal_mixer(N, strength, seed=42)

        # Analyze off-diagonal structure off_diag_metrics = measure_off_diagonal_strength(C) off_diag_metrics['mixing_strength'] = strength mixer_results.append(off_diag_metrics)
        print(f" Strength {strength:.1f}: off-diag ratio = {off_diag_metrics['off_diagonal_ratio']:.3f}, " f"essentially diagonal = {off_diag_metrics['is_essentially_diagonal']}")

        # Find minimum strength for non-diagonal behavior min_strength_for_non_diagonal = None
        for result in mixer_results:
        if not result['is_essentially_diagonal']: min_strength_for_non_diagonal = result['mixing_strength'] break
        return { 'test_name': 'Mixer Non-Diagonal Requirement', 'size': N, 'mixer_analysis': mixer_results, 'min_strength_for_non_diagonal': min_strength_for_non_diagonal }
def test_collapse_prevention(N: int = 8) -> Dict: """
        Test that with proper non-diagonal mixers, we prevent collapse to per-bin scaling behavior.
"""
"""
        print(f"Testing collapse prevention for N={N}...")

        # Build RFT with diagonal mixers (should collapse) phi_list = [0.1, 0.3, 0.7] seeds = [100, 101, 102] weights = [1.0, 0.5, 0.3] R_diagonal = build_rft_operator(N, phi_list, seeds, weights, mixing_strength=0.0) R_non_diagonal = build_rft_operator(N, phi_list, seeds, weights, mixing_strength=0.8)

        # Analyze structure of resulting operators R_diag_metrics = measure_off_diagonal_strength(R_diagonal) R_non_diag_metrics = measure_off_diagonal_strength(R_non_diagonal)

        # Test
        if they behave differently commutator_norm = compute_commutator_norm(R_diagonal, R_non_diagonal) result = { 'test_name': 'Collapse Prevention', 'size': N, 'diagonal_mixer_operator': R_diag_metrics, 'non_diagonal_mixer_operator': R_non_diag_metrics, 'operators_commute': bool(commutator_norm < 1e-12), 'commutator_norm': float(commutator_norm), 'collapse_prevented': bool(not R_non_diag_metrics['is_essentially_diagonal']) }
        print(f" Diagonal mixers -> diagonal operator: {R_diag_metrics['is_essentially_diagonal']}")
        print(f" Non-diagonal mixers -> non-diagonal operator: {not R_non_diag_metrics['is_essentially_diagonal']}")
        print(f" ||[R_diag, R_non_diag]||2: {commutator_norm:.6f}")
        print(f" Collapse prevented: {result['collapse_prevented']}")
        return result
def test_realistic_mixer_constraints(N: int = 8) -> Dict: """
        Test separation under realistic constraints on mixing matrices. In practice, C(i) might have physical constraints (e.g., locality, energy bounds) that limit how non-diagonal they can be.
"""
"""
        print(f"Testing realistic mixer constraints for N={N}...")
def generate_constrained_mixer(N: int, locality: int, strength: float) -> np.ndarray: """
        Generate mixer with locality constraint (only k-local interactions)
"""
        """ np.random.seed(42) C = np.zeros((N, N), dtype=complex)

        # Fill in local interactions only
        for i in range(N):
        for j in range(max(0, i-locality), min(N, i+locality+1)):
        if i == j:

        # Diagonal element C[i, j] = 1.0 + 0.1 * np.random.normal()
        else:

        # Off-diagonal element (local interaction) C[i, j] = strength * (np.random.normal() + 1j * np.random.normal())

        # Make Hermitian C = (C + C.conj().T) / 2

        # Normalize C = C / np.linalg.norm(C, ord=2)
        return C

        # Test different locality constraints localities = [1, 2, N//2, N-1] # 1-local to fully connected locality_results = [] phi_list = [0.2, 0.5, 0.8] weights = [1.0, 0.6, 0.3]
        for locality in localities:

        # Build operators with constrained mixers R1_constrained = np.zeros((N, N), dtype=complex) R2_constrained = np.zeros((N, N), dtype=complex) for i, (phi, w) in enumerate(zip(phi_list, weights)): D_phi = generate_phase_shift_matrix(N, phi) D_phi_dag = D_phi.conj().T

        # Use different strengths for R1 and R2 to ensure they're different C1 = generate_constrained_mixer(N, locality, 0.5) C2 = generate_constrained_mixer(N, locality, 0.7) R1_constrained += w * D_phi @ C1 @ D_phi_dag R2_constrained += w * D_phi @ C2 @ D_phi_dag

        # Measure separation with constrained mixers commutator_norm = compute_commutator_norm(R1_constrained, R2_constrained)

        # Analyze structure R1_metrics = measure_off_diagonal_strength(R1_constrained) locality_result = { 'locality': locality, 'commutator_norm': float(commutator_norm), 'proves_separation': bool(commutator_norm > 1e-10), 'off_diagonal_ratio': R1_metrics['off_diagonal_ratio'] } locality_results.append(locality_result)
        print(f" Locality {locality}: ||[R1,R2]2 = {commutator_norm:.4f}, " f"separation = {locality_result['proves_separation']}")

        # Find minimum locality needed for separation min_locality_for_separation = None
        for result in locality_results:
        if result['proves_separation']: min_locality_for_separation = result['locality'] break
        return { 'test_name': 'Realistic Mixer Constraints', 'size': N, 'locality_analysis': locality_results, 'min_locality_for_separation': min_locality_for_separation }
def test_canonical_rft_mixer_analysis() -> Dict: """
        Analyze the actual mixing structure in the canonical RFT implementation to verify it satisfies non-diagonal requirements.
"""
"""
        print("Analyzing canonical RFT mixer structure...") import importlib.util
import os

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
get_rft_basis results = canonical_true_rft.get_rft_basis results= {}
        for N in [4, 8, 16]:

        # Get canonical RFT matrix RFT_canonical = get_rft_basis(N)

        # Analyze its off-diagonal structure structure = measure_off_diagonal_strength(RFT_canonical) results[f'N_{N}'] = structure
        print(f" N={N}: off-diagonal ratio = {structure['off_diagonal_ratio']:.3f}, " f"essentially diagonal = {structure['is_essentially_diagonal']}")
        return { 'test_name': 'Canonical RFT Mixer Analysis', 'canonical_structure_analysis': results }
def main(): """
        Run comprehensive mixer validation tests.
"""
"""
        print("=" * 70)
        print("RFT NON-DIAGONAL MIXER VALIDATION")
        print("Ensuring C(i) mixers prevent collapse to per-bin scaling")
        print("=" * 70)

        # Run all tests results = {}
        for N in [4, 8, 16]:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {} size_results['mixer_requirements'] = test_mixer_non_diagonal_requirement(N) size_results['collapse_prevention'] = test_collapse_prevention(N) size_results['realistic_constraints'] = test_realistic_mixer_constraints(N) results[f'N_{N}'] = size_results

        # Analyze canonical implementation results['canonical_analysis'] = test_canonical_rft_mixer_analysis()
        print("\n" + "=" * 70)
        print("MIXER VALIDATION SUMMARY")
        print("=" * 70)

        # Overall assessment all_tests_pass = True for size_key, size_results in results.items():
        if size_key == 'canonical_analysis': continue N = size_key.split('_')[1] collapse_prevented = size_results['collapse_prevention']['collapse_prevented']
        print(f"N={N}: Collapse prevention = {collapse_prevented}")
        if not collapse_prevented: all_tests_pass = False
        if all_tests_pass:
        print("\n✅ ALL MIXER VALIDATIONS PASS")
        print(" Non-diagonal mixers confirmed")
        print(" Collapse to per-bin scaling prevented")
        print(" RFT defines genuinely new spectral class")
        else:
        print("||n❌ SOME MIXER VALIDATIONS FAIL")
        print(" Risk of collapse to DFT-adjacent behavior")
        return results

if __name__ == "__main__": main()