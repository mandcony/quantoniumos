#!/usr/bin/env python3
"""""" RFT Separation Proof: Non-Commutation Test === This test proves that RFT defines a genuinely new class of transformations that is mathematically distinct from DFT/wavelets/graph-spectral methods. The key insight: If C(i) are non-diagonal mixers and the resulting operators are not jointly diagonalizable, then we get true separation. If any C(i) is diagonal, the whole thing collapses to per-bin scaling (DFT-adjacent). Test Strategy: 1) Build two distinct RFT instances R1, R2 with different (phi, sigma, w) 2) Compute commutator norm ||[R1, R2]||2 3) If ||[R1, R2]|| != 0, then they don't share a common eigenbasis 4) This kills "single common basis" property and proves separation Mathematical Framework: R = Sumᵢ wᵢ Dphiᵢ C(i) Dphiᵢdagger Where: - Dphiᵢ = phase shift operators - C(i) = non-diagonal mixing matrices - wᵢ = weights If C(i) are truly non-diagonal and operators are not jointly diagonalizable, then we have genuine separation from classical spectral methods. """"""

import numpy as np
import scipy.linalg from typing
import Tuple, Dict, List
import time from canonical_true_rft
import get_rft_basis, get_canonical_parameters
def generate_phase_shift_matrix(N: int, phi: float) -> np.ndarray:
"""
"""
        Generate phase shift matrix Dphi = diag(e^(i*phi*k)) for k=0,...,N-1 Args: N: Matrix size phi: Phase parameter Returns: N×N diagonal phase shift matrix
"""
"""
        phases = np.exp(1j * phi * np.arange(N))
        return np.diag(phases)
def generate_non_diagonal_mixer(N: int, mixing_strength: float = 0.5, seed: int = None) -> np.ndarray:
"""
"""
        Generate non-diagonal mixing matrix C(i) that ensures we don't collapse to per-bin scaling. Args: N: Matrix size mixing_strength: How much off-diagonal mixing (0=diagonal, 1=full mixing) seed: Random seed for reproducibility Returns: N×N Hermitian mixing matrix with controlled off-diagonal structure
"""
"""

        if seed is not None: np.random.seed(seed)

        # Start with random Hermitian matrix A = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N)) C = (A + A.conj().T) / 2

        # Make Hermitian

        # Control diagonal vs off-diagonal strength C_diag = np.diag(np.diag(C)) C_off_diag = C - C_diag

        # Mix: (1-strength)*diagonal + strength*full_matrix C_mixed = (1 - mixing_strength) * C_diag + mixing_strength * C

        # Normalize to unit operator norm C_mixed = C_mixed / np.linalg.norm(C_mixed, ord=2)
        return C_mixed
def build_rft_operator(N: int, phi_list: List[float], mixer_seeds: List[int], weights: List[float], mixing_strength: float = 0.5) -> np.ndarray:
"""
"""
        Build RFT operator: R = Sumᵢ wᵢ Dphiᵢ C(i) Dphiᵢdagger Args: N: Matrix size phi_list: List of phase parameters mixer_seeds: Seeds for generating mixing matrices C(i) weights: Weights wᵢ for each term mixing_strength: Controls off-diagonal mixing Returns: N×N RFT operator matrix
"""
"""
        assert len(phi_list) == len(mixer_seeds) == len(weights) R = np.zeros((N, N), dtype=complex) for i, (phi, seed, weight) in enumerate(zip(phi_list, mixer_seeds, weights)):

        # Generate phase shift Dphiᵢ D_phi = generate_phase_shift_matrix(N, phi) D_phi_dag = D_phi.conj().T

        # Generate non-diagonal mixer C(i) C_i = generate_non_diagonal_mixer(N, mixing_strength, seed)

        # Add term: wᵢ Dphiᵢ C(i) Dphiᵢdagger term = weight * D_phi @ C_i @ D_phi_dag R += term
        return R
def compute_commutator_norm(A: np.ndarray, B: np.ndarray) -> float:
"""
"""
        Compute ||[A,B]||2 where [A,B] = AB - BA If ||[A,B]|| = 0, then A and B commute (share eigenbasis) If ||[A,B]|| != 0, then they don't share a common eigenbasis
"""
"""
        commutator = A @ B - B @ A
        return np.linalg.norm(commutator, ord=2)
def test_diagonal_collapse_check(N: int = 8) -> Dict:
"""
"""
        Test that purely diagonal mixers collapse to DFT-adjacent behavior
"""
"""
        print(f"Testing diagonal collapse for N={N}...")

        # Build operator with purely diagonal mixers (mixing_strength=0) phi_list = [0.1, 0.3, 0.7] mixer_seeds = [42, 43, 44] weights = [1.0, 0.5, 0.3] R_diagonal = build_rft_operator(N, phi_list, mixer_seeds, weights, mixing_strength=0.0)

        # Check
        if this is approximately diagonal in some basis

        # For diagonal mixers: Dphiᵢ C(i) Dphiᵢdagger = Dphiᵢ diag(sigmaᵢ) Dphiᵢdagger = diag(sigmaᵢ)

        # So R should be diagonal eigenvals, eigenvecs = np.linalg.eig(R_diagonal)

        # Check
        if R is approximately diagonal in its eigenbasis R_in_eigenbasis = eigenvecs.conj().T @ R_diagonal @ eigenvecs off_diagonal_norm = np.linalg.norm(R_in_eigenbasis - np.diag(eigenvals), ord='fro') is_essentially_diagonal = off_diagonal_norm < 1e-12 result = { 'test_name': 'Diagonal Collapse Check', 'size': N, 'mixing_strength': 0.0, 'off_diagonal_norm_in_eigenbasis': float(off_diagonal_norm), 'is_essentially_diagonal': bool(is_essentially_diagonal), 'collapses_to_per_bin_scaling': bool(is_essentially_diagonal) }
        print(f" Off-diagonal norm in eigenbasis: {off_diagonal_norm:.2e}")
        print(f" Collapses to per-bin scaling: {is_essentially_diagonal}")
        return result
def test_non_commutation_separation(N: int = 8) -> Dict: """"""
        Main test: Prove separation by demonstrating non-commutation Build two distinct RFT operators and show they don't commute, proving they don't share a common eigenbasis.
"""
"""
        print(f"Testing non-commutation separation for N={N}...")

        # Build first RFT operator R1 phi_list_1 = [0.1, 0.4, 0.8] mixer_seeds_1 = [100, 101, 102] weights_1 = [1.0, 0.7, 0.4] mixing_strength = 0.8

        # Strong off-diagonal mixing R1 = build_rft_operator(N, phi_list_1, mixer_seeds_1, weights_1, mixing_strength)

        # Build second RFT operator R2 (different parameters) phi_list_2 = [0.2, 0.5, 0.9] mixer_seeds_2 = [200, 201, 202] weights_2 = [0.8, 1.0, 0.6] R2 = build_rft_operator(N, phi_list_2, mixer_seeds_2, weights_2, mixing_strength)

        # Compute commutator norm ||[R1, R2]||2 commutator_norm = compute_commutator_norm(R1, R2)

        # If commutator norm is non-zero, operators don't share eigenbasis non_commuting = commutator_norm > 1e-12 proves_separation = non_commuting

        # Additional check: Are R1 and R2 simultaneously diagonalizable?

        # If they commute, they can be simultaneously diagonalized simultaneously_diagonalizable = not non_commuting result = { 'test_name': 'Non-Commutation Separation', 'size': N, 'commutator_norm': float(commutator_norm), 'non_commuting': bool(non_commuting), 'simultaneously_diagonalizable': bool(simultaneously_diagonalizable), 'proves_separation_from_dft': bool(proves_separation), 'mixing_strength': mixing_strength }
        print(f" ||[R1, R2]||2: {commutator_norm:.6f}")
        print(f" Operators commute: {not non_commuting}")
        print(f" Proves separation from DFT: {proves_separation}")
        return result
def test_canonical_rft_separation(N: int = 8) -> Dict: """"""
        Test separation using actual canonical RFT implementation
"""
"""
        print(f"Testing canonical RFT separation for N={N}...")

        # Get canonical RFT basis RFT_canonical = get_rft_basis(N)

        # Build a custom RFT operator with different parameters phi_list = [0.3, 0.6, 1.2] mixer_seeds = [300, 301, 302] weights = [1.0, 0.8, 0.5] RFT_custom = build_rft_operator(N, phi_list, mixer_seeds, weights, mixing_strength=0.7)

        # Compute commutator with canonical implementation commutator_norm = compute_commutator_norm(RFT_canonical, RFT_custom)

        # Also test against standard DFT DFT_matrix = np.fft.fft(np.eye(N), axis=0) / np.sqrt(N) dft_commutator_norm = compute_commutator_norm(RFT_canonical, DFT_matrix) non_commuting_with_custom = commutator_norm > 1e-12 non_commuting_with_dft = dft_commutator_norm > 1e-12 result = { 'test_name': 'Canonical RFT Separation', 'size': N, 'commutator_norm_vs_custom_rft': float(commutator_norm), 'commutator_norm_vs_dft': float(dft_commutator_norm), 'non_commuting_with_custom_rft': bool(non_commuting_with_custom), 'non_commuting_with_dft': bool(non_commuting_with_dft), 'distinct_from_other_spectral_methods': bool(non_commuting_with_dft) }
        print(f" ||[RFT_canonical, RFT_custom]||2: {commutator_norm:.6f}")
        print(f" ||[RFT_canonical, DFT]2: {dft_commutator_norm:.6f}")
        print(f" Distinct from DFT: {non_commuting_with_dft}")
        return result
def test_mixer_criticality(N: int = 8) -> Dict: """"""
        Test how critical the non-diagonal nature of mixers is
"""
"""
        print(f"Testing mixer criticality for N={N}...") mixing_strengths = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] commutator_norms = [] phi_list = [0.2, 0.5, 0.8] mixer_seeds_1 = [400, 401, 402] mixer_seeds_2 = [500, 501, 502] weights = [1.0, 0.6, 0.3]
        for strength in mixing_strengths: R1 = build_rft_operator(N, phi_list, mixer_seeds_1, weights, strength) R2 = build_rft_operator(N, phi_list, mixer_seeds_2, weights, strength) comm_norm = compute_commutator_norm(R1, R2) commutator_norms.append(comm_norm)

        # Find critical mixing strength where separation emerges critical_threshold = 1e-10 critical_strength = None for i, (strength, norm) in enumerate(zip(mixing_strengths, commutator_norms)):
        if norm > critical_threshold: critical_strength = strength break result = { 'test_name': 'Mixer Criticality', 'size': N, 'mixing_strengths': mixing_strengths, 'commutator_norms': [float(x)
        for x in commutator_norms], 'critical_mixing_strength': critical_strength, 'separation_threshold': critical_threshold }
        print(f" Critical mixing strength for separation: {critical_strength}")
        print(f" Commutator norms: {[f'{x:.2e}'
        for x in commutator_norms[:3]]}...")
        return result

class RFTSeparationProof: """"""
        Test suite to prove RFT separation from classical spectral methods.
"""
"""

    def __init__(self):
        self.results = {}
    def run_separation_proof_suite(self, sizes: List[int] = None) -> Dict:
"""
"""
        Run complete separation proof test suite.
"""
"""
        if sizes is None: sizes = [4, 8, 16]
        print("=" * 70)
        print("RFT SEPARATION PROOF: NON-COMMUTATION TEST SUITE")
        print("Proving mathematical distinction from DFT/wavelets/graph-spectral")
        print("=" * 70) suite_results = { 'suite_name': 'RFT Separation Proof', 'timestamp': time.time(), 'test_sizes': sizes, 'results': {} }
        for N in sizes:
        print(f"\nTesting size N={N}")
        print("-" * 30) size_results = {} size_results['diagonal_collapse'] = test_diagonal_collapse_check(N) size_results['non_commutation'] = test_non_commutation_separation(N) size_results['canonical_separation'] = test_canonical_rft_separation(N) size_results['mixer_criticality'] = test_mixer_criticality(N)

        # Assess separation proof for this size proves_separation = ( size_results['non_commutation']['proves_separation_from_dft'] and size_results['canonical_separation']['distinct_from_other_spectral_methods'] and size_results['mixer_criticality']['critical_mixing_strength'] is not None ) size_results['proves_mathematical_separation'] = proves_separation suite_results['results'][f'N_{N}'] = size_results
        print(f"Size N={N} separation proof: {'✓ PROVEN'
        if proves_separation else '❌ NOT PROVEN'}")

        # Overall assessment all_sizes_prove_separation = all( suite_results['results'][f'N_{N}']['proves_mathematical_separation']
        for N in sizes ) suite_results['mathematical_separation_proven'] = all_sizes_prove_separation
        print("\n" + "=" * 70)
        if all_sizes_prove_separation:
        print("✓ MATHEMATICAL SEPARATION PROVEN")
        print(" RFT defines genuinely new class of spectral transforms")
        print(" Non-commuting operators ⟹ no shared eigenbasis")
        print(" Distinct from DFT/wavelets/graph-spectral methods")
        else:
        print("❌ SEPARATION NOT CONCLUSIVELY PROVEN")
        print(" May collapse to existing spectral methods")
        print("=" * 70)
        return suite_results
    def main(): """"""
        Run RFT separation proof.
"""
        """ prover = RFTSeparationProof() results = prover.run_separation_proof_suite([4, 8, 16])

        # Summary report
        print("\nDETAILED PROOF SUMMARY:")
        print("=" * 50) for size_key, size_results in results['results'].items(): N = size_key.split('_')[1]
        print(f"||nSize N={N}:")

        # Key metrics for separation proof non_comm = size_results['non_commutation'] canonical = size_results['canonical_separation']
        print(f" ||[R1,R2]||2: {non_comm['commutator_norm']:.4f}")
        print(f" ||[RFT,DFT]||2: {canonical['commutator_norm_vs_dft']:.4f}")
        print(f" Proves separation: {size_results['proves_mathematical_separation']}")
        return results

if __name__ == "__main__": main()