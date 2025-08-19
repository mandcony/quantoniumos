"""
Epsilon_N Reproducibility Test This test ensures that the epsilonₙ values computed for the formal derivation remain stable and above the critical threshold of 1e-3, preventing future refactors from accidentally re-introducing DFT-like behavior.
"""
"""

import pytest
import numpy as np
import sys sys.path.append('.') from canonical_true_rft
import forward_true_rft
def compute_epsilon_n(N):
"""
"""
        Compute epsilonₙ = ||RS - SR_F for cyclic shift matrix S.
"""
"""

        # Standard RFT parameters weights = [0.7, 0.3] theta0_values = [0.0, np.pi/4] omega_values = [1.0, (1 + np.sqrt(5))/2]

        # Golden ratio sigma0 = 1.0 gamma = 0.3

        # Compute resonance matrix via canonical function from canonical_true_rft
import generate_resonance_kernel R = generate_resonance_kernel(N, weights, theta0_values, omega_values, sigma0, gamma)

        # Cyclic shift matrix S = np.roll(np.eye(N), 1, axis=0)

        # Compute commutator norm RS = np.dot(R, S) SR = np.dot(S, R) commutator = RS - SR epsilon_n = np.linalg.norm(commutator, 'fro')
        return epsilon_n
def test_epsilon_values_above_threshold():
"""
"""
        Test that epsilonₙ values remain above 1e-3 threshold for key sizes.
"""
"""

        # Test cases from formal derivation table test_cases = [ (8, 0.400),

        # Allow some variation from computed values (12, 0.080),

        # Minimum expected threshold (16, 0.500), (32, 0.150), (64, 0.250) ] for N, min_expected in test_cases: epsilon_n = compute_epsilon_n(N)

        # Critical check: must be >> 1e-3 to ensure non-DFT behavior assert epsilon_n > 1e-3, f"epsilonₙ({N}) = {epsilon_n:.6f} is too small (< 1e-3)"

        # Reasonable range check (allow some numerical variation) assert epsilon_n > min_expected * 0.5, f"epsilonₙ({N}) = {epsilon_n:.6f} is below expected range"
        print(f"✓ epsilon_{N} = {epsilon_n:.6f} (> {min_expected:.3f} expected, >> 1e-3 threshold)")
def test_epsilon_increases_with_complexity(): """
        Test that epsilonₙ shows reasonable variation across different N values.
"""
        """ N_values = [8, 12, 16] epsilon_values = [compute_epsilon_n(N)
        for N in N_values]

        # All should be well above machine epsilon for i, (N, eps) in enumerate(zip(N_values, epsilon_values)): assert eps > 1e-10, f"epsilonₙ({N}) = {eps} is suspiciously small"
        print(f"✓ epsilon_{N} = {eps:.6f}")

        # At least one should be significantly non-zero assert max(epsilon_values) > 0.01, "All epsilonₙ values are too small - potential DFT behavior"

if __name__ == "__main__":
print("Testing epsilonₙ reproducibility for formal derivation...") test_epsilon_values_above_threshold() test_epsilon_increases_with_complexity()
print("||n All epsilonₙ tests passed - RFT maintains non-DFT behavior!")