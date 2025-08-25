# LEGACY RFT IMPLEMENTATION - REPLACE WITH CANONICAL
import os
import importlib.util

# Load the canonical_true_rft module
spec = importlib.util.spec_from_file_location(
    "canonical_true_rft", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "04_RFT_ALGORITHMS/canonical_true_rft.py")
)
canonical_true_rft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical_true_rft)

# Import specific functions/classes
forward_true_rft = canonical_true_rft.forward_true_rft
inverse_true_rft = canonical_true_rft.inverse_true_rft

from itertools import permutations
import numpy as np
from numpy.linalg import inv, norm


def unitary_dft(n):
    j, k = np.meshgrid(np.arange(n), np.arange(n))
    F = np.exp(-2j * np.pi * j * k / n) / np.sqrt(n)
    return F


def cyclic_shift(n):
    S = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        S[i, i + 1] = 1.0
    S[n - 1, 0] = 1.0
    return S


def build_rft_matrix(forward_true_rft):
    def psi(n):
        cols = []
        for e in np.eye(n, dtype=float):
            cols.append(forward_true_rft(e))
        return np.column_stack(cols)

    return psi


def greedy_match_columns(Psi, F):
    n = Psi.shape[0]
    unused = set(range(n))
    match = [-1] * n
    for j in range(n):
        corrs = [(k, abs(np.vdot(F[:, k], Psi[:, j]))) for k in unused]
        kbest = max(corrs, key=lambda t: t[1])[0]
        match[j] = kbest
        unused.remove(kbest)
    return match


def best_diagonal_right_and_left(Psi, F, match):
    n = Psi.shape[0]
    D2 = np.zeros(n, dtype=complex)
    for j, k in enumerate(match):
        fk = F[:, k]
        pj = Psi[:, j]
        D2[k] = np.vdot(fk, pj) / np.vdot(fk, fk)
    A = F[:, match] @ np.diag(D2[match])
    D1 = np.zeros(n, dtype=complex)
    for i in range(n):
        a = A[i, :]
        p = Psi[i, :]
        denom = np.vdot(a, a)
        D1[i] = np.vdot(a, p) / denom if abs(denom) > 0 else 0.0
    return np.diag(D1), np.diag(D2)


def diagonality_measure(M):
    return norm(M - np.diag(np.diag(M))) / max(1.0, norm(M))


def sinkhorn_flatten_moduli(Psi, iters=2000, tol=1e-8):
    A = np.abs(Psi).astype(float)
    n = A.shape[0]
    r = np.ones(n)
    c = np.ones(n)
    target = np.mean(A)
    for _ in range(iters):
        row_sums = A @ c
        row_sums[row_sums == 0] = 1.0
        r = target / row_sums
        A = np.diag(r) @ A
        col_sums = A.T @ np.ones(n)
        col_sums[col_sums == 0] = 1.0
        c = target / col_sums
        A = A @ np.diag(c)
        if norm(A - target * np.ones_like(A), ord="fro") / norm(A, ord="fro") < tol:
            return True, r, c
    return False, r, c


def test_rft_not_scaled_permuted_dft():
    # Use CANONICAL True RFT implementation
    import importlib.util
import os
    
    # Load the module
    spec = importlib.util.spec_from_file_location(
        "canonical_true_rft", 
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "04_RFT_ALGORITHMS/canonical_true_rft.py")
    )
    canonical_true_rft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical_true_rft)
    forward_true_rft = canonical_true_rft.forward_true_rft

    for n in (8, 12, 16):
        Psi_builder = build_rft_matrix(forward_true_rft)
        Psi = Psi_builder(n)
        F = unitary_dft(n)
        match = greedy_match_columns(Psi, F)
        D1, D2 = best_diagonal_right_and_left(Psi, F, match)
        approx = D1 @ F[:, match] @ D2
        R = norm(Psi - approx, ord="fro") / max(1.0, norm(Psi, ord="fro"))
        assert R > 1e-3, f"RFT looked too close to scaled/perm'd DFT (residual={R:.2e})"


def test_rft_not_diagonalizing_shift():
    # Use CANONICAL True RFT implementation
    def forward_true_rft(x):
        try:
            import quantonium_core

            rft = quantonium_core.ResonanceFourierTransform(x.tolist())
            result = rft.forward_transform()
            return np.array(result, dtype=complex)
        except:
            import importlib.util
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
forward_true_rft as canonical_rft

            return canonical_rft = canonical_true_rft.forward_true_rft as canonical_rft

            return canonical_rft(x.tolist())

    for n in (8, 12, 16):
        Psi = build_rft_matrix(forward_true_rft)(n)
        S = cyclic_shift(n)
        M = inv(Psi) @ S @ Psi
        off = diagonality_measure(M)
        assert off > 1e-3, f"RFT nearly diagonalized shift (offdiag={off:.2e})"


def test_rft_moduli_not_scalable_to_uniform():
    # Use CANONICAL True RFT implementation
    def forward_true_rft(x):
        try:
            import quantonium_core

            rft = quantonium_core.ResonanceFourierTransform(x.tolist())
            result = rft.forward_transform()
            return np.array(result, dtype=complex)
        except:
            import importlib.util
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
forward_true_rft as canonical_rft

            return canonical_rft = canonical_true_rft.forward_true_rft as canonical_rft

            return canonical_rft(x.tolist())

    for n in (8, 12, 16):
        Psi = build_rft_matrix(forward_true_rft)(n)
        ok, _, _ = sinkhorn_flatten_moduli(Psi, iters=2000, tol=1e-8)
        assert (
            not ok
        ), "RFT magnitudes scaled to uniform too easily; investigate equivalence."


def test_rft_not_diagonalizing_shift():
    import importlib.util
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
forward_true_rft

    for n in = canonical_true_rft.forward_true_rft

    for n in(8, 12, 16):
        Psi = build_rft_matrix(forward_true_rft)(n)
        S = cyclic_shift(n)
        M = inv(Psi) @ S @ Psi
        off = diagonality_measure(M)
        assert off > 1e-3, f"RFT nearly diagonalized shift (offdiag={off:.2e})"


def test_rft_moduli_not_scalable_to_uniform():
    import importlib.util
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
forward_true_rft

    for n in = canonical_true_rft.forward_true_rft

    for n in(8, 12, 16):
        Psi = build_rft_matrix(forward_true_rft)(n)
        ok, _, _ = sinkhorn_flatten_moduli(Psi, iters=2000, tol=1e-8)
        assert (
            not ok
        ), "RFT magnitudes scaled to uniform too easily; investigate equivalence."
