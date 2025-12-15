"""Benchmark: φ-grid exponential RFT with frame correction.

Compares:
- φ-grid RFT (raw, naive inverse)
- φ-grid RFT (raw, frame-correct inverse)
- φ-grid RFT (Gram-normalized, unitary)
- FFT (unitary baseline)

Run:
    python benchmarks/rft_phi_frame_benchmark.py

Notes:
- The φ-grid raw exponential basis is generally non-orthogonal at finite N.
- Frame-correct inversion uses the dual-frame solve: (ΦᴴΦ)^{-1} Φᴴ x.

References (orientation):
- Oppenheim & Schafer, Discrete-Time Signal Processing (orthogonality of DFT basis)
- O. Christensen, An Introduction to Frames and Riesz Bases (dual frames)
- Encyclopaedia Britannica, Fourier analysis
"""

from __future__ import annotations

import time
import numpy as np

from algorithms.rft.core.resonant_fourier_transform import (
    rft_basis_matrix,
    rft_forward_frame,
    rft_inverse_frame,
)
from algorithms.rft.core.gram_utils import gram_matrix


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    rng = np.random.default_rng(0)
    N = 256
    x = rng.normal(size=N) + 1j * rng.normal(size=N)

    # Raw φ-grid basis
    t0 = time.perf_counter()
    Phi_raw = rft_basis_matrix(N, N, use_gram_normalization=False)
    t1 = time.perf_counter()

    # Gram-normalized (unitary) basis
    Phi_unit = rft_basis_matrix(N, N, use_gram_normalization=True)
    t2 = time.perf_counter()

    # Condition number of Gram (raw)
    G = gram_matrix(Phi_raw)
    cond = np.linalg.cond(G)

    # Naive analysis/synthesis (raw)
    X_naive = Phi_raw.conj().T @ x
    x_hat_naive = Phi_raw @ X_naive

    # Frame-correct (raw)
    X_frame = rft_forward_frame(x, Phi_raw)
    x_hat_frame = rft_inverse_frame(X_frame, Phi_raw)

    # Unitary (Gram-normalized)
    X_unit = Phi_unit.conj().T @ x
    x_hat_unit = Phi_unit @ X_unit

    # FFT baseline (unitary)
    X_fft = np.fft.fft(x, norm="ortho")
    x_hat_fft = np.fft.ifft(X_fft, norm="ortho")

    print("=== RFT φ-grid frame benchmark ===")
    print(f"N={N}")
    print(f"build Phi_raw:  {(t1 - t0)*1e3:.2f} ms")
    print(f"build Phi_unit: {(t2 - t1)*1e3:.2f} ms")
    print(f"cond(Gram(Phi_raw)) = {cond:.3e}")
    print()
    print(f"raw/naive reconstruction relerr:  {rel_err(x_hat_naive, x):.3e}")
    print(f"raw/frame reconstruction relerr:  {rel_err(x_hat_frame, x):.3e}")
    print(f"unitary reconstruction relerr:     {rel_err(x_hat_unit, x):.3e}")
    print(f"fft reconstruction relerr:         {rel_err(x_hat_fft, x):.3e}")


if __name__ == "__main__":
    main()
