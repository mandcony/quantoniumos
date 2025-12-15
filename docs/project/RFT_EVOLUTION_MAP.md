# RFT Evolution Map (How We Arrived Here)

This document is a **high-level map** of the major RFT lines in this repository, why changes were made, and which artifacts provide verification.

The guiding principle throughout: **keep claims defensible** by separating

- mathematically provable statements (unitarity, invertibility),
- empirically measured statements (compression / sparsity / leakage), and
- hypothesis statements (domain advantages on particular signal classes).

## Current “defensible core” (as of Dec 15, 2025)

### A) φ-grid exponential RFT (frame-correct / Gram-normalized)

- Construct a deterministic irrational-frequency exponential dictionary (φ-grid).
- Acknowledge finite-$N$ non-orthogonality.
- Make it mathematically complete via:
  - dual-frame coefficients: $X=(\Phi^H\Phi)^{-1}\Phi^H x$
  - Gram-normalized unitary basis: $\widetilde{\Phi}=\Phi(\Phi^H\Phi)^{-1/2}$

**Where:**
- Implementation: `algorithms/rft/core/resonant_fourier_transform.py`
- Gram math: `algorithms/rft/core/gram_utils.py`
- Legacy lock (pre-correction): `algorithms/rft/core/rft_phi_legacy.py`

**Verified by:**
- `tests/validation/test_phi_frame_normalization.py`
- `benchmarks/rft_phi_frame_benchmark.py`
- Benchmark ledger: `docs/research/benchmarks/VERIFIED_BENCHMARKS.md`

### B) φ-phase FFT (closed-form, hardware-ready)

- A **strictly unitary** transform implemented as diagonal phase operators around the FFT.
- Complexity: $O(N\log N)$ via FFT backbone.

**Where:**
- Implementation: `algorithms/rft/core/phi_phase_fft.py`
- “Canonical true” wrapper used by some harnesses: `algorithms/rft/core/canonical_true_rft.py`

**Verified by:**
- existing unitary/round-trip tests in the suite
- comparison harness: `tools/benchmarking/rft_vs_fft_benchmark.py` (selectable via `--rft-impl`)

## Timeline (high level)

### 1) Early eigenbasis / resonance-operator RFT (Toeplitz/Hermitian line)

- RFT described as eigenvectors of a Hermitian “resonance operator” $K_\phi$.
- This guarantees orthonormality by the spectral theorem (when constructed that way).

**Artifacts:**
- `docs/proofs/RFT_FORMAL_PROOFS.txt`

### 2) Shift to canonical “bits → waves” framing

- Canonical basis expressed directly as deterministic exponentials.
- Goal: make the transform definition explicit in waveform terms.

### 3) Discovery: finite-$N$ irrational exponentials are not orthogonal

- Core critique: φ-spaced exponentials are **not** orthogonal for finite $N$.
- Consequence: correlation inverse ($\Phi^H$) is not generally correct.

### 4) Frame-theoretic correction implemented

- Introduced Gram normalization and dual-frame inversion.
- Preserved backward compatibility by locking the legacy implementation.

**Artifacts:**
- Theory: `docs/theory/RFT_THEORY.md`, `docs/theory/RFT_FRAME_NORMALIZATION.md`
- Validation: `tests/validation/test_phi_frame_normalization.py`

### 5) Benchmark alignment (avoid “testing the wrong kernel”)

- Updated CSV harness to explicitly benchmark the corrected φ-frame implementation by default.

**Artifact:**
- `tools/benchmarking/rft_vs_fft_benchmark.py`

## Reproducibility checkpoints

- Finite-$N$ frame-normalized unitarity: `pytest tests/validation/test_phi_frame_normalization.py`
- Large-$N$ coherence diagnostics (raw basis):
  - `python benchmarks/rft_phi_frame_asymptotics.py --sizes 256,512,1024,2048,4096`
- Large-$N$ real-data coefficient statistics (ECG):
  - `USE_REAL_DATA=1 python benchmarks/rft_phi_nudft_realdata_eval.py --ecg --N 4096`

## Scope/limitations (explicit)

- Gram normalization as implemented is a dense $N\times N$ eigendecomposition; it is not intended for very large $N$ without further algorithmic work.
- Large-$N$ evaluations therefore use NUDFT analysis and coherence statistics rather than full Gram-normalized synthesis.
- Domain advantage is not assumed; it is treated as empirical and signal-dependent.
