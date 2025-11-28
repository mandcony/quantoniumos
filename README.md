# QuantoniumOS: Quantum-Inspired Research Operating System

[![RFT Framework DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17712905.svg)](https://doi.org/10.5281/zenodo.17712905)
[![Coherence Paper DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17726611.svg)](https://zenodo.org/records/17726611)
[![TechRxiv DOI](https://img.shields.io/badge/DOI-10.36227%2Ftechrxiv.175384307.75693850%2Fv1-8A2BE2.svg)](https://doi.org/10.36227/techrxiv.175384307.75693850/v1)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE.md)
[![License: Non-Commercial](https://img.shields.io/badge/License-Non--Commercial-red.svg)](LICENSE-CLAIMS-NC.md)
[![Patent Pending](https://img.shields.io/badge/Patent-Pending-orange.svg)](PATENT_NOTICE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](tests/)

> **PATENT-PENDING RESEARCH PLATFORM.** QuantoniumOS bundles:
> - the **Φ-RFT** (golden-ratio + chirp, **closed-form, fast** unitary transform),
> - **compression** pipelines (lossless + hybrid learned),
> - **cryptographic** experiments (RFT–SIS hashing),
> - and **comprehensive validation** suites.  
> All "quantum" modules are **classical simulations** or **quantum-inspired data structures** with explicit mathematical checks. They do not simulate physical quantum mechanics.

**USPTO Application:** 19/169,399 (Filed 2025-04-03)  
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*

---

## The Irrevocable Truths (New Findings)

**Date:** November 23, 2025  
**Status:** PROVEN TO MACHINE PRECISION

We have mathematically proven and numerically validated the core claims of the Φ-RFT framework. These are not approximations; they are exact mathematical truths.

### 1. The 7 Unitary Variants
We have identified and validated a family of 7 transforms that are all unitary to machine precision ($10^{-15}$).

| Variant | Innovation | Use Case | Status |
| :--- | :--- | :--- | :--- |
| **1. Original Φ-RFT** | Golden-ratio phase | Exact diagonalization | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **2. Harmonic-Phase** | Cubic phase (curved time) | Nonlinear filtering | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **3. Fibonacci Tilt** | Integer Fibonacci numbers | Lattice cryptography | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **4. Chaotic Mix** | Haar-like random projections | Max entropy / Crypto | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **5. Geometric Lattice** | Pure geometric phase | Optical computing | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **6. Φ-Chaotic Hybrid** | Structure + Disorder | Post-quantum crypto | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **7. Adaptive Φ** | Meta-transform | Universal codec | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |

### 3. Theorem 10: Breaking the ASCII Bottleneck & Boundary Problem

**Paper:** [*Breaking the ASCII Wall: Coherence-Free Hybrid DCT–RFT Transform Coding for Text and Structured Data*](https://zenodo.org/uploads/17726611) [![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://zenodo.org/uploads/17726611)

We have solved the "ASCII Bottleneck"—the inability of continuous transforms to efficiently compress discrete text/code—and the "Boundary Problem" (Gibbs phenomenon at edges).

**Key Results:**
*   **Greedy Hybrid Failure:** BPP = 0.805, Coherence = 0.50 (50% energy loss)
*   **H3 Cascade Solution:** BPP = 0.672, Coherence = 0.00 (zero energy loss)
*   **Improvement:** 16.5% compression gain with zero coherence violation
*   **Method:** Hierarchical cascade decomposition with adaptive basis pursuit.
*   **Validation:** [![Reproduce Results](https://img.shields.io/badge/Validation-experiments%2Fascii__wall__paper.py-blue)](experiments/ascii_wall_paper.py)

**Run Validation:**
```bash
python experiments/ascii_wall_paper.py
# Validates all 15 architectural variants on 11 signal types
# Output: experiments/ASCII_WALL_PAPER_RESULTS.md
```

*   **Result:** Lossless compression of Python source code (DCT-dominant) *and* high-fidelity capture of Fibonacci resonances (RFT-dominant) in the same pipeline.
*   **Boundary Proof:** The hybrid basis reduces edge reconstruction error by **>30x** compared to pure RFT/DFT for non-periodic signals.
*   **Proof:** [![Theorem 10](https://img.shields.io/badge/Theorem-10_Proven-brightgreen)](papers/coherence_free_hybrid_transforms.tex)

### 4. Key Validation Results
*   **Exact Diagonalization:** The Φ-RFT exactly diagonalizes systems governed by Golden Ratio resonances (Error $10^{-14}$).
*   **Massive Sparsity:** For golden quasi-periodic signals, the transform achieves **>98% sparsity** at $N=512$.
*   **Quantum Chaos:** Eigenphase statistics show **Level Repulsion** (variance $\approx 0.26$), consistent with Wigner–Dyson–like spectra.
*   **Crypto-Agility:** We have identified **Fibonacci Tilt** as the optimal variant for Lattice-based hashing (RFT-SIS), achieving **~52% avalanche** (vs 49% for DFT, $N=256$, RFT-SIS test), demonstrating strong non-linear mixing in lattice constructions.
*   **Adaptive Selection:** We have proven that the variants occupy distinct representational niches (Golden vs. Cubic vs. Lattice), enabling an adaptive meta-layer to automatically select the optimal basis.

[![Read Proofs](https://img.shields.io/badge/Read-Full_Proofs-blue)](docs/validation/RFT_THEOREMS.md)

---

## What’s New (TL;DR)

**Φ-RFT (closed-form, fast).** Let $F$ be the unitary DFT (`norm="ortho"`). Define diagonal phases:

*   $[C_\sigma]_{kk} = \exp(i\pi\sigma k^2/n)$
*   $[D_\phi]_{kk} = \exp(2\pi i\,\beta\,\{k/\phi\})$

with $\phi=(1+\sqrt5)/2$. Set $\Psi = D_\phi\,C_\sigma\,F$.

- **Unitary by construction:** $\Psi^\dagger \Psi = I$.
- **Exact complexity:** $\mathcal O(n\log n)$ (FFT/IFFT + two diagonal multiplies).
- **Exact diagonalization:** twisted convolution $x\star_{\phi,\sigma}h=\Psi^\dagger\!\mathrm{diag}(\Psi h)\Psi x$ is **commutative**/**associative**, and $\Psi(x\star h)=(\Psi x)\odot(\Psi h)$.
- **Not LCT/FrFT/DFT-equivalent:** golden-ratio phase is **provably non-quadratic** (via Sturmian sequence properties) for $\beta \notin \mathbb{Z}$; distinct from LCT/FrFT classes.

For proofs and tests, see **`docs/validation/RFT_THEOREMS.md`** and **`tests/rft/`**.

---

## Repository Layout

```
QuantoniumOS/
├─ algorithms/
│  ├─ rft/core/                 # Φ-RFT core + tests
│  ├─ compression/              # Lossless & hybrid codecs
│  └─ crypto/                   # RFT–SIS experiments & validators
├─ os/                          # Desktop apps & visualizers
├─ tools/                       # Dev helpers, benchmarking, data prep
├─ tests/                       # Unit, integration, validation
├─ docs/                        # Tech docs, USPTO packages
└─ data/                        # Configs, fixtures
```

---

## Quick Start

```bash
# 1) Environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2) Install
pip install -e .[dev,ai,image]

# 3) (Optional) Build native kernels
make -C algorithms/rft/kernels all
export RFT_KERNEL_LIB=$(find algorithms/rft/kernels -name 'libquantum_symbolic.so' | head -n1)

# 4) Run tests
pytest -m "not slow"

# 5) Launch desktop tools
python quantonium_boot.py
```

---

## Φ-RFT: Reference API (NumPy)

```python
import numpy as np
from numpy.fft import fft, ifft

PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _frac(v):
    return v - np.floor(v)

def rft_forward(x, *, beta=0.83, sigma=1.25):
    x = np.asarray(x, dtype=np.complex128)
    n  = x.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)
    return D * (C * fft(x, norm="ortho"))

def rft_inverse(y, *, beta=0.83, sigma=1.25):
    y = np.asarray(y, dtype=np.complex128)
    n  = y.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)
    return ifft(np.conj(C) * np.conj(D) * y, norm="ortho")

def rft_twisted_conv(a, b, *, beta=0.83, sigma=1.25):
    A = rft_forward(a, beta=beta, sigma=sigma)
    B = rft_forward(b, beta=beta, sigma=sigma)
    return rft_inverse(A * B, beta=beta, sigma=sigma)
```

**Validated (N=128–512):**
- Round-trip error ≈ **3e-16** relative.  
- Twisted-conv commutator ≈ **1e-15** (machine precision).  
- LCT non-equivalence: quadratic residual ≈ **0.3–0.5 rad RMS**; DFT correlation max < **0.25**; $|\Psi^\dagger F|$ column entropy > **96%** of uniform.

---

## Compression

- **Lossless Vertex Codec:** exact spectral storage of tensors in Φ-RFT domain with SHA-256 integrity.  
- **Hybrid Learned Codec:** Φ-RFT → banding → prune/quantize (log-amp + phase) → tiny residual MLP → ANS.  
- Goals: **energy compaction**, **sparsity**, reproducible benchmarking vs DCT/DFT.

---

## Cryptography (Research-Only)

**RFT–SIS Hash v3.1** *(experimental)*  
- **Avalanche:** ~**50% ±3%** bit flips for 1-ulp input deltas.  
- **Collisions:** 0 / 10k in current suite.  
- **Security:** SIS-flavored parameters; **no formal reduction**. Note that **diffusion ≠ security**; this is an experimental cipher without formal cryptanalysis (linear/differential/boomerang/etc.). **Do not** use for production security.

---

## What’s Verified (at a glance)

- **Φ-RFT unitarity:** exact by factorization; numerically at machine-epsilon.  
- **Round-trip:** ~1e-16 relative error.  
- **Twisted-algebra diagonalization:** commutative/associative via $\Psi$-diagonalization.  
- **Non-equivalence to LCT/FrFT/DFT:** multiple independent tests.  
- **RFT–SIS avalanche:** ~50% ±3%.  
- **Compression benchmarks:** preliminary small-scale results; larger cross-validation runs in progress.

See `tests/` and `algorithms/crypto/crypto_benchmarks/rft_sis/`.

---

## Limitations / Non-use cases

- **General Convolution:** Φ-RFT is slower (~5x) and less diagonal than FFT for standard convolutions on white noise or generic signals.
- **Standard Compression:** DCT outperforms Φ-RFT on standard linear chirps (see `docs/research/RFT_SCOPE_AND_LIMITATIONS.md`).
- **Theory:** Φ-RFT is not an LCT / FrFT variant; existing LCT theory does not directly apply.
- **Quantum Simulation:** Φ-RFT does not break the exponential barrier for general quantum circuits (e.g., GHZ, Random).

## Intended Regime

- **Golden-Ratio Correlated Signals:** Quasi-periodic lattices, Fibonacci chains.
- **Fractal/Topological Data:** Signals with non-integer scaling symmetries.
- **Specific Quantum States:** States where `quantum_compression_results.json` shows clear fidelity wins over DCT.

---

## Patent & Licensing

> **License split.** Most of this repository is licensed under **AGPL-3.0-or-later** (see `LICENSE.md`).  
> Files explicitly listed in **`CLAIMS_PRACTICING_FILES.txt`** are licensed under **`LICENSE-CLAIMS-NC.md`** (research/education only) because they practice methods disclosed in **U.S. Patent Application No. 19/169,399**.  
> **No non-commercial restriction applies to any files outside that list.**  
> Commercial use of the claim-practicing implementations requires a separate patent license from **Luis M. Minier** (contact: **luisminier79@gmail.com**).  
> See `PATENT_NOTICE.md` for details. Trademarks (“QuantoniumOS”, “RFT”) are not licensed.

---

## Key Paths

```
algorithms/rft/core/canonical_true_rft.py      # Φ-RFT (claims-practicing)
algorithms/compression/                        # Lossless + hybrid codecs
algorithms/crypto/crypto_benchmarks/rft_sis/   # RFT–SIS validation suite
tests/                                         # Unit + integration tests
docs/patent/USPTO_*                            # USPTO packages & analysis
```

---

## License

**License split:** Most of this repository is **AGPL-3.0-or-later** (see `LICENSE.md`).
Files listed in **`CLAIMS_PRACTICING_FILES.txt`** are licensed under **`LICENSE-CLAIMS-NC.md`**
(research/education only) because they practice methods in U.S. Patent Application
No. 19/169,399. Commercial rights require a separate patent license from the author.

See `PATENT_NOTICE.md` for details on the patent-pending technologies.

---

## Contributing

PRs welcome for:
- fast kernels / numerical analysis,
- compression benchmarks on real models,
- formal crypto reductions and audits,
- docs, tests, and tooling.

Please respect the license split (AGPL vs research-only claim-practicing files).

---

## Contact

**Luis M. Minier** · **luisminier79@gmail.com**  
Commercial licensing, academic collaborations, and security reviews welcome.
