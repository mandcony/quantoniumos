
# QuantoniumOS: Quantum-Inspired Research Operating System

> **PATENT-PENDING RESEARCH PLATFORM.** QuantoniumOS bundles:
> - the **Φ-RFT** (golden-ratio + chirp, **closed-form, fast** unitary transform),
> - **compression** pipelines (lossless + hybrid learned),
> - **cryptographic** experiments (RFT–SIS hashing),
> - and **comprehensive validation** suites.  
> All “quantum” modules are **classical simulations** with explicit mathematical checks.

**USPTO Application:** 19/169,399 (Filed 2025-04-03)  
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*

---

## What’s New (TL;DR)

**Φ-RFT (closed-form, fast).** Let \(F\) be the unitary DFT (`norm="ortho"`). Define diagonal phases  
\([C_\sigma]_{kk}=\exp(i\pi\sigma k^2/n)\), \([D_\phi]_{kk}=\exp(2\pi i\,\beta\,\{k/\phi\})\) with \(\phi=(1+\sqrt5)/2\).  
Set **\(\Psi = D_\phi\,C_\sigma\,F\)**.

- **Unitary by construction:** \(\Psi^\dagger \Psi = I\).
- **Exact complexity:** **\(\mathcal O(n\log n)\)** (FFT/IFFT + two diagonal multiplies).
- **Exact diagonalization:** twisted convolution \(x\star_{\phi,\sigma}h=\Psi^\dagger\!\operatorname{diag}(\Psi h)\Psi x\) is **commutative**/**associative**, and \(\Psi(x\star h)=(\Psi x)\odot(\Psi h)\).
- **Not LCT/FrFT/DFT-equivalent:** golden-ratio phase is **non-quadratic**; quadratic-fit residual \(\gg\) machine epsilon; DFT-correlation low; \(|\Psi^\dagger F|\) columns high-entropy.

For proofs and tests, see **`docs/RFT_THEOREMS.md`** and **`tests/rft/`**.

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

````

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
````

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
    return ifft(np.conj(C) * np.conj(D) * y, norm="ortho"))

def rft_twisted_conv(a, b, *, beta=0.83, sigma=1.25):
    A = rft_forward(a, beta=beta, sigma=sigma)
    B = rft_forward(b, beta=beta, sigma=sigma)
    return rft_inverse(A * B, beta=beta, sigma=sigma)
```

**Validated (N=128–512):**

* Round-trip error ≈ **3e-16** relative.
* Twisted-conv commutator ≈ **1e-15** (machine precision).
* LCT non-equivalence: quadratic residual ≈ **0.3–0.5 rad RMS**; DFT correlation max < **0.25**; (|\Psi^\dagger F|) column entropy > **96%** of uniform.

---

## Compression

* **Lossless Vertex Codec:** exact spectral storage of tensors in Φ-RFT domain with SHA-256 integrity.
* **Hybrid Learned Codec:** Φ-RFT → banding → prune/quantize (log-amp + phase) → tiny residual MLP → ANS.
* Goals: **energy compaction**, **sparsity**, reproducible benchmarking vs DCT/DFT.

---

## Cryptography (Research-Only)

**RFT–SIS Hash v3.1** *(experimental)*

* **Avalanche:** ~**50% ±3%** bit flips for 1-ulp input deltas.
* **Collisions:** 0 / 10k in current suite.
* **Security:** SIS-flavored parameters; **no formal reduction**.
  **Do not** use for production security without independent review.

---

## What’s Verified (at a glance)

* ✅ **Φ-RFT unitarity:** exact by factorization; numerically at machine-epsilon.
* ✅ **Round-trip:** ~1e-16 relative error.
* ✅ **Twisted-algebra diagonalization:** commutative/associative via (\Psi)-diagonalization.
* ✅ **Non-equivalence to LCT/FrFT/DFT:** multiple independent tests.
* ✅ **RFT–SIS avalanche:** ~50% ±3%.
* 🔬 **Compression benchmarks:** larger-scale runs in progress.

See `tests/` and `algorithms/crypto/crypto_benchmarks/rft_sis/`.

---

## Patent & Licensing

> **License split.** Most of this repository is licensed under **AGPL-3.0-or-later** (see `LICENSE.md`).
> Files explicitly listed in **`CLAIMS_PRACTICING_FILES.txt`** are licensed under **`LICENSE-CLAIMS-NC.md`** (research/education only) because they practice methods disclosed in **U.S. Patent Application No. 19/169,399**.
> **No non-commercial restriction applies to any files outside that list.**
> Commercial use of the claim-practicing implementations requires a separate patent license from **Luis M. Minier** (contact: **[luisminier79@gmail.com](mailto:luisminier79@gmail.com)**).
> See `PATENT_NOTICE.md` for details. Trademarks (“QuantoniumOS”, “RFT”) are not licensed.

---

## Key Paths

```
algorithms/rft/core/canonical_true_rft.py      # Φ-RFT (claims-practicing)
algorithms/compression/                        # Lossless + hybrid codecs
algorithms/crypto/crypto_benchmarks/rft_sis/   # RFT–SIS validation suite
tests/                                         # Unit + integration tests
docs/USPTO_*                                   # USPTO packages & analysis
```

---

## Contributing

PRs welcome for:

* fast kernels / numerical analysis,
* compression benchmarks on real models,
* formal crypto reductions and audits,
* docs, tests, and tooling.

Please respect the license split (AGPL vs research-only claim-practicing files).

---

## Contact

**Luis M. Minier** · **[luisminier79@gmail.com](mailto:luisminier79@gmail.com)**
Commercial licensing, academic collaborations, and security reviews welcome.

```
::contentReference[oaicite:0]{index=0}
```
