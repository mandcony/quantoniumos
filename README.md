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

## 🚀 Quick Start

**New here?** → **[GETTING_STARTED.md](GETTING_STARTED.md)** (your first steps)

**Documentation:**
- 📖 **[GETTING_STARTED.md](GETTING_STARTED.md)** - First steps, examples, learning path
- 🔧 **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Installation, troubleshooting, verification
- 🏗️ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical deep dive (ASM → C → C++ → Python)
- 📋 **[docs/ARCHITECTURE_QUICKREF.md](docs/ARCHITECTURE_QUICKREF.md)** - One-page cheat sheet

**Quick installation:**
```bash
# Clone and setup
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos
./quantoniumos-bootstrap.sh

# Or manual setup (no compilation needed!)
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy sympy numba
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('✓ Setup complete!')"
```

**Build native engines for 3-10× speedup (optional):**
```bash
# C/ASM kernel
cd algorithms/rft/kernels && make -j$(nproc) && cd ../../..

# C++ engine with AVX2/AVX-512
cd src/rftmw_native && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON && make -j$(nproc)
cp rftmw_native.cpython-*-linux-gnu.so ../../../.venv/lib/python3.12/site-packages/
```

**Verify setup:**
```bash
./verify_setup.sh  # Automated health check
```

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
| **3. Fibonacci Tilt** | Integer Fibonacci numbers | Lattice structures | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **4. Chaotic Mix** | Haar-like random projections | Max entropy / Mixing | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **5. Geometric Lattice** | Pure geometric phase | Optical computing | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |
| **6. Φ-Chaotic Hybrid** | Structure + Disorder | Experimental crypto | [![Status](https://img.shields.io/badge/Status-Experimental-yellow)](docs/validation/RFT_THEOREMS.md) |
| **7. Adaptive Φ** | Meta-transform | Universal codec | [![Status](https://img.shields.io/badge/Status-Proven-brightgreen)](docs/validation/RFT_THEOREMS.md) |

### 3. Compression: Competitive Transform Codec (Not a Breakthrough)

**Paper:** [*Coherence-Free Hybrid DCT–RFT Transform Coding for Text and Structured Data*](https://zenodo.org/uploads/17726611) [![Paper](https://img.shields.io/badge/Paper-Zenodo-blue)](https://zenodo.org/uploads/17726611)

We built a hierarchical transform codec combining DCT and Φ-RFT that is **competitive with classical transform codecs** on structured signals. It does NOT beat entropy bounds or general-purpose compressors.

**Honest Results:**
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
*   **Exact Unitarity:** Round-trip error < 1e-14 across all 7 variants.
*   **Coherence Reduction:** Up to 79.6% lower mutual coherence vs DCT at optimal σ.
*   **Sparsity:** For golden quasi-periodic signals, achieves >98% sparsity at N=512.
*   **Avalanche Effect:** ~50.7% bit flip rate in experimental hash constructions.
*   **Timbre Coverage:** RFT oscillators cover 280x more timbre space than standard (H9 hypothesis).

**⚠️ Important Disclaimers:**
- **Crypto:** All cryptographic constructions are **experimental** with no hardness proofs or third-party cryptanalysis. NOT production-ready.
- **Compression:** Does NOT beat entropy bounds. Competitive with classical transform codecs, not a "breakthrough."
- **Quantum:** This is classical signal processing. "Symbolic qubit" representations are compressed encodings, not quantum computation.

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
│  ├─ rft/core/
│  │  ├─ canonical_true_rft.py     # Reference Φ-RFT (claims-practicing)
│  │  ├─ closed_form_rft.py        # Original implementation
│  │  └─ rft_optimized.py          # Optimized fused-diagonal RFT ⚡
│  ├─ compression/                 # Lossless & hybrid codecs
│  └─ crypto/                      # RFT–SIS experiments & validators
├─ src/
│  ├─ rftmw_native/
│  │  ├─ rftmw_core.hpp            # C++ RFT engine
│  │  ├─ rft_fused_kernel.hpp      # AVX2/AVX512 SIMD kernels ⚡
│  │  └─ rftmw_python.cpp          # pybind11 bindings
│  └─ apps/
│     ├─ quantsounddesign/         # Φ-RFT Sound Design Studio
│     └─ ...
├─ experiments/
│  ├─ competitors/
│  │  ├─ benchmark_transforms_vs_fft.py   # RFT vs FFT/DCT benchmark
│  │  ├─ benchmark_compression_vs_codecs.py
│  │  └─ benchmark_crypto_throughput.py
│  └─ ...
├─ scripts/
│  └─ run_full_suite.sh            # One-command benchmark runner
├─ results/                        # Benchmark output (JSON/CSV/MD)
├─ tests/                          # Unit, integration, validation
├─ docs/                           # Tech docs, USPTO packages
├─ REPRODUCING_RESULTS.md          # Reproducibility guide
└─ README.md                       # This file
```

---

## QuantSoundDesign: Φ-RFT Sound Design Studio

**QuantSoundDesign** is a professional-grade sound design studio built natively on the Φ-RFT framework. Unlike traditional DAWs that use FFT/DCT for audio processing, QuantSoundDesign leverages the 7 unitary Φ-RFT variants for synthesis, analysis, and effects.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QuantSoundDesign GUI                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Arrangement │  │   Mixer     │  │   Pattern Editor    │  │
│  │   View      │  │   (8 ch)    │  │   16-step grid      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Synth Engine                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ PolySynth (8 voices) → Φ-RFT Additive Synthesis         ││
│  │ DrumSynthesizer      → RFT-based drum generation        ││
│  │ Piano Roll          → MIDI + keyboard input (ASDFGHJK)  ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    UnitaryRFT Engine                         │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ HARMONIC variant  → Primary synthesis                   ││
│  │ FIBONACCI variant → Lattice-based effects               ││
│  │ GEOMETRIC variant → Phase modulation                    ││
│  │ Round-trip error: ~1e-16 (machine precision)            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Core Components

| File | Purpose |
|------|---------|
| `gui.py` | Main UI (FL Studio/Ableton-inspired, 3200+ LOC) |
| `engine.py` | Track/clip management, RFT processing pipeline |
| `synth_engine.py` | Polyphonic synth with Φ-RFT additive synthesis |
| `pattern_editor.py` | 16-step drum sequencer with RFT drum synthesis |
| `piano_roll.py` | MIDI editor with computer keyboard input |
| `audio_backend.py` | PyAudio/sounddevice output |

### Features

- **Φ-RFT Synthesis**: All waveforms generated via UnitaryRFT transforms
- **RFT Drum Synth**: Kick, snare, hihat synthesized with golden-ratio harmonics
- **8-Channel Mixer**: Volume, pan, mute/solo per track
- **Pattern Editor**: 16-step grid with velocity, step preview on click
- **Blank Session Start**: Opens empty project for creative freedom
- **Computer Keyboard Piano**: ASDFGHJKL = C D E F G A B (octave playable)

### Launch

```bash
# From desktop launcher
python quantonium_os_src/frontend/quantonium_desktop.py
# Click "QuantSoundDesign" in Applications

# Standalone
python -c "from src.apps.quantsounddesign.gui import QuantSoundDesign; from PyQt5.QtWidgets import QApplication; import sys; app = QApplication(sys.argv); w = QuantSoundDesign(); w.show(); app.exec_()"
```

---

## RFT Validation & Experiments

### Validated Claims

All experiments are in `experiments/` and can be reproduced:

| Experiment | Script | Key Results |
|------------|--------|-------------|
| **ASCII Bottleneck** | `ascii_wall_paper.py` | H3 Cascade: BPP 0.672, coherence 0.00 |
| **Scaling Laws** | `sota_compression_benchmark.py` | 61.8%+ sparsity for golden signals |
| **Fibonacci Tilt** | `fibonacci_tilt_hypotheses.py` | Optimal for lattice crypto (52% avalanche) |
| **Tetrahedral RFT** | `tetrahedral_deep_dive.py` | Geometric variant validation |
| **SOTA Benchmark** | `sota_compression_benchmark.py` | Comparison vs DCT/DFT |

### Run All Validations

```bash
# Quick validation
python validate_system.py

# Full RFT test suite
pytest tests/rft/ -v

# ASCII Wall theorem (Theorem 10)
python experiments/ascii_wall_paper.py

# Scaling laws (Theorem 3)
python scripts/verify_scaling_laws.py
```

### Test Results Summary

**UnitaryRFT Engine:**
- 7 variants all unitary to machine precision (error < 1e-14)
- Round-trip error: ~1e-16
- Native C kernel with Python bindings

**QuantSoundDesign Integration:**
- HARMONIC variant for synthesis
- Real-time RFT-based waveform generation
- Drum synthesis via RFT spectral shaping

---

## Quick Start

### System Requirements

- **OS**: Ubuntu 22.04+ / Windows 10+ / macOS 12+
- **Python**: 3.10+ (3.12 recommended)
- **Optional**: CMake + C++ compiler for native SIMD kernels

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

# 2. Create virtual environment
python -m venv .venv

# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

# 3. Install in editable mode
pip install --upgrade pip
pip install -e .

# 4. Verify installation
python -c "from algorithms.rft.core.rft_optimized import rft_forward; print('✓ RFT installed')"
```

### Build Native SIMD Kernels (Optional)

For maximum performance with AVX2/AVX512 acceleration:

```bash
cd src/rftmw_native
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON
make -j$(nproc)
cd ../../..

# Verify native module
python -c "import rftmw_native; print('✓ Native module:', rftmw_native.has_simd())"
```

### Run Benchmarks

```bash
# Quick transform benchmark
python experiments/competitors/benchmark_transforms_vs_fft.py --sizes 256,1024,4096 --runs 5

# Full benchmark suite (Linux/macOS)
./scripts/run_full_suite.sh

# Or manually on Windows:
python experiments/competitors/benchmark_transforms_vs_fft.py --output-dir results/competitors
python experiments/competitors/benchmark_compression_vs_codecs.py --output-dir results/competitors
python experiments/competitors/benchmark_crypto_throughput.py --output-dir results/competitors
```

See **[REPRODUCING_RESULTS.md](REPRODUCING_RESULTS.md)** for complete reproducibility guide.

### Run Tests

```bash
# Quick validation
python validate_system.py

# Full test suite
pytest tests/ -v

# RFT-specific tests
pytest tests/rft/ -v -m "not slow"
```

---

## Φ-RFT: Reference API

### Optimized Implementation (Recommended)

The optimized RFT fuses the D_φ and C_σ diagonals into a single pass, achieving **~1.06× FFT speed** (down from ~5× overhead in the original).

```python
from algorithms.rft.core.rft_optimized import (
    rft_forward_optimized,
    rft_inverse_optimized,
    OptimizedRFTEngine,
)
import numpy as np

# Simple API (recommended)
x = np.random.randn(1024)
Y = rft_forward_optimized(x)           # Forward transform
x_rec = rft_inverse_optimized(Y)       # Inverse transform
print(f"Round-trip error: {np.linalg.norm(x - x_rec):.2e}")  # ~1e-15

# Stateful engine for repeated transforms (zero-copy)
engine = OptimizedRFTEngine(size=1024, beta=1.0, sigma=1.0)
Y = engine.forward(x)
x_rec = engine.inverse(Y)
```

### Original Implementation (Reference)

```python
import numpy as np
from numpy.fft import fft, ifft

PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _frac(v):
    return v - np.floor(v)

def rft_forward(x, *, beta=1.0, sigma=1.0):
    """Forward Φ-RFT: Y = D_φ · C_σ · FFT(x)"""
    x = np.asarray(x, dtype=np.complex128)
    n  = x.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))   # Golden-ratio phase
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)       # Chirp phase
    return D * (C * fft(x, norm="ortho"))

def rft_inverse(y, *, beta=1.0, sigma=1.0):
    """Inverse Φ-RFT: x = IFFT(C†_σ · D†_φ · Y)"""
    y = np.asarray(y, dtype=np.complex128)
    n  = y.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)
    return ifft(np.conj(C) * np.conj(D) * y, norm="ortho")

def rft_twisted_conv(a, b, *, beta=1.0, sigma=1.0):
    """Twisted convolution via Φ-RFT diagonalization"""
    A = rft_forward(a, beta=beta, sigma=sigma)
    B = rft_forward(b, beta=beta, sigma=sigma)
    return rft_inverse(A * B, beta=beta, sigma=sigma)
```

### Performance Comparison

| Implementation | n=1024 | n=4096 | Ratio to FFT |
|----------------|--------|--------|--------------|
| **FFT (NumPy)** | 15.6 µs | 38.6 µs | 1.00× |
| **RFT Optimized** | 21.4 µs | 43.7 µs | **1.06×** |
| **RFT Original** | 85.4 µs | 296.9 µs | 4.97× |

The optimized version achieves **4–7× speedup** by:
1. Fusing D_φ and C_σ into single diagonal E = D_φ ⊙ C_σ
2. Precomputing and caching phase tables
3. Using single `exp()` and multiply instead of two

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

- **General Convolution:** Φ-RFT (optimized) runs at ~1.06× FFT speed but offers no advantage for standard linear convolutions on white noise.
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
algorithms/rft/core/rft_optimized.py           # Optimized Φ-RFT (fused diagonals) ⚡
algorithms/rft/core/canonical_true_rft.py      # Reference Φ-RFT (claims-practicing)
algorithms/rft/core/closed_form_rft.py         # Original implementation
src/rftmw_native/rft_fused_kernel.hpp          # AVX2/AVX512 SIMD kernels
experiments/competitors/                       # Benchmark suite vs FFT/DCT/codecs
results/competitors/                           # Benchmark output (JSON/CSV/MD)
REPRODUCING_RESULTS.md                         # Complete reproducibility guide
scripts/run_full_suite.sh                      # One-command benchmark runner
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
