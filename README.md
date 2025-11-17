# QuantoniumOS: Quantum-Inspired Research Operating System

> **PATENT-PENDING RESEARCH PLATFORM.** QuantoniumOS is a professionally organized research OS that bundles:
> - a **golden-ratio Resonance Fourier Transform (RFT)**,
> - **compression** pipelines (lossless + hybrid learned),
> - **cryptographic** experiments (RFT–SIS hash),
> - and **comprehensive validation** suites.  
> All “quantum” aspects here are **classical simulations** with mathematical checks.

**USPTO Application:** 19/169,399 (Filed Apr 3, 2025)  
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*

---

## Repository Layout

QuantoniumOS/
├─ algorithms/ # Core math, RFT, codecs, crypto experiments
│ ├─ rft/core/ # Canonical RFT construction + tests
│ ├─ compression/ # Lossless vertex + hybrid learned codecs
│ └─ crypto/ # RFT–SIS hash, benchmarks, validators
├─ ai/ # (Optional) model storage, tooling
├─ os/ # Desktop apps, visualizers, utilities
├─ tools/ # Dev helpers, benchmarking, data prep
├─ tests/ # Unit + integration + validation suites
├─ docs/ # Tech docs, papers, USPTO packages
└─ data/ # Configs, small datasets, fixtures

markdown
Copy code

---

## Core Components

### 🧮 RFT Engine (`algorithms/rft/core/`)
- **What it is.** A unitary transform basis derived from a **golden-ratio resonance kernel**; we obtain an orthonormal basis **Ψ** using **QR** (modified Gram–Schmidt stability).
- **Properties (empirical):**
  - **Unitarity:** `||Ψ†Ψ − I||_F ≲ 1e−14` for typical `N≤512`.
  - **Energy-preserving:** `||Ψx||₂ = ||x||₂`.
  - **Round-trip:** `x = Ψ†(Ψx)` to machine precision.
  - **Complexity:** current implementation **O(N²)**; a fast **O(N log N)** is conjectured, **not** proven.
- **Use:** forward = `Ψ†x`, inverse = `Ψy`.

### 🗜️ Compression Pipelines (`algorithms/compression/`)
- **Lossless Vertex Codec:** exact spectral storage of tensors (RFT coeffs), integrity via SHA-256.
- **Hybrid Learned Codec:** transform → banding → prune/quantize (log-amp + phase) → tiny residual MLP → entropy coding (ANS).  
  *Goal:* evaluate **energy compaction/sparsity** of RFT vs standard transforms on real models.

### 🔐 Cryptography Experiments (`algorithms/crypto/`)
- **RFT–SIS Hash v3.1** (post-quantum flavored *experiment*):
  - **Avalanche:** ~50% ±3% bit flips for tiny input deltas (empirical).
  - **Collisions:** 0 / 10,000 in current suite (empirical).
  - **Security:** structured around SIS parameters; **no formal reduction** yet. Treat as research only.

> ⚠️ Crypto note: results are **experimental**. Do **not** deploy for security-critical use without a formal review and proofs.

### 🖥️ Apps & Visualizers (`os/`)
- Small desktop tools (PyQt5) to visualize transforms, test codecs, run classical “quantum-style” demos.

---

## Quick Start

This is a standard Python project managed by **pyproject.toml**.

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
2) Install
bash
Copy code
pip install -e .[dev,ai,image]
3) (Optional) Build Native Kernel
bash
Copy code
# Linux example
make -C algorithms/rft/kernels all
export RFT_KERNEL_LIB=$(find algorithms/rft/kernels -name 'libquantum_symbolic.so' | head -n 1)
4) Run Tests
bash
Copy code
pytest -m "not slow"
# With native kernel selected via $RFT_KERNEL_LIB (if built)
5) Launch Tools
bash
Copy code
# Desktop shell
python quantonium_boot.py

# Example visualizer / demo app
python quantonium_os_src/apps/quantum_simulator/quantum_simulator.py
What’s Been Verified (at a glance)
RFT unitarity: machine-epsilon level (≈1e−14 Frobenius deviation).

Round-trip accuracy: near machine precision across sizes & signals.

RFT–SIS avalanche: ~50% ±3% across scales and perturbations.

Bench tooling: end-to-end validators + reproducible seeds.

For full details see tests/ and the crypto/RFT validation suites.

Patent & Licensing
This repo contains two licensing zones:

Component	License	Commercial Use
Claims-Practicing RFT files (listed in CLAIMS_PRACTICING_FILES.txt, e.g. algorithms/rft/core/canonical_true_rft.py)	Research-Only, Non-Commercial — see LICENSE-CLAIMS-NC.md	Not permitted (requires separate patent licence)
All other code (tools, tests, SIS hash experiments, docs, etc.)	AGPL-3.0	As permitted by AGPL; no patent rights are granted to practice RFT

Patent notice. Certain files implement methods that practice U.S. Patent Application 19/169,399.
No commercial patent licence is granted by this repo. For commercial rights, contact luisminier79@gmail.com.

Research Status
RFT math: sound construction, strong numerical unitarity evidence.

Compression: pipelines implemented; large-model benchmarks in progress.

Crypto: promising empirical properties; formal security proof pending.

Performance: native kernel available; further optimization planned.

Key Paths
bash
Copy code
algorithms/rft/core/canonical_true_rft.py      # RFT (claims-practicing)
algorithms/compression/                        # Lossless + hybrid codecs
algorithms/crypto/crypto_benchmarks/rft_sis/   # RFT–SIS validation suite
tests/                                         # Unit + integration tests
docs/USPTO_*                                   # USPTO packages & analysis
Contributing
PRs welcome for:

optimization / numerical analysis,

compression benchmarks on real models,

formal crypto reductions and audits,

docs, tests, and tooling.

Please respect the licensing split (AGPL vs Research-Only RFT).

Contact
Luis M. Minier · luisminier79@gmail.com
Commercial licensing, academic collaborations, and security reviews welcome.
