# QuantoniumOS

A hybrid research platform for symbolic resonance computing and high-performance Enhanced RFT Crypto v2.

**Focus areas**: unitary Resonance Fourier Transform (RFT), geometric/manifold mappings, and a C++ Feistel engine with strong diffusion and an authenticated Python wrapper.

## About

QuantoniumOS explores the intersection of resonance-based transforms and cryptographic diffusion. It includes:

- **Symbolic RFT (unitary)** — exact reconstruction (‖x − ΨΨ†x‖₂ ≪ 1e-12) with eigen-decomposition of a resonance kernel.

- **Enhanced RFT Crypto v2** — 48-round Feistel (C++) with AES-S-box, MixColumns-like diffusion, ARX, and domain-separated key schedule. Python wrapper adds salt + HKDF + HMAC for randomized, authenticated encryption.

- **Geometric hashing** — golden-ratio manifold mappings for research on coordinate/topological descriptors.

- **Validation suite** — reproducible tests for unitary accuracy, avalanche metrics, AEAD-style behavior (via wrapper), and performance.

## Current Results (as shipped)

- **Reversibility**: ✓ (all vectors)
- **Diffusion metrics**: key-avalanche ≈ 0.527, key sensitivity ≈ 0.495, message-avalanche ≈ 0.438
- **Performance (engine-only)**: ~9.2 MB/s on small buffers (faster on larger, batched inputs)
- **Wrapper security**: randomized ciphertexts (salt), HKDF-SHA256 key split, HMAC-SHA256 tag (truncated), header versioning

⚠️ **Security note (important)**: This is a research implementation. The Feistel engine and wrapper provide good diffusion and authenticated, randomized encryption for experiments, but the system has not undergone formal cryptanalysis or external security review. Do not use for production data without independent assessment.

## Patent Application (disclosure)

- **Application No.**: 19/169,399 · **Filed**: 2025-04-03
- **Title**: Hybrid Computational Framework for Quantum and Resonance Simulation  
- **Inventor**: Luis Michael Minier

The repository demonstrates techniques related to the application claims (symbolic RFT, resonance-based crypto, geometric hashing, hybrid integration). No legal advice; filing does not imply grant. See "License & Patent Status" below for usage terms.

## Quick Start

```bash
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

# Build C++ engine (pybind11)
c++ -O3 -march=native -flto -DNDEBUG -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) enhanced_rft_crypto_bindings_v2.cpp \
  -o enhanced_rft_crypto$(python3-config --extension-suffix)

# Comprehensive validation (performance, IND-CPA-like behavior via salt, auth, avalanche)
python3 test_v2_comprehensive.py
```

**Python usage** (authenticated, randomized encryption):

```python
from wrappers.enhanced_v2_wrapper import FixedRFTCryptoV2
import secrets

k = secrets.token_bytes(32)
m = b"Confidential data"
crypto = FixedRFTCryptoV2()

ct1 = crypto.encrypt(m, k)
ct2 = crypto.encrypt(m, k)
assert ct1 != ct2  # randomized due to per-message salt

pt = crypto.decrypt(ct1, k)
assert pt == m
```

## Components

| Component | Path | Notes |
|-----------|------|-------|
| **Symbolic RFT (unitary)** | `canonical_true_rft.py` | Exact reconstruction; validation scripts included |
| **Enhanced RFT Crypto v2 (C++)** | `enhanced_rft_crypto.cpp` | 48-round Feistel with S-box, MixColumns-like diffusion, ARX |
| **Python AEAD-style wrapper** | `wrappers/enhanced_v2_wrapper.py` | Salt + HKDF-SHA256 (enc/mac) + HMAC-SHA256 tag + length header |
| **Tests & Benchmarks** | `test_v2_comprehensive.py`, `test_final_v2.py` | Engine-only vs wrapper perf, avalanche, auth/tamper checks |
| **Math background** | `MATHEMATICAL_JUSTIFICATION.md` | Resonance kernel, eigendecomposition, properties |

## License & Status

- **Research code**: open for academic/educational use under the repository license.
- **Patent application**: see details above. Commercial use of covered claims may require a license; contact the inventor.
- **Security caveat**: provided "as is," without warranty; external review recommended before any production deployment.