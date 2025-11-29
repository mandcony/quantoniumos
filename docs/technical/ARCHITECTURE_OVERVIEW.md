# QuantoniumOS Architecture Overview

## Executive Summary

QuantoniumOS is a **signal processing framework** built around the Φ-RFT (Golden Ratio Recursive Fourier Transform) algorithm. It provides:

1. **RFT Mathematical Kernel**: O(N log N) unitary transform with golden ratio parameterization
2. **Cryptographic System**: 48-round Feistel cipher (experimental, no security proofs)
3. **Desktop Environment**: PyQt5-based application suite

> **IMPORTANT SCOPE LIMITATIONS:**
> - The "symbolic quantum engine" operates on **structured, separable states only**
> - O(N) scaling for that family is trivially expected, not a breakthrough
> - This is classical signal processing, NOT general quantum simulation
> - The crypto is experimental research, NOT production-ready

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Application Layer                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │  Q-Notes    │ │   Q-Vault   │ │ QuantSoundDesign    │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│              Desktop Environment                        │
│         quantonium_desktop.py (PyQt5)                   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                Core Algorithms                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │
│  │ unitary_rft │ │ enhanced_   │ │ Structured state    │ │
│  │ (O(N log N))│ │ rft_crypto  │ │ compression         │ │
│  └─────────────┘ └─────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Core Components

### 1. RFT Mathematical Kernel

**What it is:**
- A unitary transform with golden ratio (φ) parameterization
- Complexity: O(N log N) time, O(N) space
- Mathematically proven unitary (U†U = I)
- Well-validated round-trip accuracy

**What it is NOT:**
- NOT O(N) (despite some old documentation claiming this)
- NOT a replacement for FFT in all cases
- NOT "breaking physics" or "quantum computing"

**Implementation:**
```python
class UnitaryRFT:
    """
    Golden ratio parameterized unitary transform.
    Complexity: O(N log N) time, O(N) space.
    """
    def forward(self, x): ...   # Returns spectrum
    def inverse(self, X): ...   # Reconstructs signal
    # Unitarity: ||forward(x)|| = ||x||
```

### 2. Cryptographic System

**What it is:**
- 48-round Feistel cipher with RFT-derived components
- Experimental research into φ-structured cryptography
- Authenticated encryption mode (AEAD)

**What it is NOT:**
- NOT proven secure (no formal hardness analysis)
- NOT production-ready
- NOT "post-quantum" (no hardness reduction to lattice problems)

**Status:** Experimental playground for cryptographic research.

### 3. Structured State Compression

**What it is:**
- Classical O(N) algorithm for a specific family of structured states
- Useful for φ-RFT signal processing applications

**What it is NOT:**
- NOT general quantum simulation
- NOT simulating arbitrary entangled circuits
- NOT "million-qubit quantum computing"
- NOT breaking any exponential barriers

**The reality:** When you restrict to separable, product-form, φ-structured states, O(N) scaling is trivially correct. There is no miracle here.

## Desktop Environment

The PyQt5 desktop provides:
- **Q-Notes**: Text editor with Φ-RFT backend
- **Q-Vault**: Encrypted file storage (experimental crypto)
- **QuantSoundDesign**: Audio processing with Φ-RFT

## What This Project Actually Achieves

### Real, Validated Results:
1. **Unitary Transform**: Mathematically proven, well-tested
2. **Compression Codec**: Competitive with classical transform codecs
3. **Signal Processing**: Interesting φ-structured spectral analysis

### Overstated/Deprecated Claims (see docs/archive/):
1. ~~"Million-qubit quantum simulation"~~ → Structured state compression
2. ~~"O(1) memory vs O(2^n)"~~ → O(N) for structured states (trivial)
3. ~~"Post-quantum cryptography"~~ → Experimental, no proofs
4. ~~"Breaking entropy bounds"~~ → Competitive, not breakthrough

## For Paper Submissions

When submitting to peer review, include:
- `algorithms/rft/core/*` - Core transform implementations
- `docs/validation/*` - Validation methodology
- `tests/validation/*` - Test suite
- Relevant `experiments/*` for compression/codec work

Do NOT include:
- Any "quantum" terminology without explicit scoping
- Symbolic engine hype documents (archived)
- "Breakthrough" or "million-qubit" language

## See Also

- [MATHEMATICAL_FOUNDATIONS.md](../algorithms/rft/MATHEMATICAL_FOUNDATIONS.md) - Honest math description
- [docs/archive/deprecated_overclaims/](../archive/deprecated_overclaims/) - Historical overclaims
