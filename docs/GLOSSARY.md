# Glossary of Terms

> **Purpose:** Eliminate ambiguity. Every term has a precise mathematical definition.
> **Patent Reference:** USPTO Application 19/169,399

---

## Core Terminology

### RFT (Resonant Fourier Transform)

**Definition:** A multi-carrier transform that maps discrete data into a continuous waveform domain using golden-ratio frequency and phase spacing.

**Canonical Mathematical Definition:**
```
Ψₖ(t) = exp(2πi × fₖ × t + i × θₖ)

Where:
  fₖ = (k+1) × φ       — Resonant Frequency
  θₖ = 2π × k / φ      — Golden Phase  
  φ = (1+√5)/2         — Golden Ratio ≈ 1.618
```

**Forward Transform:**
```
RFT(x)[t] = Σₖ x[k] × Ψₖ(t)
```

**What it is:** A φ-OFDM framework for wave-domain symbolic computation.

**What it is NOT:**
- Not a replacement for FFT
- Not faster asymptotically
- Not universally optimal for compression

---

### BinaryRFT (Claim 1 Implementation)

**Definition:** The symbolic transformation engine for encoding binary data as amplitude-phase modulated waveforms and performing logic operations in the wave domain.

**Key Operations:**
- `encode(value)` — Binary → Wave (BPSK on RFT carriers)
- `decode(wave)` — Wave → Binary (matched filter detection)
- `xor(w1, w2)` — XOR in wave domain
- `and_(w1, w2)` — AND in wave domain
- `or_(w1, w2)` — OR in wave domain
- `not_(w)` — NOT in wave domain (phase inversion)

---

### RFT-SIS Hash (Claim 2 Implementation)

**Definition:** A post-quantum cryptographic hash combining RFT transform features with Short Integer Solution (SIS) lattice hardness.

**Pipeline:**
```
Data → SHA3 Expansion → RFT Transform → SIS Quantization → Lattice Point → Final Hash
```

**Security Basis:** SIS lattice problem (believed quantum-resistant)

---

### Topological Hashing (Claim 3)

**Definition:** Extraction of waveform features into cryptographic signatures using geometric features and synthetic phase tags (heuristics).

**Geometric Structures:**
- Polar-to-Cartesian with golden ratio scaling
- Complex exponential coordinate generation
- Phase-winding tag generation (synthetic)
- Projection-based hash generation

---

### Hybrid Mode Integration (Claim 4)

**Definition:** Unified framework combining symbolic transform (Claim 1), cryptographic subsystem (Claim 2), and geometric structures (Claim 3) with coherent propagation across layers.

---

## Signal Processing Terms

### BPSK (Binary Phase-Shift Keying)

**Definition:** Modulation scheme where bit 0 → symbol -1, bit 1 → symbol +1.

**In RFT context:** Each bit modulates a separate RFT carrier.

### Golden Ratio (φ)

**Definition:** The unique positive number satisfying φ² = φ + 1.

**Value:** φ = (1 + √5)/2 ≈ 1.618033988749895

**Properties:**
- Self-similar: φ² = φ + 1
- Fibonacci limit: F_{n+1}/F_n → φ
- Golden angle: 2π/φ² ≈ 137.5° (complement 2π/φ ≈ 222.5°; same rotation opposite direction)

### Matched Filter Detection

**Definition:** Correlation-based symbol extraction:
```
symbol[k] = sign(Re(⟨W, Ψₖ⟩))
```

### Wave-Domain Logic

**Definition:** Logic operations (XOR, AND, OR, NOT) executed directly on waveforms without decoding to binary.

---

### Energy Compaction

**Definition:** The fraction of total signal energy captured by the first K transform coefficients.

**Metric:** 
```
η(K) = Σᵢ₌₁ᴷ |c_i|² / Σᵢ₌₁ᴺ |c_i|²
```

### Sparsity

**Definition:** The minimum number of coefficients required to capture 99% of signal energy, normalized by signal length.

**Metric:**
```
sparsity = min{K : η(K) ≥ 0.99} / N
```

Lower is better.

### PSNR (Peak Signal-to-Noise Ratio)

**Definition:** Standard reconstruction quality metric.

```
PSNR = 10 · log₁₀(MAX² / MSE) dB
```

### Avalanche Effect

**Definition:** Cryptographic property where single-bit input change causes ~50% output bits to flip.

**Target:** 50% ± 5%

---

## Cryptographic Terms

### SIS (Short Integer Solution)

**Definition:** Lattice problem: given matrix A, find short vector s such that As = 0 mod q.

**Security:** Believed resistant to quantum computers (no known polynomial-time quantum algorithm).

### Winding Number

**Definition:** Topological invariant counting total phase rotations of a complex waveform.

```
winding = (phase_unwrapped[-1] - phase_unwrapped[0]) / (2π)
```

### Euler Characteristic

**Definition:** Topological invariant: χ = V - E + F (vertices minus edges plus faces).

---

## Domain Terms

### KLT (Karhunen-Loève Transform)

**Definition:** The statistically optimal transform for a given signal class, derived from covariance eigendecomposition.

### LCT (Linear Canonical Transform)

**Definition:** A family of transforms including FFT, Fresnel, and fractional Fourier transforms, characterized by quadratic phase.

### FrFT (Fractional Fourier Transform)

**Definition:** A generalization of FFT with continuous rotation parameter in time-frequency space.

---

## Deprecated Terms

| Old Term | Current Term | Reason |
|----------|--------------|--------|
| "Quantum-inspired" | "Symbolic waveform computation" | Avoids quantum computing confusion |
| "Resonance" | "RFT" or "φ-OFDM" | Clearer technical description |
| "Operating System" | "Research framework" | Accurate description |
| "φ-phase FFT" | "BinaryRFT" | Old phase-tilted FFT, now deprecated |
| "New paradigm" | (removed) | Empty marketing language |

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| φ | Golden ratio = (1 + √5)/2 ≈ 1.618 |
| Ψₖ(t) | RFT basis function for carrier k |
| fₖ | Resonant frequency = (k+1) × φ |
| θₖ | Golden phase = 2πk / φ |
| W(t) | Complex waveform |
| N | Number of bits/carriers |
| T | Number of time samples |
| A | SIS lattice matrix |
| q | SIS modulus (3329 = Kyber prime) |
| β | Short vector bound |

---

## Patent Claims Reference

| Claim | Title | Primary Implementation |
|-------|-------|------------------------|
| 1 | Symbolic Resonance Fourier Transform Engine | `BinaryRFT` class |
| 2 | Resonance-Based Cryptographic Subsystem | `RFTSISHash` class |
| 3 | Geometric Structures for Cryptographic Waveform Hashing | Topological hash functions |
| 4 | Hybrid Mode Integration | `HybridRFTFramework` class |

---

*Last updated: December 2025*
*USPTO Application 19/169,399*
