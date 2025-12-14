# Glossary of Terms

> **Purpose:** Eliminate ambiguity. Every term has a precise mathematical definition.

---

## Core Terminology

### Φ-RFT (Phi-RFT)

**Definition:** A non-orthogonal signal transform derived from the eigenbasis of a structured autocorrelation operator.

**Mathematical Definition:**
```
K = T(R(k) · d(k))        — Toeplitz matrix from autocorrelation model
K = U Λ Uᵀ                — Eigendecomposition
RFT(x) = Uᵀ x             — Transform application
```

Where:
- `R(k) = cos(2πf₀k) + cos(2πf₀φk)` — Golden-ratio frequency pair
- `d(k) = exp(-αk)` — Exponential decay envelope
- `φ = (1 + √5)/2` — Golden ratio (≈1.618)

**What it is:** A data-independent, closed-form basis derived from operator eigendecomposition.

**What it is NOT:**
- Not quantum computing
- Not a replacement for FFT
- Not faster asymptotically
- Not universally optimal

---

### Non-Orthogonal Signal Transform

**Definition:** A linear transformation where basis vectors are not mutually perpendicular but maintain unitarity through eigendecomposition of a symmetric operator.

**Contrast with FFT:** FFT uses uniformly-spaced sinusoidal basis; Φ-RFT uses eigenvectors of a golden-ratio autocorrelation model.

---

### Phase-Modulated Basis Functions

**Definition:** Basis functions with non-quadratic phase evolution, specifically:

```
φ_k(n) = exp(i · 2π · (k·n/N + k·n·φ⁻¹ mod 1))
```

The term "resonance" in older documentation refers to this phase structure, not acoustic or mechanical resonance.

---

### Research Framework (formerly "Operating System")

**Definition:** An experimental software scaffold for running RFT algorithms, benchmarks, and demonstrations.

**What it is:** A Python package with optional native acceleration, desktop UI for demos, and mobile prototype.

**What it is NOT:**
- Not an operating system kernel
- Not a replacement for Linux/Windows
- Not production software

---

## Signal Processing Terms

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

### Coherence

**Definition:** Maximum absolute inner product between transform basis vectors and measurement vectors.

```
μ = max_{i,j} |⟨u_i, m_j⟩|
```

---

## Domain Terms

### KLT (Karhunen-Loève Transform)

**Definition:** The statistically optimal transform for a given signal class, derived from covariance eigendecomposition.

**Relevance:** Φ-RFT achieves KLT-like compaction on specific signal classes without requiring covariance estimation.

### LCT (Linear Canonical Transform)

**Definition:** A family of transforms including FFT, Fresnel, and fractional Fourier transforms, characterized by quadratic phase.

**Relevance:** Φ-RFT has non-quadratic phase structure, placing it outside the LCT family.

### FrFT (Fractional Fourier Transform)

**Definition:** A generalization of FFT with continuous rotation parameter in time-frequency space.

**Relevance:** FrFT is a special case of LCT; Φ-RFT is distinct from both.

---

## Standards and Metrics

### DIN 4150-3

**Definition:** German standard for structural vibration limits (Peak Particle Velocity in mm/s).

### ISO 10816

**Definition:** International standard for machine vibration severity (velocity in mm/s RMS).

### ISO 2631

**Definition:** Standard for human exposure to whole-body vibration.

---

## Deprecated Terms

| Old Term | New Term | Reason |
|----------|----------|--------|
| "Quantum-inspired" | "Non-orthogonal signal transform" | Avoids quantum computing confusion |
| "Resonance" | "Phase-modulated basis" | Avoids acoustic/mechanical confusion |
| "Operating System" | "Research framework" | Accurate description |
| "New paradigm" | (removed) | Empty marketing language |
| "φ-phase FFT" | (deprecated) | No sparsity advantage; use Φ-RFT |

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| φ | Golden ratio = (1 + √5)/2 ≈ 1.618 |
| K | Resonance operator (Toeplitz matrix) |
| U | Eigenvector matrix of K |
| Λ | Eigenvalue matrix of K |
| F | Standard DFT matrix |
| N | Signal length |
| f₀ | Base frequency parameter |
| α | Decay rate parameter |

---

*Last updated: December 2025*
