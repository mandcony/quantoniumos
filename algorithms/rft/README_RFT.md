# Resonant Fourier Transform (RFT) - Canonical Definition

## Overview

This document establishes the **authoritative definition** of the Resonant Fourier Transform (RFT) family. All implementations in this repository should conform to this specification.

---

## 1. Core Object: The Resonance Operator

The foundation of all RFT variants is a **Hermitian resonance operator** $K$:

$$
K = T(R(k) \cdot d(k))
$$

Where:
- $T(\cdot)$ = Toeplitz matrix constructor
- $R(k)$ = Resonance autocorrelation function (encodes the signal family structure)
- $d(k)$ = Decay/regularization function (ensures well-conditioning)

**Key Property:** $K$ is Hermitian (real symmetric for real signals), so it has a complete orthonormal eigenbasis.

---

## 2. Canonical RFT Definition

The **Resonant Fourier Transform** is defined as the eigenbasis of the resonance operator:

$$
K = U \Lambda U^T, \quad U^T U = I
$$

**Forward Transform:**
$$
\text{RFT}(x) = U^T x
$$

**Inverse Transform:**
$$
\text{RFT}^{-1}(X) = U X
$$

**Theorems (proven in `formal_framework.py`):**
1. **Hermitian:** $K = K^H$ ✓
2. **Unitarity:** $U^T U = U U^T = I$ ✓
3. **Perfect Reconstruction:** $U U^T x = x$ ✓

---

## 3. RFT Variants

### 3.1 RFT-φ (Golden Resonance)

The φ-parameterized RFT uses a resonance function based on the golden ratio:

$$
R_\phi[k] = \cos(2\pi f_0 k/N) + \cos(2\pi f_0 \phi k/N)
$$

Where $\phi = (1+\sqrt{5})/2$ is the golden ratio.

**Implementation:** `algorithms/rft/kernels/resonant_fourier_transform.py`

**Use Case:** Signals with golden quasi-periodic structure (Fibonacci-modulated, phyllotaxis patterns).

### 3.2 ARFT (Adaptive RFT)

The **Adaptive Resonant Fourier Transform** derives the operator from the signal itself:

1. Estimate autocorrelation $r_{xx}[k]$ from the signal
2. Build Toeplitz matrix $R_x = T(r_{xx})$
3. Compute eigenbasis $U_x$ of $R_x$
4. $\text{ARFT}_x(x) = U_x^T x$

**Implementation:** `algorithms/rft/kernels/operator_arft_kernel.py`

**Use Case:** Unknown or varying signal families; maximum sparsity for specific signals.

**Trade-off:** O(N³) kernel construction per signal vs. fixed O(N²) for canonical RFT.

### 3.3 RFT-MW (Multiscale/Multiwindow RFT)

RFT applied across multiple scales or frequency bands, as in the H3 codec:

- Decompose signal into structure + texture components
- Apply DCT to structure (low-frequency)
- Apply RFT to texture (resonance-structured residual)

**Implementation:** `algorithms/rft/hybrids/cascade_hybrids.py`

---

## 4. DEPRECATED: φ-Phase FFT (RFT-v0)

The **original** "RFT" in early versions of this repository was:

$$
\Psi = D_\phi C_\sigma F
$$

Where:
- $F$ = DFT matrix
- $C_\sigma$ = Diagonal complex phase matrix
- $D_\phi$ = Diagonal golden-ratio phase matrix

**This is NOT the canonical RFT.** It is a phase-shifted FFT with the property:

$$
|(\Psi x)_k| = |(Fx)_k| \quad \forall x
$$

This means it has **no sparsity or compression advantage** over standard FFT.

**Status:** Demoted to "φ-phase FFT" or "phase-tilted FFT". Should not be called "RFT" in publications.

**Files still using this:** See `CLAIMS_AUDIT_REPORT.md` for a full list.

---

## 5. Hardware: RFTPU

The **Resonant Fourier Transform Processing Unit (RFTPU)** executes:
- RFT kernels (fixed φ-parameterized)
- ARFT kernels (signal-adaptive)
- RFT-MW cascade operations

The old φ-phase FFT hardware can be relabeled as "phase-tilted FFT accelerator" but should not claim RFT branding.

---

## 6. Benchmarks

| Transform | In-Family PSNR | Out-Family PSNR | Sparsity Gain |
|-----------|----------------|-----------------|---------------|
| FFT       | Baseline       | Baseline        | 0%            |
| DCT       | +0.5 dB        | +1-2 dB         | ~10%          |
| **RFT-φ** | **+15-20 dB**  | -1 dB           | **+30-50%**   |
| ARFT      | **+30%**       | N/A (adaptive)  | **+32%**      |

See `tests/benchmarks/honest_rft_benchmark.py` for methodology.

### 6.1 Operator Variant Benchmarks

Eight operator-based RFT variants are implemented in `variants/operator_variants.py`:

| Variant | Resonance Pattern | Wins | Best For |
|---------|-------------------|------|----------|
| **RFT-Golden** | cos(2πf₀k/N) + cos(2πφf₀k/N) | 4 | Golden-ratio quasi-periodic |
| **RFT-Geometric** | Geometric frequency spacing | 3 | Harmonic series, chirps |
| **RFT-Beating** | Two close frequencies | 1 | Pure sine, AM signals |
| **RFT-Harmonic** | Sum of 1/h harmonics | 1 | Natural harmonics |
| **RFT-Hybrid-DCT** | DCT + resonance blend | 1 | White noise |
| RFT-Fibonacci | Fibonacci harmonics | 0 | Fibonacci-modulated |
| RFT-Phyllotaxis | 137.5° angular pattern | 0 | Plant spirals |
| RFT-Cascade | Multi-stage harmonics | 0 | Multi-scale |

**Domain Performance (PSNR at 10% retention):**
- **In-Family signals:** RFT wins **7/7 (100%)** vs FFT/DCT
- **Out-of-Family signals:** RFT wins **3/4 (75%)** vs FFT/DCT
- **FFT only wins on square wave** (pure integer harmonics)

See `tests/benchmarks/full_rft_variant_benchmark.py` for full results.

---

## 7. File Index

| Purpose | File |
|---------|------|
| **Canonical RFT kernel** | `algorithms/rft/kernels/resonant_fourier_transform.py` |
| **ARFT kernel** | `algorithms/rft/kernels/operator_arft_kernel.py` |
| **Operator variants** | `algorithms/rft/variants/operator_variants.py` |
| **Formal proofs** | `algorithms/rft/theory/formal_framework.py` |
| **Honest benchmark** | `tests/benchmarks/honest_rft_benchmark.py` |
| **Full variant benchmark** | `tests/benchmarks/full_rft_variant_benchmark.py` |
| **Multiscale benchmark** | `tests/benchmarks/rft_multiscale_benchmark.py` |
| **Real-world benchmark** | `tests/benchmarks/rft_realworld_benchmark.py` |

---

## 8. Citation

When citing RFT, use:

> The Resonant Fourier Transform (RFT) is defined as the orthonormal eigenbasis of a resonance operator $K$ built from structured autocorrelation. Golden ratio, Fibonacci factors, and similar structures enter as parameters of $K$, not as the definition of RFT itself.

---

*Last Updated: December 2025*
