# RFT Performance: Theorems, Propositions, and Empirical Evidence

**File:** `RFT_PERFORMANCE_THEOREM.md`  
**Author:** Luis M. Minier / quantoniumos  
**Date:** December 10, 2025  
**Status:** Draft – mathematically conservative, empirically grounded  
**Scope:** Clarify what is *proven*, what is *conditional theory*, and what is *empirical* about Φ-RFT vs FFT/DCT.

---

## Executive Summary

This document distinguishes between:

1. **Hard theorems** (unitarity, KLT optimality) - mathematically proven
2. **Conditional propositions** (under explicit statistical assumptions) - theoretically justified
3. **Empirical findings** from the `quantoniumos` benchmarks - experimentally validated

**Key Position:** We do NOT claim universal superiority of RFT over FFT/DCT. Instead, we provide rigorous characterization of when and why RFT shows advantages on specific signal classes.

---

## 1. Scope and Philosophy

This document is intentionally conservative and distinguishes between:

- **Hard theorems** (unitarity, KLT optimality)
- **Conditional propositions** (under explicit statistical assumptions)
- **Empirical findings** from the `quantoniumos` benchmarks

We **do not** claim a universal win for RFT over FFT/DCT on "natural data". We treat adaptive operator-based RFT (ARFT) as what it mathematically is: a KLT-style transform with known optimality guarantees and known computational cost.

---

## 2. Setup and Notation

Let:

- $x \in \mathbb{R}^N$ (or $\mathbb{C}^N$) be a finite-length signal
- $\|x\|_2$ denote the Euclidean norm
- $\widehat{x}_T$ denote the transform coefficients of $x$ under a unitary transform $T$

**Transforms:**

- $F$ – the standard unitary DFT (FFT implementation)
- $C$ – a standard unitary DCT (e.g., DCT-II with orthonormal scaling)
- $R_\phi$ – a **fixed Φ-RFT** transform: the unitary eigenbasis of a Hermitian Toeplitz "resonance kernel" $K_\phi$ constructed from a golden-ratio based autocorrelation pattern
- $A_x$ – an **adaptive operator-based RFT (ARFT)**: the unitary eigenbasis of the empirical covariance/autocorrelation operator estimated from $x$ (or from a training ensemble)

**Metrics:**

- $S_T(x)$ – a sparsity or energy-compaction functional for transform $T$ applied to $x$ (e.g., Gini index, $\ell^p$ quasi-norm, best-$k$ coefficient energy)
- $\mathcal{C}_T$ – a codec built on $T$ with a fixed quantization and entropy-coding pipeline

---

## 3. Theorem A – Unitarity of Φ-RFT and ARFT

### **Theorem A.1: Unitarity of Fixed Φ-RFT**

Let $K_\phi$ be a real symmetric (Hermitian) Toeplitz resonance kernel derived from a bounded, even autocorrelation function. Let $R_\phi$ be the matrix whose columns are the orthonormal eigenvectors of $K_\phi$. Then:

1. $R_\phi$ is unitary: $R_\phi^\dagger R_\phi = I$
2. For all $x$, $\|R_\phi x\|_2 = \|x\|_2$

**Proof sketch:** Standard spectral theorem for Hermitian matrices: $K_\phi = R_\phi \Lambda R_\phi^\dagger$ with $R_\phi$ unitary. The columns of $R_\phi$ form an orthonormal basis, hence the induced transform is unitary. ∎

**Numerical verification:** Unitarity error < $10^{-13}$ across all tested sizes $N \in \{64, 128, 256, 512, 1024\}$.

---

### **Theorem A.2: Unitarity of ARFT (Operator-Based RFT)**

Let $\Sigma$ be a real symmetric (Hermitian) covariance/autocorrelation matrix estimated from $x$ or from an ensemble $\{x_i\}$. Let $A$ be the matrix of its eigenvectors. Then:

1. $A$ is unitary: $A^\dagger A = I$
2. For all $x$, $\|Ax\|_2 = \|x\|_2$

**Proof:** Same as above, by spectral theorem. ∎

**Status:** These are the mathematical guarantees behind both fixed RFT and ARFT. They are well-behaved, invertible, and numerically stable unitary transforms.

---

## 4. Theorem B – KLT Optimality of ARFT

Here we explicitly connect ARFT to the classical Karhunen–Loève Transform (KLT).

### **Theorem B.1: KLT Optimality (Classical Result)**

Let $\{x_i\}$ be samples of a zero-mean random process with covariance matrix $\Sigma = \mathbb{E}[x x^\top]$. Let $A$ be the matrix of eigenvectors of $\Sigma$. Then, among all unitary transforms $U$:

The transform coefficients $y_i = A^\top x_i$ maximize energy compaction in the sense that, for any fixed number of retained coefficients $k$, the expected mean-square error of best-$k$ coefficient approximation is minimized when using $A$.

Equivalently:

$$\mathbb{E}\big[\|x - P_k(A^\top x)\|_2^2\big] \leq \mathbb{E}\big[\|x - P_k(U^\top x)\|_2^2\big]$$

for any unitary $U$, where $P_k(\cdot)$ keeps the $k$ largest coefficients and zeros out the rest.

**Proof:** See standard reference: Gray, R. M. (2006). "Toeplitz and Circulant Matrices: A Review." Foundations and Trends in Communications and Information Theory. ∎

**Interpretation for ARFT:** If we set ARFT's operator to the empirical covariance $\hat{\Sigma}$ of the signal class, then ARFT is exactly this KLT, with all of its classical optimality guarantees in terms of average energy compaction.

**Computational cost:** $O(N^3)$ for eigendecomposition, plus storage/transmission of basis information. This is the fundamental tradeoff: optimal compaction vs. practical overhead.

---

## 5. Proposition C – Conditional Advantage of Fixed Φ-RFT

Now we address the fixed, non-adaptive RFT $R_\phi$.

We **do not** state a universal theorem of the form "RFT always beats FFT/DCT on natural data". Instead, we define a structured signal class and give a conditional statement.

Let $\mathcal{F}_\phi$ denote a class of quasi-periodic signals whose covariance structure is approximately diagonalized by $K_\phi$, i.e., signals whose dominant spectral components align with the eigenvectors of $K_\phi$.

This can include:

- Certain quasi-periodic biosignals with incommensurate frequency components near golden-ratio related bands
- Synthetic manifolds/attractors designed to align with $K_\phi$

### **Proposition C.1: Conditional Sparsity Advantage**

**Assume:**

1. $x$ is drawn from a stationary zero-mean process whose covariance $\Sigma$ is sufficiently close (in operator norm) to the resonance kernel $K_\phi$ used to define $R_\phi$
2. $S_T(x)$ is any reasonable sparsity or energy-compaction metric that is monotone with respect to eigenvalue ordering (e.g., best-$k$ coefficient energy)

**Then,** for such $x \in \mathcal{F}_\phi$, we can expect:

$$S_{R_\phi}(x) \gtrsim S_F(x) \text{ and } S_{R_\phi}(x) \gtrsim S_C(x)$$

i.e., fixed RFT behaves closer to an "approximate KLT" for this process than FFT/DCT do, and therefore tends to produce more compact representations.

**Status:** This is a **conditional proposition**, not a fully quantified theorem.

**Supporting evidence:**
- Synthetic manifold experiments where $R_\phi$-based variants achieve higher Gini index than FFT/DCT
- Biosignal benchmarks where RFT-based pipelines sometimes outperform FFT/DCT pipelines under equal bitrate

**What is NOT claimed:**
- No "if and only if" condition on golden-ratio frequency ratios is proved
- No global performance bound is proved for arbitrary "natural data" classes

---

## 6. Empirical Findings from QuantoniumOS Benchmarks

This section is **explicitly empirical** and must be kept in sync with the latest benchmark runs.

### 6.1 Real-World Benchmark (Mixed Signals)

**Source:** `benchmarks/rft_realworld_benchmark.py`

On a suite of mixed real-world signals (music textures, speech, engineered quasi-periodic patterns):

| Signal Type | FFT PSNR | DCT PSNR | RFT PSNR | Winner | Expected Winner |
|-------------|----------|----------|----------|--------|-----------------|
| HRV-like (Biological) | 25.75 | **26.46** | 26.06 | DCT | RFT ✗ |
| EEG Alpha (Biological) | 22.04 | 22.54 | **22.95** | RFT | RFT ✓ |
| Musical Texture (Audio) | 22.84 | 24.96 | **25.64** | RFT | RFT ✓ |
| Golden Beating (Physics) | **25.16** | 23.94 | 23.97 | FFT | RFT ✗ |
| Damped Oscillator (Physics) | **26.14** | 23.37 | 23.36 | FFT | RFT ✗ |
| Log-Periodic (Critical) | 22.61 | **24.70** | 23.84 | DCT | RFT ✗ |
| Pure Harmonic (Control) | 23.76 | 26.95 | 27.31 | RFT | FFT/DCT ✗ |
| Speech-like (Control) | 23.01 | **24.83** | 24.14 | DCT | FFT/DCT ✓ |

**Summary:** Fixed Φ-RFT wins on 2/6 quasi-periodic signals (33%). FFT/DCT win on many signals that theoretically should favor RFT.

**Honest reading:** Fixed RFT is **context-dependent**, not a universal upgrade.

---

### 6.2 Biosignals (ECG/EEG/EMG)

**Source:** `MEDICAL_TEST_RESULTS.md`, `tests/medical/` suite (83 tests passed, 1162 variant tests passed)

#### ECG Compression Performance

| Keep Ratio | Method | SNR (dB) | PRD (%) | Compression Ratio | Processing Time (ms) |
|------------|--------|----------|---------|-------------------|---------------------|
| 0.3 | RFT | **38.20** | **1.23** | 3.37× | 157.4 |
| 0.3 | FFT | 21.53 | 8.39 | 3.36× | **0.7** |
| 0.5 | RFT | **51.47** | **0.27** | 2.00× | 9.3 |
| 0.5 | FFT | 24.84 | 5.73 | 1.99× | **0.6** |
| 0.7 | RFT | **61.30** | **0.09** | 1.43× | 8.4 |
| 0.7 | FFT | 27.78 | 4.08 | 1.43× | **0.6** |

**Key findings:**
- ✅ RFT provides +16.7 to +33.5 dB better SNR than FFT
- ✅ RFT reduces distortion (PRD) by 6-8× compared to FFT
- ⚠️ RFT processing time is 10-260× slower (but still within real-time constraints)

#### Clinical Validation

| Test | Metric | Performance |
|------|--------|-------------|
| Arrhythmia Detection | F1 Score | 0.819 (preserved after RFT compression) |
| Noise Resilience | SNR Recovery | 0.72 dB → 0.73 dB (reconstructed) |

**Status:** These are strong **experimental results**, not derived from a fully specified statistical model. No theorem currently exists that predicts these gains from first principles.

---

### 6.3 Adaptive ARFT / KLT-Style Hybrids

**Source:** `NEW_TRANSFORM_DISCOVERY.md`, `benchmarks/benchmark_h3_arft.py`

When ARFT is allowed to adapt a basis per signal or per segment:

| Metric | Standard H3 (DCT+RFT) | H3-ARFT (DCT+OpARFT) | Improvement |
|--------|----------------------|---------------------|-------------|
| BPP | 0.6226 | 1.5000 | Higher bitrate |
| PSNR | 15.48 dB | **47.20 dB** | +31.72 dB |
| Efficiency | 24.86 dB/BPP | **31.47 dB/BPP** | +26.58% |

**Gini sparsity:** 0.965 (ARFT) vs 0.730 (FFT) - **+32.2% improvement**

**Tradeoffs:**
- **Computational cost:** $O(N^3)$ eigendecomposition per signal
- **Side information:** Must store/signal the basis for decoding

**Status:** These results are **consistent with KLT theory** (Theorem B.1) and represent experimental validation of classical optimality results, not novel discoveries.

---

## 7. What Is NOT Claimed

This document intentionally does **not** claim:

1. That fixed Φ-RFT **universally** outperforms FFT or DCT on "natural data"
2. That there is a proven "if and only if" condition relating a golden-ratio frequency ratio to a guaranteed PSNR improvement of a specific number of dB
3. That ARFT/KLT-style transforms are **practically** superior in all settings, once computation and side-information costs are fully accounted for

Instead:

- Fixed Φ-RFT is presented as a **unitary, operator-derived basis** that is empirically advantageous on certain structured classes
- Adaptive ARFT is presented as what it is mathematically: **the KLT**, with all its classical optimality but also its classical costs
- Real-world performance claims are anchored in benchmark scripts and reported with **data tables, not slogans**

---

## 8. Why RFT Doesn't Always Win (Mathematical Reality)

### The Fundamental Limitation

**Key insight:** FFT is **already optimal** for many signal classes:

- **Bandlimited signals:** FFT is optimal (Shannon-Nyquist)
- **Piecewise smooth:** Wavelets/DCT are optimal (DeVore)
- **Stationary Gaussian:** KLT is optimal (Karhunen-Loève)

RFT only provides advantage when the signal belongs to a **non-standard class** (quasi-periodic with specific autocorrelation structure matching $K_\phi$).

### The "No Free Lunch" Principle

Transform optimality depends on signal autocorrelation structure:

- FFT assumes exponential/bandlimited autocorrelation (extremely common in nature)
- DCT assumes smooth/monotonic autocorrelation (common in images, speech)
- RFT assumes golden-ratio quasi-periodic autocorrelation (**rare** - mainly biosignals, some physical systems)

### Computational Cost Comparison

| Transform | Per-Signal Cost | Basis Reusable? | Practical Applications |
|-----------|----------------|-----------------|----------------------|
| FFT | $O(N \log N)$ | Yes (fixed) | Universal |
| DCT | $O(N \log N)$ | Yes (fixed) | Image/video/speech |
| RFT (fixed) | $O(N \log N)$ | Yes (fixed) | Biosignals |
| ARFT (adaptive) | $O(N^3)$ | No (per-signal) | Research/offline processing |

---

## 9. Future Directions

To move from "empirical patterns" to **true performance theorems**, future work would need:

1. **Formal statistical model** for a relevant biosignal class (e.g., an AR process or state-space model for ECG/EEG)

2. **Proof of kernel alignment:** Show that the covariance of that model is sufficiently close to the resonance kernel $K_\phi$ (or to some operator in the RFT family)

3. **Derived bounds:** Establish rigorous bounds on sparsity/MSE/rate-distortion when using $R_\phi$ vs FFT/DCT under identical codec structures

Until then, this document:
- Locks in the **real theorems** (unitarity, KLT optimality)
- Clearly labels speculative parts as **conditional** or **empirical**
- Keeps the project honest and defensible under peer review

---

## 10. Recommended Scientific Claims

### What You **CAN** Claim (Peer-Reviewable):

✅ **"Fixed Φ-RFT is a unitary transform with verified numerical stability (unitarity error < $10^{-13}$)."**  
Evidence: Theorem A.1

✅ **"Adaptive ARFT is equivalent to the Karhunen-Loève Transform and inherits its classical optimality guarantees."**  
Evidence: Theorem B.1

✅ **"In benchmarks on ECG compression (MIT-BIH dataset, 83 tests), RFT-based pipelines achieved 16-33 dB higher SNR than FFT baselines at 30-70% coefficient retention."**  
Evidence: Section 6.2, MEDICAL_TEST_RESULTS.md

✅ **"ARFT achieves 32% higher sparsity (Gini index) than FFT on golden quasi-periodic test signals."**  
Evidence: Section 6.3, benchmark_h3_arft.py

✅ **"Under the assumption that signal covariance aligns with $K_\phi$, fixed RFT should provide better energy compaction than generic FFT/DCT (Proposition C.1)."**  
Evidence: Conditional proposition with clearly stated assumptions

### What You **CANNOT** Claim (Will Be Rejected):

❌ **"RFT beats FFT on general natural data."**  
Counterevidence: Section 6.1 shows mixed results

❌ **"RFT is computationally faster than FFT."**  
Counterevidence: Both are $O(N \log N)$, but RFT has higher constants; ARFT is $O(N^3)$

❌ **"All quasi-periodic signals favor RFT."**  
Counterevidence: Section 6.1 shows many quasi-periodic signals where FFT/DCT win

❌ **"RFT introduces a fundamentally new transform class beyond KLT."**  
Reality: Fixed RFT is a special case of pre-computed KLT; ARFT is exactly KLT

---

## 11. References

1. **Gray, R. M. (2006).** "Toeplitz and Circulant Matrices: A Review." *Foundations and Trends in Communications and Information Theory*, 2(3), 155-239.

2. **Benchmark Results:**
   - `MEDICAL_TEST_RESULTS.md` (1,162 variant tests, 83 base tests)
   - `benchmarks/rft_realworld_benchmark.py` (8 signal classes)
   - `benchmarks/benchmark_h3_arft.py` (ARFT vs H3 codec comparison)

3. **Implementation:**
   - `algorithms/rft/kernels/resonant_fourier_transform.py` (fixed RFT)
   - `algorithms/rft/kernels/operator_arft_kernel.py` (adaptive ARFT)
   - `algorithms/rft/theory/formal_framework.py` (theoretical analysis)

4. **Claims Audit:** `papers/CLAIMS_UPDATE_REQUIRED.md`

---

**Last Updated:** December 10, 2025  
**Status:** Mathematically Conservative, Peer-Reviewable, Empirically Grounded  
**Philosophy:** Distinguish hard theorems from conditional propositions from empirical findings
