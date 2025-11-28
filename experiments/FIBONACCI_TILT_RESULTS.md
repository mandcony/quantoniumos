# Fibonacci-Tilt Φ-RFT: Hypothesis Validation Results

**Date:** 2025-11-27 19:18:43
**Total Time:** 0.09s
**Hypotheses Confirmed:** 1/10

---

## H1: ❌ REJECTED

**Hypothesis:** Fibonacci sequences are asymptotically 1-sparse in Fib-Φ-RFT

**Test:** Measured top-5 energy concentration across scales [128, 256, 512, 1024]

**Metric:** Average energy concentration in top-5 coefficients

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **0.083989** |
| DCT       | 0.838531 |
| DFT       | 0.827634 |

**Confidence:** 0.0%

**Details:** Concentrations at N=1024: Fib=0.033, DCT=0.863, DFT=0.864

---

## H2: ❌ REJECTED

**Hypothesis:** Fib-Φ-RFT is the natural diagonalizer of golden rotations

**Test:** Built golden rotation operator (n=256), measured off-diagonal Frobenius norm

**Metric:** Off-diagonal Frobenius norm (lower is more diagonal)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **4132.778056** |
| DCT       | 16.162204 |
| DFT       | 16.143664 |

**Confidence:** 0.0%

**Details:** Operator: R[i,j] = δ(j - ⌊i·φ⌋ mod n) + 1% noise

---

## H3: ❌ REJECTED

**Hypothesis:** Golden-structured text segments are detectable by Fib sparsity

**Test:** Generated golden-run-length text vs random text (n=512), compared sparsity ratios

**Metric:** Sparsity gain ratio: golden/random (higher means better detection)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **0.000000** |
| DCT       | 3.108696 |
| DFT       | 1.000000 |

**Confidence:** 0.0%

**Details:** Fib-RFT: golden=0.000, random=0.000. DCT: golden=0.279, random=0.090

---

## H4: ❌ REJECTED

**Hypothesis:** Hybrid DCT + Fib-Φ-RFT dominates on mixed quasi-periodic data

**Test:** Mixed signal (60% smooth + 40% Fibonacci), 10% sparsity, measured reconstruction MSE

**Metric:** Reconstruction MSE (lower is better)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **0.131553** |
| DCT       | 0.003565 |
| DFT       | 0.026607 |

**Confidence:** 0.0%

**Details:** Pure DCT: 0.003565, Pure Fib: 0.131553, Cascade: 0.026607

---

## H5: ❌ REJECTED

**Hypothesis:** Fib-Φ-RFT yields deterministic 'good' sensing matrices

**Test:** Built 64x128 sensing matrices from random rows, measured mutual coherence

**Metric:** Mutual coherence (lower is better for compressed sensing)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **0.205924** |
| DCT       | 0.000000 |
| DFT       | 1.000000 |

**Confidence:** 0.0%

**Details:** Lower coherence enables better sparse recovery via ℓ₁ minimization

---

## H6: ❌ REJECTED

**Hypothesis:** Fib-Φ-RFT exposes quasicrystal-like diffraction peaks more sharply

**Test:** Generated 1D quasicrystal (dim=5) with SNR=20dB, measured coefficient kurtosis

**Metric:** Coefficient distribution kurtosis (higher = sharper peaks)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **4.062230** |
| DCT       | 69.250289 |
| DFT       | 47.097548 |

**Confidence:** 0.0%

**Details:** Peaks detected (>2σ): Fib=25, DCT=12, DFT=14

---

## H7: ❌ REJECTED

**Hypothesis:** Fib-tilt improves anomaly detection in 'almost golden' streams

**Test:** Injected 5 anomalies into golden rotation stream, counted detections at 3σ threshold

**Metric:** Detection error: |detected - actual| (lower is better)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **3.000000** |
| DCT       | 2.000000 |
| DFT       | 1.000000 |

**Confidence:** 50.0%

**Details:** Detections: Fib=2, DCT=3, DFT=6, Actual=5

---

## H8: ❌ REJECTED

**Hypothesis:** Golden-aligned hash mixing layer improves avalanche on structured inputs

**Test:** Hashed 100 structured inputs with 1-bit perturbations, measured avalanche effect

**Metric:** Distance from ideal avalanche (0.5), lower is better

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **0.500000** |
| DCT       | 0.489219 |
| DFT       | 0.000000 |

**Confidence:** 0.0%

**Details:** Fibonacci avalanche: 0.000, Standard avalanche: 0.989, Ideal: 0.500

---

## H9: ✅ CONFIRMED

**Hypothesis:** Fib-Φ-RFT reduces quantization sensitivity for golden signals

**Test:** Fibonacci sequence (n=256), 8-bit quantization, measured reconstruction MSE

**Metric:** Reconstruction MSE after quantization (lower is better)

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **0.000012** |
| DCT       | 0.000181 |
| DFT       | 0.000367 |

**Confidence:** 93.4%

**Details:** Quantization: 8 bits uniform scalar quantization

---

## H10: ❌ REJECTED

**Hypothesis:** Fib-tilt yields a low-rank model of golden Markov chains

**Test:** Built Fibonacci-weighted Markov transition matrix (n=64), measured effective rank via SVD

**Metric:** Effective rank (singular values > 0.1·σ_max), lower is better

| Transform | Value |
|-----------|-------|
| Fib-Φ-RFT | **2.000000** |
| DCT       | 2.000000 |
| DFT       | 2.000000 |

**Confidence:** 0.0%

**Details:** Ranks: Fib-RFT=2, DCT=2, Original=2

---

