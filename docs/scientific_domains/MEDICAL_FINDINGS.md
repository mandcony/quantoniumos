# Medical Applications: Biosignals & Imaging

**Status:** 🟡 **Research / Benchmarking**
**Primary Tool:** `DCT` (for ECG), `RFT-Golden` (for MRI Research)

## Executive Summary
QuantoniumOS provides a rigorous benchmarking suite for medical signal processing.
**Key Finding:** For standard ECG compression, **DCT (Discrete Cosine Transform) remains the gold standard.** RFT variants do not offer a statistically significant advantage for routine cardiac monitoring.

## 1. ECG Analysis (Heart Signals)
**Benchmark:** MIT-BIH Arrhythmia Database (Records 100-217)
**Metric:** PRD (Percent Root-mean-square Difference) at 10% compression.

| Transform | PRD (Lower is Better) | Verdict |
| :--- | :--- | :--- |
| **DCT** | **12.43%** | 🏆 **Winner** |
| RFT-Golden | 12.54% | Tie/Loss |
| ARFT (Adaptive) | 12.79% | Loss |

> **Why?** ECG signals are well-modeled by AR(1) processes, for which DCT is asymptotically optimal (approaching the Karhunen-Loève Transform). The quasi-periodic nature of RFT does not match the specific "bursty" periodicity of the heart as well as DCT.

## 2. MRI Reconstruction (Imaging)
**Benchmark:** Shepp-Logan Phantom (Synthetic)
**Application:** Compressed Sensing (reconstructing images from few samples).

**Hypothesis:** RFT's sparsity in "edge-like" features might allow for faster MRI scans (undersampling).
**Current Status:**
- **Simulation:** RFT shows promise in reconstructing specific phantom features (ellipses) from sparse k-space data.
- **Real Data:** Not yet validated on raw k-space data from MRI machines.

## Recommendation
- **Clinical ECG:** Use **DCT**.
- **Research MRI:** Investigate **RFT** for sparse reconstruction of non-Cartesian trajectories.
