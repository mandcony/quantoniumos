# RFT Variant & Hybrid Status

> **Date:** December 16, 2025
> **Verified By:** `benchmarks/run_all_benchmarks.py`

This document tracks the implementation status, complexity, and stability of the RFT variants and hybrids.

## 🧬 RFT Variants (The Basis Functions)

| Variant Code | Name | Complexity | Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `STANDARD` | Standard RFT | $O(N \log N)$ | 🟢 **Stable** | The default implementation. |
| `HARMONIC` | Harmonic RFT | $O(N \log N)$ | 🟢 **Stable** | Optimized for audio-like signals. |
| `FIBONACCI` | Fibonacci RFT | $O(N \log N)$ | 🟢 **Stable** | Best for quasicrystals. |
| `CHAOTIC` | Chaotic RFT | $O(N \log N)$ | 🟡 **Experimental** | Used for crypto mixing. |
| `GOLDEN_EXACT` | Exact Golden Unitary | $O(N^2)$ | 🔴 **Slow** | **Do not use for N > 1000.** Fails benchmarks due to timeout. |
| `PHI_CHAOTIC` | Phi-Chaotic | $O(N \log N)$ | 🟡 **Experimental** | Variant of chaotic mixing. |
| `HYPERBOLIC` | Hyperbolic RFT | $O(N \log N)$ | 🟢 **Stable** | |
| `LOG_PERIODIC` | Log-Periodic RFT | $O(N \log N)$ | 🟢 **Stable** | |

## 🤝 Hybrids (The Compression Engines)

| Hybrid Code | Name | Status | Performance (BPP) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `H0` | Baseline Greedy | 🟢 **Stable** | 1.00 | Reference implementation. |
| `H3` | **Hierarchical Cascade** | 🟢 **Production** | **0.49** | **Best Performer.** The "Coherence-Free" architecture. |
| `H4` | Quantum Superposition | 🟡 **Research** | 8.25 | Poor compression, interesting physics. |
| `H6` | Dictionary Learning | 🟡 **Slow** | 0.91 | Very slow training time. |
| `FH1` | MultiLevel Cascade | 🟢 **Stable** | 1.00 | |
| `FH3` | Frequency Cascade | 🟢 **Stable** | 1.12 | |

## 💻 Native Extensions

| Module | Function | Status | Requirement |
| :--- | :--- | :--- | :--- |
| `rftmw_native` | Quantum Symbolic Compression | � **Verified** | Built and passing benchmarks. |
| `libquantum_symbolic.so` | ASM Kernels | 🟢 **Verified** | Assembly optimizations active. |

## ⚠️ Known Issues
1. **GOLDEN_EXACT:** The exact unitary construction is computationally expensive ($O(N^2)$ or $O(N^3)$). Use `FIBONACCI` or `STANDARD` for large signals.
2. **Native Modules:** The Python wheels are not pre-built in the repo. You must run the build script to unlock maximum performance.
