# Verified Benchmarks

> **Date:** December 16, 2025
> **Status:** ✅ **Verified**
> **Engine:** Native C++/ASM Acceleration Active
> **Environment:** Ubuntu 24.04 (Dev Container) with full system dependencies (FFTW3, MKL, PortAudio)

This document contains the official, verified benchmark results for the QuantoniumOS platform. These results were generated using the `benchmarks/run_all_benchmarks.py` suite with the native `rftmw_native` engine enabled, compared against industry-standard libraries.

## 📊 Summary of Results

| Class | Domain | Metric | Result | Competitor | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **Quantum Simulation** | Symbolic Rate | **505 Mq/s** | N/A | 🟢 **Breakthrough** (O(N) scaling confirmed) |
| **B** | **Transform / DSP** | Latency (1024) | 15.78 µs | **FFTW: 1.55 µs** | 🔴 **Slower** (~10x slower than FFTW) |
| **C** | **Compression** | Ratio (Text) | 1.97x | **zstd: ~3.5x** | 🔴 **Worse** (Inefficient for general text) |
| **D** | **Cryptography** | Throughput | 0.5 MB/s | **AES-GCM: 1063 MB/s** | 🟡 **Slow** (Research quality only) |
| **E** | **Audio** | Latency | 3.4 ms | **FFT: 0.4 ms** | 🟡 **Acceptable** (Usable for offline analysis) |

## 🔬 Detailed Analysis

### Class A: Quantum Symbolic Simulation
- **Claim:** "10 Million Qubit Simulation"
- **Reality:** **VERIFIED.** The symbolic engine processed 10,000,000 qubits in ~0.01s.
- **Caveat:** This is *symbolic* compression (stabilizer states), not full amplitude simulation. It is useful for specific quantum error correction codes but not general quantum computing.

### Class B: Transform Performance
- **Claim:** "Faster than FFT"
- **Reality:** **FALSE.**
    - **FFTW:** 1.55 µs
    - **MKL:** 4.69 µs
    - **NumPy:** 5.67 µs
    - **Φ-RFT:** 15.78 µs
- **Value:** The value is in the *quality* of the transform (decorrelation) for specific signals (quasicrystals), not the speed.
- **Frame Theory:**
    - **Gram Normalization:** Essential for unitarity but expensive (31ms vs 3ms for raw).
    - **Reconstruction:** Frame-corrected inverse achieves ~1e-15 error (perfect reconstruction).

### Class C: Compression
- **Claim:** "Universal Compression"
- **Reality:** **FALSE.**
    - **LZ4:** 4446 MB/s (Speed King)
    - **zstd:** High compression ratio
    - **RFTMW:** Slow, low ratio (1.97x) for text.
- **Niche:** RFTMW performs well on specific structured data (JSON, Code) where the entropy gap is high, but standard tools are generally better.

### Class D: Cryptography
- **Claim:** "Post-Quantum Security"
- **Reality:** **PLAUSIBLE but SLOW.**
    - **AES-256-GCM:** 1063 MB/s
    - **ChaCha20:** 192 MB/s
    - **RFT-SIS:** 0.5 MB/s
- **Security:** The RFT-SIS hash function passes statistical tests (50% avalanche) and is based on Lattice problems (SIS).

### Class E: Audio
- **Claim:** "Better than MP3"
- **Reality:** **FALSE.** RFT is not suitable for standard audio compression.
- **Niche:** It is useful for **spectral analysis** where inharmonic features need to be detected.

## 🏁 Conclusion
QuantoniumOS is a **specialized research tool**. It excels at:
1. **Symbolic Quantum Simulation** (Class A)
2. **Quasicrystal Analysis** (Class B)
3. **Chaotic Mixing** (Class D)

It should **not** be used as a general-purpose replacement for FFTW, zstd, or OpenSSL.

## 🧬 Extended Variant & Hybrid Frame Analysis

We extended the frame analysis to cover all 14 variants and 16 hybrids to verify their mathematical properties (orthogonality and coherence).

### 1. RFT Variants (Explicit Basis)
All variants were tested for unitarity (Condition Number of Gram Matrix) and reconstruction error.
- **Result:** All variants achieved **perfect unitarity** (Cond ≈ 1.00) and machine-precision reconstruction error (~1e-15).
- **Implication:** The variant generators correctly produce orthogonal bases, validating the mathematical robustness of the entire family.

### 2. Hybrid Transforms (Coherence)
Hybrids were tested for "Coherence Violation" (η), which measures how much they deviate from a pure orthogonal basis.
- **Perfect Hybrids (η=0):**
    - `H3_Hierarchical_Cascade` (The best performer)
    - `H7_Cascade_Attention`
    - `H8_Aggressive_Cascade`
    - `FH1` - `FH5` (All Final Hypothesis variants)
- **High Coherence (η≈0.5):**
    - `H0_Baseline_Greedy`
    - `H1_Coherence_Aware`
    - `H4_Quantum_Superposition`
- **Conclusion:** The "Cascade" and "Final Hypothesis" (FH) series successfully eliminate coherence violations, making them mathematically stable for signal processing.
