# QuantoniumOS: Aperiodic Signal Processing Research Platform

[![RFT Framework DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17712905.svg)](https://doi.org/10.5281/zenodo.17712905)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)

> **"The Microscope for Aperiodic Order"**

QuantoniumOS is a specialized mathematical framework for analyzing **quasi-periodic** and **aperiodic** systems. While standard tools (FFT) assume nature is periodic (repeating circles), QuantoniumOS assumes nature is self-similar (spirals, fractals, golden ratios).

This makes it a powerful research tool for specific domains where the **Golden Ratio ($\phi$)** governs the structure.

---

## 🔬 Scientific Domains & Status

| Class | Domain | Application | Status | Key Finding |
| :--- | :--- | :--- | :--- | :--- |
| **A** | **Quantum** | **Symbolic Simulation** | 🟢 **Verified** | **505 Mq/s** (Million Qubits/sec) via Native Engine. |
| **B** | **Physics** | **Quasicrystals** | 🟢 **Verified** | **RFT beats FFT** on Golden Ratio systems (perfect unitarity). |
| **C** | **Compression** | **Text/Data** | 🔴 **Debunked** | RFTMW (1.97x) is worse than gzip/zstd for general data. |
| **D** | **Crypto** | **Post-Quantum** | 🟡 **Research** | Lattice-based hash (SIS) is secure but slow (0.5 MB/s). |
| **E** | **Audio** | **Analysis** | 🟡 **Niche** | Good for spectral analysis, bad for streaming (high latency). |

👉 **[View Verified Benchmarks](docs/scientific_domains/VERIFIED_BENCHMARKS.md)**
👉 **[View Hypothesis Report](docs/scientific_domains/HYPOTHESIS_RESULTS.md)**

---

## 🧮 The Core Technology: Resonant Fourier Transform (RFT)

The **RFT** is a linear transform constructed from the eigenbasis of a resonance operator with Golden Ratio structure.

- **Standard FFT:** Basis vectors are harmonic sines/cosines ($f = k$).
- **RFT-Golden:** Basis vectors are non-harmonic, quasi-periodic resonances ($f \approx k \cdot \phi$).

### When to use RFT?
✅ **YES:** Analyzing Quasicrystals, Fibonacci chains, Penrose tilings, Turbulence onset.
❌ **NO:** Compressing MP3s, JPEGs, or standard ECGs (use DCT/FFT).

---

## 🚀 Quick Start

### 1. Reproduce All Results (Recommended)
Run the full verification suite (Builds engine, runs tests, runs benchmarks):
```bash
./reproduce_results.sh
```

### 2. Run Specific Benchmarks
```bash
# Run Class A (Quantum) and Class B (Transform)
python3 benchmarks/run_all_benchmarks.py A B

# Run with all variants and hybrids
python3 benchmarks/run_all_benchmarks.py --variants
```

---

## 📂 Repository Structure

- `algorithms/rft/` - **The Core Math.** Python implementations of RFT variants.
- `src/rftmw_native/` - **The Engine.** C++/AVX2 Native Acceleration Layer.
- `benchmarks/` - **The Evidence.** Formal benchmark suite (Classes A-E).
- `docs/scientific_domains/` - **The Science.** Verified reports and findings.
- `experiments/` - **The Lab.** Raw experimental data and hypothesis testing.

---

## ⚠️ Disclaimer

This is a **classical signal processing library**. It runs on CPUs/GPUs. It is **not** a quantum computer. The name "QuantoniumOS" is historical.

**Patents:** Aspects of the RFT algorithm are patent pending (USPTO 19/169,399). See [PATENT_NOTICE.md](PATENT_NOTICE.md).
