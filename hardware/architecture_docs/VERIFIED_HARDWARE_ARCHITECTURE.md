# Verified Hardware Architecture: QPU v2

> **Based on:** Verified Benchmarks (Dec 2025)
> **Goal:** Optimize FPGA area for *proven* capabilities, discarding failed hypotheses.

## 🧩 The Jigsaw Puzzle Solved

Our software benchmarks have clearly separated the "Breakthroughs" from the "Dead Ends". We can now design a **Quantonium Processing Unit (QPU) v2** that focuses entirely on strengths.

### 1. The "Breakthrough" Core (Class A)
*   **Software Finding:** The Symbolic Engine runs at 505 Mq/s on CPU.
*   **Hardware Implication:** This is bitwise logic (XOR/CNOT on stabilizer tableaus). It is *trivial* to implement in FPGA and will run at wire speed.
*   **Action:** Dedicate 40% of FPGA area to a **Parallel Stabilizer Engine**.
    *   *Input:* Pauli Strings.
    *   *Operation:* O(1) updates.
    *   *Target:* >10 Gq/s (Billion Qubits/sec).

### 2. The "Compression" Pipeline (Class B - H3)
*   **Software Finding:** The `H3_Hierarchical_Cascade` (FFT -> RFT -> Wavelet) is the *only* variant with $\eta=0$ (perfect math) and good compression (0.655 BPP).
*   **Hardware Implication:** We need a pipeline, not just a single kernel.
*   **Action:** Hardwire the **H3 Cascade**:
    *   Stage 1: FFT (Standard IP).
    *   Stage 2: RFT Phase Shift (The "Secret Sauce").
    *   Stage 3: Wavelet Thresholding.

### 3. The "Vault" Accelerator (Class D)
*   **Software Finding:** The `SIS_HASH` is secure (50% avalanche) but slow (0.5 MB/s) on CPU due to heavy matrix math.
*   **Hardware Implication:** Lattice cryptography (Matrix-Vector Multiplication) is *perfect* for DSP slices on an FPGA.
*   **Action:** Create a **Systolic Array** for the SIS Hash.
    *   *Target:* Boost speed from 0.5 MB/s -> 500 MB/s (1000x speedup).

### 4. The "Scope" (Class B/E)
*   **Software Finding:** RFT is great for analysis (Quasicrystals, Audio Spectra) but bad for streaming.
*   **Hardware Implication:** Keep a single, high-precision `RFT_GOLDEN` kernel for "Offline Analysis Mode".

---

## 🏗️ Proposed FPGA Layout (QPU v2)

| Module | Source Mode | Priority | Area Budget | Status |
| :--- | :--- | :--- | :--- | :--- |
| **CORE_0** | `MODE_QUANTUM_SIM` (14) | **Critical** | 40% | ✅ Verified |
| **CORE_1** | `MODE_RFT_CASCADE` (6) | **High** | 30% | ✅ Verified |
| **CORE_2** | `MODE_SIS_HASH` (12) | **High** | 20% | ⚠️ Needs Accel |
| **CORE_3** | `MODE_RFT_GOLDEN` (0) | **Medium** | 10% | ✅ Utility |

## 🗑️ Deprecated / Removed
*   `RFTMW` Text Compression (Class C) -> **DROP** (Use standard LZ4 hardware).
*   `RFT_HARMONIC` Audio Streaming -> **DROP** (Latency too high).
*   `GOLDEN_EXACT` -> **DROP** (Too expensive).

This architecture transforms the FPGA from a "Testbed of 16 Modes" into a **Focused Research Accelerator**.
