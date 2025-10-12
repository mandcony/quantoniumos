# Research: Mathematical Foundations

This document contains an overview of the core mathematical concepts that underpin QuantoniumOS. The information here is for research and theoretical understanding. For practical implementation details, refer to the source code in `src/core/` and `src/assembly/`.

---

## Resonance Fourier Transform (RFT)

**Proposition**: The RFT is a unitary matrix transformation constructed via a QR decomposition of a kernel weighted by the golden ratio (φ).

**What's Novel in RFT**:
The core idea is the specific structure of the Hermitian matrix `H_φ` used to generate the unitary operator `R_φ = exp(i * H_φ)`.

-   **Construction**: `H_φ[j,k] = φ^|j-k| * base_matrix[j,k]` where `φ = (1 + sqrt(5)) / 2`.
-   **Key Property**: The use of the golden ratio's recurrence property (`φ^n = F_n * φ + F_{n-1}`) creates a matrix with a banded structure and exponentially decaying off-diagonals.
-   **Claimed Advantage**: This structure is hypothesized to enable faster-than-standard diagonalization for certain matrix classes, though this is not fully proven in a general context.

**Status**:
-   **Unitarity**: The RFT is demonstrably unitary, with numerical error less than 1e-12. This is expected for any operator of the form `exp(iH)` where `H` is Hermitian.
-   **Distinction from DFT**: The RFT is mathematically distinct from the Discrete Fourier Transform (DFT), with a measurable Frobenius distance.
-   **Practical Advantage**: The practical advantages over established transforms like the DFT or DCT are still an active area of research and have not been conclusively proven in peer-reviewed literature.

---

## AI Model Compression

**Method**: The compression technique used in this repository is a **lossy** method based on Singular Value Decomposition (SVD), quantization, and entropy coding.

**Process**:
1.  **Decomposition**: A weight matrix `W` is decomposed using SVD: `W = U * Σ * V^T`. The matrix is then approximated by keeping only the `r` most significant singular values, where `r` is much smaller than the matrix dimensions.
2.  **φ-Quantization**: The remaining singular values `σ_i` are quantized. This is a lossy step where precision is reduced. The "φ" naming suggests the quantization levels may be related to the golden ratio, but the core concept is standard quantization.
3.  **Encoding**: The quantized values are further compressed using a standard entropy coder like Huffman coding or ANS.

**Analysis**:
-   **Lossy, Not Lossless**: This method is fundamentally **lossy**. Claims of "lossless" compression at the ratios cited in the historical appendix (e.g., 15,134:1) would violate information theory (Shannon's theorem). The measured reconstruction error (RMSE) of 3-5% confirms the lossy nature.
-   **Compression Ratio**: The high compression ratios are achieved by aggressively truncating the SVD and using low-precision quantization. This is a trade-off against model accuracy.
-   **Comparison to SOTA**: The effectiveness of this specific compression recipe compared to modern, well-established techniques like GPTQ, GGUF, or bitsandbytes has not been benchmarked.

**Conclusion**: The compression system is an experimental, lossy codec. The only end-to-end validated model is `tiny-gpt2`. Claims regarding billion-parameter models are theoretical calculations based on applying this compression scheme, not validated results from reconstructed and benchmarked models.
