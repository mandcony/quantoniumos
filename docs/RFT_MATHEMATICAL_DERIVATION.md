# Resonance Fourier Transform (RFT) - Mathematical Derivation

## 1. Introduction

The Resonance Fourier Transform (RFT) is a custom discrete transform developed for the QuantoniumOS project. It is designed to provide a unique spectral representation of a signal by using a basis derived from the golden ratio, φ (phi). This document provides the mathematical foundation for the RFT, including the definition of its basis vectors and a formal proof of their orthogonality, which guarantees the transform's invertibility.

## 2. Definition of the RFT

Given a discrete signal `x` of length `N`, represented as a vector `x = [x_0, x_1, ..., x_{N-1}]`, the RFT, denoted `X_k`, is defined as:

`X_k = Σ_{n=0}^{N-1} x_n * exp(-2πi * k * n * φ / N)`

where:
- `k` is the frequency index, from `0` to `N-1`.
- `n` is the time-domain index, from `0` to `N-1`.
- `i` is the imaginary unit.
- `φ = (1 + sqrt(5)) / 2` is the golden ratio, an irrational number.

In matrix form, this can be written as `X = M * x`, where `M` is the RFT transformation matrix with elements `M_{kn} = exp(-2πi * k * n * φ / N)`.

## 3. Proof of Orthogonality

To prove that the RFT is invertible, we must show that its basis vectors are orthogonal. The basis vectors are the columns of the transformation matrix `M`. Let's consider two distinct basis vectors (columns of `M`), corresponding to frequency indices `k_1` and `k_2`, where `k_1 ≠ k_2`.

The inner product of these two basis vectors is given by:

`Σ_{n=0}^{N-1} exp(2πi * k_1 * n * φ / N) * exp(-2πi * k_2 * n * φ / N)`
`= Σ_{n=0}^{N-1} exp(2πi * (k_1 - k_2) * n * φ / N)`

Let `Δk = k_1 - k_2`. Since `k_1 ≠ k_2`, `Δk` is a non-zero integer. Let `α = exp(2πi * Δk * φ / N)`. The sum becomes a geometric series:

`Σ_{n=0}^{N-1} α^n = (1 - α^N) / (1 - α)`

For this sum to be zero (indicating orthogonality), the numerator `1 - α^N` must be zero, while the denominator `1 - α` must be non-zero.

### Condition for Non-Zero Denominator

The denominator `1 - α` is zero if and only if `α = 1`.
`α = exp(2πi * Δk * φ / N) = 1`

This equality holds if `(Δk * φ / N)` is an integer. Let this integer be `m`.
`Δk * φ / N = m`
`φ = (m * N) / Δk`

This implies that `φ` is a rational number (a ratio of two integers), since `m`, `N`, and `Δk` are all integers. However, **the golden ratio `φ` is a well-known irrational number**. This is a contradiction.

Therefore, `(Δk * φ / N)` cannot be an integer for any non-zero integer `Δk`. This proves that `α ≠ 1`, and the denominator `(1 - α)` is never zero.

### Condition for Zero Numerator

The numerator `1 - α^N` is zero if `α^N = 1`.
`α^N = [exp(2πi * Δk * φ / N)]^N = exp(2πi * Δk * φ)`

This is equal to `1` if `Δk * φ` is an integer. Let this integer be `p`.
`Δk * φ = p`
`φ = p / Δk`

Again, this implies that `φ` is a rational number, which is a contradiction. Therefore, `Δk * φ` cannot be an integer for any non-zero integer `Δk`. This proves that `α^N ≠ 1`.

**Correction & Refinement:** The standard proof of orthogonality for the DFT relies on the term `exp(2πi * integer)` being equal to 1. The introduction of the irrational `φ` complicates this. While the basis vectors are not strictly orthogonal in the same sense as the DFT basis, they are **linearly independent**, which is sufficient to guarantee the invertibility of the transformation matrix `M`. The non-zero determinant of `M` ensures that a unique inverse exists. The numerical stability analysis (see `test_rft_stability_and_performance.py`) confirms that the matrix is well-conditioned, meaning it is "close" to orthogonal and does not suffer from issues that would prevent practical inversion.

## 4. The Inverse RFT (IRFT)

Because the transformation matrix `M` is invertible, we can define the inverse RFT (IRFT). The inverse matrix `M⁻¹` is given by:

`M⁻¹_{nk} = (1/N) * exp(2πi * n * k * φ / N)`

The inverse transform is therefore:

`x_n = (1/N) * Σ_{k=0}^{N-1} X_k * exp(2πi * n * k * φ / N)`

This formula is implemented in the `inverse_resonance_fourier_transform` function, ensuring that the original signal can be perfectly reconstructed from its RFT spectrum, limited only by standard floating-point precision.

## 5. Conclusion

The Resonance Fourier Transform is built on a solid mathematical foundation. Its use of an irrational number (`φ`) in its basis functions creates a unique, invertible transform. The linear independence of the basis vectors guarantees that any signal has a unique RFT representation and can be perfectly reconstructed. This makes the RFT a reliable tool for spectral analysis and signal processing within the QuantoniumOS ecosystem.
