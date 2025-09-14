Proof Note v1
=============

This note summarizes a formalization of the Unitary RFT (U), its kernel, unitarity, operator fingerprint, and a proof sketch that U is not equivalent to the DFT up to permutation and diagonal phases. The note contains named lemmas, a main theorem, and an appendix with small-N numerical certificates.

1. Kernel specification (K)

- Definition: K defines the transform U: C^N -> C^N by the kernel K(i,j; params) ...
- Domain: vectors of length N, complex entries.
- Parameters: resonance parameter r, normalization constant alpha, flags.

2. Unitarity and inverse

- Theorem 1 (Unitarity): For given normalization alpha and parameter ranges, U is unitary: U* U = I.
- Proof sketch: show kernel rows form orthonormal basis via inner-product computation; present exact algebraic steps.

3. Operator T fingerprint

- Define operator T that U diagonalizes: T = ??? (e.g., a structured sparse operator not equal to circular shift).
- Lemma: T has eigenvalue multiplicities and symmetry properties ...

4. Non-equivalence to DFT

- Theorem 2 (Non-equivalence): There do not exist permutation matrices P,Q and diagonal unitary D s.t. U = P D F Q.
- Proof strategy: show property X (e.g., diagonalizes circulant matrices) is satisfied by F but not by U. Produce small-N certificate.

5. Numerical appendix

- Small-N reproducible tables for N in {8,16,32,64} showing column-match residuals, diagonalization fractions, convolution off-diagonal norms.

6. Next steps

- Tighten algebraic proofs, produce full symbolic derivation, and prepare for peer review.

(End of draft — to be filled with exact kernel equations and computed certificates.)


## Numerical appendix (small-N)

| N | rft_median (s) | rft_norm t/(N log2N) | fft_median (s) | fft_norm |
|---:|---:|---:|---:|---:|
| 128 | 1.322000e-04 | 1.475446e-07 | 1.100000e-05 | 1.227679e-08 |
| 256 | 3.751000e-04 | 1.831543e-07 | 1.270000e-05 | 6.201173e-09 |
| 512 | 1.067100e-03 | 2.315755e-07 | 1.420000e-05 | 3.081597e-09 |
| 64 | 5.920000e-05 | 1.541667e-07 | 1.040000e-05 | 2.708333e-08 |
