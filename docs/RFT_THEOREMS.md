# Core Theorems of the Closed-Form Φ-RFT

Let \(F\) be the unitary DFT matrix with entries \(F_{jk} = n^{-1/2}\,\omega^{jk}\), \(\omega=e^{-2\pi i / n}\) (NumPy `norm="ortho"`). Indices are \(j, k \in \{0, \dots, n-1\}\).

**Conventions.** Congruence mod 1 means equality in \(\mathbb{R}/\mathbb{Z}\). Angles are taken mod \(2\pi\).

Define diagonal phase matrices
\[
[C_\sigma]_{kk} = \exp\!\Big(i\pi\sigma \frac{k^2}{n}\Big), \qquad
[D_\phi]_{kk}   = \exp\!\big(2\pi i\,\beta\,\{k/\phi\}\big),
\]
where \(\phi=\tfrac{1+\sqrt 5}{2}\) (golden ratio) and \(\{\cdot\}\) is fractional part.  
Set \(\Psi = D_\phi\,C_\sigma\,F\).

---

## Theorem 1 — Unitary Factorization (Symbolic Derivation)

**Statement.** The matrix \(\Psi = D_\phi C_\sigma F\) satisfies \(\Psi^\dagger \Psi = I\).

**Proof.**
1. **DFT Unitarity:** By definition, \(F\) is the normalized DFT matrix, so \(F^\dagger F = I\).
2. **Diagonal Phase Unitarity:**
   Let \(U\) be any diagonal matrix with entries \(U_{kk} = e^{i \theta_k}\) for \(\theta_k \in \mathbb{R}\).
      Then \((U^\dagger)_{jk} = \delta_{jk} e^{-i \theta_j}\).
         The product \((U^\dagger U)_{jk} = \sum_m (U^\dagger)_{jm} U_{mk} = \delta_{jk} e^{-i \theta_j} e^{i \theta_k} = \delta_{jk}\).
            Thus \(U^\dagger U = I\).
               Both \(C_\sigma\) and \(D_\phi\) are of this form.
               3. **Composition:**
                  \[
                     \begin{aligned}
                        \Psi^\dagger \Psi &= (D_\phi C_\sigma F)^\dagger (D_\phi C_\sigma F) \\
                           &= F^\dagger C_\sigma^\dagger \underbrace{D_\phi^\dagger D_\phi}_{I} C_\sigma F \\
                              &= F^\dagger \underbrace{C_\sigma^\dagger C_\sigma}_{I} F \\
                                 &= F^\dagger F = I.
                                    \end{aligned}
                                       \]
                                          \(\blacksquare\)

                                          **Inverse:** \(\Psi^{-1} = F^\dagger C_\sigma^\dagger D_\phi^\dagger\).  
                                          In code (NumPy): `ifft(conj(C)*conj(D)*y, norm="ortho")`.

                                          ---

                                          ## Theorem 2 — Exact Diagonalization of a Commutative Algebra
                                          Define Φ-RFT twisted convolution
                                          \[
                                          (x \star_{\phi,\sigma} h) \;=\; \Psi^\dagger\,\mathrm{diag}(\Psi h)\,\Psi x.
                                          \]
                                          Then
                                          \[
                                          \Psi(x \star_{\phi,\sigma} h) \;=\; (\Psi x)\odot(\Psi h).
                                          \]
                                          Hence \(\Psi\) simultaneously diagonalizes the algebra \(\mathcal A=\{\,\Psi^\dagger \mathrm{diag}(g) \Psi : g\in\mathbb C^n\,\}\), which is commutative and associative.

                                          ---

                                          ## Proposition 3 — Golden-ratio phase is not quadratic (thus not a chirp)

                                          Let \(\theta_k = 2\pi\beta \{k/\phi\}\) and \(D_\phi = \mathrm{diag}(e^{i\theta_k})\).
                                          If \(\beta \notin \mathbb{Z}\), then \(\theta_k/(2\pi)\) is not congruent mod 1 to any quadratic \(Ak^2 + Bk + C\). Hence \(D_\phi\) is not a quadratic-phase chirp \(e^{i\pi(ak^2+bk+c)/n}\).

                                          **Proof (second-difference/Sturmian).**
                                          Define the forward difference operator \(\Delta f(k) = f(k+1) - f(k)\) and second difference \(\Delta^2 f(k) = \Delta(\Delta f(k))\).
                                          With \(d_k = \lfloor \frac{k+1}{\phi} \rfloor - \lfloor \frac{k}{\phi} \rfloor \in \{0,1\}\),
                                          \[
                                          \Delta^2 \{k/\phi\} = -(d_{k+1} - d_k) \in \{-1, 0, 1\}.
                                          \]
                                          Assuming \(\beta \{k/\phi\} \equiv Ak^2 + Bk + C \pmod 1\) gives
                                          \[
                                          -\beta(d_{k+1} - d_k) \equiv 2A \pmod 1.
                                          \]
                                          Since \(d_{k+1} - d_k\) hits \(0, \pm 1\) infinitely often, we must have \(2A \equiv 0\), \(\beta \equiv 0\), and \(-\beta \equiv 0 \pmod 1\), forcing \(\beta \in \mathbb{Z}\) — contradiction. \(\blacksquare\)

                                          **Edge case.** For \(\beta \in \mathbb{Z}\) this test is inconclusive; no chirp-equivalence is claimed. We neither claim nor require chirp-equivalence when \(\beta \in \mathbb{Z}\).

                                          ## Theorem 4 — Non-LCT Nature (No parameters \(a,b,c,d\) exist)

                                          **Statement.** There exist no Linear Canonical Transform parameters \(M = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \in Sp(2, \mathbb{R})\) such that \(\Psi\) corresponds to the discrete LCT operator \(L_M\), provided \(\beta \notin \mathbb{Z}\).

                                          **Proof.**
                                          1. **Group Structure:** The set of discrete LCTs forms a group isomorphic to the metaplectic group \(Mp(2, \mathbb{R})\). This group is generated by Fourier transforms, scalings, and quadratic phase modulations (chirps).
                                          2. **Diagonal Subgroup:** Any element of this group that is a diagonal matrix must be a quadratic chirp of the form \(D_{kk} = e^{i \pi (\alpha k^2 + \gamma k + \delta)}\).
                                          3. **Contradiction:**
                                             Assume \(\Psi = D_\phi C_\sigma F\) is an LCT.
                                                Since \(C_\sigma\) (chirp) and \(F\) (DFT) are standard LCTs, their product \(L' = C_\sigma F\) is an LCT.
                                                   Since LCTs form a group, the inverse \((L')^{-1}\) is an LCT.
                                                      If \(\Psi\) is an LCT, then the product \(\Psi (L')^{-1}\) must be an LCT.
                                                         Substituting definitions:
                                                            \[
                                                               \Psi (C_\sigma F)^{-1} = (D_\phi C_\sigma F) (F^{-1} C_\sigma^{-1}) = D_\phi.
                                                                  \]
                                                                     Thus, \(D_\phi\) must be an LCT. Since \(D_\phi\) is diagonal, it must be a quadratic chirp.
                                                                        However, **Proposition 3** proves that the phase of \(D_\phi\) involves the fractional part function \(\{k/\phi\}\), which has non-vanishing second differences \(\Delta^2 \neq \text{const}\) and is provably not quadratic modulo 1.
                                                                           Therefore, \(D_\phi\) is not an LCT.
                                                                              Consequently, \(\Psi\) cannot be an LCT. \(\blacksquare\)

                                                                              **Scope.** We exclude only LCT/FrFT/metaplectic; other unitary families may share properties with \(\Psi\).

                                                                              ---

                                                                              ## Practical Tests (implemented in `tests/rft/`)
                                                                              - **Round-trip:** \(\|x - \Psi^{-1}\Psi x\|/\|x\| \approx 10^{-16}\).
                                                                              - **Commutator:** \(\|h_1\star(h_2\star x)-(h_2\star(h_1\star x))\|/\|x\| \approx 10^{-15}\).
                                                                              - **Non-equivalence:** large RMS residual to quadratic phase; low max DFT correlation; high entropy of \(\Psi^\dagger F\) columns.
                                                                              - **Sturmian Property:** `test_nonquadratic_phase.py` shows \(\ge 3\) residue classes for \(\Delta^2(\beta\{k/\phi\}) \pmod 1\) when \(\beta \notin \mathbb{Z}\).

                                                                              ---

                                                                              ### Historical Note
                                                                              An earlier formulation built \(\Psi\) via QR orthonormalization of a phase kernel. See Appendix A for details and equivalence assumptions.

                                                                              ---

                                                                              ## Appendix A — Alternative Kernel-Based Formulation (Historical)
                                                                              **Definition (Kernel Form).** Let
                                                                              \[
                                                                              K_{ij} = g_{ij}\,\exp\big(2\pi i\,\beta\, \varphi_i\, \varphi_j\big),
                                                                              \]
                                                                              with amplitude envelope \(g_{ij}\) and index embedding \(\varphi_k\) (e.g. \(\varphi_k = \{k/\phi\}\)). The transform was originally taken as
                                                                              \[
                                                                              \Psi = \mathrm{orth}(K)\quad (\text{e.g. QR, first \(n\) columns}).
                                                                              \]

                                                                              **Equivalence to Closed-Form.** Assume:
                                                                              1. (Approximate separability) \(g_{ij} \approx g_i h_j\) with low-rank residual.
                                                                              2. (Golden-ratio embedding) \(\varphi_i = \{i/\phi\}\) up to bounded perturbation \(|\delta_i| \leq \epsilon\).
                                                                              3. (Singular alignment) Leading left/right singular vectors of \(K\) align (componentwise phase) with \(D_\phi\) and \(C_\sigma F\) columns.
                                                                              Then after column normalization and global phase adjustment,
                                                                              \[
                                                                              \mathrm{orth}(K) \approx D_\phi C_\sigma F,
                                                                              \]
                                                                              with empirical Frobenius relative residual \(r_n = \|K - D_\phi C_\sigma F\|_F/\|K\|_F\) observed \(<10^{-3}\) for tested \(n\in[128,512]\). Formal bounds pending.

                                                                              **Disclaimer (Empirical Status).** The above alignment and residual are currently empirical; a proof requires bounding SVD perturbations under near-separable modulation and low-discrepancy index embeddings.

                                                                              **Practical Guidance.** For implementation and benchmarking use the closed-form \(\Psi = D_\phi C_\sigma F\): it avoids QR (\(\mathcal O(n^3)\) preprocessing), is numerically stable, and gives immediate \(\mathcal O(n\log n)\) apply complexity. The kernel view remains valuable for provenance and potential extensions (e.g. alternative envelopes \(g_{ij}\)).

                                                                              **Future Work.** Provide explicit perturbation lemma: if \(\|g_{ij} - g_i h_j\|_F \leq \eta\) and \(|\delta_i| \leq \epsilon\), then derive \(r_n = \mathcal O(\eta + \epsilon)\). Document envelope choices and their spectral effects.

                                                                              ---
                                                                               
                                                                               