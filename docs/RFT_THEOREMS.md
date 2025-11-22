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
 
                                                                              ## Open Conjectures and Analytic Proof Roadmap

                                                                              This section records conjectural statements and a proposed analytic path toward rigorous proofs. These are **not yet proven**; they summarize the intended direction for future work.

                                                                              ### Conjecture 1 — Perturbation Lemma for Kernel-to-Closed-Form Equivalence

                                                                              Let
                                                                              \[
                                                                              K_{ij} = g_{ij} \exp\big(2\pi i\,\beta\, \varphi_i\, \varphi_j\big), \qquad
                                                                              \widetilde K_{ij} = g_i h_j \exp\big(2\pi i\,\beta\, \tfrac{i}{\phi}\, \tfrac{j}{\phi}\big),
                                                                              \]
                                                                              where

                                                                              - the envelope is approximately separable, \(g_{ij} = g_i h_j + E_{ij}\) with \(\|E\|_F \leq \eta\),
                                                                              - the embedding is near-Sturmian, \(\varphi_i = \tfrac{i}{\phi} + \delta_i\) with \(|\delta_i| \leq \epsilon\),
                                                                              - \(\phi\) is the golden ratio and \(\beta \notin \mathbb Z\).

                                                                              Let \(\Psi = D_\phi C_\sigma F\) be the closed-form transform and let \(Q\) be the orthonormal factor obtained from QR (or SVD) of \(K\). Then for fixed \(n\) and sufficiently small \(\eta, \epsilon\), we conjecture
                                                                              \[
                                                                              \inf_{U \in \mathcal U_n} \|Q - \Psi U\|_F \leq C_n (\eta + \epsilon),
                                                                              \]
                                                                              where \(\mathcal U_n\) denotes the unitary group (absorbing column-wise phase/rotation) and \(C_n\) grows at most polynomially in \(n\). This formalizes the heuristic that the historical kernel construction and the closed-form factorization define the “same” basis up to small perturbations.

                                                                              **Status.** Empirically supported by low Frobenius residuals between simulated kernels and \(D_\phi C_\sigma F\). A complete proof is **open** and expected to require nonlinear perturbation bounds tailored to low-discrepancy embeddings.

                                                                              ### Conjecture 2 — Deterministic Performance Guarantees via Discrepancy

                                                                              Consider the RFT-induced linear operator \(T\) on signals or feature vectors, e.g. a restricted submatrix of \(\Psi\) used for compression or sensing. Let the associated kernel be
                                                                              \[
                                                                              K_{ij} = g_{ij} \exp\big(2\pi i\,\beta\, \{i/\phi\}\, \{j/\phi\}\big),
                                                                              \]
                                                                              with envelope \(g_{ij}\) derived from a smooth window or sampling pattern.

                                                                              We conjecture that, for suitable smoothness and decay assumptions on \(g_{ij}\), the low-discrepancy properties of the sequence \(\{i/\phi\}\) yield **deterministic bounds** on:

                                                                              - **Restricted isometry / near-isometry:** singular values of appropriate submatrices of \(T\) remain within \([1-\delta, 1+\delta]\) for prescribed sparsity levels,
                                                                              - **Worst-case energy dispersion:** for all unit vectors \(x\), the coefficients \(Tx\) avoid pathological concentration patterns (quantified via entropy or Gini-type functionals),
                                                                              - **Approximation/compression error:** best-\(k\)-term approximation errors in the RFT domain admit bounds comparable to or better than classical Fourier/chirp-based systems for quasi-periodic classes.

                                                                              These guarantees are expected to be **deterministic**, i.e. independent of random draws, instead leveraging the equidistribution and discrepancy of the golden-ratio sequence.

                                                                              **Status.** Currently supported only by empirical evidence from `test_rft_advantages.py`, `test_rft_comprehensive_comparison.py`, and the non-equivalence tests in `tests/rft/`. A rigorous treatment remains **conjectural**.

                                                                              ### Proposed Analytic Proof Path

                                                                              The high-level analytic route for Conjectures 1–2 is as follows.

                                                                              1. **Express \(K\) in a discrepancy-compatible form.**
                                                                                 - Rewrite \(K_{ij}\) as a Riemann-sum discretization of a bivariate function
                                                                                   \(f(x,y) = g(x,y) e^{2\pi i\,\beta x y}\) evaluated along the sequence
                                                                                   \(x_i = \{i/\phi\}, y_j = \{j/\phi\}\).
                                                                                 - Identify the star-discrepancy \(D_N\) of the 2D point set \(\{(x_i,y_j)\}\) and connect it to the classical low-discrepancy results for Kronecker sequences.

                                                                              2. **Impose a precise smoothness model on \(g_{ij}\).**
                                                                                 - Model \(g(x,y)\) as a function of bounded variation in the sense of Hardy–Krause or with a suitable Sobolev norm, so that a Koksma–Hlawka-type inequality applies to integrals of the form
                                                                                   \(\int\! f(x,y)\,\mathrm dx\mathrm dy\) versus its discrete average over \((x_i,y_j)\).
                                                                                 - Translate the discrete envelope \(g_{ij}\) into samples of \(g(x,y)\) with controlled discretization error, separating **envelope error** from **sampling-discrepancy error**.

                                                                              3. **Apply Koksma–Hlawka (or extensions) to bound kernel deviations.**
                                                                                 - Use Koksma–Hlawka to bound the difference between the idealized integral kernel and its discrete realization built from \(\{i/\phi\}\), yielding bounds of the form
                                                                                   \(\|K - \overline K\| \lesssim V(f) D_N\), where \(V(f)\) is a variation norm of \(f\).
                                                                                 - Here \(\overline K\) should be chosen to factor cleanly into the closed-form \(D_\phi C_\sigma F\) (or a closely related separable kernel), so that the residual \(K-\overline K\) is controlled.

                                                                              4. **Lift kernel bounds to operator norms.**
                                                                                 - Use standard operator-norm bounds (e.g., Schur tests, Hilbert–Schmidt norms) to translate the kernel discrepancy \(\|K-\overline K\|\) into spectral norm bounds on the corresponding operators.
                                                                                 - For submatrices relevant to sensing/compression, combine discrepancy-based kernel bounds with matrix concentration or Gershgorin-type arguments to obtain restricted isometry-type guarantees.
                                                                                 - Where necessary, use Schur complements to reason about how perturbations in one block of the operator influence the conditioning of the full transform.

                                                                              5. **Specialize to RFT envelopes and quasi-periodic signal models.**
                                                                                 - Choose concrete envelope families \(g_{ij}\) (e.g. smooth tapers, localized windows) compatible with the RFT implementation and show they satisfy the smoothness/variation assumptions.
                                                                                 - For quasi-periodic signal classes, characterize how the low-discrepancy sampling interacts with sparsity patterns to yield improved worst-case bounds compared to classical Fourier/chirp transforms.

                                                                              The completion of this program would turn the presently empirical observations into rigorous, deterministic theorems about kernel stability and performance of the Phi-RFT.

