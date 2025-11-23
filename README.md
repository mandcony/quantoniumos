# QuantoniumOS: Quantum-Inspired Research Operating System

> **PATENT-PENDING RESEARCH PLATFORM.** QuantoniumOS bundles:
> - the **Φ-RFT** (golden-ratio + chirp, **closed-form, fast** unitary transform),
> - **compression** pipelines (lossless + hybrid learned),
> - **cryptographic** experiments (RFT–SIS hashing),
> - and **comprehensive validation** suites.  
> All "quantum" modules are **classical simulations** or **quantum-inspired data structures** with explicit mathematical checks. They do not simulate physical quantum mechanics.

**USPTO Application:** 19/169,399 (Filed 2025-04-03)  
**Title:** *Hybrid Computational Framework for Quantum and Resonance Simulation*

---

## Core Mathematical Framework

**Φ-RFT Definition.** Let \(F\) be the unitary DFT (`norm="ortho"`). Define diagonal phases \([C_\sigma]_{kk}=\exp(i\pi\sigma k^2/n)\) and \([D_\phi]_{kk}=\exp(2\pi i\,\beta\,\{k/\phi\})\) where \(\phi=(1+\sqrt5)/2\). The transform is \(\Psi = D_\phi\,C_\sigma\,F\).

**Properties:**
- Unitary by construction: \(\Psi^\dagger \Psi = I\)
- Computational complexity: \(\mathcal O(n\log n)\) via FFT with diagonal pre/post-multiplication
- Diagonalizes twisted convolution: \(x\star_{\phi,\sigma}h=\Psi^\dagger\!\operatorname{diag}(\Psi h)\Psi x\) with commutativity and associativity
- Provably distinct from LCT/FrFT/DFT classes: golden-ratio phase is non-quadratic for \(\beta \notin \mathbb{Z}\) (Sturmian sequence analysis)

Mathematical proofs and validation tests: `docs/RFT_THEOREMS.md`, `tests/rft/`

---

## Repository Layout

```
QuantoniumOS/
├─ algorithms/
│  ├─ rft/core/                 # Φ-RFT core + tests
│  ├─ compression/              # Lossless & hybrid codecs
│  └─ crypto/                   # RFT–SIS experiments & validators
├─ os/                          # Desktop apps & visualizers
├─ tools/                       # Dev helpers, benchmarking, data prep
├─ tests/                       # Unit, integration, validation
├─ docs/                        # Tech docs, USPTO packages
└─ data/                        # Configs, fixtures
```

---

## Quick Start

```bash
# 1) Environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 2) Install
pip install -e .[dev,ai,image]

# 3) (Optional) Build native kernels
make -C algorithms/rft/kernels all
export RFT_KERNEL_LIB=$(find algorithms/rft/kernels -name 'libquantum_symbolic.so' | head -n1)

# 4) Run tests
pytest -m "not slow"

# 5) Launch desktop tools
python quantonium_boot.py
```

---

## Φ-RFT: Reference API (NumPy)

```python
import numpy as np
from numpy.fft import fft, ifft

PHI = (1.0 + np.sqrt(5.0)) / 2.0

def _frac(v):
    return v - np.floor(v)

def rft_forward(x, *, beta=0.83, sigma=1.25):
    x = np.asarray(x, dtype=np.complex128)
    n  = x.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)
    return D * (C * fft(x, norm="ortho"))

def rft_inverse(y, *, beta=0.83, sigma=1.25):
    y = np.asarray(y, dtype=np.complex128)
    n  = y.shape[0]
    k  = np.arange(n, dtype=np.float64)
    D  = np.exp(2j*np.pi*beta*_frac(k/PHI))
    C  = np.exp(1j*np.pi*sigma*(k*k)/n)
    return ifft(np.conj(C) * np.conj(D) * y, norm="ortho")

def rft_twisted_conv(a, b, *, beta=0.83, sigma=1.25):
    A = rft_forward(a, beta=beta, sigma=sigma)
    B = rft_forward(b, beta=beta, sigma=sigma)
    return rft_inverse(A * B, beta=beta, sigma=sigma)
```

**Validated Results (N=128–512):**
- Round-trip error: 3×10⁻¹⁶ relative (machine precision)
- Twisted convolution commutator: 1×10⁻¹⁵ (numerical verification of algebraic closure)
- LCT non-equivalence metrics: quadratic residual 0.3–0.5 rad RMS, DFT correlation max < 0.25, column entropy > 96% of uniform distribution

---

## Compression Codecs

**Lossless Vertex Codec:** Exact spectral representation of tensors in Φ-RFT domain with SHA-256 integrity verification.

**Hybrid Learned Codec:** Multi-stage pipeline comprising Φ-RFT transform, frequency band pruning, logarithmic amplitude and phase quantization, residual prediction via MLP, and ANS entropy coding.

Objective: Empirical comparison of energy compaction and sparsity properties against DCT/DFT baselines with reproducible benchmarking methodology.

---

## Cryptographic Constructions (Experimental)

**RFT-SIS Hash v3.1** (Research prototype without security proofs)

Empirical properties:
- Avalanche effect: 50% ± 3% bit flips for 1-ulp input perturbations
- Collision resistance: 0 collisions observed in 10,000-sample test suite
- Construction: SIS-inspired lattice parameters applied to Φ-RFT spectral domain

**Security status:** No formal cryptanalytic reduction provided. Diffusion properties do not constitute security proof. Linear, differential, boomerang, and related-key analyses not performed. Not suitable for production cryptographic applications.

---

## Hardware Implementation

**8-Point Φ-RFT FPGA Synthesis**

### WebFPGA Deployment (Lattice iCE40 HX8K)
- Design: `hardware/fpga_top.sv`
- Resource utilization: 1,884 LUT4 / 7,680 (35.68%), 599 flip-flops (11.34%)
- Timing: 21.90 MHz achieved (target: 1.00 MHz)
- Status: Bitstream generated, ready for device programming

### Icarus Verilog Simulation
Comprehensive testbench: https://www.edaplayground.com/s/4/188

**Architecture (4 modules):**

1. **CORDIC Engine:** 12-iteration CORDIC algorithm for cartesian-to-polar conversion. Implements atan lookup table (12 entries), gain compensation factor 0.6073, outputs magnitude and phase in Q1.15 fixed-point radians.

2. **Complex Multiplier:** Combinational logic implementing (a + bi)(c + di) = (ac - bd) + (ad + bc)i in 16-bit fixed-point arithmetic with appropriate scaling.

3. **8×8 RFT Kernel ROM:** Pre-computed complex coefficients representing orthonormal DFT basis scaled by 1/√8. Contains 64 entries indexed by frequency k (0-7) and sample n (0-7). DC component (k=0) uniform at 0x2D41, Nyquist (k=4) alternates ±0x2D41.

4. **RFT Middleware Engine:** State machine pipeline (IDLE → COMPUTE_RFT → EXTRACT_POLAR → OUTPUT) executing 64 complex multiply-accumulate operations, CORDIC polar extraction for 8 frequency bins, and total resonance energy computation.

**Verification Test Suite (10 patterns):**
1. Impulse (delta function): validates unitary property
2. Null input: zero vector handling
3. DC component: constant value 0x08
4. Nyquist frequency: alternating 0x00/0xFF pattern
5. Linear ramp: ascending sequence 0x00-0x07
6. Step function: half-wave discontinuity
7. Symmetric pattern: triangle wave
8. Complex pattern: hexadecimal sequence 0x0123456789ABCDEF
9. Single peak: isolated high value 0xFF at terminal byte
10. Dual peaks: endpoints 0x80

**Validated Functionality:**
- CORDIC 12-iteration cartesian-to-polar conversion
- Complex multiply-accumulate across 64 coefficient pairs
- Resonance kernel ROM with 8×8 spectral basis
- Amplitude extraction with CORDIC gain compensation
- Phase extraction in fixed-point radian representation
- Frequency domain energy summation

**Implementation Results:**
- Φ-RFT realizable in synthesizable digital logic
- CORDIC-based complex transform pipeline operational
- Resource utilization under 36% on low-cost FPGA architecture
- Timing closure achieved at 21.90 MHz
- Complete frequency domain analysis with magnitude, phase, and energy metrics

**Implementation Files:**
- `hardware/fpga_top.sv`: WebFPGA-synthesizable 8-point RFT core
- `hardware/rft_middleware_engine.sv`: Complete 4-module pipeline (Icarus Verilog)
- `hardware/quantoniumos_unified_engines.sv`: Extended system architecture (simulation)
- `hardware/makerchip_rft_closed_form.tlv`: Transaction-Level Verilog for browser-based verification
- `hardware/test_logs/`: Simulation outputs and waveform captures
- EDA Playground repository: Complete testbench with frequency domain analysis

---

## Verification Status

- Φ-RFT unitarity: Exact by algebraic factorization, numerically verified at machine epsilon
- Round-trip error: Order 10⁻¹⁶ relative (double-precision limit)
- Twisted-algebra diagonalization: Commutativity and associativity verified via Ψ-diagonalization
- Non-equivalence to LCT/FrFT/DFT: Multiple independent mathematical tests
- RFT-SIS avalanche: 50% ± 3% (experimental observation)
- Hardware synthesis: 8-point RFT implemented on WebFPGA iCE40 HX8K (21.90 MHz, 35.68% LUT utilization)
- Simulation verification: Icarus Verilog and Makerchip TL-V (EDA Playground)
- Compression benchmarks: Preliminary results on synthetic data; larger-scale validation in progress

Comprehensive validation documentation: `tests/`, `algorithms/crypto/crypto_benchmarks/rft_sis/`, `docs/reports/CLOSED_FORM_RFT_VALIDATION.md`

---

## Patent & Licensing

> **License split.** Most of this repository is licensed under **AGPL-3.0-or-later** (see `LICENSE.md`).  
> Files explicitly listed in **`CLAIMS_PRACTICING_FILES.txt`** are licensed under **`LICENSE-CLAIMS-NC.md`** (research/education only) because they practice methods disclosed in **U.S. Patent Application No. 19/169,399**.  
> **No non-commercial restriction applies to any files outside that list.**  
> Commercial use of the claim-practicing implementations requires a separate patent license from **Luis M. Minier** (contact: **luisminier79@gmail.com**).  
> See `PATENT_NOTICE.md` for details. Trademarks (“QuantoniumOS”, “RFT”) are not licensed.  
> For a scenario-by-scenario breakdown (research vs. commercial), review `docs/licensing/LICENSING_OVERVIEW.md`.

---

## Key Paths

```
algorithms/rft/core/canonical_true_rft.py      # Φ-RFT (claims-practicing)
algorithms/compression/                        # Lossless + hybrid codecs
algorithms/crypto/crypto_benchmarks/rft_sis/   # RFT–SIS validation suite
tests/                                         # Unit + integration tests
docs/USPTO_*                                   # USPTO packages & analysis
```

---

## License

**License split:** Most of this repository is **AGPL-3.0-or-later** (see `LICENSE.md`).
Files listed in **`CLAIMS_PRACTICING_FILES.txt`** are licensed under **`LICENSE-CLAIMS-NC.md`**
(research/education only) because they practice methods in U.S. Patent Application
No. 19/169,399. Commercial rights require a separate patent license from the author.

See `PATENT_NOTICE.md` for details on the patent-pending technologies.

---

## Contributing

PRs welcome for:
- fast kernels / numerical analysis,
- compression benchmarks on real models,
- formal crypto reductions and audits,
- docs, tests, and tooling.

Please respect the license split (AGPL vs research-only claim-practicing files).

---

## Contact

**Luis M. Minier** · **luisminier79@gmail.com**  
Commercial licensing, academic collaborations, and security reviews welcome.
