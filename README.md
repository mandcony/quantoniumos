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

## What’s New (TL;DR)

**Φ-RFT (closed-form, fast).** Let \(F\) be the unitary DFT (`norm="ortho"`). Define diagonal phases  
\([C_\sigma]_{kk}=\exp(i\pi\sigma k^2/n)\), \([D_\phi]_{kk}=\exp(2\pi i\,\beta\,\{k/\phi\})\) with \(\phi=(1+\sqrt5)/2\).  
Set **\(\Psi = D_\phi\,C_\sigma\,F\)**.

- **Unitary by construction:** \(\Psi^\dagger \Psi = I\).
- **Exact complexity:** **\(\mathcal O(n\log n)\)** (FFT/IFFT + two diagonal multiplies).
- **Exact diagonalization:** twisted convolution \(x\star_{\phi,\sigma}h=\Psi^\dagger\!\operatorname{diag}(\Psi h)\Psi x\) is **commutative**/**associative**, and \(\Psi(x\star h)=(\Psi x)\odot(\Psi h)\).
- **Not LCT/FrFT/DFT-equivalent:** golden-ratio phase is **provably non-quadratic** (via Sturmian sequence properties) for \(\beta \notin \mathbb{Z}\); distinct from LCT/FrFT classes.

For proofs and tests, see **`docs/RFT_THEOREMS.md`** and **`tests/rft/`**.

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

**Validated (N=128–512):**
- Round-trip error ≈ **3e-16** relative.  
- Twisted-conv commutator ≈ **1e-15** (machine precision).  
- LCT non-equivalence: quadratic residual ≈ **0.3–0.5 rad RMS**; DFT correlation max < **0.25**; \(|\Psi^\dagger F|\) column entropy > **96%** of uniform.

---

## Compression

- **Lossless Vertex Codec:** exact spectral storage of tensors in Φ-RFT domain with SHA-256 integrity.  
- **Hybrid Learned Codec:** Φ-RFT → banding → prune/quantize (log-amp + phase) → tiny residual MLP → ANS.  
- Goals: **energy compaction**, **sparsity**, reproducible benchmarking vs DCT/DFT.

---

## Cryptography (Research-Only)

**RFT–SIS Hash v3.1** *(experimental)*  
- **Avalanche:** ~**50% ±3%** bit flips for 1-ulp input deltas.  
- **Collisions:** 0 / 10k in current suite.  
- **Security:** SIS-flavored parameters; **no formal reduction**. Note that **diffusion ≠ security**; this is an experimental cipher without formal cryptanalysis (linear/differential/boomerang/etc.). **Do not** use for production security.

---

## Hardware Implementation

**8-Point Φ-RFT FPGA Core** *(synthesized & verified)*

### WebFPGA Synthesis (iCE40 HX8K)
- **File:** `hardware/fpga_top.sv`
- **LUT4 Usage:** 1,884 / 7,680 (35.68%)
- **Flip-Flops:** 599 (11.34%)
- **Achieved Frequency:** 21.90 MHz (21.9× target)
- **Bitstream:** Generated and ready for flash
- **Status:** ✅ Successfully synthesized on WebFPGA

### Complete RFT Middleware Engine (Icarus Verilog)
**EDA Playground:** https://www.edaplayground.com/s/4/188

**Architecture (4 modules):**
1. **CORDIC Engine** - 12-iteration cartesian-to-polar conversion
   - Atan lookup table with 12 entries
   - Gain factor: 0.6073 (16'h9B74)
   - Outputs magnitude + phase in fixed-point radians

2. **Complex Multiplier** - Full complex arithmetic
   - (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
   - 16-bit fixed-point Q format

3. **8×8 RFT Kernel ROM** - Pre-computed coefficients
   - 64 complex coefficients (k=0 to k=7, n=0 to n=7)
   - Orthonormal DFT basis (scaled by 1/√8)
   - DC component (k=0): all equal (0x2D41)
   - Nyquist (k=4): alternating ±0x2D41

4. **RFT Middleware Engine** - Complete pipeline
   - State machine: IDLE → COMPUTE_RFT → EXTRACT_POLAR → OUTPUT
   - Sequential MAC: 64 multiply-accumulate operations
   - CORDIC polar extraction for all 8 frequency bins
   - Total resonance energy calculation

**Test Coverage (10 patterns):**
1. Impulse (delta function) - validates unitary transform
2. Null input (all zeros)
3. DC component (constant value 0x08)
4. Nyquist frequency (alternating 0x00/0xFF)
5. Linear ramp (0x00-0x07)
6. Step function (half-wave)
7. Symmetric pattern (triangle wave)
8. Complex pattern (hex sequence 0x0123456789ABCDEF)
9. Single high value (0xFF at last byte)
10. Two peaks (endpoints 0x80)

**Verified Capabilities:**
- ✅ CORDIC: 12-iteration cartesian-to-polar conversion
- ✅ Complex multiply-accumulate with 64 coefficients
- ✅ Full 8×8 resonance kernel ROM
- ✅ Amplitude extraction with CORDIC gain compensation
- ✅ Phase extraction in fixed-point radians
- ✅ Total energy calculation across frequency domain

**What This Proves:**
- ✅ Φ-RFT implementable in real digital logic
- ✅ CORDIC-based complex transform pipeline works
- ✅ Resource-efficient (<36% LUT usage on low-cost FPGA)
- ✅ Timing closure achieved (21.90 MHz)
- ✅ Complete frequency domain analysis functional

**Files:**
- `hardware/fpga_top.sv` - WebFPGA synthesizable 8-point RFT
- `hardware/rft_middleware_engine.sv` - Complete 4-module pipeline (Icarus)
- `hardware/quantoniumos_unified_engines.sv` - Full system (simulation only)
- `hardware/makerchip_rft_closed_form.tlv` - TL-Verilog for online verification
- `hardware/test_logs/` - Simulation results with waveforms
- EDA Playground: Comprehensive testbench with frequency analysis output

---

## What's Verified (at a glance)

- ✅ **Φ-RFT unitarity:** exact by factorization; numerically at machine-epsilon.  
- ✅ **Round-trip:** ~1e-16 relative error.  
- ✅ **Twisted-algebra diagonalization:** commutative/associative via \(\Psi\)-diagonalization.  
- ✅ **Non-equivalence to LCT/FrFT/DFT:** multiple independent tests.  
- ✅ **RFT–SIS avalanche:** ~50% ±3%.  
- ✅ **Hardware synthesis:** 8-point RFT on WebFPGA (iCE40 HX8K, 21.90 MHz, 35.68% LUT usage).  
- ✅ **Simulation verification:** Icarus Verilog + Makerchip (EDA Playground).  
- 🔬 **Compression benchmarks:** preliminary small-scale results; larger cross-validation runs in progress.

See `tests/`, `algorithms/crypto/crypto_benchmarks/rft_sis/`, and `docs/reports/CLOSED_FORM_RFT_VALIDATION.md` for an end-to-end empirical summary.

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
