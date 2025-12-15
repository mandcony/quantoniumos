# QuantoniumOS – Reproducible Results Guide

This document tells you **exactly** how to go from a fresh clone of
`github.com/mandcony/quantoniumos` to the same benchmark tables and logs
shown in the repository.

---

## Quick Start: Proof & Validation CLI

**NEW (December 2025):** Use the unified CLI runner for all proofs:

```bash
# Quick validation (~2 min) - runs all fast proofs
python scripts/run_proofs.py --quick

# Full validation suite - runs everything
python scripts/run_proofs.py --full

# List all available proofs
python scripts/run_proofs.py --list

# Run specific category
python scripts/run_proofs.py --category unitarity      # Verify Ψ^H Ψ = I
python scripts/run_proofs.py --category non-equivalence # RFT ≠ permuted DFT
python scripts/run_proofs.py --category hardware       # FPGA kernel validation
python scripts/run_proofs.py --category sparsity       # Domain-specific sparsity
python scripts/run_proofs.py --category coherence      # Zero-coherence cascade
python scripts/run_proofs.py --category paper-claims   # Paper theorem validation

# Generate JSON report
python scripts/run_proofs.py --quick --report results/proof_report.json
```

---

## 1. System Requirements

Tested on:

- **OS**: Ubuntu 22.04 (Linux x86_64)
- **Python**: 3.12
- **CPU**: x86_64 with AVX2 (optional but recommended)

You need:

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake nasm python3-dev python3-venv \
    git
```

---

## 2. Clone + Python environment

```bash
git clone https://github.com/mandcony/quantoniumos.git
cd quantoniumos

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .  # uses pyproject.toml
```

If `pip install -e .` fails, please open an issue with the full error log.

---

## 3. Build native Φ-RFT engine (`rftmw_native`)

This step compiles the C++/ASM extension. If it fails, the Python fallback
still works but some benchmarks will be slower.

```bash
cd src/rftmw_native
mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DRFTMW_ENABLE_ASM=ON

make -j"$(nproc)"

# Back to repo root
cd ../../..
```

You should now have a file like:

- `src/rftmw_native/build/rftmw_native.cpython-3*.so`

Python should see it as `import rftmw_native`.

---

## 4. Quick sanity check

From the repo root:

```bash
source .venv/bin/activate

python - << 'EOF'
import numpy as np

try:
    import rftmw_native
    print("rftmw_native:", getattr(rftmw_native, "__version__", "<no __version__>"))
except ImportError as e:
    print("NO rftmw_native:", e)

from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT

x = np.random.randn(1024)
engine = CanonicalTrueRFT(size=1024, backend="python")  # pure Python reference
X = engine.forward(x)
x_rec = engine.inverse(X)
print("Python RFT L2 error:", np.linalg.norm(x - x_rec))
EOF
```

You should see an error ~1e-13 for the Python reference backend.

---

## 5. Transform benchmark vs FFT / DCT / DWT

To reproduce the transform benchmark:

```bash
source .venv/bin/activate
cd experiments/competitors

python benchmark_transforms_vs_fft.py \
  --sizes 256,1024,4096 \
  --runs 10 \
  --output-dir ../../results/competitors
```

This will generate:

- `results/competitors/transform_benchmark_*.csv`
- `results/competitors/transform_benchmark_*.json`
- `results/competitors/transform_benchmark_*.md`

**Expected results (with optimized RFT):**

| Transform | Avg Time (µs) | Ratio to FFT |
|-----------|---------------|--------------|
| FFT | ~55 | 1.00× |
| **RFT (optimized)** | ~58 | **1.06×** |
| RFT (original) | ~272 | 4.97× |
| DCT | ~39 | 0.71× |

The optimized RFT achieves **4–7× speedup** over the original by fusing D_φ and C_σ diagonals.

---

## 6. Compression benchmark vs standard codecs

To reproduce the compression results:

```bash
source .venv/bin/activate
cd experiments/competitors

python benchmark_compression_vs_codecs.py \
  --datasets ascii random \
  --runs 2 \
  --output-dir ../../results/competitors
```

Outputs:

- `results/competitors/compression_benchmark_*.csv`
- `results/competitors/compression_benchmark_*.json`
- `results/competitors/compression_benchmark_*.md`

Interpretation:

- `brotli`, `zstd`, `lzma`, `zlib` ≈ entropy
- `rftmw_ans`, `rft_vertex` = current experimental codecs with large positive entropy gap (not competitive yet)

---

## 7. Crypto throughput / avalanche

To reproduce the crypto benchmark:

```bash
source .venv/bin/activate
cd experiments/competitors

python benchmark_crypto_throughput.py \
  --sizes 1024,4096 \
  --runs 3 \
  --output-dir ../../results/competitors
```

Outputs:

- `results/competitors/crypto_benchmark_*.csv`
- `results/competitors/crypto_benchmark_*.json`
- `results/competitors/crypto_benchmark_*.md`

This shows:

- AES-GCM / ChaCha20: high throughput
- RFT cipher: near-ideal avalanche but very low throughput (toy, not production crypto)

---

## 8. Native `rftmw_native` module test

From repo root:

```bash
source .venv/bin/activate
cd src/rftmw_native

python test_rftmw_native.py
```

(If this file doesn’t exist yet, see the TODO section in the docs.)

You should see a report like:

- Φ-RFT C++ engine reconstruction error (currently failing, known bug)
- ASM kernel tests (should pass)
- QuantumSymbolicCompressor test

---

## 9. Known limitations (as of this commit)

- `rftmw_native.RFTMWEngine` forward/inverse mismatch – reconstruction error ≈ 3.24 → C++ implementation not yet unitary.
- RFTMW compression (`rftmw_ans`, `rft_vertex`) is experimental and performs significantly worse than standard codecs.
- RFT cipher is a toy: good avalanche in tests but orders of magnitude slower than AES/ChaCha.

These limitations are intentional and documented; they are not advertised as production features.

---

## 10. Mathematical Proof Validation

The following proofs can be run to validate mathematical claims:

### Unitarity Proofs (Theorem 1)

```bash
# All 12 RFT variants satisfy Ψ^H Ψ = I with error < 1e-10
python scripts/run_proofs.py --category unitarity

# Or run pytest directly
pytest tests/rft/test_variant_unitarity.py -v
```

**Expected output:** 26 tests passing, unitarity error < 1e-13 for all variants.

### Non-Equivalence Proofs (Theorem 2)

```bash
# Prove RFT ≠ permuted/phased DFT
python experiments/proofs/non_equivalence_proof.py
python experiments/proofs/non_equivalence_theorem.py

# Or via CLI
python scripts/run_proofs.py --category non-equivalence
```

**Expected output:** Golden phase is non-affine, no permutation P exists such that Ψ = Λ₁ P F Λ₂.

### Sparsity Proofs (Theorem 3)

```bash
# Verify RFT sparsity advantage on golden quasi-periodic signals
python experiments/proofs/sparsity_theorem.py

# Full sparsity benchmark (slower)
python experiments/proofs/hybrid_benchmark.py
```

**Expected output:** Domain-specific sparsity on target signals. For current metrics, see [VERIFIED_BENCHMARKS](../research/benchmarks/VERIFIED_BENCHMARKS.md).

### Zero-Coherence Cascade (Theorem 4)

```bash
# Validate H3/FH5 cascade achieves η=0 coherence
python scripts/validate_paper_claims.py

# Quick check
python scripts/run_proofs.py --category coherence
```

**Expected output:** Coherence η = 0.00 for all cascade methods.

---

## 11. Hardware Validation (FPGA/TL-Verilog)

### Python-to-FPGA Kernel Verification

Verify that Python-generated RFT kernels match the FPGA ROM values:

```bash
python scripts/run_proofs.py --category hardware
```

**Expected output:**
- All 12 kernel variants match (768 ROM entries)
- Q1.15 fixed-point conversion verified
- ±1 LSB rounding tolerance

### FPGA Synthesis (requires Yosys)

```bash
cd hardware

# Synthesize for iCE40 HX8K (WebFPGA)
yosys -p "read_verilog -sv fpga_top.sv; synth_ice40 -top fpga_top" 2>&1 | tail -30

# Expected: ~2100 LUT4s, ~380 FLOPs, timing at 27+ MHz
```

### WebFPGA Timing (December 2025 validated)

| Metric | Result |
|--------|--------|
| LUT4s | 2,160 (40.91%) |
| FLOPs | 377 (7.14%) |
| IOs | 10 (25.64%) |
| Clock | 27.62 MHz |
| Target | 12 MHz (2.3× margin) |

### TL-Verilog Simulation (Makerchip)

1. Open https://makerchip.com
2. Copy contents of `hardware/rftpu_architecture.tlv`
3. Paste and click "Compile"
4. Run 100+ cycles, verify `tile_done_bitmap` activates

---

## 12. Paper Claims Reproducibility

### ASCII Wall Paper (Coherence-Free Hybrid Transforms)

```bash
python experiments/ascii_wall_paper.py
# Output: experiments/ASCII_WALL_PAPER_RESULTS.md
```

**Expected results:**
- H3 Cascade: BPP 0.655-0.669, η = 0.00
- FH5 Entropy-Guided: BPP 0.663, η = 0.00
- 17-19% compression gain over naive hybrid

### Full Paper Claims Validation

```bash
python scripts/validate_paper_claims.py
```

Validates all claims from the coherence-free hybrid transforms paper.

---

## 13. One-Command Full Validation

For complete reproducibility in one command:

```bash
# Full proof + benchmark suite
python scripts/run_proofs.py --full --report results/full_validation_$(date +%Y%m%d).json
```

This generates a timestamped JSON report with all test results.
