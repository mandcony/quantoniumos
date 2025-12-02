# QuantoniumOS – Reproducible Results Guide

This document tells you **exactly** how to go from a fresh clone of
`github.com/mandcony/quantoniumos` to the same benchmark tables and logs
shown in the repository.

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
