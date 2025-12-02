# Getting Started with QuantoniumOS

**Your first steps with the Φ-RFT framework**

---

## ✅ Setup Complete!

Your QuantoniumOS repository is configured and ready to use.

**Current Status:**
- ✓ Virtual environment active (`.venv/`)
- ✓ Core dependencies installed (NumPy, SciPy, SymPy)
- ✓ RFT core operational (unitarity error: 2.72e-16 ✨)
- ⚠ Native extensions not built (optional for 10-40× speedup)

---

## 🎯 Quick Verification

Test that everything works:

\`\`\`bash
source .venv/bin/activate
python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np

rft = CanonicalTrueRFT(64)
x = np.random.randn(64)
y = rft.forward_transform(x)
error = rft.get_unitarity_error()

print(f'✓ RFT Core: OPERATIONAL')
print(f'  Unitarity error: {error:.2e}')
print(f'  Status: {"EXCELLENT" if error < 1e-12 else "GOOD"}')
"
\`\`\`

**Expected:** Unitarity error < 1e-12 (typically ~1e-16)

---

## 📚 Understanding the Architecture

QuantoniumOS uses a **multi-layer stack** where each layer provides different performance:

\`\`\`
Python (2-5 GB/s)    ← Always available, research-friendly
   ↓
C++ (30 GB/s)        ← Optional, production-grade
   ↓
C (8 GB/s)           ← Optional, portable fallback
   ↓
ASM (50 GB/s)        ← Optional, maximum performance
\`\`\`

**You're currently using:** Pure Python mode (no compilation needed!)

**Want 3-10× more speed?** Build the native engines:
```bash
# Build C/ASM kernel
cd algorithms/rft/kernels && make -j$(nproc) && cd ../../..

# Build C++ engine
cd src/rftmw_native && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON
make -j$(nproc)
cp rftmw_native.cpython-*-linux-gnu.so ../../../.venv/lib/python3.12/site-packages/
cd ../../..

# Verify
python -c "import rftmw_native; print(f'✓ Native engines: {rftmw_native.HAS_ASM_KERNELS}')"
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md#step-4-build-native-extensions-optional) for details.

---

## 🚀 Your First RFT Transform

\`\`\`python
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np

# Create RFT operator (size must be positive integer)
rft = CanonicalTrueRFT(size=1024)

# Generate test signal
x = np.random.randn(1024) + 1j * np.random.randn(1024)
x = x / np.linalg.norm(x)  # Normalize

# Forward transform: y = Ψx
y = rft.forward_transform(x)

# Inverse transform: x' = Ψ†y
x_reconstructed = rft.inverse_transform(y)

# Check round-trip accuracy
error = np.linalg.norm(x - x_reconstructed)
print(f"Round-trip error: {error:.2e}")  # Should be < 1e-10

# Validate unitarity: ||ΨΨ† - I|| ≈ 0
unitarity_error = rft.get_unitarity_error()
print(f"Unitarity error: {unitarity_error:.2e}")  # Should be < 1e-12
\`\`\`

---

## 🔬 Explore the 7 Unitary Variants

The Φ-RFT framework includes 7 proven unitary transforms:

\`\`\`python
from algorithms.rft.core.rft_variants import (
    RFT_VARIANT_STANDARD,        # Original Φ-RFT (golden ratio)
    RFT_VARIANT_HARMONIC,         # Cubic phase (curved time)
    RFT_VARIANT_FIBONACCI_TILT,   # Fibonacci lattice
    RFT_VARIANT_CHAOTIC_MIX,      # Haar-like random
    RFT_VARIANT_GEOMETRIC_LATTICE,# Pure geometric phase
    RFT_VARIANT_PHI_CHAOTIC,      # Hybrid (experimental)
    RFT_VARIANT_ENTROPY_GUIDED    # Adaptive meta-transform
)

# Example: Use harmonic variant
from algorithms.rft.kernels.python_bindings.unitary_rft import UnitaryRFT

rft_harmonic = UnitaryRFT(size=256, variant=RFT_VARIANT_HARMONIC)
y = rft_harmonic.forward_transform(x)
\`\`\`

**See:** \`experiments/hypothesis_testing/test_all_variants.py\`

---

## 🎵 Try the Audio Demo (QuantSoundDesign)

**QuantSoundDesign** is a professional sound design studio built on Φ-RFT:

\`\`\`bash
cd quantonium_os_src/apps/quantsounddesign/
python main.py
\`\`\`

**Features:**
- 7 RFT variants as oscillator modes
- Golden-ratio phase modulation
- 16-step pattern sequencer
- Real-time timbre morphing

---

## 📊 Run Benchmarks

Compare RFT performance across all 7 variants:

\`\`\`bash
python benchmarks/run_all_benchmarks.py
\`\`\`

**Output:** Benchmark results in \`data/scaling_results.json\`

---

## 🧪 Explore Experiments

All research experiments are reproducible:

\`\`\`bash
# ASCII Wall Paper (Theorem 10: coherence-free compression)
python experiments/ascii_wall/ascii_wall_paper.py

# Fibonacci Tilt validation
python experiments/fibonacci/fibonacci_tilt_hypotheses.py

# Scaling laws (N vs sparsity)
python scripts/verify_scaling_laws.py

# Full validation suite
python validate_system.py
\`\`\`

---

## 🧑‍💻 Development Workflow

\`\`\`bash
# Activate environment (always do this first!)
source .venv/bin/activate

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/rft/test_canonical_rft.py -v

# Profile performance
python -m cProfile -o profile.out your_script.py
python -m pstats profile.out

# Generate documentation
cd docs/
# Read markdown files (no build needed!)
\`\`\`

---

## 📖 Documentation Map

| Document | Purpose | Audience |
|:---------|:--------|:---------|
| **GETTING_STARTED.md** (this file) | First steps | Everyone |
| **SETUP_GUIDE.md** | Installation & troubleshooting | Users |
| **docs/ARCHITECTURE.md** | Technical deep dive | Developers |
| **docs/ARCHITECTURE_QUICKREF.md** | One-page cheat sheet | Quick reference |
| **QUICK_REFERENCE.md** | API reference | Developers |
| **REPRODUCING_RESULTS.md** | Benchmark reproduction | Researchers |
| **README.md** | Project overview | Everyone |

---

## 🎓 Learning Path

### Beginner: Understanding RFT
1. Read \`README.md\` (project overview)
2. Run \`verify_setup.sh\` (ensure working setup)
3. Try "Your First RFT Transform" example above
4. Explore \`docs/validation/RFT_THEOREMS.md\` (mathematical proofs)

### Intermediate: Research & Experiments
1. Read \`docs/ARCHITECTURE.md\` (understand the stack)
2. Run benchmarks: \`python benchmarks/run_all_benchmarks.py\`
3. Reproduce experiments: \`experiments/ascii_wall/ascii_wall_paper.py\`
4. Read research papers: \`papers/coherence_free_hybrid_transforms.tex\`

### Advanced: Performance Optimization
1. Read \`docs/ARCHITECTURE.md\` (multi-layer stack)
2. Build C/C++ extensions (see SETUP_GUIDE.md)
3. Profile your code: \`python -m cProfile\`
4. Implement custom variants (see \`algorithms/rft/core/rft_variants.py\`)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: \`git checkout -b feature/my-feature\`
3. Make changes and test: \`pytest tests/\`
4. Commit: \`git commit -m "Add feature"\`
5. Push: \`git push origin feature/my-feature\`
6. Open a Pull Request

**See:** \`CONTRIBUTING.md\` (if exists) or open an issue

---

## ❓ Common Questions

### Q: Do I need to compile anything?
**A:** No! Pure Python mode works out of the box. Compilation is optional for 10-40× speedup.

### Q: What's the difference between RFT and FFT?
**A:** RFT uses golden-ratio phase modulation (non-quadratic) instead of linear phase. It's mathematically distinct from FFT/DFT and optimized for quasi-periodic signals.

### Q: Can I use RFT for audio/image processing?
**A:** Yes! RFT is a general-purpose transform. See \`quantonium_os_src/apps/quantsounddesign/\` for audio examples.

### Q: Is this production-ready?
**A:** The core RFT framework is stable and validated. Crypto modules are **experimental only** (no hardness proofs). Compression is competitive but not a "breakthrough."

### Q: Can I use this commercially?
**A:** Core algorithms (RFT) are patent-pending and require a commercial license. Other code is AGPL-3.0. See \`LICENSE.md\` and \`LICENSE-CLAIMS-NC.md\`.

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'algorithms'"
**Fix:** Run from repository root and ensure virtual environment is active:
\`\`\`bash
cd /path/to/quantoniumos
source .venv/bin/activate
\`\`\`

### "Unitarity error > 1e-9"
**Fix:** Use double precision (\`np.complex128\`) and power-of-2 sizes:
\`\`\`python
x = np.random.randn(1024).astype(np.complex128)  # Not complex64
\`\`\`

### "ImportError: No module named 'rftmw'"
**Fix:** Native extensions not built. Use pure Python mode (always available) or see SETUP_GUIDE.md to build extensions.

**See full troubleshooting:** [SETUP_GUIDE.md#troubleshooting](SETUP_GUIDE.md#troubleshooting)

---

## 📬 Support

- **Issues:** https://github.com/mandcony/quantoniumos/issues
- **Discussions:** https://github.com/mandcony/quantoniumos/discussions
- **Documentation:** \`docs/\` directory
- **Email:** See \`CITATION.cff\` for contact info

---

## 🎉 You're Ready!

Your QuantoniumOS setup is complete. Start exploring:

\`\`\`bash
# Try the first example
python -c "from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT; print('✓ Ready to go!')"

# Explore experiments
ls experiments/

# Read the architecture
cat docs/ARCHITECTURE_QUICKREF.md
\`\`\`

**Happy researching! 🚀**

---

**QuantoniumOS:** Quantum-inspired research operating system  
**Φ-RFT:** Golden-ratio phase modulated unitary transform  
**DOI:** [10.5281/zenodo.17712905](https://doi.org/10.5281/zenodo.17712905)  
**Patent:** U.S. Application No. 19/169,399 (Pending)
