# QuantoniumOS Documentation Index

## Essential Documentation (READ THESE)

1. **README.md** - Project overview and quick start
2. **docs/manuals/COMPLETE_DEVELOPER_MANUAL.md** - Comprehensive technical reference
3. **docs/manuals/QUICK_START.md** - Get started in 15 minutes
4. **docs/research/TECHNICAL_SUMMARY.md** - Architecture and algorithms

## Specialized Documentation

### Development
- **docs/project/PROJECT_STRUCTURE.md** - Directory organization
- **docs/project/REPO_STRIP_VISION.md** - What to keep/strip + vision alignment
- **docs/validation/REPRODUCIBILITY.md** - How to reproduce results
- **docs/validation/RFT_VALIDATION_GUIDE.md** - Testing the RFT algorithms
- **docs/reports/CLOSED_FORM_RFT_VALIDATION.md** - Empirical evidence and novelty analysis for the closed-form Phi-RFT

### Applications

#### QuantSoundDesign (In Development)
- **src/apps/quantsounddesign/README.md** - Complete technical documentation
  - Architecture overview (gui.py → engine.py → audio_backend.py)
  - Φ-RFT integration points (UnitaryRFT + RFTMW)
  - Synth engine with RFT oscillators (280x timbre coverage)
  - Pattern editor with drum synthesis
  - Current status: Testing/In Development

### Experiments
- **experiments/README.md** - Complete experiment index
  - `hypothesis_testing/` - H1-H12 hypothesis battery (5 supported, 1 partial, 4 rejected)
  - `entropy/` - Information-theoretic analysis
  - `ascii_wall/` - ASCII Wall compression experiments
  - `fibonacci/` - Fibonacci tilt experiments
  - `tetrahedral/` - Tetrahedral geometry deep-dive
  - `sota_benchmarks/` - State-of-the-art comparisons
  - `corpus/` - Real-world corpus testing

### Hardware/FPGA
- **hardware/quantoniumos_engines_README.md** - Verilog/SystemVerilog hardware design
- **hardware/quantoniumos_unified_engines.sv** - Hardware implementation
- **hardware/CRITICAL_FIXES_REPORT.md** - Hardware synthesis fixes

### Patent/Legal
- **PATENT_NOTICE.md** - Patent information
- **LICENSE.md** - MIT License
- **LICENSE-CLAIMS-NC.md** - Patent claims license
- **CLAIMS_PRACTICING_FILES.txt** - Files practicing patent claims
- **docs/patent/USPTO_EXAMINER_RESPONSE_PACKAGE.md** - Patent response package

### Theory
- **docs/validation/RFT_THEOREMS.md** - Mathematical theorems
- **docs/research/theoretical_justifications.md** - Theoretical foundations
- **docs/MATHEMATICAL_FOUNDATIONS.md** - Math background
- **docs/theory/RFT_FRAME_NORMALIZATION.md** - Tight-frame proof + asymptotic orthogonality note (φ-grid)

### RFT Lineage
- **docs/project/RFT_EVOLUTION_MAP.md** - How the current φ-frame/Gram-corrected kernel emerged

### API/User Guides
- **docs/api/README.md** - API documentation
- **docs/user/README.md** - User guides

## Archived (Historical)
See `docs/archive/` for historical reports and planning documents.

## Scripts
- **validate_all.sh** - Run all validation tests
- **hardware/verify_fixes.sh** - Verify hardware fixes
- **hardware/generate_hardware_test_vectors.py** - Generate test data

---

## Project Structure

```
quantoniumos/
├── algorithms/          # Core RFT algorithms
│   └── rft/            # UnitaryRFT with 7 variants
├── experiments/         # All experimental validation
│   ├── ascii_wall/     # ASCII compression experiments
│   ├── hypothesis_testing/  # H1-H12 battery
│   ├── entropy/        # Information theory analysis
│   ├── fibonacci/      # Fibonacci tilt experiments
│   ├── tetrahedral/    # Geometry deep-dive
│   ├── sota_benchmarks/ # SOTA comparisons
│   └── corpus/         # Real-world testing
├── src/apps/quantsounddesign/ # QuantSoundDesign (testing)
├── hardware/            # FPGA/Verilog implementations
├── papers/              # Academic papers
├── docs/                # Documentation
└── quantonium_os_src/   # Core OS components
```

---

**Documentation Statistics:**
- Core documentation files: ~15
- Experiments with results: 7 categories
- Applications: QuantSoundDesign (testing phase)
- Total essential reading: ~8 files
- Archived historical docs: ~5

**Note**: If you're new, read in this order:
1. README.md
2. docs/manuals/QUICK_START.md
3. docs/manuals/COMPLETE_DEVELOPER_MANUAL.md
