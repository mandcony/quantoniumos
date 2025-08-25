# QuantoniumOS

**Symbolic RFT • Geometric Waveform Hash • Resonance Encryption • OS Orchestrator**

QuantoniumOS is a research-grade framework that simulates quantum-style computing with **symbolic signals**. It includes a **generalized Fourier transform engine**, a **waveform/geometry-driven crypto stack**, and end-to-end **validators**.  


---

## Quick Start

> Python 3.10+ recommended. No `.env` required. Use a virtual environment.

```bash
# 1) Create & activate a venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
# If you keep a requirements file, install it here. Otherwise:
pip install numpy scipy networkx flask

# 3) Launch a runner (pick one)
# A) Unified CLI orchestrator
python quantonium_os_unified.py

# B) Full OS launcher (same root)
python launch_quantoniumos.py
Proven Tests (run these to verify)
1) Canonical RFT self-test (unitarity & round-trip)

bash
Copy
Edit
python canonical_true_rft.py
2) Full validator bundle (math/crypto/quantum)

bash
Copy
Edit
python run_all_validators.py
Outputs land in validation_results/ and JSON artifacts at repo root (e.g., definitive_quantum_validation_results.json, quantoniumos_validation_report.json).

What’s Inside (top-level highlights)
bash
Copy
Edit
01_START_HERE/          # Onboarding & navigation
02_CORE_VALIDATORS/     # Scientific validators (quantum, unitarity, crypto) 
03_RUNNING_SYSTEMS/     # Long-running / production-style apps
04_RFT_ALGORITHMS/      # Canonical & paper-compliant RFT code
05_QUANTUM_ENGINES/     # Engine adapters / kernels
06_CRYPTOGRAPHY/        # Waveform hash + resonance encryption
07_TESTS_BENCHMARKS/    # Test infrastructure
11_QUANTONIUMOS/        # OS layer (orchestration, runners)
13_DOCUMENTATION/       # Guides, reports, papers, legal
14_CONFIGURATION/       # Config files (no .env required)
15_DEPLOYMENT/          # Packaging / deployment scripts
src/quantoniumos/       # Installable package API (engines, algorithms)
validation_results/     # Saved outputs from recent runs
Other useful root scripts you can run directly:

quantonium_hpc_pipeline.py – pipeline/orchestration

verify_system.py – quick system checks

bulletproof_quantum_kernel.py, working_quantum_kernel.py – engine shims

paper_compliant_rft_fixed.py – RFT version aligned with manuscript

launch_gui.py – optional desktop launcher (if GUI deps installed)

## Usage Notes

**Configuration:** QuantoniumOS uses code-based configuration rather than `.env` files:
- **Class initialization** - Parameters set during object creation
- **JSON files** - Stored in `14_CONFIGURATION/` directory  
- **Command-line args** - For runtime options

```python
# Example configuration
os_instance = QuantoniumOS(quantum_dimension=8, rft_size=64, enable_crypto=True)
```

**Artifacts:** Runners save machine-verifiable artifacts (JSON reports) to the repo so others can reproduce your results.

**Security Note:** This is research-grade crypto, not production-ready. For production systems, follow the guidance in your Security/README notes and use peer-reviewed libraries alongside this research code.

## Citation
If this work helps your research, please cite the QuantoniumOS project and the associated preprints/DOIs.

bibtex
Copy
Edit
@software{QuantoniumOS_2025,
  title   = {QuantoniumOS: Symbolic RFT, Geometric Waveform Hash, Resonance Encryption, and OS Framework},
  author  = {Minier, Luis},
  year    = {2025},
  url     = {https://github.com/mandcony/quantoniumos}
}