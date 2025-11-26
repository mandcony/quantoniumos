# Paper Validation Test Plan

**Date:** November 25, 2025  
**Purpose:** Complete validation checklist for IEEE paper submission

---

## Python Tests (Can Run Now)

### Core Tests
```bash
# Run all Python tests at once
python3 run_quick_paper_tests.py

# Or run individually:
python3 scripts/irrevocable_truths.py                    # Theorems 1-7
python3 scripts/verify_scaling_laws.py                   # Theorem 3 (98% sparsity)
python3 scripts/verify_ascii_bottleneck.py               # Theorem 10
python3 scripts/verify_variant_claims.py                 # Theorems 4-7
python3 scripts/verify_rate_distortion.py                # Rate-distortion curves
python3 scripts/verify_performance_and_crypto.py         # RFT-SIS avalanche
```

### Expected Outputs
- ✅ `data/scaling_results.json` - Sparsity scaling to 98.63% at N=512
- ✅ `figures/latex_data/rate_distortion.csv` - For LaTeX plots
- ✅ All variant unitarity errors < 5×10⁻¹⁴

---

## Assembly/C Tests (Can Run Now)

```bash
# Assembly validation against Python reference
python3 tests/validation/test_assembly_variants.py
python3 tests/validation/test_assembly_vs_python_comprehensive.py
```

**Note:** Some assembly tests may fail if C bindings aren't compiled. This is OK for paper - focus on Python results.

---

## Verilog/FPGA Tests (You'll Run & Screenshot)

### Option 1: Icarus Verilog (If Available)

```bash
cd hardware

# Check if simulators exist
ls -lh sim_rft sim_quantoniumos

# Run RFT middleware test
./sim_rft

# Run full QuantoniumOS engines test
./sim_quantoniumos

# Or recompile if needed
make sim
```

**Expected Output:**
- Waveform files: `quantoniumos_unified.vcd`, `quantoniumos_full.vcd`
- Test results showing unitarity verification
- Screenshot the VCD waveforms in GTKWave

### Option 2: Makerchip (Online Verilog IDE)

1. Open https://makerchip.com/
2. Load `hardware/makerchip_rft_closed_form.tlv`
3. Click "Compile" and observe waveforms
5. (Repo snapshot) A local copy of the Makerchip TLV snapshot is included as `figures/makerchip_tlv_snapshot.tlv` — you can upload that file to Makerchip or use it as a reference for debugging.
4. **Screenshot:**
   - Waveform viewer showing RFT transform
   - Log output showing test results
   - Any visualization of golden-ratio phase progression

### Option 3: WebFPGA

1. Deploy `hardware/fpga_top.sv` or `hardware/quantoniumos_unified_engines.sv`
2. Program FPGA board
3. **Screenshot:**
   - WebFPGA console output
   - Waveform analyzer
   - Resource utilization report

---

## Test Results Checklist

### For Paper Section: "Experimental Validation"

| Test | Status | Data File | Screenshot Needed |
|:-----|:-------|:----------|:------------------|
| **Python: Irrevocable Truths** | ⬜ | `data/scaling_results.json` | No |
| **Python: Scaling Laws** | ⬜ | `data/scaling_results.json` | Optional (plot) |
| **Python: ASCII Bottleneck** | ⬜ | Terminal output | Optional |
| **Python: Variant Claims** | ⬜ | Terminal output | Optional |
| **Python: Rate-Distortion** | ⬜ | `figures/latex_data/rate_distortion.csv` | Yes (plot) |
| **Python: RFT-SIS Crypto** | ⬜ | Terminal output | Optional |
| **Verilog: Icarus Sim** | ✅ | `hardware/quantoniumos_unified.vcd` | Yes (GTKWave) |
| **Verilog: Makerchip** | ⬜ | N/A | Yes (waveforms) |
| **FPGA: WebFPGA** | ✅ | `hardware/WEBFPGA_SYNTHESIS_RESULTS.md` | Yes (utilization) |

---

## Screenshot Checklist

### Required for Paper

1. **Figure: Scaling Laws**
   - Script: `python3 scripts/verify_scaling_laws.py`
   - Shows: Sparsity increasing to 98.63% at N=512
   - Format: PNG or PDF from matplotlib

2. **Figure: Rate-Distortion Curves**
   - Script: `python3 scripts/verify_rate_distortion.py`
   - Shows: Hybrid DCT+RFT beats pure DCT/RFT
   - Format: PNG or PDF

3. **Figure/Table: Variant Differentiation**
   - Script: `python3 scripts/verify_variant_claims.py`
   - Shows: 7 variants with different use cases
   - Format: Table or bar chart

4. **Screenshot: Verilog Waveforms**
   - Tool: GTKWave or Makerchip
   - Shows: RFT transform in hardware
   - Format: PNG with visible signal names

5. **Screenshot: FPGA Utilization**
   - Tool: WebFPGA or Vivado synthesis report
   - Shows: Resource usage (LUTs, FFs, DSPs)
   - Format: PNG of resource report

### Optional but Recommended

6. **Screenshot: Unitarity Validation**
   - Show terminal output from `irrevocable_truths.py`
   - Proves all 7 variants have error < 5×10⁻¹⁴

7. **Screenshot: RFT-SIS Avalanche**
   - Show cryptographic avalanche metrics
   - Proves Fibonacci Tilt > DFT for lattice hashing

---

## Data Files to Include with Paper

```
submission/
├── data/
│   ├── scaling_results.json          # Theorem 3 validation
│   └── rate_distortion.csv           # Theorem 10 validation
├── figures/
│   ├── scaling_sparsity.pdf          # Sparsity vs N plot
│   ├── rate_distortion.pdf           # RD curves
│   ├── variant_comparison.pdf        # 7 variants table/chart
│   ├── verilog_waveforms.png         # Hardware simulation (GTKWave)
│   ├── webfpga_synthesis_screenshot.png  # WebFPGA synthesis console
│   ├── makerchip_tlv_snapshot.tlv    # Makerchip TLV snapshot (archival)
│   └── fpga_utilization.png          # Resource usage
└── code/
    ├── irrevocable_truths.py         # Main validation script
    ├── verify_scaling_laws.py
    ├── verify_rate_distortion.py
    └── hardware/
        ├── makerchip_rft_closed_form.tlv
        └── quantoniumos_unified_engines.sv
```

---

## Quick Start: Run Everything Now

```bash
# 1. Run all Python tests
python3 run_quick_paper_tests.py > python_test_results.txt 2>&1

# 2. Check the output
cat python_test_results.txt

# 3. Verify data files were generated
ls -lh data/scaling_results.json
ls -lh figures/latex_data/rate_distortion.csv

# 4. For Verilog (if you have Icarus installed)
cd hardware && ./sim_rft && ./sim_quantoniumos
```

Then you handle:
- Makerchip screenshots
- WebFPGA screenshots
- Any additional plots needed for paper

---

## Paper Claims to Validate

| Claim | Test | Status |
|:------|:-----|:-------|
| "7 unitary variants to machine precision" | `irrevocable_truths.py` | ⬜ |
| "Maximum unitarity deviation below 10⁻¹⁴" | `data/scaling_results.json` | ⬜ |
| "98.63% sparsity at N=512" | `verify_scaling_laws.py` | ⬜ |
| "Solves ASCII bottleneck" | `verify_ascii_bottleneck.py` | ⬜ |
| "Fibonacci Tilt optimal for RFT-SIS" | `verify_variant_claims.py` | ⬜ |
| "O(N log N) complexity" | Code inspection + FPGA timing | ⬜ |
| "Hardware implementation validated" | Verilog simulation | ⬜ |

---

## Notes

- Python tests should all pass (critical for paper)
- Assembly tests are optional (nice-to-have)
- Verilog/FPGA tests prove hardware viability (major claim)
- Screenshots make the paper more credible

**When you're ready, run:**
```bash
python3 run_quick_paper_tests.py
```

And let me know which tests pass/fail!
