# Real Data Integration (Research-Only)

⚠️ **RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE** ⚠️

This folder holds optional fetch scripts for external research datasets. **No data is bundled.** Users must acknowledge licenses/DUAs before downloading.

## Quick Start

```bash
# 1. Fetch datasets (requires USE_REAL_DATA=1)
USE_REAL_DATA=1 python data/physionet_mitbih_fetch.py   # MIT-BIH ECG
USE_REAL_DATA=1 python data/physionet_sleepedf_fetch.py # Sleep-EDF EEG
USE_REAL_DATA=1 python data/genomics_fetch.py           # Lambda phage + PDB

# 2. Run validation CLI
USE_REAL_DATA=1 python scripts/test_real_data.py --verbose

# 3. Run specific tests
USE_REAL_DATA=1 python scripts/test_real_data.py --ecg
USE_REAL_DATA=1 python scripts/test_real_data.py --eeg
USE_REAL_DATA=1 python scripts/test_real_data.py --genomics
USE_REAL_DATA=1 python scripts/test_real_data.py --json  # CI-friendly output
```

## Datasets

| Dataset | License | Fetch Script | Status |
|---------|---------|--------------|--------|
| MIT-BIH Arrhythmia (ECG) | PhysioNet DUA | `data/physionet_mitbih_fetch.py` | ✓ Ready |
| Sleep-EDF (EEG) | PhysioNet terms | `data/physionet_sleepedf_fetch.py` | ✓ Ready |
| Lambda Phage (FASTA) | Public domain | `data/genomics_fetch.py` | ✓ Ready |
| PDB 1CRN (Protein) | CC BY 4.0 | `data/genomics_fetch.py` | ✓ Ready |
| FastMRI (MRI) | CC BY-NC 4.0 | `data/fastmri_fetch.sh` | ⏳ Needs URL |
| BraTS (Brain MRI) | CC BY 4.0 | Manual | Not implemented |
| Ninapro (EMG) | Non-commercial | Manual | Not implemented |
| TUH EEG Seizure | Non-commercial + DUA | Manual | Not implemented |

## Test Results (December 2025)

| Dataset | Samples | RFT Roundtrip Error | 30% Compression PRD |
|---------|---------|---------------------|---------------------|
| MIT-BIH ECG 100 | 650,000 @ 360 Hz | 2.61e-16 | 2.13% |
| MIT-BIH ECG 101 | 650,000 @ 360 Hz | 2.33e-16 | — |
| Sleep-EDF EEG | 7,950,000 @ 100 Hz | 3.32e-16 | 18.89% |
| Lambda Phage | 48,502 bp | 1.67e-16 | — |
| PDB 1CRN | 46 CA atoms | 1.83e-16 | — |

## CLI Reference

```
usage: test_real_data.py [-h] [--ecg] [--eeg] [--genomics] [--mri] [--verbose] [--json]

Real-Data Validation CLI for QuantoniumOS RFT

options:
  --ecg         Test MIT-BIH ECG only
  --eeg         Test Sleep-EDF EEG only
  --genomics    Test genomics data only
  --mri         Test FastMRI only (if available)
  --verbose, -v Verbose output
  --json        JSON output for CI
```

## Warnings

- **FastMRI** is CC BY-NC 4.0 (no commercial use); requires registration at fastmri.org
- **TUH EEG, Ninapro** require non-commercial + DUA; do not redistribute
- **MIT-BIH / Sleep-EDF** require PhysioNet DUA/terms acceptance
- Keep PHI out of the repo; these datasets are de-identified
- All data and software here is **strictly for research/educational use**
