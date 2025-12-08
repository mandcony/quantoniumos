"""
RFT Experiments Package
=======================

Honest validation experiments for RFT fast algorithms and routing.

Experiments:
- exp_basis_quality.py: Compare approximate bases (circulant, Lanczos) to exact
- exp_fft_correction_rank.py: Test if (U-F) is low-rank for FFT+correction approach
- exp_classifier_calibration.py: Calibrate signal classifier thresholds
- exp_router_vs_oracle.py: Measure router PSNR gap from oracle transform selection

All experiments produce reproducible results and should be re-run
before updating claims in the codebase.
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent

__all__ = [
    'EXPERIMENTS_DIR',
]
