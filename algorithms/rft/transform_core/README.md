# RFT Transform Core

This directory contains the **pure mathematical kernel** of the Resonant Fourier Transform (RFT).

- No compression
- No hashing / cryptography
- No codecs

Primary entry points:
- `resonant_fourier_transform.py` — canonical basis + forward/inverse transform
- `phi_phase_fft.py` — baseline (deprecated) φ-phase FFT reference
- `canonical_true_rft.py` — compatibility wrapper for the φ-phase FFT API
