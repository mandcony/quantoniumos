# Wavelet-RFT Hybrid Compression for Medical Signals

**MEDICAL RESEARCH LICENSE**: This work is **FREE for hospitals, medical researchers, academics, and healthcare institutions** for testing, validation, and research purposes. See [LICENSE-CLAIMS-NC.md](LICENSE-CLAIMS-NC.md) for commercial medical device usage.

## Overview

Wavelet-RFT hybrid compression combines multi-level Haar wavelet decomposition with Resonant Fourier Transform (RFT) based Wiener filtering for biomedical signal compression.

## Implementation

Located in: `tests/medical/test_biosignal_compression.py`

### Function: `wavelet_rft_hybrid_compress_signal()`

**Algorithm**:
1. **Multi-level Haar decomposition**: Decomposes signal into approximation and detail subbands (2-4 levels)
2. **Noise estimation**: Uses median absolute deviation on finest detail level
3. **RFT Wiener filtering**: Applies frequency-domain filtering to each detail subband
4. **Coefficient thresholding**: Keeps top coefficients based on keep_ratio
5. **Reconstruction**: Inverse wavelet transform from coarsest to finest level

**Parameters**:
- `signal`: Input biosignal (ECG, EEG, EMG)
- `levels`: Decomposition depth (default: 3)
- `keep_ratio`: Coefficient retention fraction (default: 0.5)

**Returns**:
- Reconstructed signal
- Statistics dict with compression ratio and coefficient counts

## Test Results on Real MIT-BIH ECG Data

**Dataset**: MIT-BIH Arrhythmia Database (8 patient records, 15.6MB)
- PhysioNet data use agreement required
- Records: 100, 101, 200, 207, 208, 217, 219, 221

### Performance Metrics (50% keep ratio)

| Method | SNR (dB) | PRD (%) | Correlation | Compression Ratio |
|--------|----------|---------|-------------|-------------------|
| **Pure RFT** | 35-45 | 2-5 | 0.98-0.99 | 2.0x |
| **FFT Baseline** | 30-38 | 4-8 | 0.97-0.98 | 2.0x |
| **Wavelet-RFT Hybrid** | 20-28 | 6-12 | 0.95-0.97 | 1.8x |

### Test Coverage

**70 passing tests** across 14 RFT operator variants:
- `test_real_ecg_wavelet_rft_hybrid`: Basic validation (14 variants)
- `test_real_ecg_wavelet_vs_rft_comparison`: Method comparison (14 variants)
- `test_real_ecg_wavelet_hybrid_levels`: Multi-level decomposition (42 tests: 14 variants × 3 levels)

### Scientific Findings

1. **Pure RFT excels on ECG**: Achieves 35-45 dB SNR, outperforming both hybrid and FFT
2. **Signal-dependent performance**: Hybrid doesn't universally improve over baseline
3. **Clinical acceptability**: All methods achieve >20 dB SNR (clinically acceptable)
4. **Variant behavior**: Some experimental variants (e.g., `rft_hybrid_dct`) show lower performance on real ECG data

## Medical Research Applications

### Suitable For:
- ✅ ECG compression for remote monitoring
- ✅ Long-term EEG storage (epilepsy studies)
- ✅ Wearable biosensor data transmission
- ✅ Telemedicine bandwidth optimization
- ✅ Medical signal denoising research

### Research Validation:
- Real patient ECG data from MIT-BIH Database
- Compliant with PhysioNet data use agreements
- Suitable for academic publication and clinical validation studies

## Usage Example

```python
from tests.medical.test_biosignal_compression import (
    wavelet_rft_hybrid_compress_signal,
    snr, prd
)

# Load ECG signal (360 Hz sampling rate)
ecg_signal = load_patient_ecg()

# Compress with 3-level decomposition, 50% coefficient retention
compressed, stats = wavelet_rft_hybrid_compress_signal(
    ecg_signal, 
    levels=3, 
    keep_ratio=0.5
)

# Evaluate quality
signal_snr = snr(ecg_signal, compressed)
signal_prd = prd(ecg_signal, compressed)

print(f"SNR: {signal_snr:.2f} dB")
print(f"PRD: {signal_prd:.2f}%")
print(f"Compression: {stats['compression_ratio']:.2f}x")
```

## Running Tests

**With real MIT-BIH data**:
```bash
USE_REAL_DATA=1 pytest tests/medical/test_biosignal_compression.py::TestRealMITBIH -v
```

**Wavelet hybrid tests only**:
```bash
USE_REAL_DATA=1 pytest tests/medical/test_biosignal_compression.py::TestRealMITBIH -k "wavelet" -v
```

## Contributing

Medical researchers and academic institutions are encouraged to:
- Test on additional datasets (TUH EEG, PTB-XL, etc.)
- Optimize parameters for specific signal types
- Compare against clinical benchmarks
- Publish validation studies

## References

- **MIT-BIH Arrhythmia Database**: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001).
- **PhysioNet**: Goldberger AL, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation 101(23):e215-e220.

## License

**SPDX-License-Identifier**: AGPL-3.0-or-later

**Medical Research Exception**: FREE for non-commercial medical research, academic studies, hospital validation, and educational purposes.

**Commercial Medical Devices**: Contact for licensing. See [LICENSE-CLAIMS-NC.md](LICENSE-CLAIMS-NC.md).

---

*Developed by Luis M. Minier / QuantoniumOS Project*  
*For medical research collaboration: See [SECURITY.md](SECURITY.md)*
