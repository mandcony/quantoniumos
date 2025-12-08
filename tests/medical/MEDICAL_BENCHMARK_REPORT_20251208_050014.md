# QuantoniumOS Medical Benchmark Report

**Generated:** 2025-12-08 05:00:14

---

## Medical Imaging Results

| Noise Type | Method | PSNR Before | PSNR After | SSIM | Time (ms) |
|------------|--------|-------------|------------|------|-----------|
| Rician_0.05 | RFT | 24.26 dB | 14.88 dB | 0.750 | 172.1 |
| Rician_0.05 | DCT | 24.26 dB | 15.94 dB | 0.814 | 11.5 |
| Rician_0.05 | Wavelet | 24.26 dB | 24.84 dB | 0.980 | 1.6 |
| Rician_0.10 | RFT | 18.21 dB | 13.93 dB | 0.680 | 48.6 |
| Rician_0.10 | DCT | 18.21 dB | 14.77 dB | 0.748 | 1.5 |
| Rician_0.10 | Wavelet | 18.21 dB | 18.36 dB | 0.909 | 1.3 |
| Poisson_500 | RFT | 32.69 dB | 15.46 dB | 0.797 | 46.9 |
| Poisson_500 | DCT | 32.69 dB | 16.91 dB | 0.860 | 1.3 |
| Poisson_500 | Wavelet | 32.69 dB | 35.32 dB | 0.998 | 1.4 |
| Rician_0.15 | RFT | 14.75 dB | 13.04 dB | 0.620 | 47.7 |
| Rician_0.15 | DCT | 14.75 dB | 13.22 dB | 0.640 | 1.4 |
| Rician_0.15 | Wavelet | 14.75 dB | 14.85 dB | 0.805 | 1.6 |
| Poisson_100 | RFT | 25.67 dB | 15.46 dB | 0.796 | 48.6 |
| Poisson_100 | DCT | 25.67 dB | 16.88 dB | 0.859 | 1.5 |
| Poisson_100 | Wavelet | 25.67 dB | 26.38 dB | 0.987 | 1.9 |

## Biosignal Compression Results

| Signal | Method | Keep Ratio | CR | SNR (dB) | PRD (%) | Corr |
|--------|--------|------------|-----|----------|---------|------|
| ECG | RFT | 0.2 | 5.02x | 19.86 | 10.17 | 0.9938 |
| ECG | FFT | 0.2 | 5.02x | 19.86 | 10.17 | 0.9938 |
| ECG | RFT | 0.3 | 3.37x | 25.47 | 5.33 | 0.9983 |
| ECG | FFT | 0.3 | 3.37x | 25.47 | 5.33 | 0.9983 |
| ECG | RFT | 0.5 | 2.00x | 28.75 | 3.65 | 0.9992 |
| ECG | FFT | 0.5 | 2.00x | 28.77 | 3.64 | 0.9992 |
| ECG | RFT | 0.7 | 1.43x | 31.70 | 2.60 | 0.9996 |
| ECG | FFT | 0.7 | 1.43x | 31.70 | 2.60 | 0.9996 |
| EEG | RFT | 0.3 | 3.37x | 27.42 | 4.25 | 0.9991 |
| EEG | FFT | 0.3 | 3.36x | 27.43 | 4.25 | 0.9991 |
| EEG | RFT | 0.5 | 2.00x | 31.65 | 2.61 | 0.9997 |
| EEG | FFT | 0.5 | 2.00x | 31.65 | 2.61 | 0.9997 |

## Genomics Transform Results

### K-mer Spectrum Analysis

| Method | Input Size | Top-10 Energy | Time (ms) |
|--------|------------|---------------|-----------|
| RFT | 10000 | 0.978 | 0.07 |
| FFT | 10000 | 0.978 | 0.01 |
| DCT | 10000 | 0.979 | 0.04 |
| RFT | 50000 | 0.997 | 0.12 |
| FFT | 50000 | 0.997 | 0.01 |
| DCT | 50000 | 0.997 | 0.07 |

### Contact Map Compression

| Size | CR | Accuracy | F1 Score | Time (ms) |
|------|-----|----------|----------|-----------|
| 64x64 | 2.00x | 0.999 | 0.997 | 7.8 |
| 128x128 | 2.00x | 1.000 | 1.000 | 19.1 |

## Security and Privacy Results

- **Avalanche Effect**: 0.000 (ideal: 0.5) ✗
- **Collision Resistance**: 0 collisions in 500 samples ✓
- **Byzantine 0%**: RFT improvement = 0.0%
- **Byzantine 20%**: RFT improvement = 84.1%
- **Byzantine 30%**: RFT improvement = 82.9%

## Edge Device Results

### ARM Cortex-M4 (STM32F4)

| Buffer Size | Estimated (ms) | Target (ms) | Status |
|-------------|----------------|-------------|--------|
| 64 | 1.0 | 50.0 | ✓ |
| 128 | 1.2 | 50.0 | ✓ |
| 256 | 1.5 | 50.0 | ✓ |
| 512 | 2.2 | 50.0 | ✓ |

### ESP32

| Buffer Size | Estimated (ms) | Target (ms) | Status |
|-------------|----------------|-------------|--------|
| 64 | 0.7 | 100.0 | ✓ |
| 128 | 0.8 | 100.0 | ✓ |
| 256 | 1.0 | 100.0 | ✓ |
| 512 | 1.5 | 100.0 | ✓ |

### Nordic nRF52840

| Buffer Size | Estimated (ms) | Target (ms) | Status |
|-------------|----------------|-------------|--------|
| 64 | 2.6 | 100.0 | ✓ |
| 128 | 3.0 | 100.0 | ✓ |
| 256 | 3.9 | 100.0 | ✓ |
| 512 | 5.7 | 100.0 | ✓ |

### Raspberry Pi Pico

| Buffer Size | Estimated (ms) | Target (ms) | Status |
|-------------|----------------|-------------|--------|
| 64 | 1.3 | 50.0 | ✓ |
| 128 | 1.5 | 50.0 | ✓ |
| 256 | 1.9 | 50.0 | ✓ |
| 512 | 2.8 | 50.0 | ✓ |
