# RFT vs FFT Benchmark Results

## Executive Summary

This report presents competitive benchmarks between QuantoniumOS RFT and standard FFT implementations.

## Speed Performance

| Size | FFT Time (s) | RFT Time (s) | Ratio |
|------|-------------|-------------|-------|
| 64 | 0.0050 | 0.0276 | 5.48x |
| 128 | 0.0099 | 0.0416 | 4.19x |
| 256 | 0.0093 | 0.1298 | 13.97x |
| 512 | 0.0107 | 0.3940 | 36.86x |
| 1024 | 0.0150 | 1.3454 | 89.43x |
| 2048 | 0.0260 | 1.0009 | 38.50x |

## Accuracy Results

| Size | FFT Error | RFT Error |
|------|-----------|----------|
| 256 | 9.42e-16 | 4.28e-15 |
| 512 | 1.20e-15 | 5.14e-15 |
| 1024 | 1.35e-15 | 6.66e-15 |

## Compression Efficiency

- Original size: 147.2 MB
- FFT compressed: 73.6 MB (MSE: 0.624944)
- RFT compressed: 0.0 MB (MSE: 1.000091)
