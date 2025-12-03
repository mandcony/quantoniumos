# RFT vs FFT Benchmark Results

## Executive Summary

This report presents competitive benchmarks between QuantoniumOS RFT and standard FFT implementations.

## Speed Performance

| Size | FFT Time (s) | RFT Time (s) | Ratio |
|------|-------------|-------------|-------|
| 64 | 0.0038 | inf | infx |
| 128 | 0.0050 | inf | infx |
| 256 | 0.0055 | inf | infx |
| 512 | 0.0074 | inf | infx |
| 1024 | 0.0101 | inf | infx |
| 2048 | 0.0185 | inf | infx |

## Accuracy Results

| Size | FFT Error | RFT Error |
|------|-----------|----------|
| 256 | 8.88e-16 | inf |
| 512 | 1.20e-15 | inf |
| 1024 | 1.23e-15 | inf |

## Compression Efficiency

- Original size: 147.2 MB
- FFT compressed: 73.6 MB (MSE: 0.624771)
- RFT compressed: 0.0 MB (MSE: 0.999668)
