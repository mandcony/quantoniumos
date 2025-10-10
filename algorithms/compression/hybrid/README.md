# RFT Hybrid Codec

The Hybrid Codec combines RFT transforms with quantization and residual prediction for advanced compression.

## Core Components

- **`rft_hybrid_codec.py`** - Main hybrid compression implementation
- **`hybrid_residual_predictor.py`** - Residual prediction algorithms

## Algorithm Overview

The hybrid codec implements a multi-stage compression pipeline:

1. **RFT Transform** - Apply Resonance Fourier Transform
2. **Quantization** - Reduce precision with controlled loss
3. **Residual Prediction** - Predict and encode prediction errors
4. **Entropy Coding** - Final compression stage

## Compression Pipeline

```
Input → RFT → Quantize → Residual → Entropy → Compressed
        ↑                    ↓
    Golden Ratio        Error Bounds
   Parameters           Validation
```

## Mathematical Foundation

The hybrid approach combines:
- **Unitary RFT transforms** for frequency domain representation
- **Adaptive quantization** based on perceptual importance
- **Predictive residuals** to minimize reconstruction error
- **Golden ratio parameterization** for optimal transform matrices

## Key Features

- **Multi-stage compression** - Combines multiple techniques
- **Quality control** - Configurable loss parameters
- **Error bounds** - Guaranteed reconstruction quality
- **Adaptive algorithms** - Content-aware optimization
- **Performance monitoring** - Real-time compression metrics

## Usage

```python
from algorithms.compression.hybrid.rft_hybrid_codec import RFTHybridCodec
from algorithms.compression.hybrid.hybrid_residual_predictor import ResidualPredictor

codec = RFTHybridCodec()
predictor = ResidualPredictor()

compressed = codec.encode(data, quality=0.95)
reconstructed = codec.decode(compressed)
```

## Integration

The hybrid codec integrates with:
- RFT core algorithms (`algorithms/rft/core/`)
- Vertex codec (`algorithms/compression/vertex/`)
- Model compression tools (`tools/compression/`)
- End-to-end validation (`tests/validation/test_rft_hybrid_codec_e2e.py`)