# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
RFT Hybrid Codec

Multi-stage compression pipeline combining RFT transforms, adaptive quantization,
and residual prediction for high-quality data compression.
"""

from .rft_hybrid_codec import RFTHybridCodec
from .hybrid_residual_predictor import TinyResidualPredictor as ResidualPredictor

__all__ = ['RFTHybridCodec', 'ResidualPredictor']