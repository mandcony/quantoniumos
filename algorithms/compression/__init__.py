"""
QuantoniumOS Compression Algorithms

This package contains advanced compression algorithms using quantum-inspired
mathematical techniques for efficient data representation.

Modules:
- vertex: Symbolic quantum state encoding using modular arithmetic
- hybrid: Multi-stage compression combining RFT, quantization, and prediction
"""

from .vertex.rft_vertex_codec import RFTVertexCodec
from .hybrid.rft_hybrid_codec import RFTHybridCodec
from .hybrid.hybrid_residual_predictor import TinyResidualPredictor as ResidualPredictor

__all__ = [
    'RFTVertexCodec',
    'RFTHybridCodec', 
    'ResidualPredictor'
]

__version__ = '1.0.0'
__author__ = 'QuantoniumOS Research Team'