"""
RFT Routing Module
==================

Provides signal classification and adaptive transform routing
for optimal sparsity.
"""

from .signal_classifier import (
    TransformType,
    SignalFeatures,
    extract_features,
    classify_signal,
    get_best_transform_for_signal,
    AdaptiveRouter,
    apply_transform,
    apply_inverse_transform,
)

__all__ = [
    'TransformType',
    'SignalFeatures',
    'extract_features',
    'classify_signal',
    'get_best_transform_for_signal',
    'AdaptiveRouter',
    'apply_transform',
    'apply_inverse_transform',
]
