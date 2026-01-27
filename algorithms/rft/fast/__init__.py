# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fast RFT Module
===============

Provides fast basis construction via caching, approximation,
and (eventually) O(N log N) algorithms.
"""

from .cached_basis import (
    get_cached_basis,
    get_fast_basis,
    rft_forward_fast,
    rft_inverse_fast,
    precompute_all_bases,
    STANDARD_SIZES,
)

__all__ = [
    'get_cached_basis',
    'get_fast_basis',
    'rft_forward_fast',
    'rft_inverse_fast',
    'precompute_all_bases',
    'STANDARD_SIZES',
]
