#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Simplified Bloom Filter Implementation - QuantoniumOS
===================================================

Bloom filter for quick resonance checking.
"""

from typing import List, Callable, Any

class SimplifiedBloomFilter:
    def __init__(self, size: int, hash_fns: List[Callable[[Any], int]]):
        self.size = size
        self.bit_array = [False] * size
        self.hash_fns = hash_fns

    def add(self, item: Any):
        for fn in self.hash_fns:
            idx = fn(item) % self.size
            self.bit_array[idx] = True

    def test(self, item: Any) -> bool:
        for fn in self.hash_fns:
            idx = fn(item) % self.size
            if not self.bit_array[idx]:
                return False
        return True

def hash1(s: Any) -> int:
    h = 0
    s_str = str(s)
    for char in s_str:
        h = (h << 5) - h + ord(char)
        h |= 0 # 32bit int simulation
    return abs(h)

def hash2(s: Any) -> int:
    h = 5381
    s_str = str(s)
    for char in s_str:
        h = (h << 5) + h + ord(char)
    return abs(h)
