#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Shard Implementation - QuantoniumOS
===================================

Shard structure for grouping geometric containers.
"""

from typing import List
from .geometric_container import GeometricContainer
from .bloom_filter import SimplifiedBloomFilter, hash1, hash2

class Shard:
    def __init__(self, containers: List[GeometricContainer]):
        self.containers = containers
        self.bloom = self._build_bloom(containers)

    def _build_bloom(self, containers: List[GeometricContainer]) -> SimplifiedBloomFilter:
        bf = SimplifiedBloomFilter(256, [hash1, hash2])
        for c in containers:
            if c.resonant_frequencies:
                bf.add(c.resonant_frequencies[0])
        return bf

    def search(self, target_freq: float, threshold: float = 0.1) -> List[GeometricContainer]:
        # Quick check with the bloom filter first
        if not self.bloom.test(target_freq):
            return []
        
        # Then do a thorough check
        results = []
        for c in self.containers:
            if not c.resonant_frequencies:
                continue
            if abs(c.resonant_frequencies[0] - target_freq) < threshold:
                results.append(c)
        return results
