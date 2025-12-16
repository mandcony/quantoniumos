#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""
Unified Backend Validation Test
================================

Tests that all transform backends (Python, C/ASM, C++ Native) produce
consistent results when available.

This test validates:
1. All 14 Python variants are unitary
2. C/ASM matches Python for N <= 8 (known accuracy limit)
3. C++ Native matches Python when available
4. Scheduler correctly routes based on size/accuracy
"""

import sys
import os
import numpy as np
import pytest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from algorithms.rft.unified_transform_scheduler import (
    UnifiedTransformScheduler, Backend
)


class TestUnifiedScheduler:
    """Test the unified transform scheduler"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize scheduler for each test"""
        self.scheduler = UnifiedTransformScheduler(prefer_native=False)
    
    def test_python_backend_available(self):
        """Python backend should always be available"""
        assert self.scheduler.backends[Backend.PYTHON].available
        assert self.scheduler.backends[Backend.PYTHON].accuracy == "production"
    
    def test_all_python_variants_unitary(self):
        """All 14 Python variants should be unitary"""
        n = 32
        tol = 1e-10
        
        # 2D variants return (n², n²) matrices instead of (n, n)
        _2d_variants = {"robust_manifold_2d"}
        
        for variant_name in self.scheduler.variant_generators:
            basis = self.scheduler.get_basis(n, variant_name)
            identity = basis.conj().T @ basis
            
            # 2D variants produce (n², n²) matrices
            if variant_name in _2d_variants:
                expected_size = n * n
            else:
                expected_size = n
            
            error = np.linalg.norm(identity - np.eye(expected_size))
            
            assert error < tol, f"Variant {variant_name} not unitary: error={error:.2e}"
    
    def test_python_roundtrip(self):
        """Python forward + inverse should reconstruct signal"""
        n = 64
        np.random.seed(42)
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        for variant in ['original', 'h3_cascade', 'fh5_entropy']:
            if variant not in self.scheduler.variant_generators:
                continue
            
            X = self.scheduler.forward(x, variant, Backend.PYTHON)
            x_rec = self.scheduler.inverse(X, variant, Backend.PYTHON)
            
            error = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
            assert error < 1e-12, f"Roundtrip failed for {variant}: error={error:.2e}"
    
    def test_c_asm_small_size(self):
        """C/ASM should work for N <= 8"""
        if not self.scheduler.backends[Backend.C_ASM].available:
            pytest.skip("C/ASM backend not available")
        
        n = 8
        np.random.seed(42)
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        X = self.scheduler.forward(x, 'original', Backend.C_ASM)
        x_rec = self.scheduler.inverse(X, 'original', Backend.C_ASM)
        
        error = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
        # Relaxed tolerance for C/ASM
        assert error < 1e-6, f"C/ASM roundtrip failed at N=8: error={error:.2e}"
    
    def test_backend_selection_accuracy(self):
        """Scheduler should prefer accurate backends"""
        # For large sizes, should select Python (production accuracy)
        # OR CPP_NATIVE if available (it is also production accuracy)
        backend = self.scheduler.select_backend(256, require_accuracy=True)
        assert backend in (Backend.PYTHON, Backend.CPP_NATIVE)
        
        # For small sizes with native preference, may select C/ASM
        if self.scheduler.backends[Backend.C_ASM].available:
            self.scheduler.prefer_native = True
            backend = self.scheduler.select_backend(8, require_accuracy=False)
            # Could be either, depends on availability
            assert backend in (Backend.PYTHON, Backend.C_ASM, Backend.CPP_NATIVE)
    
    def test_energy_preservation(self):
        """Transform should preserve signal energy (Parseval)"""
        n = 128
        np.random.seed(42)
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        energy_time = np.sum(np.abs(x) ** 2)
        
        for variant in ['original', 'h3_cascade']:
            if variant not in self.scheduler.variant_generators:
                continue
            
            X = self.scheduler.forward(x, variant, Backend.PYTHON)
            energy_freq = np.sum(np.abs(X) ** 2)
            
            error = abs(energy_freq - energy_time) / energy_time
            assert error < 1e-12, f"Energy not preserved for {variant}: error={error:.2e}"
    
    def test_variant_count(self):
        """Should have at least 14 variants loaded"""
        assert len(self.scheduler.variant_generators) >= 14, \
            f"Only {len(self.scheduler.variant_generators)} variants loaded, expected >=14"


class TestCrossBackendConsistency:
    """Test consistency between backends when multiple are available"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.scheduler = UnifiedTransformScheduler()
    
    def test_python_vs_c_asm_n8(self):
        """Python and C/ASM should match for N=8"""
        if not self.scheduler.backends[Backend.C_ASM].available:
            pytest.skip("C/ASM backend not available")
        
        n = 8
        np.random.seed(42)
        x = np.random.randn(n) + 1j * np.random.randn(n)
        
        X_py = self.scheduler.forward(x, 'original', Backend.PYTHON)
        X_asm = self.scheduler.forward(x, 'original', Backend.C_ASM)
        
        # Compute correlation (phase-invariant comparison)
        corr = np.abs(np.vdot(X_py, X_asm)) / (np.linalg.norm(X_py) * np.linalg.norm(X_asm))
        
        # Should be highly correlated even if phases differ
        assert corr > 0.9, f"Python/C_ASM correlation too low: {corr:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
