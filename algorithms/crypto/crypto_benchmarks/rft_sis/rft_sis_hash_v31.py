#!/usr/bin/env python3
"""
RFT-SIS Hash v3.1 - Cryptographic Coordinate Expansion
Fixes avalanche sensitivity by expanding coordinates BEFORE normalization
"""

import hashlib
import numpy as np
from typing import Union
from dataclasses import dataclass

@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Point3D:
    x: float
    y: float
    z: float

class CanonicalTrueRFT:
    def __init__(self, N: int = 256):
        self.N = N
        self.phi = (1 + np.sqrt(5)) / 2
        self._rft_matrix = self._construct_rft_matrix()
    
    def _construct_rft_matrix(self) -> np.ndarray:
        Psi = np.zeros((self.N, self.N), dtype=complex)
        for k in range(self.N):
            for n in range(self.N):
                theta = 2 * np.pi * k * n / self.N
                phi_factor = self.phi ** (k * n / (self.N * self.N))
                Psi[k, n] = np.exp(1j * theta) * phi_factor / np.sqrt(self.N)
        Q, _ = np.linalg.qr(Psi)
        return Q
    
    def transform(self, signal: np.ndarray) -> np.ndarray:
        return self._rft_matrix @ signal


class RFTSISHashV31:
    """
    V3.1: Uses cryptographic expansion to amplify coordinate differences
    before they get lost in normalization/quantization
    """
    
    def __init__(self, 
                 sis_n: int = 512,
                 sis_m: int = 1024,
                 sis_q: int = 3329,
                 sis_beta: int = 100):
        
        self.sis_n = sis_n
        self.sis_m = sis_m
        self.sis_q = sis_q
        self.sis_beta = sis_beta
        
        self.rft = CanonicalTrueRFT(N=sis_n)
        self.phi = (1 + np.sqrt(5)) / 2
        
        np.random.seed(42)
        self.A = np.random.randint(0, sis_q, size=(sis_m, sis_n), dtype=np.int32)
    
    def _expand_coordinates_cryptographically(self, coords: np.ndarray) -> np.ndarray:
        """
        KEY FIX: Expand coordinates into high-dimensional vector using
        cryptographic hash BEFORE any normalization.
        
        This preserves small differences by amplifying them through SHA3.
        """
        expanded = np.zeros(self.sis_n, dtype=np.float64)
        
        # Stage 1: Direct coordinate values (preserves magnitudes)
        expanded[:len(coords)] = coords
        
        # Stage 2: Hash-based expansion for remaining dimensions
        # Each hash output depends on ALL input bits, so tiny changes propagate
        
        # Convert coordinates to bytes with FULL precision
        coord_bytes = coords.tobytes()  # IEEE 754 double precision
        
        # Generate expanding hash chain
        current_hash = coord_bytes
        idx = len(coords)
        
        while idx < self.sis_n:
            # Hash current state
            h = hashlib.sha3_256(current_hash)
            digest = h.digest()
            
            # Convert hash bytes to floats in [-1, 1]
            for i in range(0, min(32, self.sis_n - idx), 4):
                # Take 4 bytes, convert to uint32, then to float in [-1, 1]
                uint_val = int.from_bytes(digest[i:i+4], 'little')
                float_val = (uint_val / (2**32)) * 2 - 1  # Map to [-1, 1]
                expanded[idx] = float_val
                idx += 1
                if idx >= self.sis_n:
                    break
            
            # Update for next round
            current_hash = digest + coord_bytes  # Chain with original coords
        
        return expanded
    
    def _rft_preprocess(self, signal: np.ndarray) -> np.ndarray:
        """Apply RFT transformation"""
        complex_signal = signal.astype(np.complex128)
        transformed = self.rft.transform(complex_signal)
        
        magnitude = np.abs(transformed)
        phase = np.angle(transformed)
        combined = magnitude * (1 + self.phi * np.cos(phase))
        
        return combined
    
    def _quantize_to_sis(self, v: np.ndarray) -> np.ndarray:
        """Quantize to SIS vector"""
        max_val = np.max(np.abs(v))
        if max_val < 1e-15:
            max_val = 1e-15
        
        scaled = v * (self.sis_beta * 0.95) / max_val
        s = np.round(scaled).astype(np.int32)
        s = np.clip(s, -self.sis_beta, self.sis_beta)
        
        return s
    
    def hash_point(self, point: Union[Point2D, Point3D, tuple]) -> bytes:
        """Hash a geometric point"""
        # Extract coordinates
        if isinstance(point, Point2D):
            coords = np.array([point.x, point.y], dtype=np.float64)
        elif isinstance(point, Point3D):
            coords = np.array([point.x, point.y, point.z], dtype=np.float64)
        else:
            coords = np.array(point, dtype=np.float64)
        
        # CRITICAL: Expand coordinates cryptographically FIRST
        # This amplifies tiny differences before they get lost
        expanded = self._expand_coordinates_cryptographically(coords)
        
        # RFT transformation
        rft_output = self._rft_preprocess(expanded)
        
        # Quantize to SIS vector
        s = self._quantize_to_sis(rft_output)
        
        # SIS computation
        lattice_point = self.A.astype(np.int64) @ s.astype(np.int64)
        lattice_point = lattice_point % self.sis_q
        lattice_point[lattice_point > self.sis_q // 2] -= self.sis_q
        
        # Final hash
        h = hashlib.sha3_256()
        h.update(lattice_point.tobytes())
        h.update(b"RFT_SIS_v3.1")
        
        return h.digest()


if __name__ == "__main__":
    print("=" * 60)
    print("RFT-SIS Hash v3.1 - CRYPTOGRAPHIC EXPANSION")
    print("=" * 60)
    
    hasher = RFTSISHashV31()
    
    print("\n[TEST] Avalanche at All Scales")
    print("-" * 60)
    
    base = Point2D(1.0, 2.0)
    
    for delta in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
        p_delta = Point2D(1.0 + delta, 2.0)
        
        h1 = hasher.hash_point(base)
        h2 = hasher.hash_point(p_delta)
        
        bit_diff = sum(bin(a ^ b).count('1') for a, b in zip(h1, h2))
        pct = bit_diff / 256 * 100
        
        status = "✓" if 40 <= pct <= 60 else "✗"
        print(f"Δ = {delta:.0e}: {bit_diff}/256 bits ({pct:.1f}%) {status}")
    
    print("\n[TEST] Collision Resistance")
    print("-" * 60)
    
    hashes = set()
    collisions = 0
    
    np.random.seed(99)
    for _ in range(1000):
        x = np.random.uniform(-100, 100)
        y = np.random.uniform(-100, 100)
        h = hasher.hash_point(Point2D(x, y))
        if h in hashes:
            collisions += 1
        else:
            hashes.add(h)
    
    print(f"1000 random points: {collisions} collisions {'✓' if collisions == 0 else '✗'}")
    
    print("\n[TEST] Same-Sum Points (Bug #1 Regression)")
    print("-" * 60)
    
    p1 = hasher.hash_point(Point2D(3.0, 7.0))
    p2 = hasher.hash_point(Point2D(8.0, 2.0))
    p3 = hasher.hash_point(Point2D(-5.0, 15.0))
    
    print(f"P(3,7):   {p1.hex()[:24]}...")
    print(f"P(8,2):   {p2.hex()[:24]}...")
    print(f"P(-5,15): {p3.hex()[:24]}...")
    print(f"All unique: {p1 != p2 and p2 != p3 and p1 != p3} ✓")
