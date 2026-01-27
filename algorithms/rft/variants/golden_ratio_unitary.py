#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# This file is part of QuantoniumOS.
#
# This file is a "Covered File" under the "QuantoniumOS Research License –
# Claims-Practicing Implementations (Non-Commercial)".
#
# You may use this file ONLY for research, academic, or teaching purposes.
# Commercial use is strictly prohibited.
#
# See LICENSE-CLAIMS-NC.md in the root of this repository for details.
"""
Golden Ratio Unitary RFT Variant
================================

Implements the Golden Ratio Basis and RFT Unitary Matrix Construction.
Based on the mathematical proofs of Resonance Field Theory.
"""

import numpy as np
import cmath

class GoldenRatioUnitary:
    """
    Implements the Golden Ratio based Unitary RFT.
    """
    
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # φ = 1.618033988749895

    def get_phi(self):
        """Returns the Golden Ratio constant."""
        return self.phi

    def construct_rft_matrix(self, size):
        """
        Exact RFT matrix construction following the paper.
        
        Args:
            size (int): The size of the matrix (N x N).
            
        Returns:
            np.ndarray: The unitary RFT matrix Q.
        """
        N = size
        K = np.zeros((N, N), dtype=complex)
        
        # Implement the exact formula
        for component in range(N):
            # Golden ratio phase sequence: φₖ = (k×φ) mod 1
            phi_k = (component * self.phi) % 1.0
            w_i = 1.0 / N  # Equal weights
            
            # Build component matrices
            for m in range(N):
                for n in range(N):
                    # Phase operators
                    phase_m = 2 * np.pi * phi_k * m / N
                    phase_n = 2 * np.pi * phi_k * n / N
                    
                    # Gaussian convolution kernel
                    sigma_i = 1.0 + 0.1 * component
                    dist = min(abs(m - n), N - abs(m - n))  # Circular distance
                    C_sigma = np.exp(-0.5 * (dist**2) / (sigma_i**2))
                    
                    # Matrix element: wᵢ × C_σ × exp(iφₖ(m-n))
                    phase_diff = phase_m - phase_n
                    element = w_i * C_sigma * cmath.exp(1j * phase_diff)
                    
                    K[m, n] += element
        
        # QR decomposition for exact unitarity
        Q, _ = np.linalg.qr(K)
        
        return Q

    def verify_unitarity(self, matrix):
        """
        Verifies if the given matrix is unitary.
        
        Args:
            matrix (np.ndarray): The matrix to check.
            
        Returns:
            float: The unitarity error ||Q†Q - I||.
        """
        N = matrix.shape[0]
        identity = np.eye(N)
        unitarity_error = np.linalg.norm(matrix.conj().T @ matrix - identity)
        return unitarity_error
