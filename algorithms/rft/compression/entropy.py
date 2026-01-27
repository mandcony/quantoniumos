# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Entropy estimation and quantization utilities for Rate-Distortion analysis.
"""
import numpy as np
from collections import Counter

def uniform_quantizer(coeffs: np.ndarray, step_size: float) -> np.ndarray:
    """
    Uniform scalar quantization with dead-zone at zero.
    """
    if step_size <= 0:
        return coeffs
    
    # Dead-zone quantization
    q = np.zeros_like(coeffs, dtype=np.int32)
    
    # Handle complex coefficients by quantizing real/imag separately
    if np.iscomplexobj(coeffs):
        q_real = np.sign(coeffs.real) * np.floor(np.abs(coeffs.real) / step_size)
        q_imag = np.sign(coeffs.imag) * np.floor(np.abs(coeffs.imag) / step_size)
        return q_real + 1j * q_imag
    else:
        return np.sign(coeffs) * np.floor(np.abs(coeffs) / step_size)

def estimate_bitrate(coeffs: np.ndarray) -> float:
    """
    Estimate the bitrate (bits per sample) required to encode the coefficients.
    
    Model:
    1. Sparsity Map: Encoded using combinatorial coding or run-length approximation.
       Cost ~ K * log2(N/K) + O(K) where K is number of non-zeros.
    2. Values: Encoded using 0-order entropy of the non-zero quantized bins.
    """
    N = coeffs.size
    if N == 0:
        return 0.0
        
    # Count non-zeros
    non_zeros = coeffs[coeffs != 0]
    K = non_zeros.size
    
    if K == 0:
        return 0.0 # All zeros cost very little (just a flag)
        
    # 1. Position Cost (Sparsity)
    # Approximation of log2(N choose K)
    # If K << N, approx K * log2(N/K) + K * 1.44
    if K == N:
        pos_bits = 0 # No position info needed if dense
    else:
        p = K / N
        # Binary entropy function H(p) * N
        h_p = -p * np.log2(p) - (1-p) * np.log2(1-p)
        pos_bits = N * h_p
        
    # 2. Value Cost (Entropy)
    # We treat real and imag parts as separate symbols if complex
    if np.iscomplexobj(non_zeros):
        symbols = np.concatenate([non_zeros.real, non_zeros.imag])
        symbols = symbols[symbols != 0] # Re-filter zeros in components
    else:
        symbols = non_zeros
        
    if symbols.size == 0:
        val_bits = 0
    else:
        counts = Counter(symbols)
        total_syms = len(symbols)
        probs = np.array(list(counts.values())) / total_syms
        entropy = -np.sum(probs * np.log2(probs))
        val_bits = total_syms * entropy
        
    total_bits = pos_bits + val_bits
    return float(total_bits)

def calculate_rd_point(signal, transform_func, inverse_func, step_size):
    """
    Calculate (Rate, Distortion) point for a given transform and step size.
    """
    # 1. Transform
    coeffs = transform_func(signal)
    
    # 2. Quantize
    q_coeffs = uniform_quantizer(coeffs, step_size)
    
    # 3. Estimate Rate
    total_bits = estimate_bitrate(q_coeffs)
    bpp = total_bits / signal.size # Bits per pixel/sample
    
    # 4. Inverse Transform (Reconstruction)
    # De-quantize (rescale)
    recon_coeffs = q_coeffs * step_size
    recon = inverse_func(recon_coeffs)
    
    # 5. Calculate Distortion (MSE)
    if np.iscomplexobj(signal):
        mse = np.mean(np.abs(signal - recon)**2)
    else:
        mse = np.mean((signal - recon.real)**2)
        
    return bpp, mse
