#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Test All RFT Variants with Wavelet Hybrid for Medical Imaging
==============================================================

Finds which RFT variant works best with wavelet decomposition for MRI denoising.

⚠️  RESEARCH USE ONLY — NOT FOR CLINICAL OR DIAGNOSTIC USE  ⚠️

This code is for research comparison purposes only and has not been
validated for medical device or clinical diagnostic applications.

Copyright (C) 2025 Luis M. Minier / quantoniumos
Licensed under AGPL-3.0-or-later
"""

import numpy as np
import time
import sys
sys.path.insert(0, "/workspaces/quantoniumos")


def shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """Generate Shepp-Logan phantom for MRI simulation."""
    phantom = np.zeros((size, size), dtype=np.float64)
    ellipses = [
        (0.0, 0.0, 0.69, 0.92, 0, 2.0),
        (0.0, -0.0184, 0.6624, 0.874, 0, -0.98),
        (0.22, 0.0, 0.11, 0.31, -18, -0.02),
        (-0.22, 0.0, 0.16, 0.41, 18, -0.02),
        (0.0, 0.35, 0.21, 0.25, 0, 0.01),
        (0.0, 0.1, 0.046, 0.046, 0, 0.01),
        (0.0, -0.1, 0.046, 0.046, 0, 0.01),
    ]
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    for x0, y0, a, b, theta_deg, intensity in ellipses:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = (x - x0) * cos_t + (y - y0) * sin_t
        y_rot = -(x - x0) * sin_t + (y - y0) * cos_t
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
        phantom[mask] += intensity
    phantom = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-10)
    return phantom


def add_rician_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    """Add Rician noise (MRI)."""
    real = image + sigma * np.random.randn(*image.shape)
    imag = sigma * np.random.randn(*image.shape)
    return np.sqrt(real**2 + imag**2)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate PSNR."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def ssim_simple(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Simplified SSIM."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x, mu_y = np.mean(original), np.mean(reconstructed)
    sigma_x, sigma_y = np.std(original), np.std(reconstructed)
    sigma_xy = np.mean((original - mu_x) * (reconstructed - mu_y))
    return ((2*mu_x*mu_y + c1) * (2*sigma_xy + c2)) / \
           ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))


# =============================================================================
# Wavelet-RFT Hybrid with Configurable Variant
# =============================================================================

def _haar_decompose_2d(image):
    rows, cols = image.shape
    if rows % 2 != 0:
        image = np.vstack([image, image[-1:, :]])
        rows += 1
    if cols % 2 != 0:
        image = np.hstack([image, image[:, -1:]])
        cols += 1
    
    h_low = (image[:, 0::2] + image[:, 1::2]) / np.sqrt(2)
    h_high = (image[:, 0::2] - image[:, 1::2]) / np.sqrt(2)
    LL = (h_low[0::2, :] + h_low[1::2, :]) / np.sqrt(2)
    LH = (h_low[0::2, :] - h_low[1::2, :]) / np.sqrt(2)
    HL = (h_high[0::2, :] + h_high[1::2, :]) / np.sqrt(2)
    HH = (h_high[0::2, :] - h_high[1::2, :]) / np.sqrt(2)
    return LL, LH, HL, HH


def _haar_reconstruct_2d(LL, LH, HL, HH, original_shape):
    rows, cols = LL.shape[0] * 2, LL.shape[1] * 2
    sqrt2_inv = 1.0 / np.sqrt(2)
    
    h_low = np.zeros((rows, LL.shape[1]))
    h_low[0::2, :] = (LL + LH) * sqrt2_inv
    h_low[1::2, :] = (LL - LH) * sqrt2_inv
    
    h_high = np.zeros((rows, HL.shape[1]))
    h_high[0::2, :] = (HL + HH) * sqrt2_inv
    h_high[1::2, :] = (HL - HH) * sqrt2_inv
    
    image = np.zeros((rows, cols))
    image[:, 0::2] = (h_low + h_high) * sqrt2_inv
    image[:, 1::2] = (h_low - h_high) * sqrt2_inv
    
    return image[:original_shape[0], :original_shape[1]]


def _soft_threshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def rft_filter_subband_variant(subband: np.ndarray, noise_var: float, 
                                variant_name: str) -> np.ndarray:
    """Apply RFT-based filtering using specified variant."""
    try:
        from algorithms.rft.variants.operator_variants import get_operator_variant
    except ImportError:
        return subband  # Fallback
    
    rows, cols = subband.shape
    
    # Get variant bases
    try:
        Phi_rows = get_operator_variant(variant_name, cols)
        Phi_cols = get_operator_variant(variant_name, rows)
    except ValueError:
        # Unknown variant, fallback to golden
        Phi_rows = get_operator_variant('rft_golden', cols)
        Phi_cols = get_operator_variant('rft_golden', rows)
    
    # 2D transform via matrix multiply (fast)
    coeffs = Phi_cols.T @ subband.astype(np.float64) @ Phi_rows
    
    # Wiener filter
    power = np.abs(coeffs) ** 2
    wiener = power / (power + noise_var + 1e-10)
    filtered = coeffs * wiener
    
    # Inverse
    result = Phi_cols @ filtered @ Phi_rows.T
    
    return result.real


def wavelet_rft_hybrid_denoise(noisy_image: np.ndarray, 
                                rft_variant: str = 'rft_golden',
                                levels: int = 2) -> np.ndarray:
    """Wavelet-RFT hybrid denoising with configurable RFT variant."""
    original_shape = noisy_image.shape
    rows, cols = noisy_image.shape
    
    # Pad
    pad_rows = (2 ** levels - rows % (2 ** levels)) % (2 ** levels)
    pad_cols = (2 ** levels - cols % (2 ** levels)) % (2 ** levels)
    if pad_rows > 0 or pad_cols > 0:
        padded = np.pad(noisy_image, ((0, pad_rows), (0, pad_cols)), mode='reflect')
    else:
        padded = noisy_image.copy()
    
    # Decompose
    coeffs_pyramid = []
    current = padded
    for level in range(levels):
        LL, LH, HL, HH = _haar_decompose_2d(current)
        coeffs_pyramid.append((LH, HL, HH, current.shape))
        current = LL
    
    processed_LL = current
    
    # Process each level
    for level in range(levels - 1, -1, -1):
        LH, HL, HH, shape = coeffs_pyramid[level]
        
        # Estimate noise
        mad = np.median(np.abs(HH - np.median(HH)))
        noise_var = (mad / 0.6745) ** 2
        
        # Apply RFT filtering with variant
        LH_filtered = rft_filter_subband_variant(LH, noise_var, rft_variant)
        HL_filtered = rft_filter_subband_variant(HL, noise_var, rft_variant)
        HH_filtered = rft_filter_subband_variant(HH, noise_var * 2, rft_variant)
        
        # BayesShrink threshold
        signal_var = max(np.var(LH) - noise_var, 1e-10)
        threshold = noise_var / np.sqrt(signal_var)
        
        LH_denoised = _soft_threshold(LH_filtered, threshold * 0.5)
        HL_denoised = _soft_threshold(HL_filtered, threshold * 0.5)
        HH_denoised = _soft_threshold(HH_filtered, threshold)
        
        processed_LL = _haar_reconstruct_2d(processed_LL, LH_denoised, 
                                             HL_denoised, HH_denoised, shape)
    
    denoised = processed_LL[:original_shape[0], :original_shape[1]]
    return np.clip(denoised, 0, 1)


def run_variant_comparison():
    """Compare all RFT variants with wavelet hybrid."""
    print("=" * 85)
    print("RFT VARIANT COMPARISON for Wavelet-RFT Hybrid Medical Denoising")
    print("=" * 85)
    
    # Get all variants
    try:
        from algorithms.rft.variants.operator_variants import OPERATOR_VARIANTS
        variants = list(OPERATOR_VARIANTS.keys())
    except ImportError:
        print("ERROR: Could not import operator variants")
        return
    
    np.random.seed(42)
    phantom = shepp_logan_phantom(128)  # Smaller for speed
    
    # Test scenarios - focus on Rician where hybrid excels
    scenarios = [
        ("Rician σ=0.10", lambda x: add_rician_noise(x, 0.10)),
        ("Rician σ=0.15", lambda x: add_rician_noise(x, 0.15)),
    ]
    
    results = []
    
    for scenario_name, noise_fn in scenarios:
        noisy = noise_fn(phantom)
        psnr_before = psnr(phantom, noisy)
        
        print(f"\n{scenario_name} (PSNR before: {psnr_before:.2f} dB)")
        print("-" * 85)
        print(f"{'Variant':<28} {'PSNR After':>12} {'Improvement':>12} {'SSIM':>8} {'Time':>10}")
        print("-" * 85)
        
        scenario_results = []
        
        for variant in variants:
            start = time.perf_counter()
            try:
                denoised = wavelet_rft_hybrid_denoise(noisy, rft_variant=variant, levels=2)
                elapsed = (time.perf_counter() - start) * 1000
                psnr_after = psnr(phantom, denoised)
                ssim_val = ssim_simple(phantom, denoised)
                improvement = psnr_after - psnr_before
                
                print(f"{variant:<28} {psnr_after:>10.2f} dB {improvement:>+10.2f} dB {ssim_val:>8.3f} {elapsed:>8.1f} ms")
                
                scenario_results.append({
                    'variant': variant,
                    'psnr_after': psnr_after,
                    'improvement': improvement,
                    'ssim': ssim_val,
                    'time_ms': elapsed,
                })
            except Exception as e:
                print(f"{variant:<28} ERROR: {e}")
        
        # Find best for this scenario
        if scenario_results:
            best = max(scenario_results, key=lambda r: r['psnr_after'])
            print(f"\n🏆 BEST for {scenario_name}: {best['variant']} ({best['psnr_after']:.2f} dB, +{best['improvement']:.2f} dB)")
            results.append((scenario_name, best))
    
    # Overall summary
    print("\n" + "=" * 85)
    print("OVERALL WINNER ANALYSIS")
    print("=" * 85)
    
    variant_wins = {}
    for scenario, best in results:
        v = best['variant']
        if v not in variant_wins:
            variant_wins[v] = {'wins': 0, 'total_improvement': 0}
        variant_wins[v]['wins'] += 1
        variant_wins[v]['total_improvement'] += best['improvement']
    
    print(f"\n{'Variant':<28} {'Wins':>6} {'Avg Improvement':>16}")
    print("-" * 52)
    for v, stats in sorted(variant_wins.items(), key=lambda x: -x[1]['wins']):
        avg_imp = stats['total_improvement'] / stats['wins']
        print(f"{v:<28} {stats['wins']:>6} {avg_imp:>+14.2f} dB")
    
    return results


if __name__ == "__main__":
    run_variant_comparison()
