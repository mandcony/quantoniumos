#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Quantonium-NC
# License: LICENSE-CLAIMS-NC.md (Research/Education Only)
"""Hybrid Φ-RFT / DCT decomposition utilities for Theorem 10 tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.fftpack import dct, idct
from numpy.fft import fft, ifft

PHI: float = (1.0 + np.sqrt(5.0)) / 2.0


def _frac(values: np.ndarray) -> np.ndarray:
    """Return fractional part of values, preserving dtype."""
    return values - np.floor(values)


def rft_forward(x: ArrayLike, beta: float = 0.83, sigma: float = 1.25) -> np.ndarray:
    """Φ-RFT forward transform for one dimensional sequences."""
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.shape[0]
    k = np.arange(n, dtype=np.float64)
    phase_golden = np.exp(2j * np.pi * beta * _frac(k / PHI))
    phase_quadratic = np.exp(1j * np.pi * sigma * (k * k) / n)
    return phase_golden * (phase_quadratic * fft(signal, norm="ortho"))


def rft_inverse(y: ArrayLike, beta: float = 0.83, sigma: float = 1.25) -> np.ndarray:
    """Inverse Φ-RFT transform matching :func:`rft_forward`."""
    coeffs = np.asarray(y, dtype=np.complex128)
    n = coeffs.shape[0]
    k = np.arange(n, dtype=np.float64)
    phase_golden = np.exp(2j * np.pi * beta * _frac(k / PHI))
    phase_quadratic = np.exp(1j * np.pi * sigma * (k * k) / n)
    return ifft(np.conj(phase_quadratic) * np.conj(phase_golden) * coeffs, norm="ortho")


@dataclass
class HybridResult:
    """Container for hybrid decomposition outputs."""

    structural: np.ndarray
    texture: np.ndarray
    residual: np.ndarray
    metadata: Dict[str, object]


def _energy(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector) ** 2)


def hybrid_decomposition(
    x: ArrayLike,
    *,
    max_iter: int = 5,
    threshold_dct: float = 0.05,
    threshold_rft: float = 0.05,
    strategy: str = "balanced",
    verbose: bool = False,
) -> HybridResult:
    """Split ``x`` into DCT-sparse structure and Φ-RFT-sparse texture."""
    signal = np.asarray(x, dtype=np.complex128)
    structural = np.zeros_like(signal)
    texture = np.zeros_like(signal)
    residual = signal.copy()

    metadata: Dict[str, object] = {
        "iterations": 0,
        "struct_energy": [],
        "texture_energy": [],
        "residual_energy": [],
        "struct_sparsity": [],
        "texture_sparsity": [],
    }

    total_energy = _energy(signal)

    for idx in range(max_iter):
        metadata["iterations"] = idx + 1

        # Determine order for this iteration
        if strategy == "rft_first":
            steps = [("rft", threshold_rft), ("dct", threshold_dct)]
        elif strategy == "dct_first":
            steps = [("dct", threshold_dct), ("rft", threshold_rft)]
        else:
            # Balanced: Default to DCT first as it captures broad structure
            steps = [("dct", threshold_dct), ("rft", threshold_rft)]

        for step_type, thresh in steps:
            if step_type == "dct":
                # --- DCT Update ---
                if np.iscomplexobj(residual):
                    coeffs_real = dct(residual.real, norm="ortho")
                    coeffs_imag = dct(residual.imag, norm="ortho")
                    dct_coeffs = coeffs_real + 1j * coeffs_imag
                else:
                    dct_coeffs = dct(residual, norm="ortho")

                dct_thresh_val = thresh * np.max(np.abs(dct_coeffs)) if dct_coeffs.size else 0.0
                mask_dct = np.abs(dct_coeffs) > dct_thresh_val
                dct_sparse = dct_coeffs * mask_dct
                
                if np.iscomplexobj(dct_sparse):
                    update_real = idct(dct_sparse.real, norm="ortho")
                    update_imag = idct(dct_sparse.imag, norm="ortho")
                    structural_update = update_real + 1j * update_imag
                else:
                    structural_update = idct(dct_sparse, norm="ortho")
                
                structural += structural_update
                residual -= structural_update
                
                metadata["struct_energy"].append(_energy(structural_update))
                metadata["struct_sparsity"].append(1.0 - (np.sum(mask_dct) / mask_dct.size))
                # Zero entry for the other component to keep lists aligned
                metadata["texture_energy"].append(0.0) 
                metadata["texture_sparsity"].append(1.0)

            elif step_type == "rft":
                # --- RFT Update ---
                rft_coeffs = rft_forward(residual)
                rft_thresh_val = thresh * np.max(np.abs(rft_coeffs)) if rft_coeffs.size else 0.0
                
                # Enforce strict sparsity for RFT to avoid capturing broadband structure (steps)
                # Only keep the top few coefficients (e.g. 5) if they exceed threshold
                # This ensures RFT only captures resonant Phi components
                sorted_indices = np.argsort(np.abs(rft_coeffs))[::-1]
                mask_rft = np.zeros_like(rft_coeffs, dtype=bool)
                
                # Keep at most 5 coefficients, but only if they are above threshold
                top_k = 5
                for i in range(min(top_k, rft_coeffs.size)):
                    idx_coeff = sorted_indices[i]
                    if np.abs(rft_coeffs[idx_coeff]) > rft_thresh_val:
                        mask_rft[idx_coeff] = True
                    else:
                        break
                
                rft_sparse = rft_coeffs * mask_rft
                
                texture_update = rft_inverse(rft_sparse)
                texture += texture_update
                residual -= texture_update

                metadata["texture_energy"].append(_energy(texture_update))
                metadata["texture_sparsity"].append(1.0 - (np.sum(mask_rft) / mask_rft.size))
                metadata["struct_energy"].append(0.0)
                metadata["struct_sparsity"].append(1.0)

        residual_energy = _energy(residual)
        metadata["residual_energy"].append(residual_energy)

        if verbose:
            struct_ratio = np.sum(metadata["struct_energy"]) / total_energy if total_energy else 0.0
            texture_ratio = np.sum(metadata["texture_energy"]) / total_energy if total_energy else 0.0
            residual_ratio = residual_energy / total_energy if total_energy else 0.0
            print(
                f"Iteration {idx + 1}: struct={struct_ratio:.2%}, "
                f"texture={texture_ratio:.2%}, residual={residual_ratio:.2%}"
            )

        # Check convergence using the minimum of the two thresholds squared
        min_thresh = min(threshold_dct, threshold_rft)
        if total_energy > 0.0 and residual_energy / total_energy < min_thresh**2:
            break

    return HybridResult(structural, texture, residual, metadata)


from scipy.stats import kurtosis

def analyze_signal(x: ArrayLike) -> Dict[str, float]:
    """Extract simple descriptive statistics used for weight heuristics."""
    signal = np.asarray(x, dtype=np.complex128)
    n = signal.size

    centered = signal - np.mean(signal)
    autocorr = np.correlate(centered, centered, mode="full")[n - 1 :]
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
    autocorr_decay = float(np.mean(np.abs(autocorr) > 0.5))

    if n > 2:
        diff_signal = np.diff(signal)
        second_diff = np.abs(np.diff(signal, n=2))
        std_diff = np.std(second_diff) if np.std(second_diff) > 0 else 1.0
        edge_density = float(np.mean(second_diff > std_diff))
        
        # Kurtosis of the first derivative helps distinguish steps (high kurtosis) from waves (low kurtosis)
        # Sine wave derivative kurtosis is ~ -1.5 (platykurtic)
        # Random steps/text derivative kurtosis is typically higher (> -1.0)
        if np.all(np.imag(signal) == 0):
             k_val = float(kurtosis(np.real(diff_signal)))
        else:
             k_val = float(kurtosis(np.abs(diff_signal)))
    else:
        edge_density = 0.0
        k_val = 0.0

    spectrum_abs = np.abs(fft(signal))
    spectrum_power = spectrum_abs ** 2
    spectrum_sum = np.sum(spectrum_power)
    if spectrum_sum <= 0:
        spectral_entropy_norm = 0.0
    else:
        p = spectrum_power / spectrum_sum
        valid = p > 1e-10
        entropy = -np.sum(p[valid] * np.log2(p[valid]))
        spectral_entropy_norm = float(entropy / np.log2(n)) if n > 1 else 0.0

    top_peaks = np.sort(spectrum_abs)[-5:]
    if top_peaks.size < 2 or np.any(top_peaks[1:] == 0):
        quasi_periodicity = 0.0
    else:
        ratios = top_peaks[:-1] / top_peaks[1:]
        # Check for ratios close to 1/PHI or 1/PHI^2
        is_phi = np.abs(ratios - (1.0 / PHI)) < 0.15
        is_phi2 = np.abs(ratios - (1.0 / PHI**2)) < 0.15
        quasi_periodicity = float(np.mean(is_phi | is_phi2))

    if n > 1:
        diff_signal = np.diff(signal)
        var_diff = np.mean(np.abs(diff_signal - np.mean(diff_signal))**2)
        smoothness = float(1.0 / (1.0 + var_diff))
    else:
        smoothness = 1.0

    return {
        "autocorr_decay": autocorr_decay,
        "edge_density": edge_density,
        "spectral_entropy": spectral_entropy_norm,
        "quasi_periodicity": quasi_periodicity,
        "smoothness": smoothness,
        "kurtosis": k_val,
    }


def predict_weights(features: Dict[str, float]) -> Dict[str, float]:
    """Heuristic allocation of DCT vs Φ-RFT preference weights."""
    # Strong edge density indicates symbolic/step-like data (ASCII, Code)
    # We prioritize this to avoid the "ASCII Bottleneck" where RFT fails.
    # However, pure sine waves also have high edge density (by this metric).
    # We use kurtosis to distinguish: Waves < -1.0, Steps > -1.0.
    
    is_step_like = features["edge_density"] > 0.45 and features["kurtosis"] > -1.2
    
    if is_step_like:
        return {"dct": 0.95, "rft": 0.05}

    if features["quasi_periodicity"] > 0.35:
        if features["spectral_entropy"] > 0.35:
            return {"dct": 0.50, "rft": 0.50}
        return {"dct": 0.20, "rft": 0.80}

    if features["edge_density"] > 0.3:
        return {"dct": 0.95, "rft": 0.05}
    if features["smoothness"] > 0.8:
        return {"dct": 0.85, "rft": 0.15}
    if features["spectral_entropy"] > 0.9:
        return {"dct": 0.50, "rft": 0.50}
    return {"dct": 0.65, "rft": 0.35}


def adaptive_hybrid_compress(
    x: ArrayLike,
    *,
    verbose: bool = False,
    max_iter: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, object]]:
    """Run adaptive hybrid decomposition and return components plus metadata."""
    features = analyze_signal(x)
    weights = predict_weights(features)

    if verbose:
        print("Signal features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        print("Predicted weights:")
        print(f"  DCT: {weights['dct']:.2f}")
        print(f"  RFT: {weights['rft']:.2f}")

    threshold_dct = 0.15 * (1.0 - weights["dct"])
    threshold_rft = 0.15 * (1.0 - weights["rft"])
    # Ensure minimum threshold
    threshold_dct = max(threshold_dct, 1e-3)
    threshold_rft = max(threshold_rft, 1e-3)

    # Determine strategy based on dominant weight
    if weights["dct"] > 0.6:
        strategy = "dct_first"
    else:
        strategy = "rft_first"

    result = hybrid_decomposition(
        x,
        max_iter=max_iter,
        threshold_dct=threshold_dct,
        threshold_rft=threshold_rft,
        strategy=strategy,
        verbose=verbose,
    )
    result.metadata["features"] = features
    result.metadata["weights"] = weights

    return result.structural, result.texture, weights, result.metadata


__all__ = [
    "PHI",
    "HybridResult",
    "adaptive_hybrid_compress",
    "analyze_signal",
    "hybrid_decomposition",
    "predict_weights",
    "rft_forward",
    "rft_inverse",
]
