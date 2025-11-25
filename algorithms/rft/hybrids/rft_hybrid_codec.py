#!/usr/bin/env python3
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
"""RFT Hybrid Codec (Quant + Residual Re-synthesis)
===================================================

End-to-end encoding/decoding of tensors using:
  * RFT transform (leveraging existing deterministic golden unitary from rft_vertex_codec)
  * Band partitioning of spectral coefficients (geometric growth)
  * Optional pruning via amplitude threshold
  * Scalar quantization of log-amplitude + uniform phase quantization
  * Residual predictor training data extraction (delta logA / delta phase)
  * Serialization of per-tensor hybrid containers + manifest-ready metadata

This module DOES NOT train the residual predictor itself (see
`hybrid_residual_predictor.train_residual_predictor`). Instead, it produces the
feature/target arrays needed for training and can apply an already serialized
predictor at decode time to reconstruct higher-fidelity coefficients.

Container Format (per tensor JSON):
{
  "type": "rft_hybrid_tensor",
  "version": 1,
  "dtype": "float32",
  "original_shape": [...],
  "numel": int,
  "bands": [
      {"id":0, "start":0, "end":B0_end, "quant": {...}},
      ...
  ],
  "codec": {
      "mode": "hybrid",
      "quant_amp_bits": 6,
      "quant_phase_bits": 5,
      "prune_threshold": 1e-4,
      "residual_predictor_ref": "predictors/global_mlp_v1.json"
  },
  "sparsity": 0.82,    # fraction pruned
  "bitrate_coeff": 11.3,# bits per kept coeff (approx)
  "kept_coeff": int,
  "total_coeff": int,
  "payload": {
      "indices": base64-of-packed-kept-indices (uint32),
      "amp_codes": base64-of-quantized-amplitudes (uint16/uint8),
      "phase_codes": base64-of-quantized-phases (uint8),
      "amp_scale": [min_log_amp, max_log_amp],
      "phase_scale": [-pi, pi]
  }
}

Decoding reconstructs coarse coefficients and (optionally) applies the residual
predictor:  logA' = logA_coarse + delta_logA  ;  phase' = clamp(phase_coarse + delta_phase).

Author: QuantoniumOS Hybrid Compression Initiative
License: MIT
"""
from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np

from algorithms.rft.core.closed_form_rft import rft_forward, rft_inverse
from algorithms.rft.compression.rft_vertex_codec import _generate_seed  # deterministic seed for size
from .hybrid_residual_predictor import TinyResidualPredictor

# ---------------------------------------------------------------------------
# Helper serialization
# ---------------------------------------------------------------------------

def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode('ascii')


def _b64_decode(data: str, dtype: np.dtype) -> np.ndarray:
    raw = base64.b64decode(data)
    return np.frombuffer(raw, dtype=dtype)


# ---------------------------------------------------------------------------
# Band partitioning
# ---------------------------------------------------------------------------

def partition_bands(length: int, growth: float = 2.0, min_band: int = 64) -> List[Tuple[int, int]]:
    bands = []
    start = 0
    size = min_band
    while start < length:
        end = min(length, start + size)
        bands.append((start, end))
        start = end
        size = int(size * growth)
    return bands


# ---------------------------------------------------------------------------
# Quantization utilities
# ---------------------------------------------------------------------------

def quantize_uniform(values: np.ndarray, bits: int, value_min: float, value_max: float) -> Tuple[np.ndarray, Dict[str, float]]:
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = (1 << bits) - 1
    if value_max <= value_min:
        # degenerate; all zeros
        codes = np.zeros_like(values, dtype=np.uint32)
        return codes, {"min": value_min, "max": value_max, "bits": bits, "levels": levels}
    norm = (values - value_min) / (value_max - value_min)
    codes = np.clip(np.rint(norm * levels), 0, levels).astype(np.uint32)
    return codes, {"min": value_min, "max": value_max, "bits": bits, "levels": levels}


def dequantize_uniform(codes: np.ndarray, meta: Dict[str, float]) -> np.ndarray:
    bits = int(meta["bits"])
    levels = (1 << bits) - 1
    value_min = meta["min"]
    value_max = meta["max"]
    if value_max <= value_min:
        return np.full_like(codes, value_min, dtype=np.float64)
    return (codes.astype(np.float64) / levels) * (value_max - value_min) + value_min


# ---------------------------------------------------------------------------
# Core spectral transform wrappers (real-valued segments)
# ---------------------------------------------------------------------------

def rft_forward_real(vec: np.ndarray) -> np.ndarray:
    # Use verified closed-form Φ-RFT
    return rft_forward(vec)


def rft_inverse_real(coeffs: np.ndarray) -> np.ndarray:
    # Use verified closed-form Φ-RFT
    rec = rft_inverse(coeffs)
    return rec.real


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
@dataclass
class HybridEncodingResult:
    container: Dict
    residual_samples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def encode_tensor_hybrid(
    tensor: np.ndarray,
    prune_threshold: float = 0.0,
    quant_amp_bits: int = 6,
    quant_phase_bits: int = 5,
    band_growth: float = 2.0,
    min_band: int = 64,
    tensor_name: str = "tensor",
    collect_residual_samples: bool = True,
) -> HybridEncodingResult:
    if tensor.dtype.kind not in "f":
        raise ValueError("Only floating tensors supported by hybrid codec")
    flat = tensor.reshape(-1).astype(np.float64)
    n = flat.size
    coeffs = rft_forward_real(flat)
    amps = np.abs(coeffs)
    phases = np.angle(coeffs)
    log_amps = np.log(np.maximum(amps, 1e-12))

    # Pruning mask
    keep_mask = amps >= prune_threshold if prune_threshold > 0 else np.ones_like(amps, dtype=bool)
    kept_indices = np.nonzero(keep_mask)[0]

    # Band partition
    bands = partition_bands(n, growth=band_growth, min_band=min_band)
    band_meta = []
    band_ids = np.zeros(n, dtype=np.int32)
    for bid, (s, e) in enumerate(bands):
        band_ids[s:e] = bid
        band_meta.append({"id": bid, "start": s, "end": e})

    kept_band_ids = band_ids[kept_indices]
    kept_log_amps = log_amps[kept_indices]
    kept_phases = phases[kept_indices]

    # Quantization ranges
    log_min = float(kept_log_amps.min()) if kept_log_amps.size else 0.0
    log_max = float(kept_log_amps.max()) if kept_log_amps.size else 0.0
    phase_min = -math.pi
    phase_max = math.pi

    amp_codes, amp_qmeta = quantize_uniform(kept_log_amps, quant_amp_bits, log_min, log_max) if kept_log_amps.size else (np.zeros(0,dtype=np.uint32), {"min":0.0, "max":0.0, "bits":quant_amp_bits, "levels":(1<<quant_amp_bits)-1})
    phase_codes, phase_qmeta = quantize_uniform(kept_phases, quant_phase_bits, phase_min, phase_max) if kept_phases.size else (np.zeros(0,dtype=np.uint32), {"min":phase_min, "max":phase_max, "bits":quant_phase_bits, "levels":(1<<quant_phase_bits)-1})

    # Coarse reconstruction for residuals
    coarse_log_amps = dequantize_uniform(amp_codes, amp_qmeta) if kept_log_amps.size else kept_log_amps
    coarse_phases = dequantize_uniform(phase_codes, phase_qmeta) if kept_phases.size else kept_phases

    delta_logA = kept_log_amps - coarse_log_amps
    delta_phase = kept_phases - coarse_phases
    # Wrap phase deltas into [-pi,pi]
    delta_phase = (delta_phase + math.pi) % (2*math.pi) - math.pi

    # Feature prep for residual predictor training
    residual_samples: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    if collect_residual_samples and kept_indices.size:
        idx_norm = kept_indices.astype(np.float64) / max(1, n-1)
        residual = np.stack([delta_logA, delta_phase], axis=1).astype(np.float32)
        residual_samples.append((tensor_name, idx_norm.astype(np.float32), kept_band_ids.astype(np.float32), coarse_log_amps.astype(np.float32), coarse_phases.astype(np.float32), residual))

    # Pack kept indices as uint32
    indices_u32 = kept_indices.astype(np.uint32)

    container = {
        "type": "rft_hybrid_tensor",
        "version": 1,
        "dtype": str(tensor.dtype),
        "original_shape": list(tensor.shape),
        "numel": int(tensor.size),
        "codec": {
            "mode": "hybrid",
            "quant_amp_bits": int(quant_amp_bits),
            "quant_phase_bits": int(quant_phase_bits),
            "prune_threshold": float(prune_threshold),
        },
        "bands": band_meta,
        "sparsity": float(1.0 - kept_indices.size / max(1, n)),
        "kept_coeff": int(kept_indices.size),
        "total_coeff": int(n),
        "payload": {
            "indices": _b64(indices_u32.view(np.uint8)),
            "amp_codes": _b64(amp_codes.astype(np.uint16).view(np.uint8)),
            "phase_codes": _b64(phase_codes.astype(np.uint16).view(np.uint8)),
            "amp_scale": [log_min, log_max],
            "phase_scale": [phase_min, phase_max],
        },
    }
    # Approx bitrate (bits per kept coeff)
    bits_amp = quant_amp_bits
    bits_phase = quant_phase_bits
    container["bitrate_coeff"] = float(bits_amp + bits_phase)
    return HybridEncodingResult(container=container, residual_samples=residual_samples)


# ---------------------------------------------------------------------------
# Decoding (coarse + residual application)
# ---------------------------------------------------------------------------

def decode_tensor_hybrid(container: Dict, predictor: Optional[TinyResidualPredictor] = None) -> np.ndarray:
    if container.get("type") != "rft_hybrid_tensor":
        raise ValueError("Not a hybrid tensor container")
    shape = tuple(container["original_shape"])
    n = int(container["total_coeff"]) if "total_coeff" in container else int(np.prod(shape))
    payload = container["payload"]
    indices_raw = _b64_decode(payload["indices"], np.uint8).view(np.uint32)
    amp_codes = _b64_decode(payload["amp_codes"], np.uint8).view(np.uint16).astype(np.uint32)
    phase_codes = _b64_decode(payload["phase_codes"], np.uint8).view(np.uint16).astype(np.uint32)
    log_min, log_max = payload["amp_scale"]
    phase_min, phase_max = payload["phase_scale"]
    kept_indices = indices_raw.astype(np.int64)

    # Dequantize coarse
    kept_logA = dequantize_uniform(amp_codes, {"min":log_min, "max":log_max, "bits":container["codec"]["quant_amp_bits"]}) if amp_codes.size else np.empty(0)
    kept_phase = dequantize_uniform(phase_codes, {"min":phase_min, "max":phase_max, "bits":container["codec"]["quant_phase_bits"]}) if phase_codes.size else np.empty(0)

    if predictor is not None and amp_codes.size:
        # Build predictor features (match training assembly)
        bands_meta = container.get("bands", [])
        # Construct band id vector
        band_ids = np.zeros(kept_indices.size, dtype=np.int32)
        for b in bands_meta:
            bid = b["id"]
            s, e = b["start"], b["end"]
            mask = (kept_indices >= s) & (kept_indices < e)
            band_ids[mask] = bid
        idx_norm = kept_indices.astype(np.float64) / max(1, n-1)
        # One-hot bands
        B = int(band_ids.max()) + 1 if band_ids.size else 1
        oh = np.zeros((kept_indices.size, B), dtype=np.float32)
        if band_ids.size:
            oh[np.arange(kept_indices.size), band_ids] = 1.0
        feat = np.stack([idx_norm, kept_logA, kept_phase], axis=1).astype(np.float32)
        feat = np.concatenate([feat, oh], axis=1)
        deltas = predictor.forward(feat).astype(np.float64)
        delta_logA = deltas[:,0]
        delta_phase = deltas[:,1]
        kept_logA = kept_logA + delta_logA
        kept_phase = kept_phase + delta_phase
        kept_phase = (kept_phase + math.pi) % (2*math.pi) - math.pi

    # Reconstruct full coefficient vector
    coeffs = np.zeros(n, dtype=np.complex128)
    if kept_indices.size:
        amps = np.exp(kept_logA)
        complex_vals = amps * (np.cos(kept_phase) + 1j * np.sin(kept_phase))
        coeffs[kept_indices] = complex_vals
    # Inverse
    flat_rec = rft_inverse_real(coeffs)
    return flat_rec.reshape(shape).astype(np.float32)


class RFTHybridCodec:
    """
    A wrapper class for the hybrid codec functions to provide a consistent interface.
    """
    def __init__(self, **kwargs):
        self.encode_kwargs = kwargs

    def encode(self, tensor: np.ndarray) -> Dict:
        return encode_tensor_hybrid(tensor, **self.encode_kwargs).container

    def decode(self, container: Dict, predictor: Optional[TinyResidualPredictor] = None) -> np.ndarray:
        return decode_tensor_hybrid(container, predictor)


__all__ = [
    "encode_tensor_hybrid",
    "decode_tensor_hybrid",
    "partition_bands",
    "RFTHybridCodec",
]
