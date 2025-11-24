#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Hybrid RFT Residual Predictor
================================

Implements a tiny feed-forward MLP that predicts residual corrections for
quantized/pruned RFT spectral coefficients (amplitude + phase) after coarse
quantization. This module is intentionally compact and dependency-light.

Design Goals
------------
* Extremely small parameter footprint (few KB typical) so storage overhead is
  amortized across many tensors.
* Deterministic serialization (pure JSON) for portability + integrity hashes.
* Flexible input feature construction (index normalization, band id, coarse
  log amplitude, coarse phase) and optional sinusoidal index encodings.
* Supports training over a heterogeneous batch of tensors in a single pass.

Predictor I/O Contract
----------------------
Inputs (per coefficient):
  idx_norm: float in [0,1]
  band_id: integer (embedded as one-hot or learned embedding) => we use one-hot
  A_log_coarse: log( max(coarse_amplitude, 1e-12) )
  phase_coarse: coarse phase in radians [-pi, pi]

Outputs:
  delta_log_amp, delta_phase (added to coarse quantities). We clamp the final
  reconstructed amplitude to >= 0 and phase to [-pi, pi].

Loss:
  Mean squared error over (log_amp + phase) residual components.

Serialization Format (JSON):
{
  "type": "rft_hybrid_residual_predictor",
  "version": 1,
  "architecture": {
      "input_dim": int,
      "hidden_dim": int,
      "output_dim": 2,
      "num_hidden_layers": int,
      "activation": "relu",
      "bands": int,
      "use_fourier_features": bool,
      "fourier_features": int
  },
  "state": {"W0": [...], "b0": [...], ...},
  "training": {"epochs": int, "samples": int, "loss": float},
  "checksum_sha256": "..."
}

All weights stored as base64-encoded raw float32 arrays (flattened) to avoid
JSON float bloat while remaining human-inspectable with small overhead.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _b64_encode_f32(arr: np.ndarray) -> str:
    arr = arr.astype(np.float32, copy=False).ravel()
    return base64.b64encode(arr.tobytes()).decode('ascii')


def _b64_decode_f32(data: str, shape: Tuple[int, ...]) -> np.ndarray:
    raw = base64.b64decode(data.encode('ascii'))
    arr = np.frombuffer(raw, dtype=np.float32)
    return arr.reshape(shape).copy()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class PredictorConfig:
    input_dim: int
    hidden_dim: int = 32
    num_hidden_layers: int = 2
    bands: int = 1
    use_fourier_features: bool = False
    fourier_features: int = 0
    activation: str = "relu"

    def to_dict(self) -> Dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_hidden_layers": self.num_hidden_layers,
            "bands": self.bands,
            "use_fourier_features": self.use_fourier_features,
            "fourier_features": self.fourier_features,
            "activation": self.activation,
        }


# ---------------------------------------------------------------------------
# Tiny MLP implementation (NumPy only)
# ---------------------------------------------------------------------------
class TinyResidualPredictor:
    """A very small MLP for residual prediction (NumPy backend)."""

    def __init__(self, config: PredictorConfig):
        self.config = config
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []  # (W, b)
        rng = np.random.default_rng(42)
        in_dim = config.input_dim
        for layer_idx in range(config.num_hidden_layers):
            W = rng.standard_normal((in_dim, config.hidden_dim)).astype(np.float32) / math.sqrt(in_dim)
            b = np.zeros((config.hidden_dim,), dtype=np.float32)
            self.layers.append((W, b))
            in_dim = config.hidden_dim
        # Output layer (2 dims: delta_log_amp, delta_phase)
        W_out = rng.standard_normal((in_dim, 2)).astype(np.float32) / math.sqrt(in_dim)
        b_out = np.zeros((2,), dtype=np.float32)
        self.layers.append((W_out, b_out))

    # Activation
    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.config.activation == "relu":
            return np.maximum(x, 0.0, out=x)
        elif self.config.activation == "tanh":
            return np.tanh(x)
        else:
            return x  # identity fallback

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for (W, b) in self.layers[:-1]:
            h = h @ W + b
            h = self._act(h)
        W_out, b_out = self.layers[-1]
        return h @ W_out + b_out

    # Simple SGD training
    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 5, batch_size: int = 4096, lr: float = 1e-3) -> Dict[str, float]:
        n = X.shape[0]
        if n == 0:
            return {"loss": float('nan'), "epochs": 0}
        rng = np.random.default_rng(123)
        for epoch in range(epochs):
            idx = rng.permutation(n)
            X_shuf = X[idx]
            Y_shuf = Y[idx]
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start+batch_size]
                yb = Y_shuf[start:start+batch_size]
                # Forward
                activations = []
                h = xb
                for (W, b) in self.layers[:-1]:
                    z = h @ W + b
                    activations.append((h, W, z))
                    h = self._act(z.copy())
                W_out, b_out = self.layers[-1]
                pred = h @ W_out + b_out
                # Loss (MSE)
                diff = pred - yb
                loss = (diff * diff).mean()
                # Backprop
                grad_pred = 2.0 * diff / diff.shape[0]
                grad_W_out = h.T @ grad_pred
                grad_b_out = grad_pred.sum(axis=0)
                # Prop through last hidden
                grad_h = grad_pred @ W_out.T
                # Update output layer
                W_out -= lr * grad_W_out
                b_out -= lr * grad_b_out
                # Hidden layers reverse
                for (h_prev, W, z) in reversed(activations):
                    if self.config.activation == "relu":
                        grad_z = grad_h * (z > 0).astype(np.float32)
                    elif self.config.activation == "tanh":
                        tanh_z = np.tanh(z)
                        grad_z = grad_h * (1 - tanh_z * tanh_z)
                    else:
                        grad_z = grad_h
                    grad_W = h_prev.T @ grad_z
                    grad_b = grad_z.sum(axis=0)
                    grad_h = grad_z @ W.T
                    W -= lr * grad_W
                    b -= lr * grad_b
            # (Optional) early stopping hooks could be added here
        final_pred = self.forward(X)
        final_loss = float(((final_pred - Y) ** 2).mean())
        return {"loss": final_loss, "epochs": epochs}

    def serialize(self, training_meta: Optional[Dict[str, float]] = None) -> Dict:
        arch = self.config.to_dict()
        state: Dict[str, Dict] = {}
        for i, (W, b) in enumerate(self.layers):
            state[f"W{i}"] = {"shape": list(W.shape), "data": _b64_encode_f32(W)}
            state[f"b{i}"] = {"shape": list(b.shape), "data": _b64_encode_f32(b)}
        payload = {
            "type": "rft_hybrid_residual_predictor",
            "version": 1,
            "architecture": arch,
            "state": state,
            "training": training_meta or {},
        }
        digest = _sha256_bytes(json.dumps(payload, sort_keys=True).encode('utf-8'))
        payload["checksum_sha256"] = digest
        return payload

    @staticmethod
    def deserialize(obj: Dict) -> "TinyResidualPredictor":
        if obj.get("type") != "rft_hybrid_residual_predictor":
            raise ValueError("Invalid predictor type")
        arch = obj["architecture"]
        config = PredictorConfig(**arch)
        model = TinyResidualPredictor(config)
        # Load weights
        state = obj["state"]
        layers: List[Tuple[np.ndarray, np.ndarray]] = []
        layer_pairs = sorted({int(k[1:]) for k in state if k.startswith('W')})
        for idx in layer_pairs:
            W_entry = state[f"W{idx}"]
            b_entry = state[f"b{idx}"]
            W = _b64_decode_f32(W_entry["data"], tuple(W_entry["shape"]))
            b = _b64_decode_f32(b_entry["data"], tuple(b_entry["shape"]))
            layers.append((W, b))
        model.layers = layers
        return model


# ---------------------------------------------------------------------------
# Dataset assembly helpers
# ---------------------------------------------------------------------------

def build_training_samples(
    tensors: Iterable[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build (X,Y) arrays for predictor training.

    Each item: (tensor_name, idx_norm, band_id, logA_coarse, phase_coarse, target_residual)
    where target_residual is (delta_logA, delta_phase).
    """
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    for _, idx_norm, band_ids, logA, phase, residual in tensors:
        feat = np.stack([idx_norm, logA, phase], axis=1)  # base features
        # One-hot band
        if band_ids.ndim != 1:
            raise ValueError("band_ids must be 1D")
        bands = int(band_ids.max()) + 1 if band_ids.size else 1
        oh = np.zeros((band_ids.size, bands), dtype=np.float32)
        if band_ids.size:
            oh[np.arange(band_ids.size), band_ids.astype(int)] = 1.0
        feat = np.concatenate([feat, oh], axis=1)
        X_list.append(feat.astype(np.float32))
        Y_list.append(residual.astype(np.float32))
    if not X_list:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)
    return X, Y


# ---------------------------------------------------------------------------
# High-level training convenience
# ---------------------------------------------------------------------------

def train_residual_predictor(
    sample_iter: Iterable[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    bands: int,
    hidden_dim: int = 32,
    epochs: int = 5,
    steps_cap: Optional[int] = None,
    lr: float = 1e-3,
) -> Dict:
    X, Y = build_training_samples(sample_iter)
    if steps_cap is not None and X.shape[0] > steps_cap:
        X = X[:steps_cap]
        Y = Y[:steps_cap]
    input_dim = X.shape[1] if X.size else (3 + bands)
    config = PredictorConfig(input_dim=input_dim, hidden_dim=hidden_dim, bands=bands)
    model = TinyResidualPredictor(config)
    stats = model.train(X, Y, epochs=epochs, lr=lr)
    payload = model.serialize({"loss": stats["loss"], "epochs": stats["epochs"], "samples": int(X.shape[0])})
    return payload


__all__ = [
    "TinyResidualPredictor",
    "PredictorConfig",
    "train_residual_predictor",
    "build_training_samples",
]
