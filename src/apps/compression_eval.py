# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Compression evaluation harness for local LLM weights.

Purpose
- Runs a small, repeatable sweep comparing wave-domain masking vs matched-parameter low-rank.
- Writes a JSON report that other apps (e.g. the chatbox) can read to decide what to enable.

Notes
- This does NOT magically eliminate datacenters. It helps you pick and validate local compression
  schemes for on-device inference.
- Wave masking uses a dense basis in this code (O(N^2) transforms). Compute wins require a fast
  transform/kernel.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class MatrixPick:
    name: str
    tensor: torch.Tensor


def construct_rft_basis(N: int, phi: float) -> torch.Tensor:
    """RFT-inspired unitary basis using QR (demo-quality unitary)."""
    n = np.arange(N)
    k = np.arange(N)
    freqs = (k + 1) * phi
    phases = 2 * np.pi * np.outer(freqs, n)
    Psi = (1 / np.sqrt(N)) * np.exp(1j * phases)
    q, _r = np.linalg.qr(Psi)
    return torch.tensor(q, dtype=torch.complex64)


def band_mask(N: int, bandwidth: int) -> torch.Tensor:
    i = torch.arange(N).view(-1, 1)
    j = torch.arange(N).view(1, -1)
    return (torch.abs(i - j) <= bandwidth)


def block_diag_mask(N: int, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    idx = torch.arange(N)
    bi = (idx // block_size).view(-1, 1)
    bj = (idx // block_size).view(1, -1)
    return bi.eq(bj)


def fro_rel_error(A: torch.Tensor, B: torch.Tensor) -> float:
    num = torch.linalg.norm(A - B)
    den = torch.linalg.norm(A)
    return (num / (den + 1e-12)).real.item()


def truncated_svd_approx(W: torch.Tensor, rank: int) -> torch.Tensor:
    if rank <= 0:
        return torch.zeros_like(W)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    Ur = U[:, :rank]
    Sr = S[:rank]
    Vhr = Vh[:rank, :]
    return (Ur * Sr) @ Vhr


def low_rank_rank_for_budget(N: int, wave_kept_entries: int) -> int:
    """Match parameter budget: each kept complex entry ~2 real scalars."""
    wave_params = 2 * int(wave_kept_entries)
    per_rank_params = (2 * N + 1)
    r = wave_params // per_rank_params
    return int(max(0, min(N, r)))


def _best_square_submatrix_from_state_dict(state_dict: Dict[str, Any], n_target: int) -> Optional[MatrixPick]:
    best: Optional[Tuple[int, str, torch.Tensor]] = None
    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            continue
        d0, d1 = tensor.shape
        min_dim = int(min(d0, d1))
        if min_dim < n_target:
            continue
        if best is None or min_dim > best[0]:
            best = (min_dim, name, tensor)
    if best is None:
        return None
    _min_dim, name, tensor = best
    return MatrixPick(name=name, tensor=tensor)


def _as_square_submatrix(pick: MatrixPick, n: int) -> torch.Tensor:
    t = pick.tensor
    N = int(min(n, t.shape[0], t.shape[1]))
    return t[:N, :N].detach().to(torch.float32).cpu()


def evaluate_weight_matrix(
    W_real: torch.Tensor,
    phi: float = (1 + np.sqrt(5)) / 2,
    bandwidths: Iterable[int] = (0, 1, 2, 4, 8, 16, 32),
    block_sizes: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
    n_inputs: int = 64,
    seed: int = 0,
) -> Dict[str, Any]:
    N = int(W_real.shape[0])
    Psi = construct_rft_basis(N, phi).to(torch.complex64)
    Wc = W_real.to(torch.complex64)

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_inputs, N, generator=g)
    y_true = (x.to(torch.complex64) @ Wc).real

    Lambda_full = Psi @ Wc @ Psi.conj().T

    def approx_wave(mask: torch.Tensor) -> torch.Tensor:
        Lambda_masked = Lambda_full * mask.to(Lambda_full.dtype)
        return Psi.conj().T @ Lambda_masked @ Psi

    band_rows: List[Dict[str, Any]] = []
    block_rows: List[Dict[str, Any]] = []

    for b in bandwidths:
        if b >= N:
            continue
        m = band_mask(N, int(b))
        kept = int(m.sum().item())
        kfrac = kept / float(N * N)

        W_wave = approx_wave(m)
        y_wave = (x.to(torch.complex64) @ W_wave).real

        r = low_rank_rank_for_budget(N, kept)
        W_lr = truncated_svd_approx(W_real, r).to(torch.complex64)
        y_lr = (x.to(torch.complex64) @ W_lr).real

        band_rows.append(
            {
                "bandwidth": int(b),
                "kept_fraction": float(kfrac),
                "kept_entries": int(kept),
                "wave": {
                    "rel_frob": float(fro_rel_error(Wc, W_wave)),
                    "out_mse": float(torch.mean((y_true - y_wave) ** 2).item()),
                },
                "low_rank": {
                    "rank": int(r),
                    "rel_frob": float(fro_rel_error(Wc, W_lr)),
                    "out_mse": float(torch.mean((y_true - y_lr) ** 2).item()),
                },
            }
        )

    for B in block_sizes:
        if B <= 0 or B > N:
            continue
        m = block_diag_mask(N, int(B))
        kept = int(m.sum().item())
        kfrac = kept / float(N * N)

        W_wave = approx_wave(m)
        y_wave = (x.to(torch.complex64) @ W_wave).real

        r = low_rank_rank_for_budget(N, kept)
        W_lr = truncated_svd_approx(W_real, r).to(torch.complex64)
        y_lr = (x.to(torch.complex64) @ W_lr).real

        block_rows.append(
            {
                "block_size": int(B),
                "kept_fraction": float(kfrac),
                "kept_entries": int(kept),
                "wave": {
                    "rel_frob": float(fro_rel_error(Wc, W_wave)),
                    "out_mse": float(torch.mean((y_true - y_wave) ** 2).item()),
                },
                "low_rank": {
                    "rank": int(r),
                    "rel_frob": float(fro_rel_error(Wc, W_lr)),
                    "out_mse": float(torch.mean((y_true - y_lr) ** 2).item()),
                },
            }
        )

    return {
        "N": N,
        "phi": float(phi),
        "banded": band_rows,
        "block_diag": block_rows,
    }


def run_model_sweep(
    model_id: str = "distilgpt2",
    n_target: int = 128,
    out_path: str = "logs/compression_report.json",
) -> Dict[str, Any]:
    """Load a local HF model, pick a large 2D matrix, run sweep, write JSON report."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    report: Dict[str, Any] = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": model_id,
        "n_target": int(n_target),
        "status": "unknown",
    }

    try:
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(model_id)
        sd = m.state_dict()
        pick = _best_square_submatrix_from_state_dict(sd, n_target=n_target)
        if pick is None:
            raise RuntimeError(f"No 2D tensor with min_dim >= {n_target} in {model_id}.")

        W = _as_square_submatrix(pick, n_target)
        report.update(
            {
                "status": "ok",
                "picked_tensor": pick.name,
                "N": int(W.shape[0]),
                "results": evaluate_weight_matrix(W),
            }
        )
    except Exception as e:
        report.update({"status": "error", "error": f"{type(e).__name__}: {e}"})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def summarize_report(report: Dict[str, Any], target_kept_fraction: float = 0.25) -> str:
    """Tiny human-readable one-liner for UI/status bars."""
    if report.get("status") != "ok":
        return f"compression eval: {report.get('status')}"

    results = report.get("results", {})
    band = results.get("banded", [])
    block = results.get("block_diag", [])

    def _best_at_frac(rows: List[Dict[str, Any]]):
        if not rows:
            return None
        # choose closest kept_fraction to target
        rows2 = sorted(rows, key=lambda r: abs(r["kept_fraction"] - target_kept_fraction))
        return rows2[0]

    b = _best_at_frac(band)
    k = _best_at_frac(block)
    parts: List[str] = []
    if b is not None:
        parts.append(
            f"band@{b['kept_fraction']:.2f}: waveMSE={b['wave']['out_mse']:.3g} lrMSE={b['low_rank']['out_mse']:.3g}"
        )
    if k is not None:
        parts.append(
            f"block@{k['kept_fraction']:.2f}: waveMSE={k['wave']['out_mse']:.3g} lrMSE={k['low_rank']['out_mse']:.3g}"
        )
    return " | ".join(parts) if parts else "compression eval: ok"
