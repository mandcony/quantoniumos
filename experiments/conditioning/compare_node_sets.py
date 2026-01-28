#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-QuantoniumOS-Claims-NC
# Copyright (C) 2026 Luis M. Minier / quantoniumos
# This file is listed in CLAIMS_PRACTICING_FILES.txt and is licensed
# under LICENSE-CLAIMS-NC.md (research/education only). Commercial
# rights require a separate patent license from the author.
"""
Compare conditioning and coherence across φ-modulated and baseline grids.

Outputs JSON with:
- Gram matrix condition number κ(G)
- Minimum angular separation (unit circle)
- Coherence μ of Gram-normalized basis
- Eigenvalue summary for G

Canonical grid (frozen): f_k = frac((k+1) * φ)
Non-canonical variant: f_k = frac(φ^{-k})

Note on scaling: log-log slopes are indicative finite-N trends; non-monotonic
behavior can reflect Diophantine resonances at Fibonacci-related sizes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PHI = (1 + np.sqrt(5)) / 2


def frac(x: np.ndarray) -> np.ndarray:
    return np.mod(x, 1.0)


def grid_equispaced(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / n


def grid_phi_modulated(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return frac((k + 1.0) * PHI)


def grid_phi_inverse(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return frac(PHI ** (-k))


def grid_irrational(n: int, alpha: float) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return frac((k + 1.0) * alpha)


def grid_random(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n, dtype=np.float64)


def grid_jittered(n: int, rng: np.random.Generator) -> np.ndarray:
    base = grid_equispaced(n)
    jitter = rng.uniform(-0.5 / n, 0.5 / n, size=n)
    return frac(base + jitter)


def min_angular_separation(freqs: np.ndarray) -> float:
    """Minimum circular distance between sorted frequencies."""
    f = np.sort(freqs)
    diffs = np.diff(f)
    wrap = 1.0 - f[-1] + f[0]
    return float(np.min(np.concatenate([diffs, [wrap]])))


def build_phi_matrix(freqs: np.ndarray, n: int) -> np.ndarray:
    """Unit-circle Vandermonde: Φ[n,k] = exp(j 2π f_k n) / sqrt(N)."""
    n_idx = np.arange(n, dtype=np.float64)
    return np.exp(1j * 2.0 * np.pi * np.outer(n_idx, freqs)) / np.sqrt(float(n))


def gram_inverse_sqrt(G: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Compute (G)^{-1/2} for Hermitian positive semidefinite G."""
    w, v = np.linalg.eigh(G)
    w = np.maximum(w, eps)
    inv_sqrt = v @ np.diag(1.0 / np.sqrt(w)) @ v.conj().T
    return inv_sqrt


def coherence(phi_tilde: np.ndarray) -> float:
    """Max absolute entry of unitary basis (coherence proxy)."""
    return float(np.max(np.abs(phi_tilde)))


def kappa_from_svd(Phi: np.ndarray) -> float:
    """Condition number via singular values of Phi (stable)."""
    try:
        s = np.linalg.svd(Phi, compute_uv=False)
        if s[-1] <= 0:
            return float("inf")
        return float(s[0] / s[-1])
    except np.linalg.LinAlgError:
        return float("inf")


def kappa_from_eigs(eigs: np.ndarray) -> float:
    """Condition number via eigenvalues of G=Phi^H Phi (clamped)."""
    eig_min = float(np.min(eigs))
    eig_max = float(np.max(eigs))
    if eig_min <= 0:
        return float("inf")
    return float(eig_max / eig_min)


def eigen_summary(G: np.ndarray) -> Dict[str, float]:
    """Eigen summary with raw/clamped min for PSD stability reporting."""
    w = np.linalg.eigvalsh(G)
    w_min_raw = float(np.min(w))
    w_clamped = np.maximum(w, 0.0)
    return {
        "min_raw": w_min_raw,
        "min_clamped": float(np.min(w_clamped)),
        "max": float(np.max(w)),
        "mean": float(np.mean(w)),
    }


def compute_metrics(freqs: np.ndarray, n: int) -> Dict[str, float]:
    Phi = build_phi_matrix(freqs, n)
    G = Phi.conj().T @ Phi
    # Symmetrize to remove numerical asymmetry
    G = 0.5 * (G + G.conj().T)
    k = kappa_from_svd(Phi)
    min_sep = min_angular_separation(freqs)
    inv_sqrt = gram_inverse_sqrt(G)
    Phi_tilde = Phi @ inv_sqrt
    mu = coherence(Phi_tilde)
    eig = eigen_summary(G)
    k_eig = kappa_from_eigs(np.maximum(np.linalg.eigvalsh(G), 0.0))
    return {
        "kappa": k,
        "kappa_eig": k_eig,
        "min_sep": min_sep,
        "coherence": mu,
        "eig_min_raw": eig["min_raw"],
        "eig_min_clamped": eig["min_clamped"],
        "eig_max": eig["max"],
        "eig_mean": eig["mean"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare conditioning across node sets")
    ap.add_argument("--sizes", default="5,10,20,32,64,128,256",
                    help="Comma-separated N values")
    ap.add_argument("--trials", type=int, default=5,
                    help="Trials for random/jittered grids")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed")
    ap.add_argument("--output", default="data/artifacts/conditioning/phi_grid_metrics.json",
                    help="Output JSON path")
    args = ap.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)

    grids = [
        ("equispaced", lambda n: grid_equispaced(n)),
        ("phi_modulated", lambda n: grid_phi_modulated(n)),
        ("phi_inverse_modulated", lambda n: grid_phi_inverse(n)),
        ("sqrt2_modulated", lambda n: grid_irrational(n, np.sqrt(2))),
        ("e_modulated", lambda n: grid_irrational(n, np.e)),
        ("pi_modulated", lambda n: grid_irrational(n, np.pi)),
    ]

    results: List[Dict[str, float]] = []

    for n in sizes:
        for name, fn in grids:
            freqs = fn(n)
            metrics = compute_metrics(freqs, n)
            results.append({
                "grid": name,
                "N": n,
                **metrics,
            })

        # Random and jittered: average over trials
        for name, gen in ("random", grid_random), ("jittered", grid_jittered):
            trial_metrics = []
            for _ in range(args.trials):
                freqs = gen(n, rng)
                trial_metrics.append(compute_metrics(freqs, n))
            avg = {
                "kappa": float(np.mean([m["kappa"] for m in trial_metrics])),
                "kappa_eig": float(np.mean([m["kappa_eig"] for m in trial_metrics])),
                "min_sep": float(np.mean([m["min_sep"] for m in trial_metrics])),
                "coherence": float(np.mean([m["coherence"] for m in trial_metrics])),
                "eig_min_raw": float(np.mean([m["eig_min_raw"] for m in trial_metrics])),
                "eig_min_clamped": float(np.mean([m["eig_min_clamped"] for m in trial_metrics])),
                "eig_max": float(np.mean([m["eig_max"] for m in trial_metrics])),
                "eig_mean": float(np.mean([m["eig_mean"] for m in trial_metrics])),
            }
            results.append({
                "grid": f"{name}_avg",
                "N": n,
                "trials": args.trials,
                **avg,
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "canonical_grid": "f_k = frac((k+1) * φ)",
        "sizes": sizes,
        "trials": args.trials,
        "seed": args.seed,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✓ Wrote metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
