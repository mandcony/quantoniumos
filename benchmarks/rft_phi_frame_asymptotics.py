#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""φ-grid exponential basis: asymptotic orthogonality diagnostics.

Computes (for increasing N):
- Mutual coherence: max_{k!=l} |<φ_k, φ_l>| for the raw φ-grid exponential basis.
- Average cross-energy: mean_{k!=l} |<φ_k, φ_l>|^2.

This uses a closed-form inner-product for complex exponentials, avoiding
building Φ explicitly (and avoiding O(N^3) Gram normalization).

Notes:
- FFT/DFT basis is orthonormal by construction, so its mutual coherence is 0
  (up to floating point error); we report the analytical baseline.
- Frequencies are folded into [0,1) in cycles/sample, matching the repo's
  square-kernel φ-grid convention.

Usage:
  python benchmarks/rft_phi_frame_asymptotics.py --sizes 256,512,1024,2048,4096 --out results/patent_benchmarks/phi_frame_asymptotics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0


def phi_frequencies(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return np.mod((k + 1.0) * PHI, 1.0)


def exp_inner_product_magnitude(n: int, delta_f: np.ndarray) -> np.ndarray:
    """|<v(f), v(f+delta_f)>| for unit-norm length-n exponentials.

    v_f[t] = exp(j 2π f t) / sqrt(n)

    <v_f, v_g> = (1/n) sum_{t=0}^{n-1} exp(j2π (g-f) t)
               = (1/n) * (1 - exp(j2π n Δf)) / (1 - exp(j2π Δf))

    Use stable magnitude form:
      |<.|.| = |sin(π n Δf)| / (n |sin(π Δf)|)
    with the limiting value 1 when Δf -> 0.
    """

    delta_f = np.asarray(delta_f, dtype=np.float64)
    num = np.abs(np.sin(np.pi * n * delta_f))
    den = n * np.abs(np.sin(np.pi * delta_f))

    out = np.empty_like(delta_f)
    eps = 1e-15
    mask = den < eps
    out[mask] = 1.0
    out[~mask] = num[~mask] / den[~mask]
    return out


@dataclass
class AsymptoticRow:
    size: int
    mu_phi: float
    mean_offdiag_abs2_phi: float
    fft_mu: float
    fft_mean_offdiag_abs2: float


def compute_stats(n: int) -> AsymptoticRow:
    f = phi_frequencies(n)

    # Pairwise frequency differences Δf in (-1,1).
    # Use broadcasting; n=4096 => 16M floats, fine.
    df = f[None, :] - f[:, None]

    mag = exp_inner_product_magnitude(n, df)

    # Exclude diagonal.
    np.fill_diagonal(mag, 0.0)

    mu_phi = float(np.max(mag))
    mean_offdiag_abs2_phi = float(np.mean(mag**2))

    # FFT basis is orthonormal: exactly zero off-diagonals.
    return AsymptoticRow(
        size=int(n),
        mu_phi=mu_phi,
        mean_offdiag_abs2_phi=mean_offdiag_abs2_phi,
        fft_mu=0.0,
        fft_mean_offdiag_abs2=0.0,
    )


def write_csv(path: Path, rows: List[AsymptoticRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "size",
                "mu_phi",
                "mean_offdiag_abs2_phi",
                "fft_mu",
                "fft_mean_offdiag_abs2",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=str, default="256,512,1024,2048,4096")
    ap.add_argument(
        "--out",
        type=str,
        default="results/patent_benchmarks/phi_frame_asymptotics.csv",
    )
    args = ap.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]

    rows: List[AsymptoticRow] = []
    for n in sizes:
        if n <= 1:
            continue
        print(f"N={n} ...")
        row = compute_stats(n)
        rows.append(row)
        print(
            f"  mu_phi={row.mu_phi:.3e}  mean_offdiag_abs2_phi={row.mean_offdiag_abs2_phi:.3e}"
        )

    out = Path(args.out)
    write_csv(out, rows)
    print("CSV written to:", out)


if __name__ == "__main__":
    main()
