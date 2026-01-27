#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""Real-world + large-N evaluation for φ-grid NUDFT vs FFT.

Goal
----
For N >= 4096, Gram normalization (dense eigendecomposition of an N×N Gram matrix)
becomes computationally impractical in pure Python/NumPy. This script therefore
focuses on *analysis-side* properties that are still meaningful and measurable:

- Coefficient sparsity proxy: k for 99% energy (k_99)
- Spectral leakage proxy: peak / L1 on an off-bin tone

It compares:
- FFT coefficients (unitary FFT via numpy/scipy conventions)
- φ-grid NUDFT coefficients (non-uniform DFT at deterministic irrational frequencies)

Datasets
--------
- ECG: MIT-BIH (if present under data/physionet/mitbih and wfdb installed)
- Audio: any user-provided WAV file path (speech or music)

Usage
-----
  # ECG (requires fetch + wfdb)
  USE_REAL_DATA=1 python benchmarks/rft_phi_nudft_realdata_eval.py --ecg --N 4096

  # Audio (speech/music): provide your own WAVs
  python benchmarks/rft_phi_nudft_realdata_eval.py --wav path/to/file.wav --N 4096

Outputs a small CSV under results/patent_benchmarks/.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

PHI = (1.0 + 5.0 ** 0.5) / 2.0
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def phi_frequencies(n: int) -> np.ndarray:
    k = np.arange(n, dtype=np.float64)
    return np.mod((k + 1.0) * PHI, 1.0)


def nufft_phi_forward(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Compute y[k] = <v_k, x> where v_k[n] = exp(j2π f_k n)/sqrt(N).

    So y = (1/sqrt(N)) * sum_n x[n] * exp(-j2π f_k n)

    This is an O(N^2) NUDFT. For N=4096 it is feasible for a small number
    of windows/signals.
    """

    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    f = np.asarray(f, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)

    # E[k,n] = exp(-j2π f_k n)
    E = np.exp(-1j * 2.0 * np.pi * np.outer(f, t))
    return (E @ x) / np.sqrt(float(n))


def forward_fft_unitary(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    n = x.shape[0]
    return np.fft.fft(x) / np.sqrt(float(n))


def energy_compaction_k(coeffs: np.ndarray, frac: float = 0.99) -> int:
    mag2 = np.abs(coeffs) ** 2
    total = float(mag2.sum())
    if total <= 0:
        return 0
    idx = np.argsort(mag2)[::-1]
    csum = np.cumsum(mag2[idx])
    return int(np.searchsorted(csum, frac * total) + 1)


def leakage_peak_over_l1(coeffs: np.ndarray) -> float:
    mag = np.abs(coeffs)
    return float(mag.max() / (mag.sum() + 1e-12))


def windows_1d(x: np.ndarray, n: int, *, max_windows: int = 8, hop: Optional[int] = None) -> Iterable[np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if hop is None:
        hop = n
    count = 0
    for start in range(0, max(0, x.shape[0] - n + 1), hop):
        w = x[start : start + n]
        if w.shape[0] != n:
            break
        w = w - float(np.mean(w))
        norm = float(np.linalg.norm(w)) + 1e-12
        yield w / norm
        count += 1
        if count >= max_windows:
            break


def make_offbin_tone(n: int, freq_cycles_per_n: float = 0.37, phase: float = 0.2) -> np.ndarray:
    t = np.arange(n)
    return np.exp(1j * (2.0 * np.pi * (freq_cycles_per_n * n) * t / n + phase))


@dataclass
class Row:
    dataset: str
    N: int
    metric: str
    rft_value: float
    fft_value: float


def eval_signal_windows(name: str, windows: Iterable[np.ndarray], n: int) -> List[Row]:
    f = phi_frequencies(n)
    rows: List[Row] = []

    k_rft_all: List[int] = []
    k_fft_all: List[int] = []

    for w in windows:
        y_phi = nufft_phi_forward(w.astype(np.complex128), f)
        y_fft = forward_fft_unitary(w.astype(np.complex128))
        k_rft_all.append(energy_compaction_k(y_phi, 0.99))
        k_fft_all.append(energy_compaction_k(y_fft, 0.99))

    if k_rft_all:
        rows.append(Row(name, n, "k_99_energy_mean", float(np.mean(k_rft_all)), float(np.mean(k_fft_all))))
        rows.append(Row(name, n, "k_99_energy_median", float(np.median(k_rft_all)), float(np.median(k_fft_all))))

    # Leakage on canonical off-bin tone (synthetic, but leakage is the point)
    tone = make_offbin_tone(n)
    tone = tone / (np.linalg.norm(tone) + 1e-12)
    y_phi_t = nufft_phi_forward(tone, f)
    y_fft_t = forward_fft_unitary(tone)
    rows.append(Row(name, n, "leakage_peak_over_l1_offbin_tone", leakage_peak_over_l1(y_phi_t), leakage_peak_over_l1(y_fft_t)))

    return rows


def load_ecg_mitbih() -> List[Tuple[str, np.ndarray]]:
    if os.environ.get("USE_REAL_DATA") != "1":
        raise RuntimeError("Set USE_REAL_DATA=1 to run real-data evaluation.")

    mitbih_dir = DATA_DIR / "physionet" / "mitbih"
    if not mitbih_dir.exists():
        raise FileNotFoundError("MIT-BIH not found. Run: USE_REAL_DATA=1 python data/physionet_mitbih_fetch.py")

    try:
        import wfdb  # type: ignore
    except ImportError as e:
        raise RuntimeError("wfdb not installed. Run: pip install wfdb") from e

    out: List[Tuple[str, np.ndarray]] = []
    for rec_id in ["100", "101"]:
        if not (mitbih_dir / f"{rec_id}.dat").exists():
            continue
        record = wfdb.rdrecord(str(mitbih_dir / rec_id))
        sig = record.p_signal[:, 0].astype(np.float64)
        out.append((f"MIT-BIH ECG {rec_id}", sig))
    if not out:
        raise FileNotFoundError("MIT-BIH records not present after fetch.")
    return out


def load_eeg_sleepedf() -> List[Tuple[str, np.ndarray]]:
    if os.environ.get("USE_REAL_DATA") != "1":
        raise RuntimeError("Set USE_REAL_DATA=1 to run real-data evaluation.")

    sleepedf_dir = DATA_DIR / "physionet" / "sleepedf"
    if not sleepedf_dir.exists():
        raise FileNotFoundError(
            "Sleep-EDF not found. Run: USE_REAL_DATA=1 python data/physionet_sleepedf_fetch.py"
        )

    try:
        import pyedflib  # type: ignore
    except ImportError as e:
        raise RuntimeError("pyedflib not installed. Run: pip install pyedflib") from e

    out: List[Tuple[str, np.ndarray]] = []
    for edf_path in sorted(sleepedf_dir.glob("*-PSG.edf")):
        f = pyedflib.EdfReader(str(edf_path))
        labels = [f.getLabel(i) for i in range(f.signals_in_file)]
        eeg_idx = labels.index("EEG Fpz-Cz") if "EEG Fpz-Cz" in labels else 0
        eeg = f.readSignal(eeg_idx).astype(np.float64)
        fs = f.getSampleFrequency(eeg_idx)
        f.close()
        out.append((f"Sleep-EDF EEG({edf_path.name})@{fs}Hz", eeg))
    if not out:
        raise FileNotFoundError("No *-PSG.edf files found after fetch.")
    return out


def load_wav(path: Path) -> Tuple[str, np.ndarray]:
    try:
        from scipy.io import wavfile  # type: ignore
    except Exception as e:
        raise RuntimeError("scipy is required to read WAV. Ensure scipy is installed.") from e

    sr, x = wavfile.read(str(path))
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[:, 0]
    # Convert integer PCM to float.
    if np.issubdtype(x.dtype, np.integer):
        maxv = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)
    return (f"WAV({path.name})@{sr}Hz", x)


def write_csv(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "N", "metric", "rft_value", "fft_value"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--max-windows", type=int, default=6)
    ap.add_argument("--hop", type=int, default=None)
    ap.add_argument("--ecg", action="store_true", help="Evaluate MIT-BIH ECG (requires USE_REAL_DATA=1, wfdb, and fetched data)")
    ap.add_argument("--eeg", action="store_true", help="Evaluate Sleep-EDF EEG (requires USE_REAL_DATA=1, pyedflib, and fetched data)")
    ap.add_argument("--wav", type=str, default=None, help="Path to a WAV file (speech or music) to evaluate")
    ap.add_argument(
        "--out",
        type=str,
        default="results/patent_benchmarks/phi_nudft_realdata_eval.csv",
    )
    args = ap.parse_args()

    n = int(args.N)
    rows: List[Row] = []

    if args.ecg:
        for name, sig in load_ecg_mitbih():
            rows.extend(
                eval_signal_windows(
                    name,
                    windows_1d(sig, n, max_windows=int(args.max_windows), hop=args.hop),
                    n,
                )
            )

    if args.eeg:
        for name, sig in load_eeg_sleepedf():
            rows.extend(
                eval_signal_windows(
                    name,
                    windows_1d(sig, n, max_windows=int(args.max_windows), hop=args.hop),
                    n,
                )
            )

    if args.wav:
        name, sig = load_wav(Path(args.wav))
        rows.extend(
            eval_signal_windows(
                name,
                windows_1d(sig, n, max_windows=int(args.max_windows), hop=args.hop),
                n,
            )
        )

    if not rows:
        raise SystemExit("No datasets selected. Use --ecg and/or --wav PATH.")

    out = Path(args.out)
    write_csv(out, rows)

    print("Wrote:", out)
    for r in rows:
        print(f"{r.dataset:>24}  {r.metric:<34}  phi={r.rft_value:.3g}  fft={r.fft_value:.3g}")


if __name__ == "__main__":
    main()
