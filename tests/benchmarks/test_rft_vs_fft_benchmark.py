#!/usr/bin/env python3
import os
import csv
import subprocess
import sys


def test_rft_vs_fft_benchmark_runs(tmp_path):
    out_csv = tmp_path / "rft_vs_fft.csv"
    cmd = [
        sys.executable,
        os.path.join(os.getcwd(), "tools", "benchmarking", "rft_vs_fft_benchmark.py"),
        "--sizes",
        "16,32",
        "--out",
        str(out_csv),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"benchmark failed: {res.stderr}\n{res.stdout}"
    assert out_csv.exists(), "CSV not created"

    # Validate header and a few rows
    with out_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        assert cols == [
            "size",
            "signal",
            "metric",
            "rft_value",
            "fft_value",
            "better",
            "rft_impl",
        ]
        rows = list(r)
        assert len(rows) >= 8  # at least a couple of metrics across signals
        assert any(row["metric"] == "frobenius_U_minus_DFT" for row in rows)
