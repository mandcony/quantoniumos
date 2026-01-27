# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_nist_sts_10gb(tmp_path: Path):
    sts_root = Path(os.getenv("STS_HOME", "/opt/sts"))
    assess_bin = Path(os.getenv("STS_ASSESS", str(sts_root / "assess")))
    if not assess_bin.exists():
        pytest.skip("NIST STS assess binary not available (set STS_HOME or STS_ASSESS)")

    bits = int(os.getenv("STS_BITS", str(10_000_000)))  # default 10M bits for CI; 10G bits nightly
    report_path = REPO_ROOT / "results" / "sts" / "finalAnalysisReport.txt"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "sts" / "run_sts.py"),
        "--bits",
        str(bits),
        "--out",
        str(report_path),
    ]
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))
    assert report_path.exists()
