#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
#
# MEDICAL RESEARCH LICENSE:
# FREE for hospitals, medical researchers, academics, and healthcare
# institutions for testing, validation, and research purposes.
# Commercial medical device use: See LICENSE-CLAIMS-NC.md
#
"""
Real Data Fixtures for Medical Tests
====================================

Provides pytest fixtures and skip decorators for optional real-dataset tests.
Tests skip cleanly when data is absent; no accidental downloads.

Datasets:
- MIT-BIH Arrhythmia (ECG) — PhysioNet DUA required
- Sleep-EDF (EEG) — PhysioNet terms required
- FastMRI (MRI) — CC BY-NC 4.0 (non-commercial)
- Genomics public: lambda phage FASTA, PDB 1CRN

Usage:
    Set USE_REAL_DATA=1 and run fetch scripts before enabling real-data tests.
"""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import pytest
import numpy as np

# -----------------------------------------------------------------------------
# Environment and Paths
# -----------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

# Check environment flag
USE_REAL_DATA = os.environ.get("USE_REAL_DATA", "0") == "1"

# Dataset paths (relative to DATA_ROOT)
MITBIH_DIR = DATA_ROOT / "physionet" / "mitbih"
SLEEPEDF_DIR = DATA_ROOT / "physionet" / "sleepedf"
FASTMRI_DIR = DATA_ROOT / "fastmri" / "knee_singlecoil_subset"
GENOMICS_DIR = DATA_ROOT / "genomics"

# -----------------------------------------------------------------------------
# Skip Decorators
# -----------------------------------------------------------------------------


def _dataset_available(path: Path) -> bool:
    """Check if dataset directory exists and contains files."""
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


# MIT-BIH Arrhythmia
MITBIH_AVAILABLE = USE_REAL_DATA and _dataset_available(MITBIH_DIR)
skip_no_mitbih = pytest.mark.skipif(
    not MITBIH_AVAILABLE,
    reason=(
        "MIT-BIH data not available. Set USE_REAL_DATA=1 and run "
        "python data/physionet_mitbih_fetch.py after accepting PhysioNet DUA."
    ),
)

# Sleep-EDF
SLEEPEDF_AVAILABLE = USE_REAL_DATA and _dataset_available(SLEEPEDF_DIR)
skip_no_sleepedf = pytest.mark.skipif(
    not SLEEPEDF_AVAILABLE,
    reason=(
        "Sleep-EDF data not available. Set USE_REAL_DATA=1 and run "
        "python data/physionet_sleepedf_fetch.py after accepting PhysioNet terms."
    ),
)

# FastMRI
FASTMRI_AVAILABLE = USE_REAL_DATA and _dataset_available(FASTMRI_DIR)
skip_no_fastmri = pytest.mark.skipif(
    not FASTMRI_AVAILABLE,
    reason=(
        "FastMRI data not available. Set USE_REAL_DATA=1 and "
        "FASTMRI_KNEE_URL=<signed_url>, then run bash data/fastmri_fetch.sh. "
        "Requires CC BY-NC 4.0 acceptance at https://fastmri.org/"
    ),
)

# Genomics public sets
LAMBDA_FASTA = GENOMICS_DIR / "lambda_phage.fasta"
PDB_1CRN = GENOMICS_DIR / "1crn.pdb"
GENOMICS_AVAILABLE = USE_REAL_DATA and (
    _dataset_available(LAMBDA_FASTA) or _dataset_available(PDB_1CRN)
)
skip_no_genomics = pytest.mark.skipif(
    not GENOMICS_AVAILABLE,
    reason=(
        "Genomics public datasets not available. Set USE_REAL_DATA=1 and run "
        "python data/genomics_fetch.py to download lambda phage FASTA and PDB 1CRN."
    ),
)

# Composite marker for any real data
skip_no_real_data = pytest.mark.skipif(
    not USE_REAL_DATA,
    reason="USE_REAL_DATA=1 not set. Real-data tests disabled.",
)

# -----------------------------------------------------------------------------
# Data Loaders
# -----------------------------------------------------------------------------


def list_mitbih_records() -> List[Path]:
    """List available MIT-BIH .dat record files."""
    if not MITBIH_AVAILABLE:
        return []
    return sorted(MITBIH_DIR.glob("*.dat"))


def load_mitbih_record(record_path: Path) -> Tuple[np.ndarray, int]:
    """
    Load a single MIT-BIH record as numpy array.

    Returns:
        (signal, sample_rate) — signal is 1D float array, sample_rate=360
    """
    try:
        import wfdb
    except ImportError:
        pytest.skip("wfdb package not installed; run: pip install wfdb")

    record_name = record_path.stem
    record_dir = str(record_path.parent)
    record = wfdb.rdrecord(record_name, pn_dir=None, sampfrom=0, sampto=None, channels=[0], pb_dir=record_dir)
    signal = record.p_signal.flatten().astype(np.float64)
    fs = record.fs
    return signal, int(fs)


def list_sleepedf_records() -> List[Path]:
    """List available Sleep-EDF .edf files."""
    if not SLEEPEDF_AVAILABLE:
        return []
    return sorted(SLEEPEDF_DIR.glob("*.edf"))


def load_sleepedf_record(edf_path: Path, duration_sec: float = 30.0) -> Tuple[np.ndarray, int]:
    """
    Load a snippet from Sleep-EDF EDF file.

    Args:
        edf_path: Path to .edf file
        duration_sec: Seconds to load (default 30s epoch)

    Returns:
        (signal, sample_rate)
    """
    try:
        import pyedflib
    except ImportError:
        pytest.skip("pyedflib not installed; run: pip install pyedflib")

    with pyedflib.EdfReader(str(edf_path)) as f:
        n_signals = f.signals_in_file
        if n_signals == 0:
            pytest.skip(f"No signals in {edf_path}")
        fs = int(f.getSampleFrequency(0))
        n_samples = min(int(duration_sec * fs), f.getNSamples()[0])
        signal = f.readSignal(0, n_samples).astype(np.float64)
    return signal, fs


def list_fastmri_slices() -> List[Path]:
    """List available FastMRI .h5 slice files."""
    if not FASTMRI_AVAILABLE:
        return []
    return sorted(FASTMRI_DIR.glob("*.h5"))


def load_fastmri_slice(h5_path: Path, slice_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a k-space slice from FastMRI HDF5.

    Returns:
        (kspace, target) — kspace is complex 2D, target is magnitude 2D
    """
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed; run: pip install h5py")

    with h5py.File(h5_path, "r") as f:
        kspace = f["kspace"][slice_idx]  # complex
        target = f.get("reconstruction_rss", f.get("reconstruction_esc", None))
        if target is not None:
            target = target[slice_idx]
        else:
            # Fallback: inverse FFT magnitude
            target = np.abs(np.fft.ifft2(kspace))
    return kspace.astype(np.complex128), target.astype(np.float64)


def load_lambda_fasta() -> str:
    """Load lambda phage FASTA sequence."""
    if not _dataset_available(LAMBDA_FASTA):
        pytest.skip("Lambda phage FASTA not available.")
    lines = LAMBDA_FASTA.read_text().splitlines()
    # Skip header lines starting with >
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    return "".join(seq_lines).upper()


def load_pdb_1crn() -> str:
    """Load PDB 1CRN structure file as text."""
    if not _dataset_available(PDB_1CRN):
        pytest.skip("PDB 1CRN not available.")
    return PDB_1CRN.read_text()


# -----------------------------------------------------------------------------
# Pytest Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mitbih_records() -> List[Path]:
    """Session-scoped list of MIT-BIH record paths."""
    return list_mitbih_records()


@pytest.fixture(scope="session")
def sleepedf_records() -> List[Path]:
    """Session-scoped list of Sleep-EDF EDF paths."""
    return list_sleepedf_records()


@pytest.fixture(scope="session")
def fastmri_slices() -> List[Path]:
    """Session-scoped list of FastMRI slice paths."""
    return list_fastmri_slices()


@pytest.fixture(scope="session")
def lambda_sequence() -> str:
    """Lambda phage genome sequence."""
    return load_lambda_fasta()


@pytest.fixture(scope="session")
def pdb_1crn_text() -> str:
    """PDB 1CRN structure text."""
    return load_pdb_1crn()
