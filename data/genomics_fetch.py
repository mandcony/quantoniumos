#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Fetch small public-domain/CC0/CC BY genomics/proteomics samples.

Downloads:
- Lambda phage reference genome (NCBI RefSeq, ~48kb, public domain)
- Crambin protein structure (PDB ID 1CRN, CC BY 4.0)

Files are stored under data/genomics/.

Usage:
    USE_REAL_DATA=1 python data/genomics_fetch.py
"""

import os
import sys
import gzip
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent / "genomics"

# Lambda phage genome - use nucleotide direct link
LAMBDA_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001416.1&rettype=fasta&retmode=text"
LAMBDA_DEST = ROOT / "lambda_phage.fasta"

# Crambin structure
PDB_URL = "https://files.rcsb.org/download/1CRN.pdb"
PDB_DEST = ROOT / "1crn.pdb"


def ensure_env():
    if os.environ.get("USE_REAL_DATA") != "1":
        print("USE_REAL_DATA not set to 1; aborting fetch to avoid accidental downloads.")
        sys.exit(1)


def download(url: str, dest: pathlib.Path, decompress: bool = False):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✓ Exists {dest}")
        return
    print(f"↓ Fetching {url}")
    try:
        tmp = str(dest) + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        if decompress:
            with gzip.open(tmp, "rb") as f_in:
                with open(dest, "wb") as f_out:
                    f_out.write(f_in.read())
            os.remove(tmp)
        else:
            os.rename(tmp, dest)
        print(f"✓ Saved {dest}")
    except Exception as exc:
        print(f"✗ Failed: {exc}")
        raise RuntimeError(f"Failed to fetch {url}") from exc


def main():
    ensure_env()
    print("Fetching public genomics datasets...")
    download(LAMBDA_URL, LAMBDA_DEST)
    download(PDB_URL, PDB_DEST)
    print(f"\nDone. Files stored under {ROOT}")
    print(f"  - Lambda phage: {LAMBDA_DEST}")
    print(f"  - PDB 1CRN: {PDB_DEST}")


if __name__ == "__main__":
    main()
