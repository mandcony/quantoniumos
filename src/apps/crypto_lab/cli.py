#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-Quantonium-NC
"""
Crypto Lab CLI - Experimental wave-crypto/memory playground
Part of QuantoniumOS Wavespace Workspace

WARNING: This module is for RESEARCH PURPOSES ONLY.
Do NOT use for production cryptography.
"""

import sys
import os

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def main() -> None:
    """Main entry point for Crypto Lab."""
    print("=" * 60)
    print("  Crypto Lab: Experimental wave-crypto/memory playground")
    print("=" * 60)
    print()
    print("  ⚠️  WARNING: RESEARCH ONLY - NOT FOR PRODUCTION USE")
    print()
    print("  Status: Loaded (stub v1)")
    print()
    print("  Features (coming soon):")
    print("    • Wave-domain key mixing")
    print("    • RFT-based hashing experiments")
    print("    • Holographic memory encoding/decoding")
    print("    • Associative wave memory probes")
    print()
    print("  Run with: python -m src.apps.crypto_lab.cli")
    print("=" * 60)


if __name__ == "__main__":
    main()
