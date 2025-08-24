#!/usr/bin/env python
"""
Crypto Engine Builder
==================
Utility script to build the Quantum Cryptography Engine.
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from 04_RFT_ALGORITHMS.paper_compliant_rft_fixed import FixedRFTCryptoBindings


def build_crypto_engine(mode="release", verbose=False, output_dir=None):
    """Build the Quantum Cryptography Engine"""
    if verbose:
        print(f"Building Quantum Cryptography Engine in {mode} mode")

    # Initialize the RFT crypto bindings
    rft_crypto = FixedRFTCryptoBindings()
    rft_crypto.init_engine()

    if verbose:
        print("RFT Crypto Bindings initialized")

    # Create output directory if specified
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Return success
    return {
        "status": "SUCCESS",
        "mode": mode,
        "rft_crypto": rft_crypto,
        "output_dir": output_dir,
    }


def main():
    """Main entry point for the build utility"""
    parser = argparse.ArgumentParser(
        description="Build the Quantum Cryptography Engine"
    )
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    mode = "debug" if args.debug else "release"
    result = build_crypto_engine(mode, args.verbose, args.output)

    if result["status"] == "SUCCESS":
        print("Crypto Engine build complete.")
        return 0
    else:
        print("Crypto Engine build failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
