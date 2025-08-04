#!/usr/bin/env python3
"""
QuantoniumOS Result Verifier

This script verifies the integrity and signature of QuantoniumOS verification results.

Usage:
    python verify_result.py --result=RESULT_FILE --public-key=PUBLIC_KEY_FILE
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.verification.third_party_replication import (
    ExperimentalResult, 
    ResultSignature
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Verify QuantoniumOS result integrity')
    parser.add_argument('--result', required=True, help='Path to result file')
    parser.add_argument('--public-key', required=True, help='Path to public key file')
    args = parser.parse_args()
    
    # Load result
    try:
        result = ExperimentalResult.load(args.result)
    except Exception as e:
        print(f"Error loading result: {e}")
        return
    
    # Load public key
    try:
        with open(args.public_key, 'rb') as f:
            public_key_pem = f.read()
    except Exception as e:
        print(f"Error loading public key: {e}")
        return
    
    # Verify hash
    print("Verifying result hash...")
    if result.verify_hash():
        print("Hash verification: PASSED")
    else:
        print("Hash verification: FAILED")
        return
    
    # Verify signature if available
    if result.signature:
        print("Verifying result signature...")
        
        signer = ResultSignature()
        signer.load_public_key(public_key_pem)
        
        if signer.verify_signature(result):
            print("Signature verification: PASSED")
        else:
            print("Signature verification: FAILED")
    else:
        print("No signature found in result")
    
    # Print result summary
    print("\nResult Summary:")
    print(f"Experiment: {result.experiment_name}")
    print(f"Timestamp: {result.timestamp}")
    
    if 'results' in result.results:
        for category, category_results in result.results['results'].items():
            print(f"\n{category.capitalize()} Results:")
            for key, value in category_results.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
