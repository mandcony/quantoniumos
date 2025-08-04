#!/usr/bin/env python3
"""
QuantoniumOS Result Comparator

This script compares two QuantoniumOS verification results.

Usage:
    python compare_results.py --result1=RESULT_FILE1 --result2=RESULT_FILE2 [--tolerance=TOLERANCE]
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
    VerificationProtocol
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compare QuantoniumOS results')
    parser.add_argument('--result1', required=True, help='Path to first result file')
    parser.add_argument('--result2', required=True, help='Path to second result file')
    parser.add_argument('--tolerance', type=float, default=1e-6, 
                      help='Numerical tolerance for floating-point comparison')
    args = parser.parse_args()
    
    # Create verification protocol
    protocol = VerificationProtocol(
        docker_image="quantoniumos/verification:latest",
        output_dir="verification_results"
    )
    
    # Compare results
    comparison = protocol.compare_results(
        result1_path=args.result1,
        result2_path=args.result2,
        tolerance=args.tolerance
    )
    
    # Print comparison results
    if 'error' in comparison:
        print(f"Error: {comparison['error']}")
        return
    
    print("Results Comparison:")
    print(f"Overall Match: {'PASS' if comparison['match'] else 'FAIL'}")
    print(f"Parameters Match: {'Yes' if comparison['parameters_match'] else 'No'}")
    
    if comparison['differences']:
        print("\nDifferences:")
        for key, diff in comparison['differences'].items():
            print(f"  {key}:")
            if diff['status'] == 'missing_in_result1':
                print(f"    Missing in result1, value in result2: {diff['value2']}")
            elif diff['status'] == 'missing_in_result2':
                print(f"    Missing in result2, value in result1: {diff['value1']}")
            else:
                print(f"    Value in result1: {diff['value1']}")
                print(f"    Value in result2: {diff['value2']}")
                if 'difference' in diff:
                    print(f"    Absolute difference: {diff['difference']}")
    else:
        print("\nNo differences found between results.")
    
    # Compare environments
    env1 = comparison['environment']['result1']
    env2 = comparison['environment']['result2']
    
    print("\nEnvironment Comparison:")
    print(f"  Result1 Docker image: {env1.get('docker_image', 'N/A')}")
    print(f"  Result2 Docker image: {env2.get('docker_image', 'N/A')}")
    print(f"  Result1 Git commit: {env1.get('commit_hash', 'N/A')}")
    print(f"  Result2 Git commit: {env2.get('commit_hash', 'N/A')}")

if __name__ == "__main__":
    main()
