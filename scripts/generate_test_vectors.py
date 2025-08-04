#!/usr/bin/env python3
"""
QuantoniumOS Test Vector Generator

This script generates standardized test vectors for QuantoniumOS cryptographic primitives
and publishes them to the public test vector directory.

These test vectors serve as a public reference for validating implementations across
different platforms and languages.
"""

import os
import sys
import shutil
from datetime import datetime

# Fix import paths - add the project root to Python's module search path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Import cryptanalysis module
from core.testing.cryptanalysis import generate_public_test_vectors

def main():
    """Generate and publish test vectors"""
    print("="*80)
    print("QuantoniumOS Test Vector Generator")
    print("="*80)
    
    # Generate test vectors
    result = generate_public_test_vectors()
    
    # Create public directory if it doesn't exist
    public_dir = os.path.join('docs', 'public_test_vectors')
    os.makedirs(public_dir, exist_ok=True)
    
    # Copy vector files to public directory
    source_dir = 'public_test_vectors'
    for filename in os.listdir(source_dir):
        if filename.endswith('.json') or filename.endswith('.html') or filename.endswith('.md'):
            src_file = os.path.join(source_dir, filename)
            dst_file = os.path.join(public_dir, filename)
            shutil.copy2(src_file, dst_file)
    
    # Create index.md file for the docs
    index_file = os.path.join(public_dir, 'index.md')
    
    with open(index_file, 'w') as f:
        f.write("# QuantoniumOS Public Test Vectors\n\n")
        f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This page provides standardized test vectors for QuantoniumOS cryptographic primitives. ")
        f.write("These vectors allow developers to validate their implementations against the reference implementation.\n\n")
        
        f.write("## Available Test Vectors\n\n")
        
        f.write("### Resonance Encryption\n\n")
        f.write("- [HTML Format](vectors_ResonanceEncryption_encryption_latest.html)\n")
        f.write("- [JSON Format](vectors_ResonanceEncryption_encryption_latest.json)\n\n")
        
        f.write("### Geometric Waveform Hash\n\n")
        f.write("- [HTML Format](vectors_GeometricWaveformHash_hash_latest.html)\n")
        f.write("- [JSON Format](vectors_GeometricWaveformHash_hash_latest.json)\n\n")
        
        f.write("## Using Test Vectors\n\n")
        f.write("To validate your implementation:\n\n")
        f.write("1. Process each input with your implementation\n")
        f.write("2. Compare your output with the expected output\n")
        f.write("3. Your implementation is correct if all outputs match\n\n")
        
        f.write("## Validation Results\n\n")
        f.write("- Encryption test vectors: ")
        f.write("PASSED" if result['encryption_validation'].passed else "FAILED")
        f.write("\n")
        
        f.write("- Hash test vectors: ")
        f.write("PASSED" if result['hash_validation'].passed else "FAILED")
        f.write("\n\n")
        
        f.write("[Detailed Test Vector Report](test_vector_summary_latest.md)\n")
    
    print("\nTest vectors successfully generated and published!")
    print(f"Public test vectors available at: {public_dir}")
    print(f"Documentation index: {index_file}")

if __name__ == "__main__":
    main()
