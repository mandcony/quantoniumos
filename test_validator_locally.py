#!/usr/bin/env python3
"""
Script to run the test vector validator locally
"""
import os
import sys
import subprocess
import json
import tempfile

def main():
    print("QuantoniumOS Validator Local Test")
    print("---------------------------------")
    
    # Path to test vectors
    test_vector_path = os.path.join(os.getcwd(), "public_test_vectors", "vectors_ResonanceEncryption_encryption_latest.json")
    if not os.path.exists(test_vector_path):
        print(f"Error: Test vector file not found at {test_vector_path}")
        return 1
    
    print(f"Found test vectors at: {test_vector_path}")
    
    # Create temporary directory for results
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        result_path = tmp.name
    
    print("Checking for test vector validator...")
    cargo_dir = os.path.join(os.getcwd(), "resonance-core-rs")
    
    # Look for the validator in several possible locations
    possible_paths = [
        os.path.join(cargo_dir, "target", "release", "test_vector_validator.exe"),
        os.path.join(cargo_dir, "target", "release", "test_vector_validator"),
        os.path.join(cargo_dir, "target", "debug", "test_vector_validator.exe"),
        os.path.join(cargo_dir, "target", "debug", "test_vector_validator"),
        os.path.join("build", "bin", "test_vector_validator.exe"),
        os.path.join("build", "bin", "test_vector_validator")
    ]
    
    validator_path = None
    for path in possible_paths:
        if os.path.exists(path):
            validator_path = path
            break
    
    if not validator_path:
        print("Error: Could not find the test_vector_validator binary.")
        print("Please build the Rust project first with:")
        print("  cd resonance-core-rs && cargo build --release")
        return 1
    
    print(f"Found validator at: {validator_path}")
    
    # Run the validator
    print("Running validator on test vectors...")
    
    validate_result = subprocess.run(
        [validator_path, test_vector_path, result_path],
        capture_output=True,
        text=True
    )
    
    print("Validator output:")
    print("-" * 40)
    print(validate_result.stdout)
    
    if validate_result.stderr:
        print("Validator errors:")
        print("-" * 40)
        print(validate_result.stderr)
    
    print("-" * 40)
    
    # Check validation results
    if validate_result.returncode != 0:
        print("Validation failed")
        return 1
    
    print(f"Validation succeeded! Results written to {result_path}")
    
    # Load and display summary
    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        print("\nResults summary:")
        print(f"Total: {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        
    except Exception as e:
        print(f"Error reading results: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
