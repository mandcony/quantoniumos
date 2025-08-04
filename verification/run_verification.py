#!/usr/bin/env python3
"""
QuantoniumOS Verification Runner

This script runs the QuantoniumOS verification process in a Docker container
and processes the results.

Usage:
    python run_verification.py [--test_suite=SUITE] [--iterations=N]
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.verification.third_party_replication import (
    DockerEnvironment, 
    ExperimentalResult, 
    ResultSignature, 
    VerificationProtocol
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run QuantoniumOS verification')
    parser.add_argument('--test_suite', default='all', 
                      choices=['all', 'encryption', 'hash', 'scheduler'],
                      help='Test suite to run')
    parser.add_argument('--iterations', type=int, default=100,
                      help='Number of iterations for benchmarks')
    args = parser.parse_args()
    
    # Create verification protocol
    protocol = VerificationProtocol(
        docker_image="quantoniumos/verification:latest",
        output_dir="verification_results"
    )
    
    # Generate keys for result signing
    keys_dir = os.path.join("verification", "keys")
    os.makedirs(keys_dir, exist_ok=True)
    
    private_key_path = os.path.join(keys_dir, "private_key.pem")
    public_key_path = os.path.join(keys_dir, "public_key.pem")
    
    if not os.path.exists(private_key_path) or not os.path.exists(public_key_path):
        print("Generating new key pair for result signing...")
        private_key_pem, public_key_pem = protocol.result_signer.generate_key_pair()
        
        with open(private_key_path, 'wb') as f:
            f.write(private_key_pem)
        
        with open(public_key_path, 'wb') as f:
            f.write(public_key_pem)
    else:
        print("Loading existing key pair...")
        with open(private_key_path, 'rb') as f:
            private_key_pem = f.read()
        
        with open(public_key_path, 'rb') as f:
            public_key_pem = f.read()
        
        protocol.result_signer.load_private_key(private_key_pem)
        protocol.result_signer.load_public_key(public_key_pem)
    
    # Prepare Docker environment
    print("Preparing Docker environment...")
    dockerfile_path = os.path.join("docker", "Dockerfile")
    protocol.prepare_environment(dockerfile_path)
    
    # Run verification
    print(f"Running verification with test suite: {args.test_suite}, iterations: {args.iterations}...")
    
    # Set up volumes for Docker
    volumes = {
        os.path.abspath("verification_results"): {
            'bind': '/app/results',
            'mode': 'rw'
        }
    }
    
    # Run verification
    result = protocol.run_verification(
        experiment_name=f"quantoniumos_verification_{args.test_suite}",
        verification_script="verify_quantoniumos.py",
        parameters={
            "test_suite": args.test_suite,
            "iterations": args.iterations
        },
        volumes=volumes
    )
    
    if result:
        print("Verification completed successfully.")
        print(f"Results saved to {protocol.output_dir}")
    else:
        print("Verification failed.")

if __name__ == "__main__":
    main()
