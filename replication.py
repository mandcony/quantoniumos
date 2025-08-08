#!/usr/bin/env python3
"""
QuantoniumOS Replication Framework Main Runner

This script provides a unified interface to run all replication-related
tasks from a single command-line tool.

Usage:
    python replication.py [command] [options]

Commands:
    build       Build the Docker verification image
    verify      Run the verification process
    compare     Compare two verification results
    report      Generate a verification report
    all         Run the full end-to-end demo
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir))
# Replaced sys.path manipulation with proper namespace import
from quantoniumos.core import *

# Import core modules
try:
    from core.verification.third_party_replication import (
        DockerEnvironment, 
        ExperimentalResult, 
        ResultSignature, 
        VerificationProtocol
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def build_docker_image(args):
    """Build the Docker verification image"""
    
    print("Building Docker verification image...")
    
    # Check if Docker is available
    try:
        docker_env = DockerEnvironment()
        docker_env.client.info()
    except Exception as e:
        print(f"Docker is not available: {e}")
        return False
    
    # Build Docker image
    dockerfile_path = os.path.join("docker", "Dockerfile")
    if not os.path.exists(dockerfile_path):
        print(f"Dockerfile not found at {dockerfile_path}")
        return False
    
    image = docker_env.build_image(dockerfile_path, "quantoniumos/verification:latest")
    
    if not image:
        print("Failed to build Docker image")
        return False
    
    print("Docker image built successfully: quantoniumos/verification:latest")
    return True

def run_verification(args):
    """Run the verification process"""
    
    print("Running verification process...")
    
    # Create verification protocol
    protocol = VerificationProtocol(
        docker_image="quantoniumos/verification:latest",
        output_dir="verification_results"
    )
    
    # Generate keys for result signing if they don't exist
    keys_dir = os.path.join("verification", "keys")
    os.makedirs(keys_dir, exist_ok=True)
    
    private_key_path = os.path.join(keys_dir, "private_key.pem")
    public_key_path = os.path.join(keys_dir, "public_key.pem")
    
    if not os.path.exists(private_key_path) or not os.path.exists(public_key_path):
        print("Generating new key pair...")
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
    
    # Set up volumes for Docker
    volumes = {
        os.path.abspath("verification_results"): {
            'bind': '/app/results',
            'mode': 'rw'
        }
    }
    
    # Run verification
    test_suite = args.test_suite if hasattr(args, 'test_suite') else "all"
    iterations = args.iterations if hasattr(args, 'iterations') else 100
    
    print(f"Running verification with test suite: {test_suite}, iterations: {iterations}...")
    
    result = protocol.run_verification(
        experiment_name=f"quantoniumos_verification_{test_suite}",
        verification_script="verify_quantoniumos.py",
        parameters={
            "test_suite": test_suite,
            "iterations": iterations
        },
        volumes=volumes
    )
    
    if result:
        print("Verification completed successfully.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(
            "verification_results", 
            f"verification_{test_suite}_{timestamp}.json"
        )
        result.save(result_path)
        print(f"Results saved to {result_path}")
    else:
        print("Verification failed.")
    
    return result is not None

def compare_results(args):
    """Compare two verification results"""
    
    if not args.result1 or not args.result2:
        print("Error: Both --result1 and --result2 are required")
        return False
    
    print(f"Comparing results: {args.result1} and {args.result2}...")
    
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
        return False
    
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
    
    # Save comparison to file
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            "verification_results", 
            f"comparison_{timestamp}.json"
        )
    
    with open(output_path, 'w') as f:
        import json
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison saved to {output_path}")
    
    return True

def generate_report(args):
    """Generate a verification report"""
    
    if not args.result:
        print("Error: --result is required")
        return False
    
    print(f"Generating report for result: {args.result}...")
    
    # Load result
    try:
        result = ExperimentalResult.load(args.result)
    except Exception as e:
        print(f"Error loading result: {e}")
        return False
    
    # Create report directory if it doesn't exist
    report_dir = os.path.join("verification_results", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create report filename
    if args.output:
        report_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"report_{timestamp}.md")
    
    # Generate report content
    report_content = f"""# QuantoniumOS Verification Report

## Overview

- **Report Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Experiment:** {result.experiment_name}
- **Timestamp:** {result.timestamp}
- **Hash:** {result.hash}
- **Signature:** {result.signature[:20]}... if result.signature else "Not signed"

## Environment

- **Docker Image:** {result.environment.get('docker_image', 'N/A')}
- **Commit Hash:** {result.environment.get('commit_hash', 'N/A')}

## Results

"""
    
    # Add results based on structure
    if 'results' in result.results:
        # Docker verification result structure
        for category, category_results in result.results.get('results', {}).items():
            report_content += f"### {category.capitalize()}\n\n"
            for key, value in category_results.items():
                report_content += f"- **{key}:** {value}\n"
            report_content += "\n"
    else:
        # Local result structure
        for category, category_results in result.results.items():
            report_content += f"### {category.capitalize()}\n\n"
            if isinstance(category_results, dict):
                for key, value in category_results.items():
                    report_content += f"- **{key}:** {value}\n"
            else:
                report_content += f"- {category_results}\n"
            report_content += "\n"
    
    # Add parameters
    report_content += "## Parameters\n\n"
    for key, value in result.parameters.items():
        report_content += f"- **{key}:** {value}\n"
    
    # Add verification environment
    report_content += "\n## Verification Environment\n\n"
    report_content += f"- **Python Version:** {sys.version.split()[0]}\n"
    report_content += f"- **OS:** {os.name}\n"
    report_content += f"- **Date:** {datetime.now().strftime('%Y-%m-%d')}\n"
    
    # Write report to file
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report saved to {report_path}")
    
    return True

def run_all(args):
    """Run the full end-to-end demo"""
    
    print("Running full end-to-end demo...")
    
    # Run the integration demo
    try:
        subprocess.run(["python", "integration_demo.py"], check=True)
        return True
    except subprocess.CalledProcessError:
        print("Error running integration demo")
        return False

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='QuantoniumOS Replication Framework')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build the Docker verification image')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Run the verification process')
    verify_parser.add_argument('--test-suite', choices=['all', 'encryption', 'hash', 'scheduler'],
                             default='all', help='Test suite to run')
    verify_parser.add_argument('--iterations', type=int, default=100,
                             help='Number of iterations for benchmarks')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two verification results')
    compare_parser.add_argument('--result1', required=True, help='Path to first result file')
    compare_parser.add_argument('--result2', required=True, help='Path to second result file')
    compare_parser.add_argument('--tolerance', type=float, default=1e-6,
                              help='Numerical tolerance for floating-point comparison')
    compare_parser.add_argument('--output', help='Path to output comparison file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate a verification report')
    report_parser.add_argument('--result', required=True, help='Path to result file')
    report_parser.add_argument('--output', help='Path to output report file')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run the full end-to-end demo')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        success = build_docker_image(args)
    elif args.command == 'verify':
        success = run_verification(args)
    elif args.command == 'compare':
        success = compare_results(args)
    elif args.command == 'report':
        success = generate_report(args)
    elif args.command == 'all':
        success = run_all(args)
    else:
        parser.print_help()
        return
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
