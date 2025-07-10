#!/usr/bin/env python3
"""
Quantonium OS - Command Line Interface

A simple CLI tool to interact with the Quantonium OS Cloud Runtime API.
"""

import argparse
import json
import os
import sys

import requests

# Base configuration
BASE_URL = "http://localhost:5000"
DEFAULT_API_KEY = "default_dev_key"

# Get API key from environment or use default
API_KEY = os.environ.get("QUANTONIUM_API_KEY", DEFAULT_API_KEY)

# Headers for all requests
headers = {"Content-Type": "application/json", "X-API-Key": API_KEY}


def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))


def get_status():
    """Check the API status"""
    response = requests.get(f"{BASE_URL}/", headers=headers)
    return response.json()


def encrypt_data(plaintext, key):
    """Encrypt data using the symbolic encryption endpoint"""
    payload = {"plaintext": plaintext, "key": key}
    response = requests.post(f"{BASE_URL}/encrypt", headers=headers, json=payload)
    return response.json()


def analyze_waveform(waveform_str):
    """Perform RFT on a waveform"""
    # Convert string of comma-separated values to list of floats
    waveform = [float(x) for x in waveform_str.split(",")]

    payload = {"waveform": waveform}
    response = requests.post(f"{BASE_URL}/simulate/rft", headers=headers, json=payload)
    return response.json()


def get_entropy(amount):
    """Generate quantum-inspired entropy"""
    payload = {"amount": amount}
    response = requests.post(
        f"{BASE_URL}/entropy/sample", headers=headers, json=payload
    )
    return response.json()


def unlock_container(waveform_str, hash_val):
    """Attempt to unlock a symbolic container"""
    # Convert string of comma-separated values to list of floats
    waveform = [float(x) for x in waveform_str.split(",")]

    payload = {"waveform": waveform, "hash": hash_val}
    response = requests.post(
        f"{BASE_URL}/container/unlock", headers=headers, json=payload
    )
    return response.json()


def main():
    # Create main parser
    parser = argparse.ArgumentParser(description="Quantonium OS Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    status_parser = subparsers.add_parser("status", help="Check API status")

    # Encrypt command
    encrypt_parser = subparsers.add_parser("encrypt", help="Encrypt data")
    encrypt_parser.add_argument("plaintext", help="Text to encrypt")
    encrypt_parser.add_argument("key", help="Encryption key")

    # RFT command
    rft_parser = subparsers.add_parser(
        "rft", help="Perform Resonance Fourier Transform"
    )
    rft_parser.add_argument(
        "waveform", help="Comma-separated waveform values (e.g. '0.1,0.5,0.9')"
    )

    # Entropy command
    entropy_parser = subparsers.add_parser("entropy", help="Generate entropy")
    entropy_parser.add_argument(
        "--amount", type=int, default=32, help="Amount of entropy to generate (1-1024)"
    )

    # Container command
    container_parser = subparsers.add_parser(
        "unlock", help="Unlock a symbolic container"
    )
    container_parser.add_argument(
        "waveform", help="Comma-separated waveform values (e.g. '0.2,0.7,0.3')"
    )
    container_parser.add_argument("hash", help="Hash for verification")

    # Parse arguments
    args = parser.parse_args()

    try:
        # Execute requested command
        if args.command == "status":
            result = get_status()
            print_json(result)

        elif args.command == "encrypt":
            result = encrypt_data(args.plaintext, args.key)
            print_json(result)

        elif args.command == "rft":
            result = analyze_waveform(args.waveform)
            print_json(result)

        elif args.command == "entropy":
            result = get_entropy(args.amount)
            print_json(result)

        elif args.command == "unlock":
            result = unlock_container(args.waveform, args.hash)
            print_json(result)

        else:
            parser.print_help()

    except requests.exceptions.RequestException as e:
        print(f"ERROR: API request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
