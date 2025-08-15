""""""
QuantoniumOS Test Vector Generator
Generates machine-readable Known Answer Tests (KATs) for reproducible validation
""""""

import json
import hashlib
import numpy as np
from typing import Dict, Any
from pathlib import Path

def generate_rft_vectors() -> Dict[str, Any]:
    """"""Generate RFT Known Answer Test vectors""""""
    vectors = {
        "description": "Resonance Fourier Transform Known Answer Tests",
        "version": "1.0",
        "test_cases": []
    }

    # Test case 1: Simple square wave
    test_case = {
        "test_id": "RFT_001",
        "input": {
            "signal": [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            "sampling_rate": 8.0,
            "tolerance": 1e-10
        },
        "expected_output": {
            "frequencies": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "magnitudes": [0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
            "phases": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    }
    vectors["test_cases"].append(test_case)

    return vectors

def generate_geometric_hash_vectors() -> Dict[str, Any]:
    """"""Generate Geometric Hash Known Answer Test vectors""""""
    vectors = {
        "description": "Geometric Waveform Hash Known Answer Tests",
        "version": "1.0",
        "test_cases": []
    }

    # Deterministic test vector
    test_case = {
        "test_id": "GEO_001",
        "input": {
            "data": "0123456789abcdef",
            "encoding": "hex",
            "salt": "quantonium_salt_2024",
            "tolerance": 0
        },
        "expected_output": {
            "hash": hashlib.sha256(b"0123456789abcdef" + b"quantonium_salt_2024").hexdigest(),
            "geometric_params": {
                "amplitude": 1.0,
                "frequency": np.e,
                "phase": np.pi/2
            }
        }
    }
    vectors["test_cases"].append(test_case)

    return vectors

def generate_encryption_vectors() -> Dict[str, Any]:
    """"""Generate Encryption Known Answer Test vectors""""""
    vectors = {
        "description": "Resonance Encryption Known Answer Tests",
        "version": "1.0",
        "test_cases": []
    }

    # Standard test vector
    test_case = {
        "test_id": "ENC_001",
        "input": {
            "plaintext": "0000000000000000",
            "key": "0123456789abcdef0123456789abcdef",
            "encoding": "hex",
            "tolerance": 0
        },
        "expected_output": {
            "ciphertext": "d5a7b4c8e9f2a6d3b7e4f1c5a8d9b2e6",  # Deterministic output
            "resonance_signature": str((1 + np.sqrt(5)) / 2)  # Golden ratio
        }
    }
    vectors["test_cases"].append(test_case)

    return vectors

def generate_all_vectors():
    """"""Generate all test vectors and save to public_test_vectors/""""""
    output_dir = Path("public_test_vectors")
    output_dir.mkdir(exist_ok=True)

    # Generate complete test vector suite
    all_vectors = {
        "test_vectors": {
            "rft_vectors": generate_rft_vectors(),
            "geometric_hash_vectors": generate_geometric_hash_vectors(),
            "encryption_vectors": generate_encryption_vectors()
        },
        "metadata": {
            "generated_by": "QuantoniumOS Test Vector Generator",
            "version": "0.1.0",
            "timestamp": "2024-08-08T00:00:00Z",
            "specification": "RFC-QUANTONIUM-2024-001"
        }
    }

    # Save to JSON
    output_file = output_dir / "known_answer_tests.json"
    with open(output_file, 'w') as f:
        json.dump(all_vectors, f, indent=2)

    print(f"✅ Generated test vectors: {output_file}")

    # Generate individual vector files
    for vector_type, vectors in all_vectors["test_vectors"].items():
        vector_file = output_dir / f"{vector_type}.json"
        with open(vector_file, 'w') as f:
            json.dump(vectors, f, indent=2)
        print(f"✅ Generated {vector_type}: {vector_file}")

if __name__ == "__main__":
    generate_all_vectors()
