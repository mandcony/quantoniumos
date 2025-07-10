"""
Quantum Routes Module

This module defines the API routes for the QuantoniumOS quantum applications.
It integrates with the quantum engine adapter to provide a unified interface.
"""

import base64
import json
import logging
import os

from flask import Blueprint, Response, jsonify, request

from encryption.quantum_engine_adapter import quantum_adapter

# Create blueprint with prefix
quantum_api = Blueprint("quantum_api", __name__, url_prefix="/api/quantum")

# Set up logging
logger = logging.getLogger(__name__)


@quantum_api.route("/initialize", methods=["POST"])
def initialize_quantum_engine():
    """Initialize the quantum engine with the specified parameters."""
    try:
        data = request.json
        max_qubits = data.get("max_qubits", 150)
        connect_encryption = data.get("connect_encryption", True)

        # Initialize the quantum engine
        result = quantum_adapter.initialize(max_qubits, connect_encryption)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error initializing quantum engine: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/encrypt", methods=["POST"])
def encrypt_data():
    """Encrypt data using quantum-inspired encryption."""
    try:
        data = request.json
        plaintext = data.get("plaintext", "")
        key = data.get("key", "symbolic-key")

        # Perform encryption
        result = quantum_adapter.encrypt(plaintext, key)
        return jsonify({"success": True, "encrypted": result})
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/decrypt", methods=["POST"])
def decrypt_data():
    """Decrypt data using quantum-inspired encryption."""
    try:
        data = request.json
        ciphertext = data.get("ciphertext", "")
        key = data.get("key", "symbolic-key")

        # Perform decryption
        result = quantum_adapter.decrypt(ciphertext, key)
        return jsonify({"success": True, "decrypted": result})
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/entropy", methods=["POST"])
def generate_entropy():
    """Generate quantum-inspired entropy."""
    try:
        data = request.json
        amount = data.get("amount", 32)

        # Generate entropy
        result = quantum_adapter.generate_entropy(amount)
        return jsonify({"success": True, "entropy": result})
    except Exception as e:
        logger.error(f"Entropy generation error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/rft", methods=["POST"])
def perform_rft():
    """Apply Resonance Fourier Transform to a waveform."""
    try:
        data = request.json
        waveform = data.get("waveform", [])

        # Apply RFT
        result = quantum_adapter.apply_rft(waveform)
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.error(f"RFT error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/irft", methods=["POST"])
def perform_irft():
    """Apply Inverse Resonance Fourier Transform."""
    try:
        data = request.json
        frequency_data = data.get("frequency_data", {})

        # Apply inverse RFT
        result = quantum_adapter.apply_irft(frequency_data)
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.error(f"Inverse RFT error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/container/unlock", methods=["POST"])
def unlock_container():
    """Attempt to unlock a container using a waveform."""
    try:
        data = request.json
        waveform = data.get("waveform", [])
        container_hash = data.get("container_hash", "")
        key = data.get("key", "symbolic-key")

        # Unlock container
        result = quantum_adapter.unlock_container(waveform, container_hash, key)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Container unlock error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@quantum_api.route("/benchmark", methods=["POST"])
def run_benchmark():
    """Run a benchmark of the quantum engine capabilities."""
    try:
        data = request.json
        max_qubits = data.get("max_qubits", 150)
        run_full_benchmark = data.get("full_benchmark", False)

        # Run the benchmark using our adapter
        result = quantum_adapter.run_benchmark(max_qubits, run_full_benchmark)

        # Format results for CSV if requested
        if data.get("format") == "csv" and "perturbation_results" in result:
            perturbation_data = result.get("perturbation_results", [])

            # Generate CSV
            csv_lines = ["id,perturbation,fidelity,resonance"]
            for entry in perturbation_data:
                csv_lines.append(
                    f"{entry['id']},{entry['perturbation']:.6f},{entry['fidelity']:.6f},{entry['resonance']:.6f}"
                )

            csv_content = "\n".join(csv_lines)

            # Return as CSV
            return Response(
                csv_content,
                mimetype="text/csv",
                headers={
                    "Content-disposition": "attachment; filename=perturbation_results.csv"
                },
            )

        # Return as JSON by default
        return jsonify(result)
    except Exception as e:
        logger.error(f"Benchmark error: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
