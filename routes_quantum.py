"""
Quantonium OS - Quantum API Routes

Direct, optimized routes for QuantoniumOS quantum apps with proper engine integration.
"""

import os
import base64
import json
import hashlib
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify, Response

from encryption.quantum_engine_adapter import quantum_adapter

# Initialize the blueprint
quantum_api = Blueprint("quantum_api", __name__, url_prefix="/api/quantum")

@quantum_api.route("/initialize", methods=["POST"])
def initialize():
    """Initialize the quantum system with desired settings."""
    try:
        data = request.json
        max_qubits = data.get("max_qubits", 150)
        connect_encryption = data.get("connect_encryption", True)
        
        # Enforcing limits (150 qubits maximum)
        max_qubits = min(max(max_qubits, 1), 150)
        
        # Generate a new engine ID
        engine_id = hashlib.md5(os.urandom(16)).hexdigest()[:16]
        
        return jsonify({
            "success": True,
            "engine_id": engine_id,
            "max_qubits": max_qubits
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Initialization failed: {str(e)}"
        }), 400

@quantum_api.route("/encrypt", methods=["POST"])
def encrypt():
    """Encrypt data using the quantum engine."""
    try:
        data = request.json
        plaintext = data.get("plaintext", "")
        key = data.get("key", "")
        
        if not plaintext or not key:
            return jsonify({
                "success": False,
                "error": "Missing plaintext or key"
            }), 400
            
        result = quantum_adapter.encrypt(plaintext, key)
        
        return jsonify({
            "success": True,
            "ciphertext": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Encryption failed: {str(e)}"
        }), 500

@quantum_api.route("/decrypt", methods=["POST"])
def decrypt():
    """Decrypt data using the quantum engine."""
    try:
        data = request.json
        ciphertext = data.get("ciphertext", "")
        key = data.get("key", "")
        
        if not ciphertext or not key:
            return jsonify({
                "success": False,
                "error": "Missing ciphertext or key"
            }), 400
            
        result = quantum_adapter.decrypt(ciphertext, key)
        
        return jsonify({
            "success": True,
            "plaintext": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Decryption failed: {str(e)}"
        }), 500

@quantum_api.route("/entropy", methods=["POST"])
def entropy():
    """Generate quantum entropy."""
    try:
        data = request.json
        amount = int(data.get("amount", 32))
        
        # Limit amount to reasonable bounds
        amount = min(max(amount, 1), 1024)
        
        result = quantum_adapter.generate_entropy(amount)
        
        return jsonify({
            "success": True,
            "entropy": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Entropy generation failed: {str(e)}"
        }), 500

@quantum_api.route("/rft", methods=["POST"])
def rft():
    """Apply the Resonance Fourier Transform to a waveform."""
    try:
        data = request.json
        waveform = data.get("waveform", [])
        
        if not waveform:
            return jsonify({
                "success": False,
                "error": "Missing waveform data"
            }), 400
            
        result = quantum_adapter.apply_rft(waveform)
        
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"RFT application failed: {str(e)}"
        }), 500

@quantum_api.route("/irft", methods=["POST"])
def irft():
    """Apply the Inverse Resonance Fourier Transform."""
    try:
        data = request.json
        frequency_data = data.get("frequency_data", {})
        
        if not frequency_data or "frequencies" not in frequency_data:
            return jsonify({
                "success": False,
                "error": "Missing or invalid frequency data"
            }), 400
            
        result = quantum_adapter.apply_irft(frequency_data)
        
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"IRFT application failed: {str(e)}"
        }), 500

@quantum_api.route("/container", methods=["POST"])
def container():
    """Attempt to unlock a container with a waveform."""
    try:
        data = request.json
        waveform = data.get("waveform", [])
        container_hash = data.get("hash", "")
        key = data.get("key", "")
        
        if not waveform or not container_hash:
            return jsonify({
                "success": False,
                "error": "Missing waveform or container hash"
            }), 400
            
        result = quantum_adapter.unlock_container(waveform, container_hash, key)
        
        return jsonify({
            "success": result.get("success", False),
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Container operation failed: {str(e)}"
        }), 500

@quantum_api.route("/benchmark", methods=["POST"])
def benchmark():
    """Run a quantum system benchmark."""
    try:
        data = request.json
        max_qubits = data.get("max_qubits", 150)
        run_full_benchmark = data.get("connect_encryption", False)
        
        # Ensure we don't exceed system limits
        max_qubits = min(max(max_qubits, 1), 150)
        
        # Run the benchmark using our adapter
        result = quantum_adapter.run_benchmark(max_qubits, run_full_benchmark)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Benchmark failed: {str(e)}"
        }), 500