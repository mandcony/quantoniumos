"""
Routes package for QuantoniumOS Flask application
"""

from flask import Blueprint, jsonify, request
import sys
import os

# Add RFT algorithms to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))

# Create main API blueprint
api = Blueprint('api', __name__, url_prefix='/api/v1')

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "QuantoniumOS API",
        "version": "1.0.0"
    })

@api.route('/rft/forward', methods=['POST'])
def rft_forward():
    """RFT forward transform endpoint"""
    try:
        import canonical_true_rft
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        input_data = data['input']
        result = canonical_true_rft.forward_true_rft(input_data)
        
        return jsonify({
            "status": "success",
            "input_length": len(input_data),
            "output_length": len(result),
            "result": result.tolist() if hasattr(result, 'tolist') else list(result)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/rft/inverse', methods=['POST'])
def rft_inverse():
    """RFT inverse transform endpoint"""
    try:
        import canonical_true_rft
        data = request.get_json()
        
        if not data or 'input' not in data:
            return jsonify({"error": "Missing 'input' field"}), 400
        
        input_data = data['input']
        result = canonical_true_rft.inverse_true_rft(input_data)
        
        return jsonify({
            "status": "success",
            "input_length": len(input_data),
            "output_length": len(result),
            "result": result.tolist() if hasattr(result, 'tolist') else list(result)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Encryption/Decryption functions
def encrypt(data, key=None):
    """Basic encryption placeholder"""
    # This is a placeholder - implement actual RFT-based encryption
    import base64
    import json
    
    if isinstance(data, (dict, list)):
        data = json.dumps(data)
    
    encoded = base64.b64encode(str(data).encode()).decode()
    return f"RFT_ENCRYPTED:{encoded}"

def decrypt(encrypted_data, key=None):
    """Basic decryption placeholder"""
    # This is a placeholder - implement actual RFT-based decryption
    import base64
    import json
    
    if not encrypted_data.startswith("RFT_ENCRYPTED:"):
        raise ValueError("Invalid encrypted data format")
    
    encoded = encrypted_data[14:]  # Remove "RFT_ENCRYPTED:" prefix
    decoded = base64.b64decode(encoded.encode()).decode()
    
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return decoded

@api.route('/encrypt', methods=['POST'])
def encrypt_endpoint():
    """Encryption endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        encrypted = encrypt(data)
        return jsonify({
            "status": "success",
            "encrypted": encrypted
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/decrypt', methods=['POST'])
def decrypt_endpoint():
    """Decryption endpoint"""
    try:
        data = request.get_json()
        if not data or 'encrypted' not in data:
            return jsonify({"error": "Missing 'encrypted' field"}), 400
        
        decrypted = decrypt(data['encrypted'])
        return jsonify({
            "status": "success",
            "decrypted": decrypted
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Entropy functions (placeholders)
def sample_entropy(data, m=2, r=0.2):
    """Sample entropy calculation placeholder"""
    import numpy as np
    
    if not isinstance(data, (list, tuple, np.ndarray)):
        data = list(data)
    
    N = len(data)
    if N < m + 1:
        return 0.0
    
    # Simple placeholder calculation
    return abs(hash(str(data))) % 1000 / 1000.0

def entropy_stream(data_stream):
    """Entropy stream calculation placeholder"""
    if hasattr(data_stream, '__iter__'):
        return [sample_entropy(chunk) for chunk in data_stream]
    else:
        return sample_entropy(data_stream)

@api.route('/entropy/sample', methods=['POST'])
def sample_entropy_endpoint():
    """Sample entropy calculation endpoint"""
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({"error": "Missing 'data' field"}), 400
        
        entropy = sample_entropy(data['data'])
        return jsonify({
            "status": "success",
            "entropy": entropy,
            "data_length": len(data['data'])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
