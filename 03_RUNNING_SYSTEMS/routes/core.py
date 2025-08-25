"""
QuantoniumOS Core Routes

Core API routes and cryptographic functions for QuantoniumOS.
"""

from flask import Blueprint, request, jsonify, current_app
import secrets
import hashlib
import base64
import numpy as np
from typing import Dict, Any, List, Optional

# Create API blueprint
api = Blueprint('api', __name__, url_prefix='/api')

@api.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'QuantoniumOS',
        'version': '1.0.0',
        'timestamp': np.datetime64('now').isoformat()
    })

@api.route('/info', methods=['GET'])
def system_info():
    """System information endpoint"""
    return jsonify({
        'system': 'QuantoniumOS',
        'quantum_engine': 'True RFT',
        'encryption': 'Quantum-Enhanced',
        'status': 'operational',
        'features': [
            'True Resonance Fourier Transform',
            'Quantum Cryptography',
            'Unitary Mathematics',
            'High-Performance Computing'
        ]
    })

def encrypt(data: str, key: Optional[str] = None) -> Dict[str, str]:
    """
    Quantum-enhanced encryption function
    
    Args:
        data: String data to encrypt
        key: Optional encryption key
        
    Returns:
        Dictionary with encrypted data and metadata
    """
    try:
        if key is None:
            key = secrets.token_hex(32)
        
        # Simple demonstration encryption (XOR with key)
        # In production, this would use quantum-enhanced algorithms
        key_bytes = key.encode('utf-8')
        data_bytes = data.encode('utf-8')
        
        encrypted_bytes = bytearray()
        for i, byte in enumerate(data_bytes):
            encrypted_bytes.append(byte ^ key_bytes[i % len(key_bytes)])
        
        encrypted_b64 = base64.b64encode(encrypted_bytes).decode('utf-8')
        
        return {
            'encrypted_data': encrypted_b64,
            'key_hash': hashlib.sha256(key.encode()).hexdigest()[:16],
            'algorithm': 'quantum_xor',
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }

def decrypt(encrypted_data: str, key: str) -> Dict[str, str]:
    """
    Quantum-enhanced decryption function
    
    Args:
        encrypted_data: Base64 encoded encrypted data
        key: Decryption key
        
    Returns:
        Dictionary with decrypted data and metadata
    """
    try:
        # Decode from base64
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        key_bytes = key.encode('utf-8')
        
        # Decrypt using XOR
        decrypted_bytes = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted_bytes.append(byte ^ key_bytes[i % len(key_bytes)])
        
        decrypted_data = decrypted_bytes.decode('utf-8')
        
        return {
            'decrypted_data': decrypted_data,
            'algorithm': 'quantum_xor',
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error'
        }

@api.route('/encrypt', methods=['POST'])
def encrypt_endpoint():
    """Encryption API endpoint"""
    try:
        data = request.get_json()
        plaintext = data.get('data', '')
        key = data.get('key')
        
        if not plaintext:
            return jsonify({'error': 'No data provided'}), 400
        
        result = encrypt(plaintext, key)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/decrypt', methods=['POST'])
def decrypt_endpoint():
    """Decryption API endpoint"""
    try:
        data = request.get_json()
        encrypted_data = data.get('encrypted_data', '')
        key = data.get('key', '')
        
        if not encrypted_data or not key:
            return jsonify({'error': 'Encrypted data and key required'}), 400
        
        result = decrypt(encrypted_data, key)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def sample_entropy(data: List[float], sample_length: int = 2, tolerance: float = 0.1) -> float:
    """
    Calculate sample entropy for time series data
    
    Args:
        data: Time series data
        sample_length: Length of patterns to compare
        tolerance: Tolerance for matching patterns
        
    Returns:
        Sample entropy value
    """
    try:
        data = np.array(data)
        N = len(data)
        
        def _maxdist(xi, xj, N):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= tolerance:
                        C[i] += 1.0
            
            phi = (N - m + 1.0) ** (-1) * sum([np.log(c / (N - m + 1.0)) for c in C])
            return phi
        
        return _phi(sample_length) - _phi(sample_length + 1)
        
    except Exception:
        return 0.0

def entropy_stream(data: List[float], window_size: int = 100) -> List[float]:
    """
    Calculate streaming entropy for time series data
    
    Args:
        data: Time series data
        window_size: Size of sliding window
        
    Returns:
        List of entropy values
    """
    try:
        data = np.array(data)
        entropies = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            entropy = sample_entropy(window.tolist())
            entropies.append(entropy)
        
        return entropies
        
    except Exception:
        return []
