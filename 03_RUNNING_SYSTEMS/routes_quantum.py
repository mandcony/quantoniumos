"""
QuantoniumOS Quantum Routes

Quantum computing and True RFT API endpoints for QuantoniumOS.
"""

from flask import Blueprint, request, jsonify
import numpy as np
import sys
import os
from typing import Dict, Any, List, Optional

# Add the RFT algorithms to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '04_RFT_ALGORITHMS'))

try:
    from canonical_true_rft import TrueResonanceFourierTransform
    RFT_AVAILABLE = True
except ImportError:
    RFT_AVAILABLE = False

# Create quantum API blueprint
quantum_api = Blueprint('quantum', __name__, url_prefix='/api/quantum')

@quantum_api.route('/status', methods=['GET'])
def quantum_status():
    """Get quantum engine status"""
    return jsonify({
        'quantum_engine': 'True RFT',
        'available': RFT_AVAILABLE,
        'algorithms': ['true_rft', 'unitary_transform'],
        'status': 'operational' if RFT_AVAILABLE else 'unavailable',
        'precision': '1e-15'
    })

@quantum_api.route('/transform', methods=['POST'])
def quantum_transform():
    """Perform True RFT transformation"""
    try:
        if not RFT_AVAILABLE:
            return jsonify({'error': 'True RFT engine not available'}), 503
        
        data = request.get_json()
        input_data = data.get('data', [])
        transform_type = data.get('type', 'forward')
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to complex array if needed
        if isinstance(input_data[0], (list, tuple)):
            # Complex data as [real, imag] pairs
            signal = np.array([complex(x[0], x[1]) for x in input_data])
        else:
            # Real data
            signal = np.array(input_data, dtype=complex)
        
        # Create RFT instance
        rft = TrueResonanceFourierTransform(len(signal))
        
        if transform_type == 'forward':
            result = rft.forward_transform(signal)
        else:
            result = rft.inverse_transform(signal)
        
        # Convert back to JSON-serializable format
        result_data = [[float(x.real), float(x.imag)] for x in result]
        
        return jsonify({
            'result': result_data,
            'transform_type': transform_type,
            'length': len(result),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@quantum_api.route('/validate', methods=['POST'])
def quantum_validate():
    """Validate quantum transformation unitarity"""
    try:
        if not RFT_AVAILABLE:
            return jsonify({'error': 'True RFT engine not available'}), 503
        
        data = request.get_json()
        input_data = data.get('data', [])
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert to complex array
        if isinstance(input_data[0], (list, tuple)):
            signal = np.array([complex(x[0], x[1]) for x in input_data])
        else:
            signal = np.array(input_data, dtype=complex)
        
        # Create RFT instance and validate
        rft = TrueResonanceFourierTransform(len(signal))
        validation = rft.validate_unitarity(signal)
        
        return jsonify({
            'validation': validation,
            'unitary': validation['is_unitary'],
            'error': validation['reconstruction_error'],
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@quantum_api.route('/benchmark', methods=['POST'])
def quantum_benchmark():
    """Benchmark quantum operations"""
    try:
        if not RFT_AVAILABLE:
            return jsonify({'error': 'True RFT engine not available'}), 503
        
        data = request.get_json()
        size = data.get('size', 1024)
        iterations = data.get('iterations', 10)
        
        # Generate test signal
        signal = np.random.random(size) + 1j * np.random.random(size)
        
        # Create RFT instance
        rft = TrueResonanceFourierTransform(size)
        
        # Benchmark forward transform
        import time
        start_time = time.time()
        for _ in range(iterations):
            result = rft.forward_transform(signal)
        forward_time = (time.time() - start_time) / iterations
        
        # Benchmark inverse transform
        start_time = time.time()
        for _ in range(iterations):
            reconstructed = rft.inverse_transform(result)
        inverse_time = (time.time() - start_time) / iterations
        
        # Validate reconstruction
        validation = rft.validate_unitarity(signal)
        
        return jsonify({
            'benchmark': {
                'size': size,
                'iterations': iterations,
                'forward_time': forward_time,
                'inverse_time': inverse_time,
                'total_time': forward_time + inverse_time,
                'throughput': size / (forward_time + inverse_time)
            },
            'validation': validation,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@quantum_api.route('/algorithms', methods=['GET'])
def quantum_algorithms():
    """List available quantum algorithms"""
    algorithms = []
    
    if RFT_AVAILABLE:
        algorithms.extend([
            {
                'name': 'True Resonance Fourier Transform',
                'id': 'true_rft',
                'description': 'Unitary frequency domain transformation',
                'properties': ['unitary', 'reversible', 'energy_conserving']
            },
            {
                'name': 'Quantum Encryption',
                'id': 'quantum_encrypt',
                'description': 'Quantum-enhanced cryptographic operations',
                'properties': ['secure', 'quantum_resistant']
            }
        ])
    
    return jsonify({
        'algorithms': algorithms,
        'count': len(algorithms),
        'quantum_engine_available': RFT_AVAILABLE
    })
