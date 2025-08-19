"""
Quantum computation routes for QuantoniumOS Flask application
"""

from flask import Blueprint, jsonify, request
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), '04_RFT_ALGORITHMS'))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core'))

# Create quantum API blueprint
quantum_api = Blueprint('quantum', __name__, url_prefix='/api/quantum')

@quantum_api.route('/health', methods=['GET'])
def quantum_health():
    """Quantum API health check"""
    return jsonify({
        "status": "healthy",
        "service": "Quantum API",
        "version": "1.0.0"
    })

@quantum_api.route('/grover', methods=['POST'])
def grover_search():
    """Grover's algorithm search endpoint"""
    try:
        # Import quantum modules from organized structure
        from core.python.quantum import grover_amplification
        
        data = request.get_json()
        if not data or 'target' not in data or 'database' not in data:
            return jsonify({"error": "Missing 'target' or 'database' fields"}), 400
        
        target = data['target']
        database = data['database']
        
        # Placeholder Grover's algorithm simulation
        result = {
            "target": target,
            "database_size": len(database),
            "found": target in database,
            "iterations": int(3.14159 * (len(database) ** 0.5) / 4) if len(database) > 0 else 0,
            "probability": 0.95 if target in database else 0.0
        }
        
        return jsonify({
            "status": "success",
            "result": result,
            "algorithm": "Grover's Search"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@quantum_api.route('/amplitude', methods=['POST'])
def quantum_amplitude():
    """Quantum amplitude calculation endpoint"""
    try:
        from core.python.quantum import symbolic_amplitude
        
        data = request.get_json()
        if not data or 'state' not in data:
            return jsonify({"error": "Missing 'state' field"}), 400
        
        state = data['state']
        
        # Placeholder amplitude calculation
        amplitudes = []
        for i, component in enumerate(state):
            amp = complex(component) if isinstance(component, (int, float)) else complex(component['real'], component.get('imag', 0))
            amplitudes.append({
                "index": i,
                "amplitude": {"real": amp.real, "imag": amp.imag},
                "magnitude": abs(amp),
                "phase": amp.__phase__() if hasattr(amp, '__phase__') else 0
            })
        
        return jsonify({
            "status": "success",
            "amplitudes": amplitudes,
            "normalization": sum(abs(complex(c) if isinstance(c, (int, float)) else complex(c['real'], c.get('imag', 0)))**2 for c in state)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@quantum_api.route('/search', methods=['POST'])
def quantum_search():
    """Quantum search algorithm endpoint"""
    try:
        from core.python.quantum import symbolic_quantum_search
        
        data = request.get_json()
        if not data or 'query' not in data or 'space' not in data:
            return jsonify({"error": "Missing 'query' or 'space' fields"}), 400
        
        query = data['query']
        search_space = data['space']
        
        # Placeholder quantum search
        results = []
        for item in search_space:
            if query.lower() in str(item).lower():
                results.append({
                    "item": item,
                    "relevance": 0.9,
                    "quantum_probability": 0.85
                })
        
        return jsonify({
            "status": "success",
            "query": query,
            "results": results[:10],  # Top 10 results
            "total_found": len(results),
            "search_space_size": len(search_space)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@quantum_api.route('/qubit/state', methods=['POST'])
def qubit_state():
    """Multi-qubit state management endpoint"""
    try:
        from core.python.quantum import multi_qubit_state
        
        data = request.get_json()
        if not data or 'qubits' not in data:
            return jsonify({"error": "Missing 'qubits' field"}), 400
        
        num_qubits = data['qubits']
        initial_state = data.get('initial_state', 'all_zero')
        
        # Create quantum state representation
        state_size = 2 ** num_qubits
        
        if initial_state == 'all_zero':
            state = [1.0] + [0.0] * (state_size - 1)
        elif initial_state == 'superposition':
            state = [1.0 / (state_size ** 0.5)] * state_size
        else:
            state = data.get('custom_state', [1.0] + [0.0] * (state_size - 1))
        
        return jsonify({
            "status": "success",
            "num_qubits": num_qubits,
            "state_size": state_size,
            "state": state,
            "initial_state": initial_state
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
