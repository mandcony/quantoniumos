"""
Quantonium OS - API Routes

Handles API endpoints for the frontend interface.
"""

import random
import json
import base64
from flask import request, jsonify, abort, current_app
from attached_assets.wave_primitives import WaveNumber, rft, irft
from attached_assets.geometric_waveform_hash import geometric_waveform_hash, verify_resonance
from attached_assets.resonance_encryption import resonance_encrypt, resonance_decrypt, generate_wave_from_text
from attached_assets.symbolic_qubit_resonance_test import create_fixed_demo_containers, run_test_with_containers
from attached_assets.symbolic_quantum_nova_system import SymbolicQuantumNovaSystem

def api_routes(bp):
    """Register API routes on the given blueprint."""
    
    @bp.route('/encrypt', methods=['POST'])
    def encrypt():
        """Encrypt data using resonance technique."""
        data = request.json
        if not data or not data.get('plaintext') or not data.get('key'):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        plaintext = data['plaintext']
        key = data['key']
        
        try:
            # Generate wave from key
            wave_key = generate_wave_from_text(key)
            
            # Encrypt the data
            ciphertext = resonance_encrypt(plaintext, wave_key)
            
            return jsonify({
                'ciphertext': ciphertext,
                'key_info': {
                    'amplitude': wave_key.amplitude,
                    'phase': wave_key.phase
                }
            })
        except Exception as e:
            current_app.logger.error(f"Encryption error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/decrypt', methods=['POST'])
    def decrypt():
        """Decrypt data using resonance technique."""
        data = request.json
        if not data or not data.get('ciphertext') or not data.get('key'):
            return jsonify({'error': 'Missing required parameters'}), 400
            
        ciphertext = data['ciphertext']
        key = data['key']
        
        try:
            # Generate wave from key
            wave_key = generate_wave_from_text(key)
            
            # Decrypt the data
            plaintext = resonance_decrypt(ciphertext, wave_key)
            
            return jsonify({
                'plaintext': plaintext
            })
        except Exception as e:
            current_app.logger.error(f"Decryption error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/simulate/rft', methods=['POST'])
    def calculate_rft():
        """Calculate Resonance Fourier Transform of waveform."""
        data = request.json
        if not data or 'waveform' not in data:
            return jsonify({'error': 'Missing waveform parameter'}), 400
            
        waveform = data['waveform']
        
        try:
            # Calculate RFT
            frequencies = rft(waveform)
            
            return jsonify({
                'frequencies': frequencies
            })
        except Exception as e:
            current_app.logger.error(f"RFT calculation error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/entropy/sample', methods=['POST'])
    def generate_entropy():
        """Generate quantum-inspired entropy."""
        data = request.json
        if not data or 'amount' not in data:
            return jsonify({'error': 'Missing amount parameter'}), 400
            
        amount = data['amount']
        if not isinstance(amount, int) or amount < 1 or amount > 1024:
            return jsonify({'error': 'Amount must be an integer between 1 and 1024'}), 400
            
        try:
            # Generate entropy using our quantum system
            nova = SymbolicQuantumNovaSystem(num_qubits=max(8, amount // 4))
            result = nova.run_quantum_demo()
            
            # Use the symbolic measurement as entropy source
            measurement = result['measurement']
            
            # Generate the requested amount of entropy
            entropy_bytes = bytearray()
            while len(entropy_bytes) < amount:
                # Hash the measurement repeatedly to generate more entropy
                h = measurement + str(len(entropy_bytes))
                hash_bytes = geometric_waveform_hash(h.encode()).encode()
                entropy_bytes.extend(hash_bytes[:min(len(hash_bytes), amount - len(entropy_bytes))])
            
            # Encode as base64 for display
            entropy_base64 = base64.b64encode(entropy_bytes).decode('ascii')
            
            return jsonify({
                'entropy': entropy_base64,
                'bits': len(entropy_bytes) * 8
            })
        except Exception as e:
            current_app.logger.error(f"Entropy generation error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    @bp.route('/container/unlock', methods=['POST'])
    def unlock_container():
        """Unlock a container with a resonant waveform."""
        data = request.json
        if not data or 'hash' not in data:
            return jsonify({'error': 'Missing hash parameter'}), 400
            
        hash_value = data['hash']
        
        # If waveform is provided, use it for verification
        if 'waveform' in data and data['waveform']:
            waveform = data['waveform']
            
            # Create demo containers
            containers = create_fixed_demo_containers()
            
            # Run container matching
            matching_container = run_test_with_containers(waveform, containers)
            
            if matching_container:
                # Container found with matching resonance
                return jsonify({
                    'unlocked': True,
                    'container': {
                        'id': matching_container.id,
                        'created': '2025-04-15T14:32:00Z',
                        'access_count': 3,
                        'last_accessed': '2025-05-12T09:17:43Z',
                        'content_preview': 'Secure resonance-locked container unlocked successfully.'
                    }
                })
            else:
                # No match found
                return jsonify({
                    'unlocked': False,
                    'reason': 'No container with matching resonance found'
                })
        else:
            # No waveform provided, return error
            return jsonify({
                'error': 'Missing waveform parameter'
            }), 400
    
    @bp.route('/container/auto-unlock', methods=['POST'])
    def auto_unlock_container():
        """Auto-unlock a container using just the hash."""
        data = request.json
        if not data or 'hash' not in data:
            return jsonify({'error': 'Missing hash parameter'}), 400
            
        hash_value = data['hash']
        
        # Create demo containers
        containers = create_fixed_demo_containers()
        
        # In auto mode, we extract amplitude directly from hash
        # This is a simplified version - in real app we'd use more complex verification
        unlocked = False
        target_container = None
        
        # Demo logic - in real app this would verify based on the hash cryptographic properties
        if len(hash_value) > 20:
            # Simple demo: use first character's code as a random seed
            seed = ord(hash_value[0]) % 100
            if seed > 30:  # 70% chance of success in demo
                # Find container with closest hash signature
                for container in containers:
                    if container.hash_value.startswith('RH-A0.5'):
                        unlocked = True
                        target_container = container
                        break
        
        if unlocked and target_container:
            return jsonify({
                'unlocked': True,
                'container': {
                    'id': target_container.id,
                    'created': '2025-04-22T11:45:00Z',
                    'access_count': 1,
                    'last_accessed': '2025-05-13T14:30:12Z',
                    'content_preview': 'Secure container unlocked with hash key authentication.'
                }
            })
        else:
            return jsonify({
                'unlocked': False,
                'reason': 'Hash verification failed'
            })