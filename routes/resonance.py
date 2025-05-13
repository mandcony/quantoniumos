"""
Quantonium OS - Resonance Routes

Handles container operations, resonance encryption, and quantum operations.
"""

from flask import request, jsonify
from flask_login import login_required, current_user
import json
import uuid
import base64

from app import db
from models import Container
from attached_assets.geometric_waveform_hash import geometric_waveform_hash
from attached_assets.symbolic_qubit_resonance_test import create_symbolic_containers

def resonance_routes(bp):
    @bp.route('/containers', methods=['GET'])
    @login_required
    def get_containers():
        try:
            # Get all containers belonging to the current user
            containers = Container.query.filter_by(owner_id=current_user.id).all()
            
            container_list = []
            for container in containers:
                container_list.append({
                    "id": container.id,
                    "name": container.name,
                    "created_at": container.created_at.isoformat(),
                    "resonant_frequencies": json.loads(container.resonant_frequencies) if container.resonant_frequencies else [],
                    "hash_value": container.hash_value,
                    "locked": container.locked
                })
            
            return jsonify({"containers": container_list}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get containers: {str(e)}"}), 500

    @bp.route('/containers', methods=['POST'])
    @login_required
    def create_container():
        try:
            data = request.json
            
            if not data or not data.get('name') or not data.get('vertices'):
                return jsonify({"error": "Missing required fields"}), 400
            
            # Generate hash from vertices
            container_data = json.dumps(data['vertices']).encode()
            hash_value = geometric_waveform_hash(container_data)
            
            # Create resonance frequencies
            resonant_frequencies = []
            for i in range(3):  # Generate 3 resonant frequencies
                freq = float(hash_value[i]) / 255.0  # Normalize to 0-1
                resonant_frequencies.append(round(freq, 3))
            
            # Create container
            new_container = Container()
            new_container.name = data['name']
            new_container.owner_id = current_user.id
            new_container.vertices = json.dumps(data['vertices'])
            new_container.resonant_frequencies = json.dumps(resonant_frequencies)
            new_container.hash_value = hash_value
            new_container.locked = True
            
            db.session.add(new_container)
            db.session.commit()
            
            return jsonify({
                "id": new_container.id,
                "name": new_container.name,
                "resonant_frequencies": resonant_frequencies,
                "hash_value": hash_value
            }), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Failed to create container: {str(e)}"}), 500

    @bp.route('/containers/<container_id>/unlock', methods=['POST'])
    @login_required
    def unlock_container(container_id):
        try:
            data = request.json
            
            if not data or not data.get('waveform'):
                return jsonify({"error": "Missing waveform data"}), 400
            
            # Find container
            container = Container.query.get(container_id)
            
            if not container:
                return jsonify({"error": "Container not found"}), 404
            
            # Verify ownership
            if container.owner_id != current_user.id:
                return jsonify({"error": "Unauthorized"}), 403
            
            # Verify waveform matches container frequencies
            container_freqs = json.loads(container.resonant_frequencies) if container.resonant_frequencies else []
            waveform = data['waveform']
            
            # Simple resonance check (in a real implementation this would be more complex)
            resonance_match = False
            if len(waveform) >= len(container_freqs):
                matches = 0
                for i, freq in enumerate(container_freqs):
                    if abs(waveform[i] - freq) < 0.1:  # Threshold for frequency match
                        matches += 1
                resonance_match = matches >= len(container_freqs) * 0.8  # 80% match required
            
            if not resonance_match:
                return jsonify({
                    "success": False,
                    "message": "Resonance mismatch - container remains locked"
                }), 200
            
            # Unlock container
            container.locked = False
            db.session.commit()
            
            return jsonify({
                "success": True,
                "message": "Container unlocked successfully"
            }), 200
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Failed to unlock container: {str(e)}"}), 500

    @bp.route('/encrypt', methods=['POST'])
    @login_required
    def encrypt_data():
        try:
            data = request.json
            
            if not data or not data.get('plaintext') or not data.get('container_id'):
                return jsonify({"error": "Missing required fields"}), 400
            
            # Find container
            container = Container.query.get(data['container_id'])
            
            if not container:
                return jsonify({"error": "Container not found"}), 404
            
            # Verify ownership
            if container.owner_id != current_user.id:
                return jsonify({"error": "Unauthorized"}), 403
            
            # Check if container is unlocked
            if container.locked:
                return jsonify({"error": "Container is locked"}), 403
            
            # Encrypt data using container hash as key
            from attached_assets.symbolic_quantum_nova_system import SymbolicQuantumNovaSystem
            from attached_assets.geometric_container import GeometricContainer
            
            # Create a temporary container object for the encryption process
            temp_container = GeometricContainer(container.name, json.loads(container.vertices))
            temp_container.resonant_frequencies = json.loads(container.resonant_frequencies)
            temp_container.id = container.id
            
            # Encrypt using the quantum nova system
            nova = SymbolicQuantumNovaSystem(num_qubits=50)
            encrypted = nova.encrypt_data(temp_container, data['plaintext'])
            
            return jsonify({
                "ciphertext": base64.b64encode(encrypted.encode()).decode(),
                "container_id": container.id
            }), 200
        except Exception as e:
            return jsonify({"error": f"Encryption failed: {str(e)}"}), 500

    @bp.route('/decrypt', methods=['POST'])
    @login_required
    def decrypt_data():
        try:
            data = request.json
            
            if not data or not data.get('ciphertext') or not data.get('container_id'):
                return jsonify({"error": "Missing required fields"}), 400
            
            # Find container
            container = Container.query.get(data['container_id'])
            
            if not container:
                return jsonify({"error": "Container not found"}), 404
            
            # Verify ownership
            if container.owner_id != current_user.id:
                return jsonify({"error": "Unauthorized"}), 403
            
            # Check if container is unlocked
            if container.locked:
                return jsonify({"error": "Container is locked"}), 403
            
            # Decrypt data using container hash as key
            from attached_assets.symbolic_quantum_nova_system import SymbolicQuantumNovaSystem
            from attached_assets.geometric_container import GeometricContainer
            
            # Create a temporary container object for the decryption process
            temp_container = GeometricContainer(container.name, json.loads(container.vertices))
            temp_container.resonant_frequencies = json.loads(container.resonant_frequencies)
            temp_container.id = container.id
            
            # Decrypt using the quantum nova system
            nova = SymbolicQuantumNovaSystem(num_qubits=50)
            ciphertext = base64.b64decode(data['ciphertext']).decode()
            decrypted = nova.decrypt_data(temp_container, ciphertext)
            
            return jsonify({
                "plaintext": decrypted,
                "container_id": container.id
            }), 200
        except Exception as e:
            return jsonify({"error": f"Decryption failed: {str(e)}"}), 500

    @bp.route('/demo/containers', methods=['GET'])
    def get_demo_containers():
        """Generate and return demo containers for testing"""
        try:
            containers = create_symbolic_containers()
            
            container_list = []
            for container in containers:
                container_list.append({
                    "id": container.id,
                    "name": container.id,
                    "resonant_frequencies": container.resonant_frequencies,
                    "vertices": container.vertices
                })
            
            return jsonify({"containers": container_list}), 200
        except Exception as e:
            return jsonify({"error": f"Failed to get demo containers: {str(e)}"}), 500