"""
Quantonium OS - API Route Definitions

Defines all token-protected endpoints to access symbolic stack modules.
"""

from flask import Blueprint, request, jsonify
from core.protected.symbolic_interface import get_interface
from models import EncryptRequest, RFTRequest, EntropyRequest, ContainerUnlockRequest
from utils import validate_api_key, reject_unauthorized, sign_response

api = Blueprint("api", __name__)
symbolic = get_interface()

@api.before_request
def require_auth():
    # Skip API key validation for the root endpoint
    if request.endpoint == "api.root_status" and request.method == "GET":
        return None
        
    # Require API key for all other API endpoints
    if not validate_api_key(request):
        return reject_unauthorized()

@api.route("/", methods=["GET"])
def root_status():
    return jsonify({"name": "Quantonium OS Cloud Runtime", "status": "operational", "version": "1.0.0"})

@api.route("/encrypt", methods=["POST"])
def encrypt():
    data = EncryptRequest(**request.get_json())
    result = symbolic.encrypt(data.plaintext, data.key)
    return jsonify(sign_response({"ciphertext": result}))

@api.route("/simulate/rft", methods=["POST"])
def simulate_rft():
    data = RFTRequest(**request.get_json())
    result = symbolic.analyze_waveform(data.waveform)
    return jsonify(sign_response({"frequencies": result}))

@api.route("/entropy/sample", methods=["POST"])
def sample_entropy():
    data = EntropyRequest(**request.get_json())
    result = symbolic.get_entropy(data.amount)
    return jsonify(sign_response({"entropy": result}))

@api.route("/container/unlock", methods=["POST"])
def unlock():
    data = ContainerUnlockRequest(**request.get_json())
    result = symbolic.verify_container(data.waveform, data.hash)
    return jsonify(sign_response({"unlocked": result}))
