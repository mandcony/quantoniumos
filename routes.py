import os
import time
import logging
from functools import wraps
from flask import Blueprint, request, jsonify, current_app

from models import (
    EncryptRequest, EncryptResponse,
    RFTRequest, RFTResponse,
    EntropySampleRequest, EntropySampleResponse,
    ContainerUnlockRequest, ContainerUnlockResponse
)
from utils import sign_response
from core.protected.resonance_encrypt import encrypt_data
from core.protected.resonance_fourier import perform_rft
from core.protected.entropy_qrng import generate_entropy
from core.protected.symbolic_container import unlock_container

# Create logger
logger = logging.getLogger(__name__)

# Create blueprint
api_blueprint = Blueprint('api', __name__)

# API Key middleware
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        expected_api_key = os.environ.get('QUANTONIUM_API_KEY', 'dev-token')
        
        if not api_key or api_key != expected_api_key:
            logger.warning("Invalid or missing API key")
            return jsonify({
                "error": "Unauthorized: Invalid or missing API key",
                "status": "error",
                "timestamp": int(time.time()),
                "code": 401
            }), 401
        return f(*args, **kwargs)
    return decorated_function

# Error handling for invalid JSON
@api_blueprint.errorhandler(400)
def handle_bad_request(e):
    response = {
        "error": "Bad request: Invalid JSON payload",
        "status": "error",
        "timestamp": int(time.time()),
        "code": 400
    }
    return jsonify(response), 400

# Error handling for Pydantic validation errors
@api_blueprint.errorhandler(422)
def handle_validation_error(e):
    response = {
        "error": f"Validation error: {str(e)}",
        "status": "error",
        "timestamp": int(time.time()),
        "code": 422
    }
    return jsonify(response), 422

# Routes
@api_blueprint.route('/encrypt', methods=['POST'])
@require_api_key
def encrypt():
    try:
        # Validate request with Pydantic
        req_data = request.get_json()
        encrypt_req = EncryptRequest(**req_data)
        
        # Process with the protected module
        ciphertext = encrypt_data(encrypt_req.plaintext, encrypt_req.key)
        
        # Prepare and sign response
        response_data = EncryptResponse(
            ciphertext=ciphertext,
            status="success"
        ).dict()
        
        return jsonify(sign_response(response_data))
        
    except Exception as e:
        logger.error(f"Error in /encrypt: {str(e)}")
        error_response = {
            "error": str(e),
            "status": "error",
            "timestamp": int(time.time())
        }
        return jsonify(sign_response(error_response)), 500

@api_blueprint.route('/simulate/rft', methods=['POST'])
@require_api_key
def simulate_rft():
    try:
        # Validate request with Pydantic
        req_data = request.get_json()
        rft_req = RFTRequest(**req_data)
        
        # Process with the protected module
        frequency_data = perform_rft(rft_req.waveform)
        
        # Prepare and sign response
        response_data = RFTResponse(
            frequency_data=frequency_data,
            status="success"
        ).dict()
        
        return jsonify(sign_response(response_data))
        
    except Exception as e:
        logger.error(f"Error in /simulate/rft: {str(e)}")
        error_response = {
            "error": str(e),
            "status": "error",
            "timestamp": int(time.time())
        }
        return jsonify(sign_response(error_response)), 500

@api_blueprint.route('/entropy/sample', methods=['POST'])
@require_api_key
def entropy_sample():
    try:
        # Validate request with Pydantic
        req_data = request.get_json() or {}
        entropy_req = EntropySampleRequest(**req_data)
        
        # Process with the protected module
        entropy_data = generate_entropy(entropy_req.amount)
        
        # Prepare and sign response
        response_data = EntropySampleResponse(
            entropy=entropy_data,
            status="success"
        ).dict()
        
        return jsonify(sign_response(response_data))
        
    except Exception as e:
        logger.error(f"Error in /entropy/sample: {str(e)}")
        error_response = {
            "error": str(e),
            "status": "error",
            "timestamp": int(time.time())
        }
        return jsonify(sign_response(error_response)), 500

@api_blueprint.route('/container/unlock', methods=['POST'])
@require_api_key
def container_unlock():
    try:
        # Validate request with Pydantic
        req_data = request.get_json()
        unlock_req = ContainerUnlockRequest(**req_data)
        
        # Process with the protected module
        is_unlocked = unlock_container(unlock_req.waveform, unlock_req.hash)
        
        # Prepare and sign response
        response_data = ContainerUnlockResponse(
            unlocked=is_unlocked,
            status="success"
        ).dict()
        
        return jsonify(sign_response(response_data))
        
    except Exception as e:
        logger.error(f"Error in /container/unlock: {str(e)}")
        error_response = {
            "error": str(e),
            "status": "error",
            "timestamp": int(time.time())
        }
        return jsonify(sign_response(error_response)), 500
