import time
import hashlib
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

def sign_response(data: dict) -> dict:
    """
    Signs the response payload with a timestamp and SHA-256 hash.
    
    Args:
        data: The response data dictionary
        
    Returns:
        Modified dictionary with timestamp and signature
    """
    # Add timestamp if not present
    if 'timestamp' not in data:
        data['timestamp'] = int(time.time())
    
    # Create a copy without the signature field (in case it exists)
    payload_to_sign = {k: v for k, v in data.items() if k != 'signature'}
    
    # Convert to a canonical string representation
    canonical_payload = json.dumps(payload_to_sign, sort_keys=True)
    
    # Generate SHA-256 hash
    hash_obj = hashlib.sha256(canonical_payload.encode('utf-8'))
    data['signature'] = hash_obj.hexdigest()
    
    return data
