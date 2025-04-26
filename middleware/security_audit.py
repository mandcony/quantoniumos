"""
Quantonium OS - Security Audit Middleware

Implements NIST 800-53 compliant security audit middleware for tracking
and verifying security-relevant events. This module provides a comprehensive
audit trail for compliance and verification purposes without exposing
proprietary algorithms.
"""

import time
import logging
import functools
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Union
from flask import request, g, current_app

# Import the enhanced security logging
from utils.security_logger import (
    log_security_event, 
    SecurityEventType,
    SecurityOutcome,
    ImpactLevel,
    log_audit_event,
    sanitize_sensitive_data
)

# Configure logger
logger = logging.getLogger("quantonium_audit")
logger.setLevel(logging.INFO)

# Ensure audit directory exists
audit_dir = os.path.join(os.getcwd(), "logs", "audit")
os.makedirs(audit_dir, exist_ok=True)

# Create a list of sensitive parameters that should be redacted in audit logs
SENSITIVE_PARAMS = [
    'password', 'token', 'secret', 'key', 'api_key', 'apikey', 
    'credential', 'jwt', 'private_key', 'auth', 'waveform', 'hash'
]

# List of audit-exempt paths (health checks, static resources)
AUDIT_EXEMPT_PATHS = [
    '/static/', 
    '/health', 
    '/status',
    '/wave_ui/'
]

def is_path_exempt(path: str) -> bool:
    """
    Check if a path is exempt from audit logging.
    
    Args:
        path: The request path
        
    Returns:
        True if path is exempt, False otherwise
    """
    return any(path.startswith(exempt) for exempt in AUDIT_EXEMPT_PATHS)

def get_client_info() -> Dict[str, str]:
    """
    Get client information from the request.
    
    Returns:
        Dictionary with client information
    """
    return {
        'ip': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', 'Unknown'),
        'referer': request.headers.get('Referer', 'Direct'),
        'method': request.method,
        'path': request.path,
        'endpoint': request.endpoint or 'Unknown'
    }

def sanitize_request_data() -> Dict[str, Any]:
    """
    Create a sanitized copy of request data for audit logs.
    
    Returns:
        Dictionary with sanitized request data
    """
    data = {}
    
    # Include form data if present
    if request.form:
        form_data = request.form.to_dict()
        # Remove sensitive data
        data['form'] = sanitize_sensitive_data(form_data)
    
    # Include JSON data if present
    if request.is_json:
        try:
            json_data = request.get_json(silent=True)
            if json_data and isinstance(json_data, dict):
                # Remove sensitive data
                data['json'] = sanitize_sensitive_data(json_data)
        except Exception:
            data['json_error'] = "Could not parse JSON data"
    
    # Include query parameters if present
    if request.args:
        args_data = request.args.to_dict()
        # Remove sensitive data
        data['args'] = sanitize_sensitive_data(args_data)
    
    return data

def audit_request(include_response: bool = False) -> Callable:
    """
    Decorator to audit API requests and optionally responses.
    
    Args:
        include_response: Whether to include response data in the audit log
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Skip audit for exempt paths
            if is_path_exempt(request.path):
                return f(*args, **kwargs)
            
            # Generate correlation ID for request/response correlation
            correlation_id = str(uuid.uuid4())
            g.correlation_id = correlation_id
            
            # Get timestamp before processing
            start_time = time.time()
            request_time = datetime.utcnow().isoformat()
            
            # Get client information
            client_info = get_client_info()
            
            # Sanitize request data for audit log
            request_data = sanitize_request_data()
            
            # Extract user ID if available
            user_id = getattr(g, 'user', {}).get('id', 'anonymous')
            
            # Log the request
            log_audit_event(
                action="api_request",
                message=f"API request to {request.path}",
                user_id=user_id,
                resource=request.path,
                outcome=SecurityOutcome.UNKNOWN,  # We don't know the outcome yet
                metadata={
                    "correlation_id": correlation_id,
                    "request_time": request_time,
                    "client": client_info,
                    "request_data": request_data,
                    "endpoint": request.endpoint or 'Unknown'
                },
                client_ip=request.remote_addr
            )
            
            try:
                # Execute the original function
                response = f(*args, **kwargs)
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Skip detailed response logging for binary responses or large responses
                if include_response and hasattr(response, 'status_code'):
                    # For API responses, log key details
                    log_audit_event(
                        action="api_response",
                        message=f"API response from {request.path}",
                        user_id=user_id,
                        resource=request.path,
                        outcome=SecurityOutcome.SUCCESS if response.status_code < 400 else SecurityOutcome.FAILURE,
                        metadata={
                            "correlation_id": correlation_id,
                            "status_code": response.status_code,
                            "processing_time_ms": int(processing_time * 1000),
                            "response_size": len(response.get_data()) if hasattr(response, 'get_data') else 'Unknown',
                            "content_type": response.content_type if hasattr(response, 'content_type') else 'Unknown'
                        },
                        client_ip=request.remote_addr
                    )
                
                return response
                
            except Exception as e:
                # Log the exception
                log_security_event(
                    event_type=SecurityEventType.SYSTEM_ERROR,
                    message=f"Error processing request to {request.path}: {str(e)}",
                    outcome=SecurityOutcome.FAILURE,
                    level=logging.ERROR,
                    target_resource=request.path,
                    metadata={
                        "correlation_id": correlation_id,
                        "exception": str(e),
                        "exception_type": type(e).__name__,
                        "client": client_info
                    },
                    security_labels=["api-error", "exception"],
                    impact_level=ImpactLevel.MODERATE,
                    user_id=user_id,
                    source_ip=request.remote_addr,
                    correlation_id=correlation_id,
                    include_stack_trace=True
                )
                # Re-raise the exception to be handled by error handlers
                raise
                
        return wrapper
    return decorator

def audit_privileged_action(action_name: str, resource_type: str) -> Callable:
    """
    Decorator to audit privileged actions (admin/system operations).
    
    Args:
        action_name: Name of the action being performed
        resource_type: Type of resource being affected
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Extract user ID if available
            user_id = getattr(g, 'user', {}).get('id', 'system')
            
            # Extract resource ID from kwargs if available
            resource_id = kwargs.get('id', None) or kwargs.get(f'{resource_type}_id', None)
            resource = f"{resource_type}/{resource_id}" if resource_id else resource_type
            
            # Log the privileged action
            log_security_event(
                event_type=SecurityEventType.PRIVILEGED_FUNCTION,
                message=f"Privileged action: {action_name} on {resource}",
                outcome=SecurityOutcome.UNKNOWN,  # Don't know outcome yet
                level=logging.INFO,
                target_resource=resource,
                metadata={
                    "action": action_name,
                    "resource_type": resource_type,
                    "resource_id": resource_id,
                    "function": f.__name__,
                    "kwargs": {k: v for k, v in kwargs.items() if k not in SENSITIVE_PARAMS}
                },
                security_labels=["privileged-action", action_name, resource_type],
                impact_level=ImpactLevel.MODERATE,
                user_id=user_id,
                source_ip=request.remote_addr if request else None
            )
            
            try:
                # Execute the original function
                result = f(*args, **kwargs)
                
                # Log successful outcome
                log_security_event(
                    event_type=SecurityEventType.PRIVILEGED_FUNCTION,
                    message=f"Privileged action succeeded: {action_name} on {resource}",
                    outcome=SecurityOutcome.SUCCESS,
                    level=logging.INFO,
                    target_resource=resource,
                    metadata={"action": action_name, "resource_type": resource_type, "resource_id": resource_id},
                    security_labels=["privileged-action", action_name, resource_type, "success"],
                    impact_level=ImpactLevel.MODERATE,
                    user_id=user_id,
                    source_ip=request.remote_addr if request else None
                )
                
                return result
                
            except Exception as e:
                # Log failed outcome
                log_security_event(
                    event_type=SecurityEventType.PRIVILEGED_FUNCTION,
                    message=f"Privileged action failed: {action_name} on {resource}: {str(e)}",
                    outcome=SecurityOutcome.FAILURE,
                    level=logging.ERROR,
                    target_resource=resource,
                    metadata={
                        "action": action_name, 
                        "resource_type": resource_type, 
                        "resource_id": resource_id,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    security_labels=["privileged-action", action_name, resource_type, "failure"],
                    impact_level=ImpactLevel.HIGH,
                    user_id=user_id,
                    source_ip=request.remote_addr if request else None,
                    include_stack_trace=True
                )
                # Re-raise the exception to be handled by error handlers
                raise
                
        return wrapper
    return decorator

def audit_crypto_operation(operation_type: str) -> Callable:
    """
    Decorator to audit cryptographic operations.
    Specifically designed for monitoring and validating cryptographic operations
    without exposing algorithm details.
    
    Args:
        operation_type: Type of cryptographic operation
        
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Extract user ID if available
            user_id = getattr(g, 'user', {}).get('id', 'anonymous')
            
            # Safe metadata that doesn't expose sensitive values or algorithms
            safe_metadata = {
                "operation_type": operation_type,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add information about the input types without exposing values
            for k, v in kwargs.items():
                if k not in SENSITIVE_PARAMS:
                    if isinstance(v, (str, int, float, bool)):
                        safe_metadata[f"param_{k}_type"] = type(v).__name__
                    elif v is None:
                        safe_metadata[f"param_{k}_type"] = "None"
                    else:
                        safe_metadata[f"param_{k}_type"] = f"{type(v).__name__}"
            
            # Log the start of the crypto operation
            log_security_event(
                event_type=SecurityEventType.CRYPTO_OPERATION,
                message=f"Cryptographic operation: {operation_type}",
                outcome=SecurityOutcome.UNKNOWN,
                level=logging.INFO,
                target_resource=operation_type,
                metadata=safe_metadata,
                security_labels=["crypto", operation_type],
                impact_level=ImpactLevel.MODERATE,
                user_id=user_id,
                source_ip=request.remote_addr if request else None
            )
            
            try:
                # Execute the original function
                result = f(*args, **kwargs)
                
                # Log successful crypto operation without exposing result values
                result_metadata = {
                    "operation_type": operation_type,
                    "success": True
                }
                
                # Include information about the result type but not the actual values
                if result is not None:
                    if isinstance(result, (str, int, float, bool)):
                        result_metadata["result_type"] = type(result).__name__
                        if isinstance(result, str):
                            result_metadata["result_length"] = len(result)
                    elif isinstance(result, dict):
                        result_metadata["result_type"] = "dict"
                        result_metadata["result_keys"] = list(result.keys())
                    elif isinstance(result, list):
                        result_metadata["result_type"] = "list"
                        result_metadata["result_length"] = len(result)
                
                log_security_event(
                    event_type=SecurityEventType.CRYPTO_OPERATION,
                    message=f"Cryptographic operation succeeded: {operation_type}",
                    outcome=SecurityOutcome.SUCCESS,
                    level=logging.INFO,
                    target_resource=operation_type,
                    metadata=result_metadata,
                    security_labels=["crypto", operation_type, "success"],
                    impact_level=ImpactLevel.MODERATE,
                    user_id=user_id,
                    source_ip=request.remote_addr if request else None
                )
                
                return result
                
            except Exception as e:
                # Log failed crypto operation
                log_security_event(
                    event_type=SecurityEventType.CRYPTO_OPERATION,
                    message=f"Cryptographic operation failed: {operation_type}: {str(e)}",
                    outcome=SecurityOutcome.FAILURE,
                    level=logging.ERROR,
                    target_resource=operation_type,
                    metadata={
                        "operation_type": operation_type,
                        "error": str(e),
                        "error_type": type(e).__name__
                    },
                    security_labels=["crypto", operation_type, "failure"],
                    impact_level=ImpactLevel.HIGH,
                    user_id=user_id,
                    source_ip=request.remote_addr if request else None,
                    include_stack_trace=True
                )
                # Re-raise the exception to be handled by error handlers
                raise
                
        return wrapper
    return decorator

def initialize_audit_middleware(app) -> None:
    """
    Initialize the audit middleware for a Flask application.
    
    Args:
        app: Flask application
    """
    # Set up before_request handler to log all requests
    @app.before_request
    def audit_all_requests():
        # Skip audit for exempt paths
        if is_path_exempt(request.path):
            return None
        
        # Generate correlation ID for request tracking
        correlation_id = str(uuid.uuid4())
        g.correlation_id = correlation_id
        
        # Get client information
        client_info = get_client_info()
        
        # Extract user ID if available
        user_id = getattr(g, 'user', {}).get('id', 'anonymous')
        
        # Log basic request information
        logger.info(
            f"Request: {request.method} {request.path} from {request.remote_addr} "
            f"(User-Agent: {client_info['user_agent'][:50]}...)"
        )
        
        return None