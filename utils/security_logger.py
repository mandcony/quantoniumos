"""
Quantonium OS - Security Logging Module

Implements centralized security logging with rich contextual metadata and 
consistent formatting for security-related events across the application.
This module is designed to complement the existing JSON logging system
while providing enhanced security context for auditing and threat detection.
"""

import os
import json
import time
import uuid
import socket
import logging
import inspect
import platform
import traceback
import hashlib
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from flask import request, g, current_app
from functools import wraps

# Import JSON logger to maintain compatibility
from utils.json_logger import JSONFormatter

# Set up the security logger
security_logger = logging.getLogger("quantonium_security")

# Security event types
class SecurityEventType:
    """Standard security event types for consistent logging"""
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    ACCESS_DENIED = "ACCESS_DENIED"
    API_KEY = "API_KEY"
    RATE_LIMIT = "RATE_LIMIT"
    INPUT_VALIDATION = "INPUT_VALIDATION"
    CONFIGURATION = "CONFIGURATION"
    JWT = "JWT"
    CONTAINER_ACCESS = "CONTAINER_ACCESS"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    ENCRYPTION = "ENCRYPTION"
    SECRET_MANAGEMENT = "SECRET_MANAGEMENT"
    KEY_ROTATION = "KEY_ROTATION"
    SYSTEM = "SYSTEM"

# Security outcomes
class SecurityOutcome:
    """Standard security outcomes for consistent logging"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    BLOCKED = "BLOCKED"
    WARNING = "WARNING"
    INFO = "INFO"

class SecurityJSONFormatter(JSONFormatter):
    """
    Extended JSON formatter that adds security-specific fields
    to log records while maintaining compatibility with the base formatter.
    """
    
    def format(self, record):
        """
        Format the log record as a JSON object with enhanced security fields.
        """
        # Get the standard formatted data from parent class
        log_data = json.loads(super(SecurityJSONFormatter, self).format(record))
        
        # Add security-specific fields if present
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
            
        if hasattr(record, 'event_id'):
            log_data['event_id'] = record.event_id
            
        if hasattr(record, 'outcome'):
            log_data['outcome'] = record.outcome
            
        if hasattr(record, 'security_labels'):
            log_data['security_labels'] = record.security_labels
            
        if hasattr(record, 'target_resource'):
            log_data['target_resource'] = record.target_resource
            
        if hasattr(record, 'source_module'):
            log_data['source_module'] = record.source_module
            
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
            
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
            
        return json.dumps(log_data)

def setup_security_logger(app, log_dir="/tmp/logs", log_level=logging.INFO):
    """
    Set up specialized security logging with enhanced metadata.
    
    Args:
        app: Flask application instance
        log_dir: Directory to store security log files
        log_level: Logging level to use
    """
    global security_logger
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure the Security JSON formatter
    security_formatter = SecurityJSONFormatter()
    
    # Set up timed rotating file handler (rotate daily, keep 30 days for security)
    security_log_file = os.path.join(log_dir, 'quantonium_security.log')
    security_file_handler = TimedRotatingFileHandler(
        security_log_file,
        when='D',  # Daily rotation
        interval=1,
        backupCount=30  # Keep logs longer for security audit
    )
    
    # Create the log file to ensure it exists
    with open(security_log_file, 'a'):
        pass
    
    security_file_handler.setFormatter(security_formatter)
    security_file_handler.setLevel(log_level)
    
    # Add the handler to the security logger
    security_logger.addHandler(security_file_handler)
    security_logger.setLevel(log_level)
    
    # Also add a console handler for development environments
    if app.config.get('DEBUG', False):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(security_formatter)
        console_handler.setLevel(log_level)
        security_logger.addHandler(console_handler)
    
    # Add security logger to the app for easy access
    app.security_logger = security_logger
    
    return security_file_handler

def log_security_event(
    event_type,
    message,
    outcome=SecurityOutcome.INFO,
    level=logging.INFO,
    target_resource=None,
    user_id=None,
    metadata=None,
    security_labels=None,
    exception=None
):
    """
    Log a security event with standardized structure and metadata.
    
    Args:
        event_type: Type of security event (use SecurityEventType constants)
        message: Human-readable message describing the event
        outcome: Outcome of the event (use SecurityOutcome constants)
        level: Log level (logging.INFO, logging.WARNING, etc.)
        target_resource: Resource being accessed or modified
        user_id: ID of the user performing the action (if known)
        metadata: Dict of additional event-specific information
        security_labels: List of security labels for categorization
        exception: Exception object if event is related to an error
    
    Returns:
        event_id: Unique identifier for the logged event
    """
    global security_logger
    
    # Generate a unique ID for the event
    event_id = str(uuid.uuid4())
    
    # Get the calling function details
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals['__name__']
    function_name = frame.f_code.co_name
    line_number = frame.f_lineno
    
    # Get correlation ID from request context or generate new one
    correlation_id = None
    
    # First, check if we're in an application context
    try:
        if hasattr(g, 'correlation_id'):
            correlation_id = g.correlation_id
        
        # Check if we're in a request context
        try:
            if request and request.headers:
                correlation_id = request.headers.get('X-Correlation-ID')
                if correlation_id:
                    # Store in g for other logs to use
                    g.correlation_id = correlation_id
        except RuntimeError:
            # Not in a request context
            pass
    except RuntimeError:
        # Not in an application context
        pass
    
    # If no correlation_id, generate one
    if not correlation_id:
        correlation_id = f"auto-{uuid.uuid4()}"
    
    # Get request information if available
    request_path = None
    request_method = None
    request_ip = None
    api_key_prefix = None
    
    if request:
        request_path = request.path
        request_method = request.method
        request_ip = request.remote_addr
        
        # Extract API key information for logging (safely)
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer ') and len(auth_header) > 15:
            # Extract first 8 chars of the token (safe to log)
            api_key_prefix = auth_header[7:15] + "..."
    
    # Get user ID from context if not provided
    if user_id is None and hasattr(g, 'user') and g.user:
        user_id = g.user.get('id')
    
    # Create the record
    record = security_logger.makeRecord(
        name=security_logger.name,
        level=level,
        fn=function_name,
        lno=line_number,
        msg=message,
        args=(),
        exc_info=exception
    )
    
    # Add standard security metadata
    record.event_type = event_type
    record.event_id = event_id
    record.outcome = outcome
    record.source_module = f"{module_name}.{function_name}"
    record.correlation_id = correlation_id
    
    # Add request context if available
    if request_path:
        record.request_path = request_path
    if request_method:
        record.request_method = request_method
    if request_ip:
        record.request_ip = request_ip
    if api_key_prefix:
        record.api_key_prefix = api_key_prefix
    
    # Add user context if available
    if user_id:
        record.user_id = user_id
    
    # Add target resource if provided
    if target_resource:
        record.target_resource = target_resource
    
    # Add security labels if provided
    if security_labels:
        record.security_labels = security_labels
    
    # Add additional metadata if provided
    if metadata:
        for key, value in metadata.items():
            setattr(record, key, value)
    
    # Log the record
    security_logger.handle(record)
    
    return event_id

def log_auth_success(user_id, message=None, metadata=None):
    """Log successful authentication events"""
    if message is None:
        message = f"Authentication succeeded for user {user_id}"
    
    return log_security_event(
        event_type=SecurityEventType.AUTHENTICATION,
        message=message,
        outcome=SecurityOutcome.SUCCESS,
        level=logging.INFO,
        user_id=user_id,
        metadata=metadata
    )

def log_auth_failure(user_id=None, message=None, metadata=None, reason=None):
    """Log failed authentication events"""
    if message is None:
        message = f"Authentication failed" + (f" for user {user_id}" if user_id else "")
    
    if reason and not metadata:
        metadata = {"reason": reason}
    elif reason:
        metadata = metadata.copy()
        metadata["reason"] = reason
    
    return log_security_event(
        event_type=SecurityEventType.AUTHENTICATION,
        message=message,
        outcome=SecurityOutcome.FAILURE,
        level=logging.WARNING,
        user_id=user_id,
        metadata=metadata
    )

def log_access_denied(resource, user_id=None, message=None, metadata=None):
    """Log access denied events"""
    if message is None:
        message = f"Access denied to {resource}" + (f" for user {user_id}" if user_id else "")
    
    return log_security_event(
        event_type=SecurityEventType.ACCESS_DENIED,
        message=message,
        outcome=SecurityOutcome.BLOCKED,
        level=logging.WARNING,
        user_id=user_id,
        target_resource=resource,
        metadata=metadata
    )

def log_rate_limit_exceeded(client_ip, resource, message=None, metadata=None):
    """Log rate limit exceeded events"""
    if message is None:
        message = f"Rate limit exceeded for {resource} from {client_ip}"
    
    if not metadata:
        metadata = {}
    
    metadata["client_ip"] = client_ip
    
    return log_security_event(
        event_type=SecurityEventType.RATE_LIMIT,
        message=message,
        outcome=SecurityOutcome.BLOCKED,
        level=logging.WARNING,
        target_resource=resource,
        metadata=metadata
    )

def log_input_validation_failure(resource, validation_errors, user_id=None, message=None, metadata=None):
    """Log input validation failure events"""
    if message is None:
        message = f"Input validation failed for {resource}"
    
    if not metadata:
        metadata = {}
    
    metadata["validation_errors"] = validation_errors
    
    return log_security_event(
        event_type=SecurityEventType.INPUT_VALIDATION,
        message=message,
        outcome=SecurityOutcome.FAILURE,
        level=logging.WARNING,
        user_id=user_id,
        target_resource=resource,
        metadata=metadata
    )

def log_suspicious_activity(activity_type, details, user_id=None, client_ip=None, message=None, metadata=None):
    """Log suspicious activity events"""
    if message is None:
        message = f"Suspicious activity detected: {activity_type}"
    
    if not metadata:
        metadata = {}
    
    metadata["activity_type"] = activity_type
    metadata["details"] = details
    
    if client_ip:
        metadata["client_ip"] = client_ip
    
    return log_security_event(
        event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
        message=message,
        outcome=SecurityOutcome.WARNING,
        level=logging.WARNING,
        user_id=user_id,
        metadata=metadata,
        security_labels=["potential-security-threat"]
    )

def log_container_access(container_hash, outcome, user_id=None, message=None, metadata=None):
    """Log container access events"""
    if message is None:
        message = f"Container access attempt: {outcome} for hash {container_hash[:8]}..."
    
    return log_security_event(
        event_type=SecurityEventType.CONTAINER_ACCESS,
        message=message,
        outcome=outcome,
        level=logging.INFO if outcome == SecurityOutcome.SUCCESS else logging.WARNING,
        user_id=user_id,
        target_resource=f"container:{container_hash[:8]}",
        metadata=metadata
    )

def log_key_rotation(key_type, outcome, user_id=None, message=None, metadata=None):
    """Log key rotation events"""
    if message is None:
        message = f"Key rotation: {outcome} for {key_type}"
    
    return log_security_event(
        event_type=SecurityEventType.KEY_ROTATION,
        message=message,
        outcome=outcome,
        level=logging.INFO if outcome == SecurityOutcome.SUCCESS else logging.WARNING,
        user_id=user_id,
        target_resource=f"key_rotation:{key_type}",
        metadata=metadata
    )

def log_jwt_event(action, outcome, user_id=None, message=None, metadata=None):
    """Log JWT token events"""
    if message is None:
        message = f"JWT {action}: {outcome}" + (f" for user {user_id}" if user_id else "")
    
    return log_security_event(
        event_type=SecurityEventType.JWT,
        message=message,
        outcome=outcome,
        level=logging.INFO if outcome == SecurityOutcome.SUCCESS else logging.WARNING,
        user_id=user_id,
        metadata=metadata
    )

def security_audit(event_type=None):
    """
    Decorator for logging security-sensitive function calls with 
    automatic capturing of function arguments, return values, and exceptions.
    
    Args:
        event_type: Type of security event (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the actual event type or use the function name
            actual_event_type = event_type or f"{func.__module__}.{func.__name__}"
            
            # Build metadata about function call
            metadata = {
                "function": func.__name__,
                "module": func.__module__,
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys()) if kwargs else []
            }
            
            # Log the function call
            event_id = log_security_event(
                event_type=actual_event_type,
                message=f"Executing security-sensitive function: {func.__name__}",
                outcome=SecurityOutcome.INFO,
                level=logging.INFO,
                metadata=metadata
            )
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Log successful execution
                log_security_event(
                    event_type=actual_event_type,
                    message=f"Successfully executed {func.__name__}",
                    outcome=SecurityOutcome.SUCCESS,
                    level=logging.INFO,
                    metadata={
                        "correlation_event_id": event_id,
                        "has_result": result is not None
                    }
                )
                
                return result
                
            except Exception as e:
                # Log the exception
                log_security_event(
                    event_type=actual_event_type,
                    message=f"Exception in {func.__name__}: {str(e)}",
                    outcome=SecurityOutcome.FAILURE,
                    level=logging.ERROR,
                    metadata={
                        "correlation_event_id": event_id,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e)
                    },
                    exception=e
                )
                
                # Re-raise the exception
                raise
                
        return wrapper
    
    return decorator