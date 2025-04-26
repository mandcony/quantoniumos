"""
Quantonium OS - Security Audit Middleware - NIST Compliant

This module implements NIST SP 800-53 AU (Audit and Accountability) compliant
middleware for tracking and logging security-relevant events in the application.
It provides comprehensive logging for security monitoring, forensic analysis,
and compliance reporting.
"""

import logging
import time
import uuid
import json
import traceback
from typing import Dict, List, Any, Optional
from functools import wraps

# Import Flask types if available
try:
    from flask import request, g, current_app, Response
    from flask.wrappers import Request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    # Mock classes for type checking when Flask is not available
    class Request:
        def __init__(self):
            self.path = ""
            self.method = ""
            self.remote_addr = ""
            self.headers = {}
    class Response:
        def __init__(self):
            self.status_code = 200
    request = Request()
    g = type('g', (), {})()
    current_app = type('current_app', (), {'logger': logging.getLogger()})()

# Import the security logger functionality
from utils.security_logger import (
    log_security_event,
    SecurityEventType,
    SecurityOutcome,
    ImpactLevel,
    sanitize_sensitive_data
)

# Configure module logger
logger = logging.getLogger("quantonium_security_audit")

# Constants for security auditing
AUDIT_HEADER = "X-Quantonium-Audit-ID"
SENSITIVE_PARAMS = [
    'password', 'token', 'key', 'secret', 'credential', 
    'auth', 'apikey', 'api_key'
]
HIGH_RISK_ENDPOINTS = [
    '/api/auth/login',
    '/api/auth/register',
    '/api/auth/reset-password',
    '/api/v1/admin',
    '/api/quantum/circuit',
    '/api/container/unlock'
]
RESOURCES_TO_SKIP = [
    '/static/',
    '/api/health',
    '/favicon.ico'
]

class AuditMiddleware:
    """
    WSGI middleware for NIST 800-53 compliant audit logging.
    
    Implements controls from the AU (Audit and Accountability) family:
    - AU-2: Audit Events
    - AU-3: Content of Audit Records
    - AU-4: Audit Storage Capacity
    - AU-5: Response to Audit Processing Failures
    - AU-6: Audit Review, Analysis, and Reporting
    - AU-8: Time Stamps
    - AU-9: Protection of Audit Information
    - AU-12: Audit Generation
    """
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """
        Initialize the middleware with a Flask application.
        
        Args:
            app: Flask application
        """
        self.app = app
        # Register before/after request handlers
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Initialize audit storage
        if not hasattr(app, 'extensions'):
            app.extensions = {}
        app.extensions['audit_middleware'] = self
        
        # Log initialization
        logger.info("NIST 800-53 compliant audit middleware initialized")
    
    def before_request(self):
        """
        Process request before it reaches the view function.
        
        This hooks into the Flask before_request signal to capture
        request details for audit logging.
        """
        if not HAS_FLASK:
            return
        
        # Skip static resources and health checks to reduce log volume
        path = request.path
        if any(path.startswith(skip) for skip in RESOURCES_TO_SKIP):
            return
        
        # Generate or use existing audit ID
        audit_id = request.headers.get(AUDIT_HEADER, '')
        if not audit_id:
            audit_id = str(uuid.uuid4())
        
        # Store audit context in the Flask g object
        g.audit_id = audit_id
        g.audit_start_time = time.time()
        g.audit_path = path
        g.audit_method = request.method
        
        # Determine impact level based on endpoint sensitivity
        impact_level = ImpactLevel.LOW
        for high_risk in HIGH_RISK_ENDPOINTS:
            if path.startswith(high_risk):
                impact_level = ImpactLevel.MODERATE
                break
        
        # Record sensitive headers except auth headers which should be protected
        safe_headers = {}
        for name, value in request.headers.items():
            lower_name = name.lower()
            if lower_name == 'authorization' or lower_name == 'x-api-key' or any(s in lower_name for s in SENSITIVE_PARAMS):
                safe_headers[name] = '[REDACTED]'
            else:
                safe_headers[name] = value
        
        # Create a sanitized version of request parameters
        if request.method in ('POST', 'PUT', 'PATCH') and hasattr(request, 'get_json'):
            try:
                json_data = request.get_json(silent=True)
                if json_data:
                    # Sanitize sensitive fields in the payload
                    safe_request_data = sanitize_sensitive_data(json_data)
                else:
                    # Try to get form data
                    form_data = request.form.to_dict() if hasattr(request, 'form') else {}
                    safe_request_data = sanitize_sensitive_data(form_data)
            except Exception:
                safe_request_data = {'error': 'Unable to parse request payload'}
        else:
            # For GET requests, sanitize the query parameters
            args = request.args.to_dict() if hasattr(request, 'args') else {}
            safe_request_data = sanitize_sensitive_data(args)
        
        # Log request audit event
        log_security_event(
            event_type=SecurityEventType.AUDIT,
            message=f"Request {request.method} {path}",
            outcome=SecurityOutcome.UNKNOWN,  # Outcome not known yet
            target_resource=path,
            metadata={
                'request_type': 'http',
                'audit_id': audit_id,
                'method': request.method,
                'headers': safe_headers,
                'client_ip': request.remote_addr,
                'query_params': request.args.to_dict() if hasattr(request, 'args') else {},
                'request_data': safe_request_data
            },
            security_labels=['http-request', request.method.lower()],
            impact_level=impact_level,
            source_ip=request.remote_addr,
            correlation_id=audit_id
        )
    
    def after_request(self, response):
        """
        Process response before it is sent to the client.
        
        This hooks into the Flask after_request signal to capture
        response details for audit logging.
        
        Args:
            response: Flask response object
            
        Returns:
            Unmodified response
        """
        if not HAS_FLASK:
            return response
        
        # Skip static resources and health checks to reduce log volume
        path = getattr(g, 'audit_path', request.path)
        if any(path.startswith(skip) for skip in RESOURCES_TO_SKIP):
            return response
        
        # Calculate response time
        start_time = getattr(g, 'audit_start_time', time.time())
        response_time = time.time() - start_time
        
        # Get audit context from Flask g object
        audit_id = getattr(g, 'audit_id', str(uuid.uuid4()))
        
        # Determine outcome based on status code
        status_code = response.status_code
        if 200 <= status_code < 300:
            outcome = SecurityOutcome.SUCCESS
        elif 400 <= status_code < 500:
            if status_code == 401:
                outcome = SecurityOutcome.FAILURE
            elif status_code == 403:
                outcome = SecurityOutcome.BLOCKED
            else:
                outcome = SecurityOutcome.WARNING
        elif 500 <= status_code < 600:
            outcome = SecurityOutcome.FAILURE
        else:
            outcome = SecurityOutcome.UNKNOWN
        
        # Determine impact level based on status code and endpoint
        impact_level = ImpactLevel.LOW
        for high_risk in HIGH_RISK_ENDPOINTS:
            if path.startswith(high_risk):
                impact_level = ImpactLevel.MODERATE
                break
        
        # Elevation to HIGH impact for critical errors or security issues
        if status_code in (401, 403):
            impact_level = ImpactLevel.MODERATE
        elif status_code >= 500:
            impact_level = ImpactLevel.HIGH
        
        # Add audit ID to response headers for correlation
        try:
            response.headers[AUDIT_HEADER] = audit_id
        except Exception:
            # Some response types might not support header modification
            pass
        
        # For JSON responses, capture a safe version of the response data
        response_data = None
        if response.mimetype == 'application/json' and hasattr(response, 'get_data'):
            try:
                data = response.get_data(as_text=True)
                if data:
                    parsed = json.loads(data)
                    # Sanitize any sensitive fields in the response
                    response_data = sanitize_sensitive_data(parsed)
            except Exception:
                response_data = {'error': 'Unable to parse response payload'}
        
        # Log response audit event
        log_security_event(
            event_type=SecurityEventType.AUDIT,
            message=f"Response {getattr(g, 'audit_method', request.method)} {path} - {status_code}",
            outcome=outcome,
            target_resource=path,
            metadata={
                'request_type': 'http',
                'audit_id': audit_id,
                'status_code': status_code,
                'response_time_ms': int(response_time * 1000),
                'content_type': response.mimetype,
                'content_length': response.content_length,
                'response_data': response_data
            },
            security_labels=['http-response', f'status-{status_code}'],
            impact_level=impact_level,
            source_ip=request.remote_addr if hasattr(request, 'remote_addr') else None,
            correlation_id=audit_id
        )
        
        return response

def audit_action(action: str, resource_name: str = None):
    """
    Decorator for auditing specific actions within views.
    
    Use this decorator on route functions to add detailed audit
    information about specific actions being performed.
    
    Args:
        action: Name of the action being performed
        resource_name: Optional name of the resource being affected
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not HAS_FLASK:
                return f(*args, **kwargs)
            
            # Extract resource ID from view arguments if available
            resource_id = None
            for arg_name in ['id', 'uuid', 'identifier']:
                if arg_name in kwargs:
                    resource_id = kwargs[arg_name]
                    break
            
            # Determine resource name from function parameters
            resource = resource_name
            if not resource:
                resource = f.__name__
            
            # Add audit context to g for the after_request handler
            g.audit_action = action
            g.audit_resource = resource
            g.audit_resource_id = resource_id
            
            # Log pre-execution audit event
            audit_id = getattr(g, 'audit_id', str(uuid.uuid4()))
            log_security_event(
                event_type=SecurityEventType.AUDIT,
                message=f"Action '{action}' started on {resource}" + 
                         (f" ID {resource_id}" if resource_id else ""),
                outcome=SecurityOutcome.UNKNOWN,
                target_resource=resource,
                metadata={
                    'audit_id': audit_id,
                    'action': action,
                    'resource': resource,
                    'resource_id': resource_id
                },
                security_labels=['action-audit', action],
                correlation_id=audit_id
            )
            
            try:
                # Execute the wrapped function
                result = f(*args, **kwargs)
                
                # Log post-execution audit event on success
                log_security_event(
                    event_type=SecurityEventType.AUDIT,
                    message=f"Action '{action}' completed successfully on {resource}" +
                             (f" ID {resource_id}" if resource_id else ""),
                    outcome=SecurityOutcome.SUCCESS,
                    target_resource=resource,
                    metadata={
                        'audit_id': audit_id,
                        'action': action,
                        'resource': resource,
                        'resource_id': resource_id
                    },
                    security_labels=['action-audit', action, 'success'],
                    correlation_id=audit_id
                )
                
                return result
                
            except Exception as e:
                # Log post-execution audit event on failure
                log_security_event(
                    event_type=SecurityEventType.AUDIT,
                    message=f"Action '{action}' failed on {resource}" +
                             (f" ID {resource_id}" if resource_id else ""),
                    outcome=SecurityOutcome.FAILURE,
                    level=logging.ERROR,
                    target_resource=resource,
                    metadata={
                        'audit_id': audit_id,
                        'action': action,
                        'resource': resource,
                        'resource_id': resource_id,
                        'error': str(e),
                        'error_type': e.__class__.__name__,
                        'traceback': traceback.format_exc()
                    },
                    security_labels=['action-audit', action, 'failure'],
                    impact_level=ImpactLevel.MODERATE,
                    correlation_id=audit_id,
                    include_stack_trace=True
                )
                
                # Re-raise the exception
                raise
        
        return wrapped
    
    return decorator

def initialize_audit_middleware(app):
    """
    Initialize the audit middleware with a Flask application.
    
    This function is the main entry point for setting up the NIST 800-53
    compliant audit middleware in a Flask application.
    
    Args:
        app: Flask application
        
    Returns:
        The initialized middleware instance
    """
    middleware = AuditMiddleware(app)
    return middleware