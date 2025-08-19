"""
Security logging utilities for QuantoniumOS Flask application
"""

import logging
from typing import Dict, Any, Optional
from .json_logger import setup_json_logger

def setup_security_logger(name_or_app=None, level: int = logging.WARNING, log_dir: str = None, log_level: int = None) -> logging.Logger:
    """Set up security-specific logger"""
    
    # Handle both Flask app and string name
    if hasattr(name_or_app, 'logger'):  # Flask app
        name = f"{name_or_app.logger.name}.security"
        actual_level = log_level or level
    else:
        name = name_or_app or 'quantonium.security'
        actual_level = log_level or level
    
    return setup_json_logger(name, actual_level)

def log_security_event(logger: logging.Logger, event_type: str, 
                      severity: str, details: Dict[str, Any], 
                      user_id: Optional[str] = None, ip_address: Optional[str] = None):
    """Log security-related events"""
    logger.warning(
        f"Security Event: {event_type}",
        extra={
            'event_type': 'security_event',
            'security_event_type': event_type,
            'severity': severity,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details
        }
    )

def log_auth_attempt(logger: logging.Logger, username: str, success: bool, 
                    ip_address: Optional[str] = None, user_agent: Optional[str] = None):
    """Log authentication attempts"""
    event_type = 'auth_success' if success else 'auth_failure'
    logger.info(
        f"Authentication {'successful' if success else 'failed'} for user: {username}",
        extra={
            'event_type': event_type,
            'username': username,
            'success': success,
            'ip_address': ip_address,
            'user_agent': user_agent
        }
    )

def log_access_denied(logger: logging.Logger, resource: str, user_id: Optional[str] = None,
                     reason: str = 'insufficient_permissions', ip_address: Optional[str] = None):
    """Log access denied events"""
    logger.warning(
        f"Access denied to resource: {resource}",
        extra={
            'event_type': 'access_denied',
            'resource': resource,
            'user_id': user_id,
            'reason': reason,
            'ip_address': ip_address
        }
    )

def log_suspicious_activity(logger: logging.Logger, activity_type: str, 
                          details: Dict[str, Any], user_id: Optional[str] = None,
                          ip_address: Optional[str] = None):
    """Log suspicious activities"""
    logger.warning(
        f"Suspicious activity detected: {activity_type}",
        extra={
            'event_type': 'suspicious_activity',
            'activity_type': activity_type,
            'user_id': user_id,
            'ip_address': ip_address,
            'details': details
        }
    )

def log_rate_limit_exceeded(logger: logging.Logger, endpoint: str, 
                           ip_address: str, attempts: int, window: int):
    """Log rate limit violations"""
    logger.warning(
        f"Rate limit exceeded for endpoint {endpoint}",
        extra={
            'event_type': 'rate_limit_exceeded',
            'endpoint': endpoint,
            'ip_address': ip_address,
            'attempts': attempts,
            'window_seconds': window
        }
    )

# Default security logger
security_logger = setup_security_logger()
