"""
QuantoniumOS Security Logger

Security-focused logging utilities for QuantoniumOS.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from .json_logger import JSONFormatter

class SecurityFormatter(JSONFormatter):
    """Security-focused JSON formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format security log record as JSON"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'security_category': getattr(record, 'security_category', 'general'),
            'severity': getattr(record, 'severity', 'medium'),
            'source_ip': getattr(record, 'source_ip', 'unknown'),
            'user_id': getattr(record, 'user_id', 'anonymous'),
            'event_type': getattr(record, 'event_type', 'security_event')
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra security fields
        for key, value in record.__dict__.items():
            if key.startswith('security_') or key in ['threat_level', 'action_taken', 'affected_resource']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

def setup_security_logger(name: str = 'quantonium_security', level: str = 'WARNING') -> logging.Logger:
    """Setup security logger for QuantoniumOS"""
    
    # Create security logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with security formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(SecurityFormatter())
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def log_authentication_attempt(logger: logging.Logger, success: bool, username: str, ip: str):
    """Log authentication attempt"""
    logger.warning("Authentication attempt", extra={
        'security_category': 'authentication',
        'event_type': 'auth_attempt',
        'success': success,
        'username': username,
        'source_ip': ip,
        'severity': 'low' if success else 'medium'
    })

def log_authorization_failure(logger: logging.Logger, resource: str, user_id: str, ip: str):
    """Log authorization failure"""
    logger.error("Authorization failure", extra={
        'security_category': 'authorization',
        'event_type': 'auth_failure',
        'affected_resource': resource,
        'user_id': user_id,
        'source_ip': ip,
        'severity': 'high'
    })

def log_suspicious_activity(logger: logging.Logger, activity: str, details: Dict[str, Any]):
    """Log suspicious activity"""
    logger.critical("Suspicious activity detected", extra={
        'security_category': 'threat_detection',
        'event_type': 'suspicious_activity',
        'activity_type': activity,
        'security_details': details,
        'severity': 'critical',
        'action_taken': 'monitoring'
    })

def log_data_access(logger: logging.Logger, resource: str, user_id: str, operation: str):
    """Log data access"""
    logger.info("Data access", extra={
        'security_category': 'data_access',
        'event_type': 'data_operation',
        'affected_resource': resource,
        'user_id': user_id,
        'operation': operation,
        'severity': 'low'
    })

def log_security_configuration_change(logger: logging.Logger, change: str, user_id: str):
    """Log security configuration change"""
    logger.warning("Security configuration change", extra={
        'security_category': 'configuration',
        'event_type': 'config_change',
        'change_description': change,
        'user_id': user_id,
        'severity': 'high'
    })
