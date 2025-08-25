"""
QuantoniumOS JSON Logger

JSON-formatted logging utilities for QuantoniumOS.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

class JSONFormatter(logging.Formatter):
    """JSON formatter for log records"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

def setup_json_logger(name: str = 'quantonium', level: str = 'INFO') -> logging.Logger:
    """Setup JSON logger for QuantoniumOS"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger

def log_request(logger: logging.Logger, request_data: Dict[str, Any]):
    """Log HTTP request details"""
    logger.info("HTTP Request", extra={
        'event_type': 'http_request',
        'method': request_data.get('method'),
        'path': request_data.get('path'),
        'user_agent': request_data.get('user_agent'),
        'ip_address': request_data.get('ip_address'),
        'content_length': request_data.get('content_length')
    })

def log_response(logger: logging.Logger, response_data: Dict[str, Any]):
    """Log HTTP response details"""
    logger.info("HTTP Response", extra={
        'event_type': 'http_response',
        'status_code': response_data.get('status_code'),
        'content_length': response_data.get('content_length'),
        'duration_ms': response_data.get('duration_ms')
    })

def log_error(logger: logging.Logger, error_data: Dict[str, Any]):
    """Log error details"""
    logger.error("Error occurred", extra={
        'event_type': 'error',
        'error_type': error_data.get('error_type'),
        'error_message': error_data.get('error_message'),
        'stack_trace': error_data.get('stack_trace'),
        'context': error_data.get('context')
    })

def log_security_event(logger: logging.Logger, event_data: Dict[str, Any]):
    """Log security event"""
    logger.warning("Security Event", extra={
        'event_type': 'security',
        'security_event': event_data.get('event'),
        'ip_address': event_data.get('ip_address'),
        'user_id': event_data.get('user_id'),
        'details': event_data.get('details')
    })

def log_quantum_operation(logger: logging.Logger, operation_data: Dict[str, Any]):
    """Log quantum operation"""
    logger.info("Quantum Operation", extra={
        'event_type': 'quantum_operation',
        'operation': operation_data.get('operation'),
        'input_size': operation_data.get('input_size'),
        'duration_ms': operation_data.get('duration_ms'),
        'success': operation_data.get('success'),
        'error_rate': operation_data.get('error_rate')
    })
