"""
JSON logging utilities for QuantoniumOS Flask application
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
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
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

def setup_json_logger(name_or_app=None, level: int = logging.INFO, log_dir: str = None, log_level: int = None) -> logging.Logger:
    """Set up JSON logger for the application"""
    
    # Handle both Flask app and string name
    if hasattr(name_or_app, 'logger'):  # Flask app
        name = name_or_app.logger.name
        actual_level = log_level or level
    else:
        name = name_or_app or 'quantonium'
        actual_level = log_level or level
    
    logger = logging.getLogger(name)
    logger.setLevel(actual_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(actual_level)
    console_handler.setFormatter(JSONFormatter())
    
    logger.addHandler(console_handler)
    
    # Add file handler if log_dir is specified
    if log_dir:
        import os
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'quantonium.log'))
        file_handler.setLevel(actual_level)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def log_api_request(logger: logging.Logger, method: str, path: str, 
                   status_code: int, response_time: float, **kwargs):
    """Log API request details"""
    logger.info(
        f"API {method} {path} - {status_code}",
        extra={
            'event_type': 'api_request',
            'http_method': method,
            'path': path,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2),
            **kwargs
        }
    )

def log_rft_operation(logger: logging.Logger, operation: str, 
                     input_size: int, output_size: int, 
                     processing_time: float, **kwargs):
    """Log RFT operation details"""
    logger.info(
        f"RFT {operation} - input:{input_size} output:{output_size}",
        extra={
            'event_type': 'rft_operation',
            'operation': operation,
            'input_size': input_size,
            'output_size': output_size,
            'processing_time_ms': round(processing_time * 1000, 2),
            **kwargs
        }
    )

def log_error(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None):
    """Log error with context"""
    logger.error(
        f"Error: {str(error)}",
        extra={
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {}
        },
        exc_info=True
    )

# Default logger instance
default_logger = setup_json_logger()
