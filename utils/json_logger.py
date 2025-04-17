"""
Quantonium OS - JSON Logger Module

Implements structured JSON logging with timed rotation and custom formatting.
"""

import os
import json
import time
import logging
import hashlib
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from flask import request, g

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log messages as JSON objects
    with standardized fields for API request auditing.
    """
    
    def format(self, record):
        """
        Format the log record as a JSON object with standardized fields.
        """
        # Create the base JSON structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "epoch_ms": int(time.time() * 1000),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        
        # Add request information if available
        if hasattr(record, 'request_path'):
            log_data['route'] = record.request_path
        
        if hasattr(record, 'status_code'):
            log_data['status'] = record.status_code
        
        if hasattr(record, 'elapsed_ms'):
            log_data['elapsed_ms'] = record.elapsed_ms
        
        if hasattr(record, 'request_ip'):
            log_data['ip'] = record.request_ip
        
        if hasattr(record, 'api_key_prefix'):
            log_data['api_key_prefix'] = record.api_key_prefix
        
        if hasattr(record, 'sha256_body') and record.sha256_body:
            log_data['sha256_body'] = record.sha256_body
        
        # Add exception info if available
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        return json.dumps(log_data)

def setup_json_logger(app, log_dir="logs", log_level=logging.INFO):
    """
    Set up JSON logging with TimedRotatingFileHandler.
    
    Args:
        app: Flask application instance
        log_dir: Directory to store log files
        log_level: Logging level to use
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure the JSON formatter
    json_formatter = JSONFormatter()
    
    # Set up timed rotating file handler (rotate daily, keep 14 days)
    log_file_path = os.path.join(log_dir, 'quantonium_api.log')
    file_handler = TimedRotatingFileHandler(
        log_file_path,
        when='midnight',
        interval=1,
        backupCount=14
    )
    
    # Create the log file to ensure it exists
    with open(log_file_path, 'a'):
        pass
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(log_level)
    
    # Add the handler to the application logger
    app.logger.addHandler(file_handler)
    app.logger.setLevel(log_level)
    
    # Set up request logging middleware
    @app.before_request
    def before_request():
        # Store start time for performance tracking
        g.start_time = time.perf_counter()
    
    @app.after_request
    def after_request(response):
        # Skip logging for certain paths if needed
        if request.path == '/api/health':
            return response
        
        # Calculate request duration
        if hasattr(g, 'start_time'):
            elapsed_ms = int((time.perf_counter() - g.start_time) * 1000)
        else:
            elapsed_ms = 0
        
        # Hash request body if present
        sha256_body = None
        if request.data:
            sha256_body = hashlib.sha256(request.data).hexdigest()
        
        # Extract API key prefix (first 8 chars) if present
        api_key_prefix = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer ') and len(auth_header) > 15:
            # Extract first 8 chars of the token (safe to log)
            api_key_prefix = auth_header[7:15] + "..."
        
        # Create a new LogRecord with request information
        record = app.logger.makeRecord(
            name=app.logger.name,
            level=logging.INFO,
            fn='flask_request',
            lno=0,
            msg=f"Request: {request.method} {request.path}",
            args=(),
            exc_info=None
        )
        
        # Add request information to the LogRecord
        record.request_path = request.path
        record.status_code = response.status_code
        record.elapsed_ms = elapsed_ms
        record.request_ip = request.remote_addr
        record.api_key_prefix = api_key_prefix
        record.sha256_body = sha256_body
        
        # Log the record
        app.logger.handle(record)
        
        # Add timing header to response
        response.headers['X-Request-Time'] = str(elapsed_ms)
        
        return response
    
    return file_handler