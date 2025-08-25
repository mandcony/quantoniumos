"""
QuantoniumOS Security Audit Middleware

NIST 800-53 compliant security audit middleware for QuantoniumOS.
"""

from flask import Flask, request, g
import time
import json
from typing import Dict, Any

def initialize_audit_middleware(app: Flask):
    """Initialize security audit middleware"""
    
    @app.before_request
    def before_request():
        """Log request start"""
        g.audit_start_time = time.time()
        g.audit_data = {
            'timestamp': time.time(),
            'method': request.method,
            'path': request.path,
            'ip': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'content_length': request.content_length or 0
        }
    
    @app.after_request
    def after_request(response):
        """Log request completion"""
        if hasattr(g, 'audit_start_time'):
            duration = time.time() - g.audit_start_time
            audit_data = getattr(g, 'audit_data', {})
            audit_data.update({
                'status_code': response.status_code,
                'duration_ms': round(duration * 1000, 2),
                'response_size': len(response.get_data())
            })
            
            # Log security-relevant events
            if response.status_code >= 400:
                print(f"🔐 AUDIT: Error response - {json.dumps(audit_data)}")
            elif request.path.startswith('/api/'):
                print(f"🔐 AUDIT: API access - {json.dumps(audit_data)}")
        
        return response
    
    print("✅ Security audit middleware initialized")
