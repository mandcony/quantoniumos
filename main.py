"""
Quantonium OS - Flask App Entrypoint

Initializes the Flask app and registers symbolic API routes with security middleware.
"""

import os
import time
import logging
import platform
from datetime import datetime
from flask import Flask, send_from_directory, redirect, jsonify, g, request
from routes import api
from auth.routes import auth_api
from security import configure_security
from utils.json_logger import setup_json_logger
from auth.models import db, APIKey, APIKeyAuditLog

# Configure global logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantonium_app")

# Track application start time for uptime reporting
APP_START_TIME = time.time()
APP_VERSION = "1.1.0"  # Version tracking for API

def create_app():
    app = Flask(__name__, static_folder='static')
    
    # Configure PostgreSQL database
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    
    # Initialize database
    db.init_app(app)
    
    # Configure security middleware (replaces permissive CORS)
    talisman, limiter = configure_security(app)
    
    app.config["JSON_SORT_KEYS"] = False  # Maintain insertion order
    app.config["DEBUG"] = False  # Disable debug mode for security
    
    # Disable strict trailing slashes to prevent redirect loops
    app.url_map.strict_slashes = False
    
    # Set up structured JSON logging
    setup_json_logger(app, log_dir="logs", log_level=logging.INFO)
    
    # Create tables on startup
    with app.app_context():
        db.create_all()
    
    # Register the API blueprints with URL prefix
    # This separates API routes from static content routes
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(auth_api, url_prefix='/api/auth')
    
    # API health check endpoint (not rate limited)
    @app.route('/api/health')
    @limiter.exempt
    def health_check():
        uptime_seconds = int(time.time() - APP_START_TIME)
        return jsonify({
            "status": "ok",
            "version": APP_VERSION,
            "uptime_s": uptime_seconds,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "host": platform.node()
        })
    
    # Serve embed widget
    @app.route('/embed')
    def serve_widget():
        return send_from_directory('static', 'quantonium-widget.html')
    
    # Route to help users find the embed page
    @app.route('/widget')
    def widget_redirect():
        return redirect('/embed')
        
    # Serve embed demo instructions
    @app.route('/embed-demo')
    def embed_demo():
        return send_from_directory('static', 'embed-demo.html')
    
    # Simplified Resonance Encryption page 
    @app.route('/resonance-encrypt')
    def resonance_encrypt():
        return send_from_directory('static', 'resonance-encrypt.html')
        
    # Comprehensive frontend for Squarespace embedding
    @app.route('/frontend')
    def frontend():
        return send_from_directory('static', 'quantonium-frontend.html')
        
    # Root route redirects to demo
    @app.route('/')
    def root():
        return redirect('/embed-demo')
        
    # Status endpoint without auth
    @app.route('/status')
    def status():
        uptime_seconds = int(time.time() - APP_START_TIME)
        return jsonify({
            "name": "Quantonium OS Cloud Runtime", 
            "status": "operational", 
            "version": APP_VERSION,
            "uptime_s": uptime_seconds
        })
    
    # Metrics endpoint - protected by API key
    @app.route('/api/metrics')
    def metrics():
        import os
        import psutil
        
        # Get process information
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get rate limiter stats (if available from the limiter implementation)
        limiter_stats = {}
        if hasattr(limiter, 'limiter'):
            limiter_stats = {
                "limit": "60 per minute",
                "endpoints": len(limiter.limiter.get_application_limits()),
                "exempt_routes": len(limiter.exempt_routes)
            }
        
        # Build metrics response
        metrics_data = {
            "process": {
                "memory_rss_bytes": memory_info.rss,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()),
                "threads": process.num_threads()
            },
            "uptime_s": int(time.time() - APP_START_TIME),
            "rate_limiter": limiter_stats,
            "version": APP_VERSION
        }
        
        return jsonify(metrics_data)
    
    return app

# ✅ Gunicorn uses this
app = create_app()

# ✅ Development mode only - disabled in production
if __name__ == "__main__":
    # Note: This is for development only, should use gunicorn in production
    logger.warning("Running in development mode. Use gunicorn for production.")
    app.run(host="0.0.0.0", port=8080)
