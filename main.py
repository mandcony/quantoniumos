"""
Quantonium OS - Flask App Entrypoint

Initializes the Flask app and registers symbolic API routes with security middleware.
"""

import os
import logging
from flask import Flask, send_from_directory, redirect, jsonify
from routes import api
from security import configure_security

# Configure global logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantonium_app")

def create_app():
    app = Flask(__name__, static_folder='static')
    
    # Configure security middleware (replaces permissive CORS)
    talisman, limiter = configure_security(app)
    
    app.config["JSON_SORT_KEYS"] = False  # Maintain insertion order
    app.config["DEBUG"] = False  # Disable debug mode for security
    
    # Register the API blueprint with URL prefix
    # This separates API routes from static content routes
    app.register_blueprint(api, url_prefix='/api')
    
    # API health check endpoint (not rate limited)
    @app.route('/api/health')
    @limiter.exempt
    def health_check():
        return jsonify({"status": "ok"})
    
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
        return jsonify({
            "name": "Quantonium OS Cloud Runtime", 
            "status": "operational", 
            "version": "1.0.0"
        })
    
    return app

# ✅ Gunicorn uses this
app = create_app()

# ✅ Development mode only - disabled in production
if __name__ == "__main__":
    # Note: This is for development only, should use gunicorn in production
    logger.warning("Running in development mode. Use gunicorn for production.")
    app.run(host="0.0.0.0", port=8080)
