"""
Quantonium OS - Flask App Entrypoint

Initializes the Flask app and registers symbolic API routes.
"""

import os
from flask import Flask, send_from_directory, redirect, jsonify
from flask_cors import CORS
from routes import api

def create_app():
    app = Flask(__name__, static_folder='static')
    
    # Enable CORS for all routes to allow cross-origin requests
    # This allows embedding in Squarespace
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    app.config["JSON_SORT_KEYS"] = False  # Maintain insertion order
    # Register the API blueprint with URL prefix
    # This separates API routes from static content routes
    app.register_blueprint(api, url_prefix='/api')
    
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
    
    # Simplified Resonance Encryption page (for Squarespace embedding)
    @app.route('/resonance-encrypt')
    def resonance_encrypt():
        return send_from_directory('static', 'resonance-encrypt.html')
        
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

# ✅ Replit manual run fallback
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
