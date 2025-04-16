"""
Quantonium OS - Flask App Entrypoint

Initializes the Flask app and registers symbolic API routes.
"""

import os
from flask import Flask, send_from_directory, redirect
from flask_cors import CORS
from routes import api

def create_app():
    app = Flask(__name__, static_folder='static')
    
    # Enable CORS for all routes to allow cross-origin requests
    # This allows embedding in Squarespace
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    app.config["JSON_SORT_KEYS"] = False  # Maintain insertion order
    app.register_blueprint(api)
    
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
        
    # Root route redirects to demo
    @app.route('/')
    def root():
        return redirect('/embed-demo')
    
    return app

# ✅ Gunicorn uses this
app = create_app()

# ✅ Replit manual run fallback
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
