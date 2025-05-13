"""
Quantonium OS - Flask App Configuration

Initializes the Flask application and SQLAlchemy database connection.
"""

import os
import logging
from flask import Flask, Blueprint, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from werkzeug.security import generate_password_hash
from flask_cors import CORS

# Initialize SQLAlchemy
db = SQLAlchemy()
login_manager = LoginManager()

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Configure database
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SESSION_SECRET", "dev-key-change-in-production")
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # Enable CORS for all routes
    CORS(app)
    
    with app.app_context():
        # Import models
        from models import User, Email, Container
        
        # Create tables
        db.create_all()
        
        # Check if we need to create a default admin account
        if not User.query.filter_by(email="admin@quantonium.os").first():
            try:
                default_admin = User()
                default_admin.email = "admin@quantonium.os"
                default_admin.username = "admin"
                default_admin.password_hash = generate_password_hash("quantum123") 
                db.session.add(default_admin)
                db.session.commit()
                print("Default admin account created")
            except Exception as e:
                print(f"Error creating default admin: {e}")
        
        # Create and register blueprints
        from routes.auth import auth_routes
        auth_bp = Blueprint('auth', __name__, url_prefix='/auth')
        auth_routes(auth_bp)
        app.register_blueprint(auth_bp)
        
        from routes.email import email_routes
        email_bp = Blueprint('email', __name__, url_prefix='/email')
        email_routes(email_bp)
        app.register_blueprint(email_bp)
        
        from routes.resonance import resonance_routes
        resonance_bp = Blueprint('resonance', __name__, url_prefix='/resonance')
        resonance_routes(resonance_bp)
        app.register_blueprint(resonance_bp)
        
        from routes.api import api_routes
        api_bp = Blueprint('api', __name__, url_prefix='/api')
        api_routes(api_bp)
        app.register_blueprint(api_bp)
        
        # Add routes to serve the frontend
        @app.route('/')
        def home():
            return app.send_static_file('index.html')
            
        @app.route('/quantum-browser')
        def quantum_browser():
            return app.send_static_file('quantonium-frontend.html')
            
        @app.route('/quantum-mail')
        def quantum_mail():
            return 'Email Application Coming Soon!'
            
        @app.route('/quantum-grid')
        def quantum_grid():
            return app.send_static_file('quantum_grid/index.html')
            
        @app.route('/qubit-visualizer')
        def qubit_visualizer():
            return app.send_static_file('qubit_ui/index.html')
            
        @app.route('/wave-visualizer')
        def wave_visualizer():
            return app.send_static_file('wave_ui/index.html')
            
        @app.route('/quantum-notes')
        def quantum_notes():
            return 'Notes Application Coming Soon!'
            
        @app.route('/quantum-vault')
        def quantum_vault():
            return 'Secure Vault Application Coming Soon!'
            
        @app.route('/wave-composer')
        def wave_composer():
            return 'Wave Composer Application Coming Soon!'
            
        @app.route('/wave-debugger')
        def wave_debugger():
            return 'Wave Debugger Application Coming Soon!'
            
        @app.route('/settings')
        def settings():
            return 'Settings Application Coming Soon!'
            
        @app.route('/terminal')
        def terminal():
            return 'Terminal Application Coming Soon!'
        
        return app

# Load user for login manager
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))