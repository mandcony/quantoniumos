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
        
        # Add a home route
        @app.route('/')
        def home():
            return 'Welcome to Quantonium OS - the cutting-edge quantum computing platform!'
        
        return app

# Load user for login manager
@login_manager.user_loader
def load_user(user_id):
    from models import User
    return User.query.get(int(user_id))