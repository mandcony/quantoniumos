"""
Authentication routes for QuantoniumOS Flask application
"""

from flask import Blueprint, jsonify, request, session
import hashlib
import time
import os

# Create auth API blueprint
auth_api = Blueprint('auth', __name__, url_prefix='/auth')

# Simple in-memory user store (replace with database in production)
users_db = {
    "admin": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "created": time.time()
    },
    "user": {
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(), 
        "role": "user",
        "created": time.time()
    }
}

@auth_api.route('/login', methods=['POST'])
def login():
    """User login endpoint"""
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"error": "Missing username or password"}), 400
        
        username = data['username']
        password = data['password']
        
        # Check if user exists
        if username not in users_db:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Verify password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if password_hash != users_db[username]['password_hash']:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Create session
        session['user_id'] = username
        session['role'] = users_db[username]['role']
        session['login_time'] = time.time()
        
        return jsonify({
            "status": "success",
            "message": "Login successful",
            "user": {
                "username": username,
                "role": users_db[username]['role']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_api.route('/logout', methods=['POST'])
def logout():
    """User logout endpoint"""
    try:
        session.clear()
        return jsonify({
            "status": "success",
            "message": "Logout successful"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_api.route('/status', methods=['GET'])
def auth_status():
    """Check authentication status"""
    try:
        if 'user_id' in session:
            return jsonify({
                "authenticated": True,
                "user": {
                    "username": session['user_id'],
                    "role": session.get('role', 'user'),
                    "login_time": session.get('login_time')
                }
            })
        else:
            return jsonify({
                "authenticated": False
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@auth_api.route('/register', methods=['POST'])
def register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({"error": "Missing username or password"}), 400
        
        username = data['username']
        password = data['password']
        role = data.get('role', 'user')
        
        # Check if user already exists
        if username in users_db:
            return jsonify({"error": "User already exists"}), 409
        
        # Validate password strength (basic)
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        # Create new user
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        users_db[username] = {
            "password_hash": password_hash,
            "role": role if role in ['admin', 'user'] else 'user',
            "created": time.time()
        }
        
        return jsonify({
            "status": "success",
            "message": "User registered successfully",
            "user": {
                "username": username,
                "role": users_db[username]['role']
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def require_auth(f):
    """Decorator to require authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def require_admin(f):
    """Decorator to require admin role"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        if session.get('role') != 'admin':
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated_function
