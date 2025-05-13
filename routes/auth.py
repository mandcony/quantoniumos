"""
Quantonium OS - Authentication Routes

Handles user registration, login, logout and token management.
"""

from flask import request, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
import uuid

from app import db
from models import User

def auth_routes(bp):
    @bp.route('/register', methods=['POST'])
    def register():
        data = request.json
        
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"error": "Missing email or password"}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({"error": "Email already registered"}), 409
        
        # Generate username if not provided
        username = data.get('username', data['email'].split('@')[0])
        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            username = f"{username}_{uuid.uuid4().hex[:6]}"
        
        try:
            # Create new user
            new_user = User()
            new_user.email = data['email']
            new_user.username = username
            new_user.password_hash = generate_password_hash(data['password'])
            
            db.session.add(new_user)
            db.session.commit()
            
            return jsonify({"message": "User registered successfully"}), 201
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Registration failed: {str(e)}"}), 500

    @bp.route('/login', methods=['POST'])
    def login():
        data = request.json
        
        if not data or not data.get('email') or not data.get('password'):
            return jsonify({"error": "Missing email or password"}), 400
        
        # Find user
        user = User.query.filter_by(email=data['email']).first()
        
        if not user or not check_password_hash(user.password_hash, data['password']):
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Login user with Flask-Login
        login_user(user)
        
        # Generate JWT token
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, current_app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username
            }
        }), 200

    @bp.route('/logout', methods=['POST'])
    @login_required
    def logout():
        logout_user()
        return jsonify({"message": "Logout successful"}), 200

    @bp.route('/profile', methods=['GET'])
    @login_required
    def profile():
        return jsonify({
            "id": current_user.id,
            "email": current_user.email,
            "username": current_user.username
        }), 200