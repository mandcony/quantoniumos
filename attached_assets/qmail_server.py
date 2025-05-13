import os
import json
import uuid
import sqlite3
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_socketio import SocketIO, emit
from datetime import datetime
import bcrypt
import pyotp
from cryptography.hazmat.primitives import serialization

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "your_secure_random_key_here")
jwt = JWTManager(app)

# Database setup
DB_PATH = r"C:\quantonium_os\apps\quantonium_mail.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                password TEXT,
                totp_secret TEXT,
                public_key TEXT,
                token TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                owner TEXT,
                folder TEXT,
                sender TEXT,
                recipient TEXT,
                subject TEXT,
                body TEXT,
                date TEXT,
                attachment BLOB
            )
        """)
        conn.commit()

init_db()

# Password hashing
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# User Registration
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    public_key = data.get("public_key")

    if not email or not password or not public_key:
        return jsonify({"error": "Missing email, password, or public key"}), 400

    hashed_password = hash_password(password)
    totp_secret = pyotp.random_base32()

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return jsonify({"error": "Email already exists"}), 400
        cursor.execute(
            "INSERT INTO users (email, password, totp_secret, public_key, token) VALUES (?, ?, ?, ?, NULL)",
            (email, hashed_password, totp_secret, public_key)
        )
        conn.commit()

    print(f"User {email} registered. TOTP Secret: {totp_secret}")  # Log to console for now
    return jsonify({"message": "User registered. Check server console for TOTP secret."}), 200

# Login with JWT and 2FA
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    totp_code = data.get("totp_code")

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password, totp_secret FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user or not verify_password(password, user[0]):
            return jsonify({"error": "Invalid credentials"}), 401

        totp = pyotp.TOTP(user[1])
        if not totp.verify(totp_code):
            return jsonify({"error": "Invalid 2FA code"}), 401

        access_token = create_access_token(identity=email)
        cursor.execute("UPDATE users SET token = ? WHERE email = ?", (access_token, email))
        conn.commit()

        return jsonify({"message": "Login successful", "token": access_token}), 200

# Fetch Public Key
@app.route("/get_public_key/<email>", methods=["GET"])
@jwt_required()
def get_public_key(email):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT public_key FROM users WHERE email = ?", (email,))
        result = cursor.fetchone()
        if result:
            return jsonify({"public_key": result[0]}), 200
        return jsonify({"error": "User not found"}), 404

# Fetch Emails
@app.route("/fetch_emails/<owner>", methods=["GET"])
@jwt_required()
def fetch_emails(owner):
    current_user = get_jwt_identity()
    if current_user != owner:
        return jsonify({"error": "Unauthorized access"}), 403

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, folder, sender, subject, body, date FROM emails WHERE owner = ?", (owner,))
        emails = cursor.fetchall()

    results = [{"id": e[0], "folder": e[1], "sender": e[2], "subject": e[3], "body": e[4], "date": e[5]} for e in emails]
    return jsonify(results), 200

# Send Email
@app.route("/send_email", methods=["POST"])
@jwt_required()
def send_email():
    data = request.json
    sender = data.get("sender")
    recipient = data.get("recipient")
    subject = data.get("subject")
    body = data.get("body")
    attachment = data.get("attachment")

    if not sender or not recipient or not subject or not body:
        return jsonify({"error": "Missing fields"}), 400

    timestamp = datetime.now().isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emails (id, owner, folder, sender, recipient, subject, body, date, attachment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), sender, "Sent", sender, recipient, subject, body, timestamp, attachment))
        cursor.execute("""
            INSERT INTO emails (id, owner, folder, sender, recipient, subject, body, date, attachment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), recipient, "Inbox", sender, recipient, subject, body, timestamp, attachment))
        conn.commit()

    socketio.emit("new_email", {"recipient": recipient, "message": "New email received"})
    return jsonify({"message": "Email sent securely"}), 200

if __name__ == "__main__":
    socketio.run(app, host="127.0.0.1", port=5001, debug=True)