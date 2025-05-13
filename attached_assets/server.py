# Improved Email Server: server.py

import os
import json
import hashlib
import uuid
import datetime
from flask import Flask, request, jsonify
import jwt
from functools import wraps
from resonance_encryption import resonance_encrypt, resonance_decrypt, WaveNumber

SECRET_KEY = 'your-very-secure-secret-key'

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
USER_CREDENTIALS = os.path.join(BASE_DIR, "users.json")
MAIL_STORE = os.path.join(BASE_DIR, "mail_store.json")

# Ensure files exist
for path, default in [(USER_CREDENTIALS, {}), (MAIL_STORE, [])]:
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(default, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({"error": "Token missing!"}), 401
        try:
            jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired!"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token!"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    email, password = data.get("email"), data.get("password")
    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    with open(USER_CREDENTIALS, "r+") as f:
        users = json.load(f)
        if email in users:
            return jsonify({"error": "User already exists"}), 409
        users[email] = {"password": hash_password(password)}
        f.seek(0)
        json.dump(users, f)
    return jsonify({"message": "User registered successfully"}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email, password = data.get("email"), data.get("password")

    with open(USER_CREDENTIALS, "r") as f:
        users = json.load(f)
        user = users.get(email)
        if user and user["password"] == hash_password(password):
            token = jwt.encode({
                'user': email,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)
            }, SECRET_KEY, algorithm="HS256")
            return jsonify({"message": "Login successful", "token": token}), 200

    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/send_email", methods=["POST"])
@token_required
def send_email():
    data = request.json
    required_fields = ["sender", "recipient", "subject", "body"]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    with open(MAIL_STORE, "r+") as f:
        mail_data = json.load(f)

        encrypted_subject = resonance_encrypt(data["subject"], WaveNumber(1.23, 0.45))
        encrypted_body = resonance_encrypt(data["body"], WaveNumber(1.23, 0.45))

        timestamp = datetime.datetime.now().isoformat()

        # Sender copy
        mail_data.append({
            "id": str(uuid.uuid4()),
            "owner": data["sender"],
            "folder": "Sent",
            "sender": data["sender"],
            "subject": encrypted_subject,
            "body": encrypted_body,
            "date": timestamp
        })

        # Recipient copy
        mail_data.append({
            "id": str(uuid.uuid4()),
            "owner": data["recipient"],
            "folder": "Inbox",
            "sender": data["sender"],
            "subject": encrypted_subject,
            "body": encrypted_body,
            "date": timestamp
        })

        f.seek(0)
        json.dump(mail_data, f)

    return jsonify({"message": "Email sent securely!"}), 200

@app.route("/fetch_emails/<owner>", methods=["GET"])
@token_required
def fetch_emails(owner):
    with open(MAIL_STORE, "r") as f:
        emails = [
            {**email,
             "subject": resonance_decrypt(email["subject"], WaveNumber(1.23, 0.45)),
             "body": resonance_decrypt(email["body"], WaveNumber(1.23, 0.45))}
            for email in json.load(f) if email["owner"] == owner
        ]
    return jsonify(emails), 200

if __name__ == '__main__':
    app.run(debug=True)
