"""
Quantonium OS - Flask App Entrypoint

Initializes the Flask app and registers symbolic API routes.
"""

from flask import Flask
from routes import api

def create_app():
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False  # Maintain insertion order
    app.register_blueprint(api)
    return app

# ✅ Gunicorn uses this
app = create_app()

# ✅ Replit manual run fallback
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
