"""
Quantonium OS - Flask App Entrypoint

This is the main entry point for the QuantoniumOS application.
It imports and uses the Flask app defined in app.py
"""

from app import create_app

# Create the Flask application
app = create_app()
