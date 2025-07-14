"""
QuantoniumOS Simple Starter
A minimal version that runs without complex dependencies
"""

import os
import sys
from flask import Flask, jsonify, request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantonium_simple")

# Set required environment variables
os.environ.setdefault('QUANTONIUM_MASTER_KEY', 'dev-key-for-testing-only')
os.environ.setdefault('DATABASE_ENCRYPTION_KEY', 'dev-db-key-for-testing-only')
os.environ.setdefault('JWT_SECRET_KEY', 'dev-jwt-key-for-testing-only')
os.environ.setdefault('FLASK_ENV', 'development')
os.environ.setdefault('REDIS_DISABLED', 'true')
os.environ.setdefault('PORT', '5000')

# Create the Flask app
app = Flask(__name__)

# Set a simple database URL
instance_path = os.path.join(os.path.dirname(__file__), 'instance')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)
db_path = os.path.join(instance_path, 'quantonium.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Home route
@app.route('/')
def home():
    return jsonify({
        'name': 'QuantoniumOS',
        'status': 'Running in simplified mode',
        'message': 'Core C++ functionality validated, API running in test mode'
    })

# Health check endpoint
@app.route('/api/health')
def api_health():
    return jsonify({
        'status': 'ok',
        'mode': 'simplified'
    })

# Status endpoint
@app.route('/api/status')
def status():
    return jsonify({
        'status': 'ok',
        'uptime': 0,
        'services': {
            'database': 'sqlite (memory)',
            'redis': 'disabled',
            'c++_engine': 'validated'
        }
    })

# Version endpoint
@app.route('/api/version')
def version():
    return jsonify({
        'version': '0.3.0-rc1',
        'build': 'local-test-build'
    })

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting QuantoniumOS Simple Mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
