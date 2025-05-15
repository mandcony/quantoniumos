"""
Simplified entry point for QuantoniumOS deployment
This makes sure the app is correctly initialized for gunicorn
"""

import os
import time
import logging
from flask_cors import CORS
from main import create_app

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantonium_app")

# Create the Flask application
app = create_app()

# Add CORS headers to fix iframe loading in deployment
CORS(app, resources={r"/*": {"origins": "*"}})

# Add special headers for all responses to fix iframe issues
@app.after_request
def add_headers(response):
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

if __name__ == "__main__":
    # Only used for direct execution, not for gunicorn
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting QuantoniumOS on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)