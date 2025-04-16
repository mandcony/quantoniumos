import os
import logging
from flask import Flask
from routes import api_blueprint

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.secret_key = os.environ.get("SESSION_SECRET", "quantonium-dev-key")
    
    # Register blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')
    
    @app.route('/')
    def index():
        return {
            "name": "Quantonium OS Cloud Runtime",
            "version": "1.0.0",
            "status": "operational"
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    # Bind to 0.0.0.0:5000 for external access
    app.run(host="0.0.0.0", port=5000, debug=True)
