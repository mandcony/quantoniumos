"""
Configuration module for QuantoniumOS Flask application.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FlaskConfig:
    """Flask application configuration"""
    SECRET_KEY: str = "quantonium-development-key-change-in-production"
    DEBUG: bool = True
    TESTING: bool = False
    
    # Server configuration
    HOST: str = "127.0.0.1"
    PORT: int = 5000
    
    # RFT Configuration
    RFT_ENGINE: str = "canonical"  # canonical, cpp, python
    RFT_CACHE_SIZE: int = 1000
    
    # Security settings
    SESSION_COOKIE_SECURE: bool = False  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    
    # API Configuration
    API_VERSION: str = "v1"
    MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB max file upload
    
    # Development settings
    DEVELOPMENT_MODE: bool = True
    LOG_LEVEL: str = "INFO"

# Global configuration instance
config = FlaskConfig()

def get_config() -> FlaskConfig:
    """Get the current configuration instance"""
    return config

def update_config(**kwargs) -> None:
    """Update configuration values"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: Unknown configuration key: {key}")

def load_from_env() -> None:
    """Load configuration from environment variables"""
    global config
    
    # Override with environment variables if they exist
    if os.getenv("FLASK_SECRET_KEY"):
        config.SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    
    if os.getenv("FLASK_DEBUG"):
        config.DEBUG = os.getenv("FLASK_DEBUG").lower() == "true"
    
    if os.getenv("FLASK_HOST"):
        config.HOST = os.getenv("FLASK_HOST")
    
    if os.getenv("FLASK_PORT"):
        try:
            config.PORT = int(os.getenv("FLASK_PORT"))
        except ValueError:
            print("Warning: Invalid FLASK_PORT value, using default")
    
    if os.getenv("RFT_ENGINE"):
        config.RFT_ENGINE = os.getenv("RFT_ENGINE")

# Load environment configuration on import
load_from_env()

# Export commonly used values
DEBUG = config.DEBUG
SECRET_KEY = config.SECRET_KEY
HOST = config.HOST
PORT = config.PORT
