"""
Environment Configuration Loader

Provides configuration management for the QuantoniumOS Flask application.
Handles environment variables, default settings, and configuration validation.
"""

import os
from typing import Dict, Any, Optional

class Config:
    """Configuration class for QuantoniumOS Flask application."""
    
    def __init__(self):
        """Initialize configuration with defaults and environment overrides."""
        
        # Core application settings
        self.SECRET_KEY = os.environ.get('SECRET_KEY', 'quantonium-rft-secret-key-dev-2025')
        self.DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
        self.TESTING = os.environ.get('TESTING', 'False').lower() == 'true'
        
        # Flask settings
        self.HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
        self.PORT = int(os.environ.get('FLASK_PORT', '5000'))
        
        # RFT algorithm settings
        self.RFT_DEFAULT_N = int(os.environ.get('RFT_DEFAULT_N', '64'))
        self.RFT_DEFAULT_M = int(os.environ.get('RFT_DEFAULT_M', '2'))
        self.RFT_DEFAULT_SIGMA0 = float(os.environ.get('RFT_DEFAULT_SIGMA0', '1.0'))
        self.RFT_DEFAULT_GAMMA = float(os.environ.get('RFT_DEFAULT_GAMMA', '0.3'))
        
        # API settings
        self.API_VERSION = os.environ.get('API_VERSION', 'v1')
        self.MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', '16777216'))  # 16MB
        
        # Security settings
        self.CSRF_ENABLED = os.environ.get('CSRF_ENABLED', 'True').lower() == 'true'
        self.WTF_CSRF_ENABLED = self.CSRF_ENABLED
        
        # Database settings (if needed)
        self.DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///quantonium.db')
        
        # Logging settings
        self.LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        self.LOG_FILE = os.environ.get('LOG_FILE', 'quantonium.log')
        
        # Performance settings
        self.ENABLE_C_EXTENSIONS = os.environ.get('ENABLE_C_EXTENSIONS', 'True').lower() == 'true'
        self.CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
        
        # Development settings
        self.DEVELOPMENT_MODE = self.DEBUG
        self.AUTO_RELOAD = os.environ.get('AUTO_RELOAD', str(self.DEBUG)).lower() == 'true'
    
    def get_rft_config(self) -> Dict[str, Any]:
        """Get RFT-specific configuration."""
        return {
            'default_n': self.RFT_DEFAULT_N,
            'default_m': self.RFT_DEFAULT_M,
            'default_sigma0': self.RFT_DEFAULT_SIGMA0,
            'default_gamma': self.RFT_DEFAULT_GAMMA,
            'enable_c_extensions': self.ENABLE_C_EXTENSIONS
        }
    
    def get_flask_config(self) -> Dict[str, Any]:
        """Get Flask-specific configuration."""
        return {
            'SECRET_KEY': self.SECRET_KEY,
            'DEBUG': self.DEBUG,
            'TESTING': self.TESTING,
            'MAX_CONTENT_LENGTH': self.MAX_CONTENT_LENGTH,
            'WTF_CSRF_ENABLED': self.WTF_CSRF_ENABLED
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate required settings
            if not self.SECRET_KEY:
                raise ValueError("SECRET_KEY is required")
            
            if self.PORT < 1 or self.PORT > 65535:
                raise ValueError("PORT must be between 1 and 65535")
            
            if self.RFT_DEFAULT_N < 1:
                raise ValueError("RFT_DEFAULT_N must be positive")
            
            if self.RFT_DEFAULT_M < 1:
                raise ValueError("RFT_DEFAULT_M must be positive")
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False
    
    def __repr__(self):
        """String representation of configuration."""
        return f"<Config DEBUG={self.DEBUG} HOST={self.HOST} PORT={self.PORT}>"
    
    def __getitem__(self, key):
        """Dictionary-style access to configuration values."""
        # Handle special key mappings
        if key == 'database_uri':
            return self.DATABASE_URL
        
        # Try to get the attribute
        if hasattr(self, key):
            return getattr(self, key)
        elif hasattr(self, key.upper()):
            return getattr(self, key.upper())
        else:
            raise KeyError(f"Configuration key '{key}' not found")
    
    def __setitem__(self, key, value):
        """Dictionary-style setting of configuration values."""
        if key == 'database_uri':
            self.DATABASE_URL = value
        elif hasattr(self, key):
            setattr(self, key, value)
        elif hasattr(self, key.upper()):
            setattr(self, key.upper(), value)
        else:
            setattr(self, key.upper(), value)
    
    def get(self, key, default=None):
        """Get configuration value with default."""
        try:
            return self[key]
        except KeyError:
            return default

# Global configuration instance
config = Config()

# Validate configuration on import
if not config.validate():
    print("⚠️  Configuration validation failed. Using defaults.")

# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development configuration."""
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    def __init__(self):
        super().__init__()
        self.DEBUG = False
        self.TESTING = False
        self.SECRET_KEY = os.environ.get('SECRET_KEY')
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY environment variable is required in production")

class TestingConfig(Config):
    """Testing configuration."""
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.TESTING = True
        self.DATABASE_URL = 'sqlite:///:memory:'

# Configuration factory
def get_config(config_name: Optional[str] = None) -> Config:
    """Get configuration based on environment."""
    config_name = config_name or os.environ.get('FLASK_ENV', 'development').lower()
    
    configs = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig
    }
    
    return configs.get(config_name, DevelopmentConfig)()

if __name__ == "__main__":
    print("QuantoniumOS Environment Configuration")
    print("=" * 40)
    print(f"Configuration: {config}")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Host: {config.HOST}")
    print(f"Port: {config.PORT}")
    print(f"RFT Config: {config.get_rft_config()}")
    print(f"Validation: {'✅ PASSED' if config.validate() else '❌ FAILED'}")
