"""
QuantoniumOS Environment Configuration Loader

Loads environment variables and configuration settings for QuantoniumOS.
"""

import os
from typing import Dict, Any, Optional

class Config:
    """Configuration management for QuantoniumOS"""
    
    def __init__(self):
        self._config = {}
        self._load_defaults()
        self._load_from_env()
    
    def _load_defaults(self):
        """Load default configuration values"""
        self._config = {
            'SECRET_KEY': 'quantonium-dev-key-change-in-production',
            'DEBUG': True,
            'HOST': '0.0.0.0',
            'PORT': 5000,
            'DATABASE_URL': 'quantonium.db',
            'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB
            'UPLOAD_FOLDER': 'uploads',
            'ALLOWED_EXTENSIONS': {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'},
            'SESSION_TIMEOUT': 3600,  # 1 hour
            'QUANTUM_ENGINE_ENABLED': True,
            'RFT_PRECISION': 1e-15,
            'CORS_ORIGINS': ['http://localhost:3000', 'http://127.0.0.1:3000'],
            'API_RATE_LIMIT': '100/hour',
            'LOG_LEVEL': 'INFO',
            'LOG_FORMAT': 'json'
        }
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'QUANTONIUM_SECRET_KEY': 'SECRET_KEY',
            'QUANTONIUM_DEBUG': 'DEBUG',
            'QUANTONIUM_HOST': 'HOST',
            'QUANTONIUM_PORT': 'PORT',
            'QUANTONIUM_DATABASE_URL': 'DATABASE_URL',
            'QUANTONIUM_MAX_CONTENT_LENGTH': 'MAX_CONTENT_LENGTH',
            'QUANTONIUM_UPLOAD_FOLDER': 'UPLOAD_FOLDER',
            'QUANTONIUM_SESSION_TIMEOUT': 'SESSION_TIMEOUT',
            'QUANTONIUM_QUANTUM_ENGINE_ENABLED': 'QUANTUM_ENGINE_ENABLED',
            'QUANTONIUM_RFT_PRECISION': 'RFT_PRECISION',
            'QUANTONIUM_CORS_ORIGINS': 'CORS_ORIGINS',
            'QUANTONIUM_API_RATE_LIMIT': 'API_RATE_LIMIT',
            'QUANTONIUM_LOG_LEVEL': 'LOG_LEVEL',
            'QUANTONIUM_LOG_FORMAT': 'LOG_FORMAT'
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if config_key in ['DEBUG', 'QUANTUM_ENGINE_ENABLED']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif config_key in ['PORT', 'MAX_CONTENT_LENGTH', 'SESSION_TIMEOUT']:
                    value = int(value)
                elif config_key == 'RFT_PRECISION':
                    value = float(value)
                elif config_key == 'CORS_ORIGINS':
                    value = [origin.strip() for origin in value.split(',')]
                
                self._config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration with dictionary"""
        self._config.update(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting"""
        self._config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator"""
        return key in self._config

# Global configuration instance
config = Config()

# Environment detection
def is_development() -> bool:
    """Check if running in development mode"""
    return config.get('DEBUG', True)

def is_production() -> bool:
    """Check if running in production mode"""
    return not is_development()

def get_database_url() -> str:
    """Get database URL"""
    return config.get('DATABASE_URL', 'quantonium.db')

def get_secret_key() -> str:
    """Get secret key for Flask sessions"""
    return config.get('SECRET_KEY', 'quantonium-dev-key-change-in-production')

def get_cors_origins() -> list:
    """Get CORS allowed origins"""
    return config.get('CORS_ORIGINS', ['http://localhost:3000'])
