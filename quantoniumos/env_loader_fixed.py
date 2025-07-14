# env_loader_fixed.py
"""
QuantoniumOS Environment Loader (Fixed)
This module loads environment variables from .env file and provides
fallbacks for required configuration variables.
"""

import os
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quantonium_env")

def load_env_vars():
    """Load environment variables from .env file"""
    env_file = Path(__file__).parent / '.env'
    env_vars = {}
    
    if not env_file.exists():
        logger.warning(f".env file not found at {env_file}, using defaults")
        return {}
    
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                    
                # Skip lines without equals sign
                if '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key and value:  # Only process if both key and value exist
                    env_vars[key] = value
                    # Set environment variable if not already set
                    if key not in os.environ:
                        os.environ[key] = value
        
        logger.info(f"Loaded {len(env_vars)} variables from .env file")
        return env_vars
    
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        return {}

def setup():
    """Set up environment with reasonable defaults"""
    # Load .env file if it exists
    env_vars = load_env_vars()
    
    # Set database URI
    if 'SQLALCHEMY_DATABASE_URI' not in os.environ:
        db_path = os.path.join(os.path.dirname(__file__), 'instance', 'quantonium.db')
        os.environ['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    
    # Set master key
    if 'QUANTONIUM_MASTER_KEY' not in os.environ:
        os.environ['QUANTONIUM_MASTER_KEY'] = 'dev-key-for-testing-only'
    
    # Set database encryption key
    if 'DATABASE_ENCRYPTION_KEY' not in os.environ:
        os.environ['DATABASE_ENCRYPTION_KEY'] = 'dev-db-key-for-testing-only'
    
    # Set JWT secret key
    if 'JWT_SECRET_KEY' not in os.environ:
        os.environ['JWT_SECRET_KEY'] = 'dev-jwt-key-for-testing-only'
    
    # Disable Redis
    if 'REDIS_DISABLED' not in os.environ:
        os.environ['REDIS_DISABLED'] = 'true'
    
    # Set Flask environment
    if 'FLASK_ENV' not in os.environ:
        os.environ['FLASK_ENV'] = 'development'
    
    # Set port
    if 'PORT' not in os.environ:
        os.environ['PORT'] = '5000'
    
    return {
        'database_uri': os.environ.get('SQLALCHEMY_DATABASE_URI'),
        'master_key': os.environ.get('QUANTONIUM_MASTER_KEY'),
        'db_encryption_key': os.environ.get('DATABASE_ENCRYPTION_KEY'),
        'jwt_secret': os.environ.get('JWT_SECRET_KEY'),
        'redis_disabled': os.environ.get('REDIS_DISABLED'),
        'flask_env': os.environ.get('FLASK_ENV'),
        'port': os.environ.get('PORT')
    }

# When imported, set up the environment
config = setup()

if __name__ == "__main__":
    # When run directly, print the configuration
    print("QuantoniumOS Environment Configuration:")
    for key, value in config.items():
        if 'key' in key or 'secret' in key:
            # Mask sensitive values
            print(f"{key}: {'*' * 10}")
        else:
            print(f"{key}: {value}")
