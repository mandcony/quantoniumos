# env_loader.py
"""
QuantoniumOS Environment Loader

This module loads environment variables from .env file and provides
fallbacks for required configuration variables.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("quantonium_env")

def load_dotenv() -> Dict[str, str]:
    """
    Load variables from .env file if it exists
    """
    env_vars = {}
    env_file = Path(__file__).parent / '.env'
    
    if env_file.exists():
        logger.info(f"Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Skip lines that don't have an equals sign
                if '=' not in line:
                    logger.warning(f"Skipping invalid line in .env file: {line}")
                    continue
                    
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
                
                # Don't set these in the environment if they're already set
                if key not in os.environ:
                    os.environ[key] = value
                    
        logger.info(f"Loaded {len(env_vars)} variables from .env file")
    else:
        logger.warning(".env file not found, using existing environment variables")
        
    return env_vars

def get_database_uri() -> str:
    """
    Get database URI with fallback to SQLite
    """
    database_url = os.environ.get("SQLALCHEMY_DATABASE_URI")
    if not database_url:
        # Try DATABASE_URL for compatibility with some hosting platforms
        database_url = os.environ.get("DATABASE_URL")
        
    if not database_url:
        logger.warning("No database URI found. Falling back to SQLite database.")
        # Use the instance folder for SQLite database
        instance_path = Path(__file__).parent / 'instance'
        if not instance_path.exists():
            instance_path.mkdir(exist_ok=True)
        database_url = f"sqlite:///{instance_path}/quantonium.db"
        
    # If the URL starts with postgres:// (older format), convert to postgresql://
    if database_url and database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
        
    return database_url

def configure_environment():
    """
    Configure and validate the environment
    """
    # Load environment variables from .env if present
    env_vars = load_dotenv()
    
    # Configure database
    db_uri = get_database_uri()
    os.environ['SQLALCHEMY_DATABASE_URI'] = db_uri
    
    # Set default values for required variables
    if 'JWT_SECRET_KEY' not in os.environ:
        logger.warning("JWT_SECRET_KEY not set, using a random secret (tokens will be invalidated on restart)")
        import secrets
        os.environ['JWT_SECRET_KEY'] = secrets.token_urlsafe(32)
    
    # Configure Redis
    if 'REDIS_DISABLED' not in os.environ:
        os.environ['REDIS_DISABLED'] = 'true'
        
    return {
        'database_uri': db_uri,
        'jwt_secret': os.environ.get('JWT_SECRET_KEY'),
        'redis_disabled': os.environ.get('REDIS_DISABLED') == 'true',
        'master_key': os.environ.get('QUANTONIUM_MASTER_KEY'),
    }

# When imported, automatically configure the environment
config = configure_environment()
