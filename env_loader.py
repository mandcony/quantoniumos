"""
QuantoniumOS - Environment Configuration Loader
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
dotenv_path = Path('.env')
if dotenv_path.exists():
    logger.info("Loading environment from .env file")
    load_dotenv(dotenv_path)
else:
    logger.warning(".env file not found, using system environment variables")

# Default configuration with fallbacks
config = {
    'database_uri': os.environ.get('DATABASE_URL', 'sqlite:///quantonium_dev.db'),
    'secret_key': os.environ.get('QUANTONIUM_SECRET_KEY', 'dev_secret_key'),
    'log_level': os.environ.get('LOG_LEVEL', 'INFO'),
    'debug': os.environ.get('QUANTONIUM_DEBUG', 'false').lower() in ('true', '1', 'yes')
}
