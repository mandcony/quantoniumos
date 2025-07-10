#!/usr/bin/env python3
"""
Database initialization script for CI/CD environments
Creates all necessary tables and sets up initial data
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from auth.models import APIKey, db
from main import create_app


def init_database():
    """Initialize the database with all tables"""

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing database...")

    # Create Flask app
    app = create_app()

    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            logger.info("Database tables created successfully")

            # Check if we have any API keys
            api_key_count = APIKey.query.count()
            logger.info(f"Found {api_key_count} existing API keys")

            # Create a test API key if none exist
            if api_key_count == 0:
                logger.info("Creating test API key for CI/CD...")
                test_key, raw_key = APIKey.create(
                    name="CI/CD Test Key",
                    description="Automatically created test key for CI/CD",
                    permissions="api:read api:write admin:read",
                    is_admin=True,
                )
                logger.info(f"Created test API key with prefix: {test_key.key_prefix}")

                # Save the key to environment or file for CI/CD use
                env_file = project_root / ".env.test"
                with open(env_file, "w") as f:
                    f.write(f"QUANTONIUM_API_KEY={raw_key}\n")
                    f.write(f"TEST_API_KEY_ID={test_key.key_id}\n")
                logger.info(f"Test credentials saved to {env_file}")

            logger.info("Database initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False


if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
