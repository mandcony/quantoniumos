#!/usr/bin/env python3
"""
Database migration to add missing columns to api_keys table
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def migrate_database():
    """Migrate the database to match the current model"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting database migration...")
    
    from main import create_app
    from auth.models import db
    
    # Create Flask app
    app = create_app()
    
    with app.app_context():
        try:
            # Get database engine
            engine = db.engine
            
            # Check what tables exist
            inspector = db.inspect(engine)
            tables = inspector.get_table_names()
            logger.info(f"Existing tables: {tables}")
            
            if 'api_keys' in tables:
                # Check existing columns
                columns = [col['name'] for col in inspector.get_columns('api_keys')]
                logger.info(f"Existing api_keys columns: {columns}")
                
                # Add missing columns if they don't exist
                missing_columns = []
                
                expected_columns = {
                    'last_rotated_at': 'DATETIME',
                    'jwt_secrets': 'JSON'
                }
                
                for col_name, col_type in expected_columns.items():
                    if col_name not in columns:
                        missing_columns.append((col_name, col_type))
                
                if missing_columns:
                    logger.info(f"Adding missing columns: {missing_columns}")
                    
                    for col_name, col_type in missing_columns:
                        try:
                            if col_name == 'jwt_secrets':
                                # For JSON column, use TEXT with default empty array
                                sql = f"ALTER TABLE api_keys ADD COLUMN {col_name} TEXT DEFAULT '[]'"
                            else:
                                sql = f"ALTER TABLE api_keys ADD COLUMN {col_name} {col_type}"
                            
                            engine.execute(sql)
                            logger.info(f"Added column {col_name}")
                        except Exception as e:
                            logger.warning(f"Could not add column {col_name}: {e}")
                else:
                    logger.info("All required columns exist")
            else:
                # Create tables from scratch
                logger.info("Creating tables from scratch...")
                db.create_all()
            
            logger.info("Database migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            return False

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)
