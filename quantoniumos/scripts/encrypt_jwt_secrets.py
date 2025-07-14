#!/usr/bin/env python3
"""
Quantonium OS - JWT Secret Encryption Migration Script

This script encrypts all existing JWT secrets in the database.
Run this script once after deploying the encrypted secrets feature.
"""

import os
import sys
import logging
from datetime import datetime
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask
from auth.models import db, APIKey, APIKeyAuditLog
from auth.secret_manager import encrypt_secret, decrypt_secret

# Create a minimal Flask app context
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

@contextmanager
def app_context():
    """Provide a Flask application context."""
    with app.app_context():
        yield

def encrypt_api_key_secrets():
    """
    Encrypt all API key JWT secrets in the database.
    Only encrypts secrets that aren't already encrypted.
    """
    with app_context():
        # Get all active API keys
        keys = APIKey.query.all()
        logger.info(f"Found {len(keys)} API keys to check")
        
        encrypted_count = 0
        skipped_count = 0
        error_count = 0
        
        for key in keys:
            try:
                # Try to decrypt the secret first
                decrypted = decrypt_secret(key.jwt_secret)
                
                if decrypted:
                    # Already encrypted (since we could decrypt it)
                    logger.debug(f"Key {key.key_id} already has an encrypted secret, skipping")
                    skipped_count += 1
                    continue
                
                # Not encrypted yet
                encrypted = encrypt_secret(key.jwt_secret)
                
                # Update the key
                key.jwt_secret = encrypted
                
                # Log the action
                APIKeyAuditLog.log(
                    key,
                    'secret_encrypted',
                    details=f"JWT secret encrypted during migration at {datetime.utcnow().isoformat()}"
                )
                
                encrypted_count += 1
                logger.info(f"Encrypted JWT secret for key {key.key_id}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error encrypting secret for key {key.key_id}: {str(e)}")
        
        # Commit all changes
        db.session.commit()
        
        return {
            'total': len(keys),
            'encrypted': encrypted_count,
            'skipped': skipped_count,
            'errors': error_count
        }

def main():
    """
    Run the JWT secret encryption migration.
    """
    logger.info("Starting JWT secret encryption migration")
    
    try:
        results = encrypt_api_key_secrets()
        
        logger.info("Migration completed successfully")
        logger.info(f"Total keys: {results['total']}")
        logger.info(f"Newly encrypted: {results['encrypted']}")
        logger.info(f"Already encrypted: {results['skipped']}")
        logger.info(f"Errors: {results['errors']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())