"""
Q-Mail Integration Module

This module integrates the Q-Mail application with the QuantoniumOS quantum engines.
It provides secure email encryption using resonance-based cryptography.
"""

import os
import sys
import base64
import logging
import json
from typing import List, Dict, Any, Optional

# Configure path to include the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the quantum adapter helper
try:
    from utils.quantum_adapter_helper import encrypt_data, decrypt_data, generate_quantum_entropy
    HAS_QUANTUM_ADAPTER = True
    logging.info("Successfully imported quantum adapter for q_mail")
except ImportError:
    HAS_QUANTUM_ADAPTER = False
    logging.warning("Could not import quantum adapter, falling back to local encryption")
    # Import local alternatives if quantum adapter is not available
    from encryption.resonance_encryption import resonance_encrypt as encrypt_data
    from encryption.resonance_encryption import resonance_decrypt as decrypt_data
    from encryption.resonance_encryption import generate_entropy


class SecureMailHandler:
    """
    Handler for secure mail operations using quantum-inspired encryption.
    """
    
    def __init__(self, user_key: Optional[str] = None):
        """
        Initialize the secure mail handler.
        
        Args:
            user_key: Optional user encryption key
        """
        self.user_key = user_key or os.environ.get("QMAIL_KEY", "default-symbolic-key")
        
    def encrypt_mail(self, mail_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Encrypt a mail message using resonance-based encryption.
        
        Args:
            mail_content: Dictionary with mail content including subject, body, attachments
            
        Returns:
            Dictionary with encrypted mail content
        """
        try:
            # Serialize mail content to JSON
            mail_json = json.dumps(mail_content)
            
            # Encrypt the JSON string
            encrypted_content = encrypt_data(mail_json, self.user_key)
            
            # Return as a secure mail package
            return {
                "encrypted": True,
                "content": encrypted_content,
                "metadata": {
                    "encryption": "resonance",
                    "timestamp": mail_content.get("timestamp"),
                    "sender": mail_content.get("sender"),
                    "recipient": mail_content.get("recipient")
                }
            }
        except Exception as e:
            logging.error(f"Mail encryption error: {str(e)}")
            return {
                "encrypted": False,
                "error": f"Encryption failed: {str(e)}",
                "content": mail_content
            }
            
    def decrypt_mail(self, encrypted_mail: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt a mail message using resonance-based encryption.
        
        Args:
            encrypted_mail: Dictionary with encrypted mail content
            
        Returns:
            Dictionary with decrypted mail content
        """
        try:
            # Check if mail is actually encrypted
            if not encrypted_mail.get("encrypted", False):
                return encrypted_mail.get("content", {})
                
            # Get encrypted content
            encrypted_content = encrypted_mail.get("content", "")
            
            # Decrypt the content
            decrypted_json = decrypt_data(encrypted_content, self.user_key)
            
            # Parse the JSON string
            decrypted_mail = json.loads(decrypted_json)
            
            return decrypted_mail
        except Exception as e:
            logging.error(f"Mail decryption error: {str(e)}")
            return {
                "error": f"Decryption failed: {str(e)}",
                "subject": "Encrypted Message (Decryption Failed)",
                "body": "The encrypted message could not be decrypted."
            }
            
    def generate_mail_key(self) -> str:
        """
        Generate a new secure key for mail encryption.
        
        Returns:
            Base64-encoded key string
        """
        if HAS_QUANTUM_ADAPTER:
            # Use quantum entropy for key generation
            return generate_quantum_entropy(32)
        else:
            # Fallback to local entropy generation
            entropy = generate_entropy(32)
            return base64.b64encode(entropy).decode('utf-8')


# Create a singleton instance
mail_handler = SecureMailHandler()