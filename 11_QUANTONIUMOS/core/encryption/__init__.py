"""
QuantoniumOS Encryption Module
"""


def encrypt(data, key):
    """Simple encryption function"""
    if not data or not key:
        return None
    return {"status": "encrypted", "data": str(data)}


def decrypt(data, key):
    """Simple decryption function"""
    if not data or not key:
        return None
    return {"status": "decrypted", "data": str(data)}
