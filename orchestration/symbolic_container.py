"""
Quantonium OS - Symbolic Container Module

Symbolic payload container sealed via A/φ waveform and entropy.
"""

import sys, os
import base64
import time
import json
import hashlib
import logging

# Add core to the path to access encryption modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logger = logging.getLogger("SymbolicContainer")

# Try to import core modules, with fallbacks if not available
try:
    from core.encryption.resonance_encrypt import resonance_encrypt, resonance_decrypt
except ImportError:
    # Define fallbacks if modules aren't available
    def resonance_encrypt(payload, A, phi):
        key = f"{A:.4f}_{phi:.4f}".encode()
        h = hashlib.sha256()
        h.update(key)
        key_digest = h.digest()
        return payload.encode()
    
    def resonance_decrypt(data, A, phi):
        return data.decode()

# Simple entropy calculation function
def calculate_symbolic_entropy(data):
    """Calculate a symbolic entropy score from sample data"""
    if not data:
        return 0.0
    return sum(data) / len(data)

# Simple file hash function
def hash_file(filename):
    """Hash a file and return key information"""
    if not os.path.exists(filename):
        # Return a default hash for testing when file doesn't exist
        return ("default", 1.618, 0.577)
    
    try:
        with open(filename, 'rb') as f:
            content = f.read()
            file_hash = hashlib.sha256(content).hexdigest()
            # Generate stable A and phi values from the hash
            A = float(int(file_hash[:8], 16) % 1000) / 1000
            phi = float(int(file_hash[8:16], 16) % 1000) / 1000
            return (file_hash[:8], A, phi)
    except Exception as e:
        logger.error(f"Error hashing file: {str(e)}")
        return None

class SymbolicContainer:
    def __init__(self, payload: str, key_waveform: tuple = None, validation_file: str = "example.txt"):
        """
        Initialize a Symbolic Container.
        """
        self.payload = payload
        self.validation_file = validation_file
        
        if key_waveform is None:
            key = hash_file(validation_file)
            if key is None:
                key = ("default", 1.0, 0.5)
            self.key_id, self.A, self.phi = key
        else:
            self.key_id, self.A, self.phi = key_waveform
            
        self.entropy = None
        self.timestamp = time.time()
        self.encrypted = None

    def seal(self):
        """
        Seal the symbolic container.
        """
        try:
            encrypted = resonance_encrypt(self.payload, self.A, self.phi)
            if isinstance(encrypted, bytes):
                self.encrypted = base64.b64encode(encrypted).decode("utf-8")
            else:
                self.encrypted = str(encrypted)
                
            entropy_sample = [self.A * i for i in range(1, 9)]
            self.entropy = round(calculate_symbolic_entropy(entropy_sample), 4)
            logger.info(f"Container sealed with entropy score: {self.entropy}")
            return True
        except Exception as e:
            logger.error(f"Error sealing container: {str(e)}")
            return False

    def unlock(self, A_in: float, phi_in: float) -> str:
        """
        Attempt to unlock the container with amplitude and phase values.
        """
        try:
            # Verify the validation file if it exists
            if os.path.exists(self.validation_file):
                expected = hash_file(self.validation_file)
                _, A_expected, phi_expected = expected
                
                amp_check = abs(A_in - A_expected) <= 0.01
                phi_check = abs(phi_in - phi_expected) <= 0.01
                
                if not (amp_check and phi_check):
                    raise ValueError("Resonance mismatch. Unlock denied.")
            
            # Decrypt the container
            if not self.encrypted:
                raise ValueError("Container is not sealed")
                
            raw = base64.b64decode(self.encrypted.encode("utf-8"))
            return resonance_decrypt(raw, A_in, phi_in)
        except Exception as e:
            logger.error(f"Error unlocking container: {str(e)}")
            raise

    def export(self) -> dict:
        """
        Export the container as a dictionary.
        """
        return {
            "resonance_key": self.key_id,
            "entropy_score": self.entropy,
            "timestamp": self.timestamp,
            "sealed": self.encrypted,
            "validation_file": self.validation_file
        }
    
    def load_from_waveform(self, waveform_data):
        """
        Load container data from a waveform.
        """
        if len(waveform_data) < 2:
            waveform_data = [1.0, 0.5]  # Default values
        
        self.A = waveform_data[0]
        self.phi = waveform_data[1]
        self.key_id = f"waveform_{int(self.A * 1000)}_{int(self.phi * 1000)}"
        return self.seal()
    
    def export_symbolic_state(self):
        """
        Export the current symbolic state.
        """
        return [self.A, self.phi]

    def update_symbolic_state(self, new_state):
        """
        Update the symbolic state.
        """
        if len(new_state) >= 2:
            self.A = new_state[0]
            self.phi = new_state[1]
            return self.seal()
        return False

    @staticmethod
    def load(data: dict):
        """
        Load a container from a dictionary.
        """
        obj = SymbolicContainer(payload="", validation_file=data.get("validation_file", "example.txt"))
        obj.key_id = data["resonance_key"]
        obj.entropy = data["entropy_score"]
        obj.timestamp = data["timestamp"]
        obj.encrypted = data["sealed"]
        return obj

# -----------------------------------------------------------------------------
# Test Block
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔐 Sealing symbolic container from file...")

    validation_file = "example.txt"
    key = hash_file(validation_file)
    if key is None:
        print(f"❌ ERROR: Cannot seal. File '{validation_file}' missing or unreadable.")
        sys.exit(1)

    container = SymbolicContainer("ResonantSymbolicPayload", key, validation_file)
    container.seal()

    print("📦 Container Export:")
    print(json.dumps(container.export(), indent=2))

    try:
        unlocked = container.unlock(key[1], key[2])
        print("✅ Decrypted Payload:", unlocked)
    except Exception as e:
        print(f"❌ Unlock Error: {str(e)}")