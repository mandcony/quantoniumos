import json
import os

class Config:
    def __init__(self, filename=None):
        """Initializes configuration, loading from settings.json or default values."""
        if not filename:
            filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), "settings.json")
        self.filename = filename
        self.data = self.read()

    def read(self):
        """Reads configuration data from the JSON file."""
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default settings if file is missing or corrupted
            return {
                "resonance_frequency": 1.0,
                "resonance_spread": 0.1,
                "default_qubits": 3,
                "encryption_mode": "resonance_hashing"
            }

    def save(self, data):
        """Saves updated configuration data to the JSON file."""
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=2)
        self.data = data  # Fixed this line

# TESTING CONFIG SYSTEM
if __name__ == "__main__":
    config = Config()
    print("Current Config:", config.data)

    # Example: Update a setting
    config.data["default_qubits"] = 5
    config.save(config.data)

    print("Updated Config:", config.read())
