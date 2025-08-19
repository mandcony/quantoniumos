import os
import json


class Config:
    def __init__(self, filename=None):
        """Initializes configuration, loading from settings.json or default values."""
        if not filename:
            self.filename = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "settings.json"
            )
        else:
            self.filename = filename
        self.data = self.read()

    def read(self):
        """Reads configuration data from the JSON file."""
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Default configuration
            return {
                "default_qubits": 3,
                "default_amplitude": 1.0,
                "default_phase": 0.0,
                "log_level": "INFO",
                "dll_path": os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "bin", 
                    "engine_core.dll"
                ),
                "use_gpu": False,
                "max_threads": 4
            }

    def save(self, data):
        """Saves configuration data to the JSON file."""
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=4)
        return data


# TESTING CONFIG SYSTEM
if __name__ == "__main__":
    config = Config()
    print("Current Config:", config.data)
    
    # Example: Update a setting
    config.data["default_qubits"] = 5
    config.save(config.data)
    print("Updated Config:", config.read()) 