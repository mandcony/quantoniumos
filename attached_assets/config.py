import json
import os

class Config:
    def __init__(self, filename=None):
        if not filename:
            filename = os.path.join(os.path.abspath(os.path.dirname(__file__)), "settings.json")
        self.filename = filename
        self.data = self.read()

    def read(self):
        try:
            with open(self.filename, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"resonance_frequency": 1.0, "resonance_spread": 0.1}

    def save(self, data):
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=2)
        self.data = data