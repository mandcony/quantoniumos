"""
Configuration Module

Provides configuration for QuantoniumOS components.
"""

import json
import os
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for QuantoniumOS."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or defaults.
        
        Args:
            config_path: Optional path to a JSON configuration file
        """
        self.data = {}
        
        # Default values
        self.data = {
            "resonance_frequency": 3.7876781241365785,
            "resonance_spread": 0.6285302475952592,
            "system": {
                "max_processes": 100,
                "sample_rate": 44100,
                "buffer_size": 1024
            },
            "paths": {
                "data": "data",
                "logs": "logs",
                "temp": "temp"
            }
        }
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load(config_path)
    
    def load(self, config_path: str) -> bool:
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(config_path, 'r') as f:
                file_data = json.load(f)
                
            # Update defaults with file data
            self._update_dict(self.data, file_data)
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save(self, config_path: str) -> bool:
        """
        Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the JSON configuration
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.data, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.
        
        Args:
            key: The configuration key (can use dot notation for nested keys)
            default: Default value if the key is not found
            
        Returns:
            The configuration value or the default
        """
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key.
        
        Args:
            key: The configuration key (can use dot notation for nested keys)
            value: The value to set
        """
        keys = key.split('.')
        current = self.data
        
        # Navigate to the right level
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
            
        # Set the value at the final level
        current[keys[-1]] = value
    
    def _update_dict(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
