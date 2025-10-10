"""
QuantoniumOS Utilities Package
=============================
Centralized utilities for path management, imports, and configuration.
"""

from .paths import paths, get_project_root, setup_python_path
from .imports import imports, setup_quantonium_imports, get_kernel, get_core_module, get_app
from .config import config, get_app_registry, get_app_config, get_build_config

__all__ = [
    # Path management
    'paths',
    'get_project_root', 
    'setup_python_path',
    
    # Import management
    'imports',
    'setup_quantonium_imports',
    'get_kernel',
    'get_core_module', 
    'get_app',
    
    # Configuration management
    'config',
    'get_app_registry',
    'get_app_config',
    'get_build_config'
]


def initialize_quantonium():
    """
    Initialize the complete QuantoniumOS environment
    
    Returns:
        bool: True if initialization successful
    """
    try:
        # Setup paths
        setup_python_path()
        
        # Setup imports
        success = setup_quantonium_imports()
        
        # Create config files if needed
        config.create_config_files()
        
        print("✓ QuantoniumOS environment initialized successfully")
        return success
        
    except Exception as e:
        print(f"✗ QuantoniumOS initialization failed: {e}")
        return False


if __name__ == "__main__":
    # Test full initialization
    initialize_quantonium()
