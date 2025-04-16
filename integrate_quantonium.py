#!/usr/bin/env python3
"""
Quantonium OS - Integration Script

This script extracts and integrates the proprietary HPC modules from
quantonium_v2.zip into the Quantonium OS Cloud Runtime.

Usage:
    python integrate_quantonium.zip path/to/quantonium_v2.zip

Notes:
    This script should be executed once to set up the proprietary modules.
    After integration, the API will automatically use the advanced HPC functions.
"""

import os
import sys
import shutil
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("integrate_quantonium")

def verify_structure(extracted_path):
    """Verify the extracted zip has the expected structure."""
    required_dirs = [
        'core/HPC',
        'interface',
        'orchestration'
    ]
    
    required_files = [
        'core/HPC/symbolic_eigen.py',
        'core/HPC/ccp_engine.py',
        'core/HPC/symbolic_container.py',
        'orchestration/quantum_nova_system.py',
        'interface/quantum_os_bindings.py',
        'interface/engine_core_bindings.py'
    ]
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(extracted_path, dir_path)
        if not os.path.isdir(full_path):
            logger.error(f"Required directory not found: {dir_path}")
            return False
    
    # Check files
    for file_path in required_files:
        full_path = os.path.join(extracted_path, file_path)
        if not os.path.isfile(full_path):
            logger.error(f"Required file not found: {file_path}")
            return False
    
    return True

def integrate_modules(extracted_path, target_path='.'):
    """Integrate the extracted modules into the target directory."""
    # Directories to copy
    dirs_to_copy = [
        'core/HPC',
        'interface',
        'orchestration'
    ]
    
    for dir_path in dirs_to_copy:
        src = os.path.join(extracted_path, dir_path)
        dst = os.path.join(target_path, dir_path)
        
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        # Copy files
        if os.path.isdir(src):
            logger.info(f"Copying directory: {dir_path}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            logger.error(f"Source directory not found: {src}")
    
    # Handle any C/C++ files that need compilation
    compile_cpp_modules(target_path)
    
    return True

def compile_cpp_modules(base_path):
    """Compile C/C++ modules if present."""
    cpp_dirs = [
        'interface',
        'core/HPC'
    ]
    
    for dir_path in cpp_dirs:
        full_path = os.path.join(base_path, dir_path)
        
        # Check for CMakeLists.txt
        cmake_file = os.path.join(full_path, 'CMakeLists.txt')
        if os.path.isfile(cmake_file):
            logger.info(f"Found CMakeLists.txt in {dir_path}, compiling...")
            
            # Create a build directory
            build_dir = os.path.join(full_path, 'build')
            os.makedirs(build_dir, exist_ok=True)
            
            # Run CMake and make
            os.chdir(build_dir)
            os.system('cmake ..')
            os.system('make')
            os.chdir(base_path)
            
            logger.info(f"Compilation complete for {dir_path}")
    
    # Check for setup.py files for Pybind11 modules
    for dir_path in cpp_dirs:
        full_path = os.path.join(base_path, dir_path)
        setup_file = os.path.join(full_path, 'setup.py')
        
        if os.path.isfile(setup_file):
            logger.info(f"Found setup.py in {dir_path}, building...")
            
            # Run setup.py
            os.chdir(full_path)
            os.system('pip install -e .')
            os.chdir(base_path)
            
            logger.info(f"Build complete for {dir_path}")

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/quantonium_v2.zip")
        return 1
    
    zip_path = sys.argv[1]
    
    if not os.path.isfile(zip_path):
        logger.error(f"File not found: {zip_path}")
        return 1
    
    # Create a temporary directory for extraction
    temp_dir = "temp_quantonium"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Extract the zip file
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Verify the structure
        if not verify_structure(temp_dir):
            logger.error("Verification failed. The zip file does not contain the expected structure.")
            return 1
        
        # Integrate the modules
        logger.info("Integrating HPC modules...")
        if integrate_modules(temp_dir):
            logger.info("Integration complete!")
        else:
            logger.error("Integration failed.")
            return 1
        
    finally:
        # Clean up
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
    
    logger.info("Quantonium OS HPC modules have been successfully integrated.")
    logger.info("Restart the application for changes to take effect.")
    return 0

if __name__ == "__main__":
    sys.exit(main())