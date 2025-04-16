#!/usr/bin/env python3
"""
Quantonium OS - Eigen Library Downloader

This script downloads and extracts the Eigen C++ matrix library
which is required for the HPC modules in Quantonium OS.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("download_eigen")

EIGEN_VERSION = "3.4.0"
EIGEN_URL = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.zip"
EIGEN_DIR = "Eigen"

def download_eigen():
    """Download and extract Eigen library."""
    # Create directory if it doesn't exist
    if not os.path.exists(EIGEN_DIR):
        os.makedirs(EIGEN_DIR)
    
    target_file = os.path.join(EIGEN_DIR, f"eigen-{EIGEN_VERSION}.zip")
    target_dir = os.path.join(EIGEN_DIR, f"eigen-{EIGEN_VERSION}")
    
    # Check if already downloaded and extracted
    if os.path.exists(target_dir):
        logger.info(f"Eigen {EIGEN_VERSION} already downloaded and extracted.")
        return True
    
    # Download if not exists
    if not os.path.exists(target_file):
        logger.info(f"Downloading Eigen {EIGEN_VERSION}...")
        try:
            urllib.request.urlretrieve(EIGEN_URL, target_file)
            logger.info(f"Download complete: {target_file}")
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            return False
    
    # Extract
    logger.info(f"Extracting Eigen {EIGEN_VERSION}...")
    try:
        with zipfile.ZipFile(target_file, 'r') as zip_ref:
            zip_ref.extractall(EIGEN_DIR)
        logger.info(f"Extraction complete: {target_dir}")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        return False

def main():
    if download_eigen():
        logger.info("Eigen library is ready for use with Quantonium OS.")
        return 0
    else:
        logger.error("Failed to prepare Eigen library.")
        return 1

if __name__ == "__main__":
    sys.exit(main())