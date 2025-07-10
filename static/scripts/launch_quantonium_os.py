#!/usr/bin/env python
"""
QuantoniumOS Desktop Launcher

This script launches the full QuantoniumOS desktop environment
by starting the quantonium_os_main.py module from the attached_assets directory.
"""
import logging
import os
import subprocess
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("QuantoniumLauncher")


def find_python_executable():
    """Find the Python executable path."""
    return sys.executable


def find_asset_directory():
    """Find the QuantoniumOS files directory."""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the parent directory (static)
    static_dir = os.path.dirname(current_dir)

    # Navigate to the root directory
    root_dir = os.path.dirname(static_dir)

    # Find the attached_assets directory
    assets_dir = os.path.join(root_dir, "attached_assets")

    if not os.path.exists(assets_dir):
        logger.error(f"Assets directory not found at {assets_dir}")
        return None

    return assets_dir


def launch_quantonium_os():
    """Launch the QuantoniumOS desktop environment."""
    logger.info("🔹 Starting QuantoniumOS Desktop Environment...")

    # Find Python executable and asset directory
    python_exec = find_python_executable()
    assets_dir = find_asset_directory()

    if not assets_dir:
        logger.error("❌ Could not find QuantoniumOS assets directory")
        return False

    # Prepare command to launch QuantoniumOS
    main_script = os.path.join(assets_dir, "quantonium_os_main.py")

    if not os.path.exists(main_script):
        logger.error(f"❌ QuantoniumOS main script not found at {main_script}")
        return False

    logger.info(f"✅ Found QuantoniumOS main script at {main_script}")

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = assets_dir + os.pathsep + env.get("PYTHONPATH", "")

    try:
        # Launch QuantoniumOS as a subprocess
        logger.info("🚀 Launching QuantoniumOS...")
        process = subprocess.Popen(
            [python_exec, main_script],
            env=env,
            cwd=assets_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give it a moment to start
        time.sleep(1)

        # Check if process is still running
        if process.poll() is None:
            logger.info("✅ QuantoniumOS launched successfully!")
            return True
        else:
            # Process has terminated, get output
            stdout, stderr = process.communicate()
            logger.error(f"❌ QuantoniumOS failed to launch: {stderr}")
            logger.debug(f"Output: {stdout}")
            return False

    except Exception as e:
        logger.error(f"❌ Error launching QuantoniumOS: {str(e)}")
        return False


if __name__ == "__main__":
    success = launch_quantonium_os()
    if success:
        logger.info("QuantoniumOS desktop environment is now running.")
    else:
        logger.error("Failed to launch QuantoniumOS.")
        sys.exit(1)
