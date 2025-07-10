"""
QuantoniumOS Launcher

This script launches the QuantoniumOS desktop environment by importing
and running the actual QuantoniumOS code from attached_assets.
"""

import importlib.util
import os
import sys


def launch_quantonium_os(env_config=None):
    """
    Import the quantonium_os_main module from attached_assets
    and launch the QuantoniumOS desktop environment.

    Args:
        env_config: Optional dictionary with environment configuration.
                   If not provided, setup_environment() will be called.
    """
    # Set up environment if not already done
    if env_config is None:
        env_config = setup_environment()

    # Get the path to the quantonium_os_main.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = env_config.get("assets_dir") or os.path.join(
        current_dir, "attached_assets"
    )
    os_main_path = os.path.join(assets_dir, "quantonium_os_main.py")

    # Check if the file exists
    if not os.path.exists(os_main_path):
        print(f"ERROR: quantonium_os_main.py not found at {os_main_path}")
        return False

    try:
        # Add the attached_assets directory to sys.path
        sys.path.insert(0, assets_dir)

        # Add the root directory to sys.path as well
        sys.path.insert(0, current_dir)

        # Create a directory for app files if it doesn't exist
        app_dir = os.path.join(assets_dir, "apps")
        os.makedirs(app_dir, exist_ok=True)

        # Set up a logs directory if it doesn't exist
        log_dir = env_config.get("log_dir") or os.path.join(assets_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Log the launch
        print(f"QuantoniumOS: Launching from {os_main_path}")
        if env_config.get("headless"):
            print("QuantoniumOS: Running in headless mode. GUI will not be visible.")

        # Import and run the code from quantonium_os_main.py
        spec = importlib.util.spec_from_file_location(
            "quantonium_os_main", os_main_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # The module is executed, which should launch the QuantoniumOS
        print("QuantoniumOS: Successfully launched!")
        return True

    except ImportError as e:
        print(f"ERROR: Failed to import QuantoniumOS - {str(e)}")
        if "PyQt5" in str(e):
            print("PyQt5 is required. Install it with: pip install pyqt5 qtawesome")
        return False

    except Exception as e:
        print(f"ERROR: Failed to launch QuantoniumOS - {str(e)}")
        return False


def setup_environment():
    """
    Set up the environment for running QuantoniumOS desktop applications in containers.
    """
    # Configure headless mode for Qt applications if needed
    if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        print("QuantoniumOS: Running in headless mode with offscreen rendering")
    else:
        print(f"QuantoniumOS: Display available at {os.environ['DISPLAY']}")

    # Set up matplotlib backend
    try:
        import matplotlib

        if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            matplotlib.use("Agg")
            print("QuantoniumOS: Set matplotlib backend to Agg for headless operation")
        else:
            matplotlib.use("QtAgg")
            print(
                "QuantoniumOS: Set matplotlib backend to QtAgg for windowed operation"
            )
    except ImportError:
        print("QuantoniumOS: Matplotlib not found, skipping backend configuration")

    # Create logs directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "attached_assets")
    log_dir = os.path.join(assets_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    return {
        "assets_dir": assets_dir,
        "log_dir": log_dir,
        "headless": os.environ.get("QT_QPA_PLATFORM") == "offscreen",
    }


if __name__ == "__main__":
    # Set up the environment first
    env_config = setup_environment()
    print(
        f"QuantoniumOS: Environment configured, assets directory: {env_config['assets_dir']}"
    )

    # Launch the OS
    launch_quantonium_os()
