#!/usr/bin/env python
"""
QuantoniumOS Desktop Environment Launcher

This script launches the full QuantoniumOS desktop environment by 
initializing both the web API and desktop UI components.

Usage:
    python launch_quantonium_os.py [--web-only | --desktop-only]
    
Options:
    --web-only      Launch only the web API components
    --desktop-only  Launch only the desktop UI components
"""
import os
import sys
import subprocess
import argparse
import logging
import time
import signal
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("quantonium_launcher.log")
    ]
)
logger = logging.getLogger("QuantoniumLauncher")

# Global process objects for cleanup
web_process = None
desktop_process = None

def signal_handler(sig, frame):
    """Handle termination signals by cleaning up processes."""
    logger.info("Received termination signal, shutting down...")
    cleanup_processes()
    sys.exit(0)

def cleanup_processes():
    """Clean up any running processes."""
    global web_process, desktop_process
    
    if web_process:
        logger.info("Terminating web API process...")
        try:
            web_process.terminate()
            web_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating web process: {str(e)}")
            try:
                web_process.kill()
            except:
                pass
    
    if desktop_process:
        logger.info("Terminating desktop UI process...")
        try:
            desktop_process.terminate()
            desktop_process.wait(timeout=5)
        except Exception as e:
            logger.error(f"Error terminating desktop process: {str(e)}")
            try:
                desktop_process.kill()
            except:
                pass

def launch_web_api():
    """Launch the QuantoniumOS web API."""
    global web_process
    
    logger.info("🌐 Starting QuantoniumOS Web API...")
    
    # Set up paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    web_script = os.path.join(root_dir, "main.py")
    
    if not os.path.exists(web_script):
        logger.error(f"❌ Web API script not found at {web_script}")
        return False
    
    logger.info(f"✅ Found Web API script at {web_script}")
    
    # Set environment variables
    env = os.environ.copy()
    
    try:
        # Launch web API as a subprocess
        logger.info("🚀 Launching QuantoniumOS Web API...")
        web_process = subprocess.Popen(
            [sys.executable, web_script],
            env=env,
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if web_process.poll() is None:
            logger.info("✅ QuantoniumOS Web API launched successfully!")
            return True
        else:
            # Process has terminated, get output
            stdout, stderr = web_process.communicate()
            logger.error(f"❌ QuantoniumOS Web API failed to launch: {stderr}")
            logger.debug(f"Output: {stdout}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error launching Web API: {str(e)}")
        return False

def launch_desktop_ui():
    """Launch the QuantoniumOS desktop UI."""
    global desktop_process
    
    logger.info("🖥️ Starting QuantoniumOS Desktop UI...")
    
    # Set up paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(root_dir, "attached_assets")
    desktop_script = os.path.join(assets_dir, "quantonium_os_main.py")
    
    if not os.path.exists(desktop_script):
        logger.error(f"❌ Desktop UI script not found at {desktop_script}")
        return False
    
    logger.info(f"✅ Found Desktop UI script at {desktop_script}")
    
    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = assets_dir + os.pathsep + env.get("PYTHONPATH", "")
    
    try:
        # Launch desktop UI as a subprocess
        logger.info("🚀 Launching QuantoniumOS Desktop UI...")
        desktop_process = subprocess.Popen(
            [sys.executable, desktop_script],
            env=env,
            cwd=assets_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process is still running
        if desktop_process.poll() is None:
            logger.info("✅ QuantoniumOS Desktop UI launched successfully!")
            return True
        else:
            # Process has terminated, get output
            stdout, stderr = desktop_process.communicate()
            logger.error(f"❌ QuantoniumOS Desktop UI failed to launch: {stderr}")
            logger.debug(f"Output: {stdout}")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error launching Desktop UI: {str(e)}")
        return False

def main():
    """Main function to parse arguments and launch components."""
    parser = argparse.ArgumentParser(description="Launch QuantoniumOS components")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--web-only", action="store_true", help="Launch only the web API")
    group.add_argument("--desktop-only", action="store_true", help="Launch only the desktop UI")
    
    args = parser.parse_args()
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_processes)
    
    logger.info("==========================================")
    logger.info("QuantoniumOS Launcher Starting")
    logger.info("==========================================")
    
    web_success = True
    desktop_success = True
    
    # Launch components based on arguments
    if args.web_only:
        web_success = launch_web_api()
    elif args.desktop_only:
        desktop_success = launch_desktop_ui()
    else:
        # Launch both components
        web_success = launch_web_api()
        desktop_success = launch_desktop_ui()
    
    # Check if any components failed to launch
    if not web_success or not desktop_success:
        if not web_success:
            logger.error("Failed to launch QuantoniumOS Web API.")
        if not desktop_success:
            logger.error("Failed to launch QuantoniumOS Desktop UI.")
        
        # If only one component was requested and it failed, exit with error
        if (args.web_only and not web_success) or (args.desktop_only and not desktop_success):
            return 1
    
    # Keep the launcher running to maintain subprocesses
    logger.info("QuantoniumOS components are now running.")
    logger.info("Press Ctrl+C to shutdown all components.")
    
    try:
        # Keep the main process running
        while True:
            # Check if processes are still alive
            if web_process and web_process.poll() is not None:
                logger.error("Web API process has terminated unexpectedly.")
                if args.web_only:
                    return 1
            
            if desktop_process and desktop_process.poll() is not None:
                logger.error("Desktop UI process has terminated unexpectedly.")
                if args.desktop_only:
                    return 1
            
            # Sleep to reduce CPU usage
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        cleanup_processes()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())