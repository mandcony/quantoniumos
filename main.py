#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QuantoniumOS Main Entry Point
This is the main entry point for the QuantoniumOS system.
"""

import json
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("quantoniumos.log"), logging.StreamHandler()],
)
logger = logging.getLogger("QuantoniumOS")

# Import core components
try:
    import apps.quantonium_app_wrapper as app_wrapper
    from bulletproof_quantum_kernel import BulletproofQuantumKernel
    from quantonium_design_system import DesignSystem
    from quantonium_os_unified import QuantoniumOSUnified
    from quantoniumos import QuantoniumOS
    from topological_quantum_kernel import TopologicalQuantumKernel
    from working_quantum_kernel import WorkingQuantumKernel
except ImportError as e:
    logger.error(f"Failed to import core components: {e}")
    sys.exit(1)


def initialize_system():
    """Initialize the QuantoniumOS system."""
    logger.info("Initializing QuantoniumOS...")

    # Initialize the quantum kernels
    bulletproof_kernel = BulletproofQuantumKernel(num_qubits=8)
    topological_kernel = TopologicalQuantumKernel()
    working_kernel = WorkingQuantumKernel()

    # Initialize design system
    design_system = DesignSystem()

    # Initialize unified OS
    unified_os = QuantoniumOSUnified()

    # Load quantum kernels into unified OS
    unified_os.load_component("quantum_engine", bulletproof_kernel)
    unified_os.load_component("rft_engine", topological_kernel)
    unified_os.initialize()

    # Initialize the main system
    system = QuantoniumOS()

    return system


def start_system():
    """Start the QuantoniumOS system."""
    logger.info("Starting QuantoniumOS...")

    try:
        # Initialize the system
        system = initialize_system()

        # Start the system
        system.start()

        # Log startup information
        logger.info(
            f"QuantoniumOS started successfully at {datetime.now().isoformat()}"
        )

        return system
    except Exception as e:
        logger.error(f"Failed to start QuantoniumOS: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    print("===== QuantoniumOS =====")
    print("Initializing system...")

    # Start the system
    system = start_system()

    # Try to launch the system monitor
    try:
        if hasattr(app_wrapper, "launch_system_monitor"):
            app_wrapper.launch_system_monitor()
        else:
            print("System monitor not available.")
    except Exception as e:
        print(f"Could not launch system monitor: {e}")

    print("System initialized and running.")
    print("Type 'help' for available commands or 'exit' to quit.")

    # Simple command loop
    while True:
        cmd = input("QuantoniumOS> ").strip().lower()

        if cmd == "exit":
            print("Shutting down QuantoniumOS...")
            system.stop()  # Use stop() instead of shutdown()
            break
        elif cmd == "help":
            print("Available commands:")
            print("  help     - Show this help message")
            print("  status   - Show system status")
            print("  apps     - List available applications")
            print("  launch   - Launch an application")
            print("  exit     - Exit QuantoniumOS")
        elif cmd == "status":
            status = system.start()  # Use start() instead of get_status()
            print(json.dumps(status, indent=2))
        elif cmd == "apps":
            # Hardcoded list since app_wrapper doesn't have list_apps
            apps = [
                "quantum_simulator",
                "quantum_crypto",
                "q_browser",
                "q_mail",
                "q_notes",
                "q_vault",
                "rft_visualizer",
                "system_monitor",
            ]
            print("Available applications:")
            for app in apps:
                print(f"  - {app}")
        elif cmd.startswith("launch "):
            app_name = cmd.split(" ", 1)[1]
            print(f"Launching {app_name}...")
            try:
                # We know launch_app needs app_class, app_name, and app_title
                print(f"Launch functionality not available in command-line mode")
                print(
                    f"Please run the specific launcher directly: python apps/launch_{app_name}.py"
                )
            except Exception as e:
                print(f"Error launching {app_name}: {e}")
        else:
            print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
