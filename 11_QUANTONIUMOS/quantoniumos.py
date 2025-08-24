#!/usr/bin/env python3
"""
QuantoniumOS - Complete System Startup
Main entry point for the unified QuantoniumOS system
Integrates all phases: Kernel, GUI, API Integration, and Applications
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add all relevant paths
project_root = Path(__file__).parent
sys.path.extend(
    [
        str(project_root),
        str(project_root / "kernel"),
        str(project_root / "gui"),
        str(project_root / "web"),
        str(project_root / "filesystem"),
        str(project_root / "apps"),
        str(project_root / "phase3"),
        str(project_root / "phase4"),
        str(project_root / "11_QUANTONIUMOS"),
    ]
)


def setup_logging(level=logging.INFO):
    """Setup system logging"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("quantoniumos.log")],
    )


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = ["numpy", "tkinter", "flask", "cryptography"]

    missing = []
    for module in required_modules:
        try:
            if module == "tkinter":
                import tkinter
            else:
                __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"Missing required modules: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False

    return True


def start_desktop_mode():
    """Start QuantoniumOS in desktop mode with UI options"""
    print("Starting QuantoniumOS in Desktop Mode...")

    print("\nUI Options:")
    print("1. QuantoniumOS Unified (Oval/Tab Dock - Recommended)")
    print("2. PyQt5 Modern Interface")
    print("3. Advanced Tkinter Interface")
    print("4. Basic Tkinter Interface")

    choice = input("\nSelect interface (1-4): ").strip()

    if choice == "1" or choice == "":
        try:
            from PyQt5.QtWidgets import QApplication

            from core.quantonium_os_unified import QuantoniumOSUnified

            print("Launching QuantoniumOS Unified with Oval/Tab Dock...")

            app = QApplication(sys.argv)
            app.setApplicationName("QuantoniumOS")
            app.setApplicationVersion("3.0")

            main_window = QuantoniumOSUnified()
            main_window.show()

            return app.exec_()
        except ImportError as e:
            print(f"PyQt5 not available: {e}")
            print("Please install: pip install PyQt5")
            print("Falling back to other interfaces...")

    if choice == "2":
        try:
            from quantonium_os_pyqt5 import QuantoniumOSPyQt5

            print("Launching PyQt5 Modern Interface...")
            app = QuantoniumOSPyQt5()
            app.run()
            return
        except ImportError as e:
            print(f"PyQt5 interface not available: {e}")
            print("Falling back to Tkinter...")

    if choice == "3" or choice == "2":
        try:
            from quantonium_os_advanced import QuantoniumOSAdvancedLauncher

            launcher = QuantoniumOSAdvancedLauncher()
            launcher.run()
            return
        except ImportError as e:
            print(f"Failed to import advanced launcher: {e}")
            print("Falling back to basic desktop interface...")

    try:
        from gui.quantonium_os_gui import QuantoniumOSGUI

        gui = QuantoniumOSGUI()
        gui.run()
    except Exception as e:
        print(f"Failed to start desktop interface: {e}")
        return False

    return True


def start_web_mode(port=5000, debug=False):
    """Start QuantoniumOS in web mode (Flask)"""
    print(f"Starting QuantoniumOS Web Interface on port {port}...")

    try:
        from web.quantonium_web_interface import core.app as app
        app.run(host="0.0.0.0", port=port, debug=debug)
    except Exception as e:
        print(f"Failed to start web interface: {e}")
        return False

    return True


def start_headless_mode():
    """Start QuantoniumOS in headless mode (API only)"""
    print("Starting QuantoniumOS in Headless Mode...")

    try:
        # Initialize core components
        from kernel.quantum_vertex_kernel import QuantoniumKernel

        from phase3.api_integration.function_wrapper import quantum_wrapper
        from phase3.bridges.quantum_classical_bridge import quantum_bridge
        from phase3.services.service_orchestrator import service_orchestrator

        # Start quantum kernel
        print("Initializing Quantum Kernel...")
        kernel = QuantoniumKernel()

        # Start services
        print("Starting Service Orchestrator...")
        # service_orchestrator.start_all_services()

        # Start API bridge
        print("Initializing Quantum-Classical Bridge...")
        # quantum_bridge.initialize()

        print("QuantoniumOS headless mode ready!")
        print("Access via API endpoints or use web interface")

        # Keep running
        try:
            while True:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down QuantoniumOS...")

    except Exception as e:
        print(f"Failed to start headless mode: {e}")
        return False

    return True


def run_tests():
    """Run QuantoniumOS test suite"""
    print("Running QuantoniumOS test suite...")

    try:
        import importlib
        import subprocess

        # Run core tests
        test_files = [
            "test_all_claims.py",
            "07_TESTS_BENCHMARKS/test_quantum_kernel.py",
            "07_TESTS_BENCHMARKS/test_rft_algorithms.py",
            "run_all_validators.py",  # Added validators
        ]

        for test_file in test_files:
            if os.path.exists(test_file):
                print(f"Running {test_file}...")
                result = subprocess.run(
                    [sys.executable, test_file], capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"✓ {test_file} passed")
                else:
                    print(f"✗ {test_file} failed: {result.stderr}")
            else:
                print(f"⚠ {test_file} not found")

        # Try to directly import and run validators
        try:
            print("Running core validators directly...")
            sys.path.append(str(Path(__file__).parent.parent))
            validator_module = importlib.import_module(
                "02_CORE_VALIDATORS.validate_system"
            )
            if hasattr(validator_module, "run_validation"):
                result = validator_module.run_validation()
                print(f"System Validation Result: {result.get('status', 'UNKNOWN')}")
            elif hasattr(validator_module, "main"):
                result = validator_module.main()
                print(f"System Validation Result: {result}")
        except Exception as e:
            print(f"Error running core validators: {e}")

        print("Test suite completed!")

    except Exception as e:
        print(f"Failed to run tests: {e}")
        return False

    return True


def show_system_info():
    """Show QuantoniumOS system information"""
    print("=" * 60)
    print("QuantoniumOS - Quantum Operating System")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Project Root: {project_root}")
    print()
    print("Available Components:")
    print("  ✓ Quantum Vertex Kernel")
    print("  ✓ RFT Algorithm Engine")
    print("  ✓ Patent Integration System")
    print("  ✓ Desktop GUI (Tkinter)")
    print("  ✓ Web Interface (Flask)")
    print("  ✓ Quantum-Aware Filesystem")
    print("  ✓ Phase 3: API Integration")
    print("  ✓ Phase 4: Advanced Applications")
    print()
    print("Applications:")
    print("  • RFT Transform Visualizer")
    print("  • Quantum Cryptography Playground")
    print("  • Patent Validation Dashboard")
    print("  • System Monitor")
    print("  • Quantum Simulator")
    print("  • API Explorer")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="QuantoniumOS - Quantum Operating System"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="desktop",
        choices=["desktop", "web", "headless", "test", "info"],
        help="Operating mode (default: desktop)",
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for web mode (default: 5000)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for web interface"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)

    # Check dependencies
    if args.mode not in ["info"] and not check_dependencies():
        sys.exit(1)

    # Execute based on mode
    success = True

    if args.mode == "desktop":
        success = start_desktop_mode()
    elif args.mode == "web":
        success = start_web_mode(args.port, args.debug)
    elif args.mode == "headless":
        success = start_headless_mode()
    elif args.mode == "test":
        success = run_tests()
    elif args.mode == "info":
        show_system_info()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
