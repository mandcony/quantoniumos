#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QuantoniumOS Main Entry Point

This is the main entry point for the QuantoniumOS system.
Execute this file to launch the complete quantum operating system.
"""

import argparse
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def run_validators():
    """Run all core validators"""
    try:
        print("Running core validators...")
        validator_script = os.path.join(project_root, "run_all_validators.py")
        if os.path.exists(validator_script):
            import subprocess

            result = subprocess.run(
                [sys.executable, validator_script], capture_output=False
            )
            return result.returncode == 0
        else:
            print(f"Validator script not found: {validator_script}")
            return False
    except Exception as e:
        print(f"Error running validators: {e}")
        return False


def wire_engines():
    """Wire all engines to validators"""
    try:
        print("Wiring engines to validators...")
        wire_script = os.path.join(project_root, "wire_validators.py")
        if os.path.exists(wire_script):
            import subprocess

            result = subprocess.run([sys.executable, wire_script], capture_output=False)
            return result.returncode == 0
        else:
            print(f"Wire script not found: {wire_script}")
            return False
    except Exception as e:
        print(f"Error wiring engines: {e}")
        return False


def main():
    """Main entry point for QuantoniumOS."""
    parser = argparse.ArgumentParser(
        description="QuantoniumOS Quantum Operating System"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run all system validators before launching",
    )
    parser.add_argument(
        "--validators-only",
        action="store_true",
        help="Run only the system validators and exit",
    )
    parser.add_argument(
        "--wire",
        action="store_true",
        help="Wire all engines to validators before launching",
    )

    args = parser.parse_args()

    if args.validators_only:
        return 0 if run_validators() else 1

    print("\n" + "=" * 80)
    print(" QuantoniumOS Quantum Operating System ".center(80, "="))
    print("=" * 80 + "\n")

    if args.validate:
        validation_success = run_validators()
        print("Validation " + ("PASSED" if validation_success else "FAILED"))

    if args.wire:
        wire_success = wire_engines()
        print("Engine wiring " + ("COMPLETED" if wire_success else "FAILED"))

    # Try to launch the main application
    try:
        # Import the main app module
        print("\nLaunching QuantoniumOS...")
        try:
            from apps.quantonium_app_wrapper import launch_app

            print("Launching main application...")
            launch_app()
        except ImportError:
            print("Warning: Main application not available.")
            try:
                # Try to import from the 11_QUANTONIUMOS folder
                sys.path.append(os.path.join(project_root, "11_QUANTONIUMOS"))
                import quantoniumos

                print("Starting core system...")
                quantoniumos.main()
            except ImportError:
                print("Warning: Core system not available. Starting in headless mode.")

    except Exception as e:
        print(f"Error launching QuantoniumOS: {e}")
        return 1

    print("\nQuantoniumOS startup complete")
    return 0
    print("=" * 80 + "\n")

    print("Initializing QuantoniumOS...")

    # Run validators if requested
    if args.validate:
        print("Running system validation before launch...")
        if not run_validators():
            print("Validation failed. Continuing with launch anyway...")

    try:
        # Import the main application
        from _11_QUANTONIUMOS import quantoniumos

        # Run the system
        quantoniumos.launch()

        print("\nQuantoniumOS launched successfully!")
        return 0
    except ImportError as e:
        print(f"Error importing QuantoniumOS modules: {e}")
        print("\nPlease ensure the project is properly installed and organized.")
        return 1
    except Exception as e:
        print(f"Error launching QuantoniumOS: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
