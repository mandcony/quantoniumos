"""
Build Resonance Engine - Redirector
"""
import argparse
import os
import sys

# Add the project root to the path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Build Resonance Engine Redirector")
    parser.add_argument("--help", action="store_true", help="Show help message")
    parser.add_argument("--validate", action="store_true", help="Validate the build")
    args, unknown = parser.parse_known_args()

    if args.help:
        parser.print_help()
        return

    # Redirect to the actual build utility
    try:
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "10_UTILITIES",
            ),
        )
        from build_resonance_engine import core.main as mainas build_main

        # Launch the build utility
        print("Redirecting to build_resonance_engine...")
        build_main()
        print("Build process completed.")

    except ImportError:
        print("Redirecting from alternative path...")
        # Try the alternative path
        build_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "10_UTILITIES",
            "build_resonance_engine.py",
        )

        if not os.path.exists(build_path):
            print(f"Error: Could not find build utility at {build_path}")
            return

        # Construct command-line arguments to pass through
        cmd_args = []
        if args.validate:
            cmd_args.append("--validate")

        # Pass through any other arguments
        cmd_args.extend(unknown)

        # Update sys.argv for the script being executed
        sys.argv = [build_path] + cmd_args

        with open(build_path) as f:
            exec(f.read())

        print("Build process completed via exec.")


if __name__ == "__main__":
    main()
