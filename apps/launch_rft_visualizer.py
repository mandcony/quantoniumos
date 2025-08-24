"""
Launch RFT Visualizer - Redirector
"""
import os
import sys

# Add the project root to the path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Attempt to import from the main apps directory
try:
    from apps.launch_rft_visualizer import launch_rft_visualizer

    # Launch the application
    print("Launching QuantoniumOS RFT Visualizer...")
    result = launch_rft_visualizer()
    print("RFT Visualizer launched successfully.")

except ImportError:
    print("Launching from alternative path...")
    # Try the alternative path
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "apps",
        ),
    )

    # Use execfile equivalent for Python 3
    launcher_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "apps",
        "launch_rft_visualizer.py",
    )

    with open(launcher_path) as f:
        exec(f.read())

    print("RFT Visualizer launched successfully via exec.")
