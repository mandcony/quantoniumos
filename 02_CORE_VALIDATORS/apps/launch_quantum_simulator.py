"""
Launch Quantum Simulator - Redirector
"""
import os
import sys

# Add the project root to the path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Attempt to import from the main apps directory
try:
    from apps.launch_quantum_simulator import launch_quantum_simulator

    # Launch the quantum simulator
    print("Launching QuantoniumOS Quantum Simulator...")
    result = launch_quantum_simulator()
    print("Quantum Simulator launched successfully.")

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
        "launch_quantum_simulator.py",
    )

    with open(launcher_path) as f:
        exec(f.read())

    print("Quantum Simulator launched successfully via exec.")
