"""
Launch Q Vault - Redirector
"""
import os
import sys

# Add the project root to the path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Attempt to import from the main apps directory
try:
    from apps.launch_q_vault import launch_q_vault

    # Launch the application
    print("Launching QuantoniumOS Q Vault...")
    result = launch_q_vault()
    print("Q Vault launched successfully.")

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
        "launch_q_vault.py",
    )

    with open(launcher_path) as f:
        exec(f.read())

    print("Q Vault launched successfully via exec.")
