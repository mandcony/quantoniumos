#!/usr/bin/env python3
"""
QuantoniumOS Deployment Emergency Fix Script

This script addresses common issues with Replit deployments that refuse to connect.
It specifically focuses on the 'site refused to connect' error.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fix_deployment")


def check_replit_file():
    """Ensure .replit file has correct deployment configuration."""
    if not os.path.exists(".replit"):
        logger.error(".replit file not found!")
        logger.info("Cannot modify .replit file directly due to Replit restrictions")
        logger.info("Please update it manually through the Replit interface")
        return False

    logger.info("Found .replit file - note that we cannot modify it directly")
    logger.info(
        "Please manually update your deployment settings in the Replit interface:"
    )
    logger.info("1. Change deployment command to: gunicorn --bind 0.0.0.0:5000 app:app")
    logger.info("2. Change deploymentTarget to: cloudrun")

    return True


def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import flask

        logger.info("Flask is installed ✓")
    except ImportError:
        logger.error("Flask is not installed!")
        logger.info("Please install Flask using Replit's package manager")

    try:
        import gunicorn

        logger.info("Gunicorn is installed ✓")
    except ImportError:
        logger.error("Gunicorn is not installed!")
        logger.info("Please install Gunicorn using Replit's package manager")

    return True


def create_gitignore():
    """Create or update .gitignore to prevent issues with deployment."""
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w") as f:
            f.write(
                """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
.env

# Replit specific
.replit.app

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
"""
            )
        logger.info("Created .gitignore file")
    return True


def fix_main_entry():
    """Check main.py to ensure it has correct app creation and binding."""
    if not os.path.exists("main.py"):
        logger.error("main.py not found!")
        return False

    with open("main.py", "r") as f:
        content = f.read()

    # Check if it's importing app from elsewhere
    if "from app import app" in content:
        logger.info("main.py imports app from app.py, this is good")
        return True

    # Check for app creation and run block
    if "if __name__ == '__main__'" not in content:
        logger.warning("No __main__ block in main.py, adding one")
        with open("main.py", "a") as f:
            f.write(
                """

# Make sure we have a proper entrypoint for gunicorn
if __name__ == '__main__':
    # This block is only used when running the app directly, not through gunicorn
    app.run(host='0.0.0.0', port=5000)
"""
            )
        logger.info("Added __main__ block to main.py")

    return True


def main():
    logger.info("Running emergency deployment fix")

    # Run all fixes
    check_replit_file()
    check_dependencies()
    create_gitignore()
    fix_main_entry()

    logger.info("=== DEPLOYMENT FIX INSTRUCTIONS ===")
    logger.info("Please follow these steps:")
    logger.info("1. In Replit dashboard, click the gear icon (Settings)")
    logger.info("2. Go to the 'Deployments' tab")
    logger.info("3. Change 'Command' to: gunicorn --bind 0.0.0.0:5000 app:app")
    logger.info("4. Change 'Deployment platform' to: Cloud Run")
    logger.info("5. Click Save/Update")
    logger.info("6. Click 'Stop' on the current deployment")
    logger.info("7. Click 'Deploy' to create a new deployment")
    logger.info("8. After deployment completes, try accessing your app again")
    logger.info("")
    logger.info("If it still doesn't work, try accessing these URLs:")
    logger.info("- https://quantum-shield-luisminier79.replit.app/health")
    logger.info("- https://quantum-shield-luisminier79.replit.app/deployment-test")

    return True


if __name__ == "__main__":
    main()
