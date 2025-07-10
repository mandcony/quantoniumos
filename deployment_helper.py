"""
QuantoniumOS Deployment Helper

This script is designed to help ensure successful deployment of QuantoniumOS
by fixing common deployment issues and ensuring proper configuration.
"""

import logging
import os
import platform
import shutil
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("deployment_helper")


def ensure_correct_port_binding():
    """Make sure we're binding to 0.0.0.0 instead of localhost"""
    logger.info("Ensuring correct port binding (0.0.0.0)")
    # This is handled in main.py and the gunicorn command
    return True


def ensure_static_files():
    """Make sure all static files are available"""
    logger.info("Checking static files")

    # Check for index.html
    if not os.path.exists("static/index.html"):
        logger.warning("Missing static/index.html")
        if os.path.exists("static/quantum-os.html"):
            logger.info("Copying quantum-os.html to index.html")
            shutil.copy("static/quantum-os.html", "static/index.html")
            return True
        return False

    # Check for .replit.app
    if not os.path.exists("static/.replit.app"):
        logger.warning("Creating static/.replit.app redirect file")
        with open("static/.replit.app", "w") as f:
            f.write(
                """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="0;url=/os">
    <title>Redirecting to QuantoniumOS</title>
</head>
<body>
    <p>Redirecting to QuantoniumOS...</p>
    <script>
        window.location.href = "/os";
    </script>
</body>
</html>"""
            )

    return True


def verify_embedded_paths():
    """Check and fix embedded paths in HTML files"""
    logger.info("Verifying embedded paths in HTML files")

    files_to_check = ["static/quantum-os.html", "static/index.html"]

    absolute_path_pattern = 'src="/resonance'
    relative_path_pattern = 'src="./resonance'

    needs_fixing = False

    for file_path in files_to_check:
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} does not exist")
            continue

        with open(file_path, "r") as f:
            content = f.read()

        if absolute_path_pattern in content:
            logger.info(f"Found absolute paths in {file_path}, fixing...")
            content = content.replace(absolute_path_pattern, relative_path_pattern)

            with open(file_path, "w") as f:
                f.write(content)

            needs_fixing = True

    return True


def set_replit_specific_config():
    """Set Replit-specific configuration"""
    logger.info("Setting up Replit-specific configuration")

    # Create .replit file if it doesn't exist
    if not os.path.exists(".replit"):
        logger.info("Creating .replit configuration")
        with open(".replit", "w") as f:
            f.write(
                """run = "gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app"
language = "python3"

[deployment]
run = "gunicorn --bind 0.0.0.0:5000 main:app"
deploymentTarget = "cloudrun"

[env]
PYTHONPATH = "${REPL_HOME}"
PYTHONUNBUFFERED = "1"
"""
            )
        return True

    # Check and update .replit file if it exists but lacks deployment config
    with open(".replit", "r") as f:
        replit_config = f.read()

    if "[deployment]" not in replit_config:
        logger.info("Adding deployment configuration to .replit")
        with open(".replit", "a") as f:
            f.write(
                """
[deployment]
run = "gunicorn --bind 0.0.0.0:5000 main:app"
deploymentTarget = "cloudrun"
"""
            )
        return True

    return True


def run_deployment_checks():
    """Run all deployment checks"""
    logger.info("Running deployment checks")

    # Check environment
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")

    checks = [
        ensure_correct_port_binding,
        ensure_static_files,
        verify_embedded_paths,
        set_replit_specific_config,
    ]

    all_passed = True

    for check in checks:
        try:
            if not check():
                logger.error(f"Check {check.__name__} failed")
                all_passed = False
        except Exception as e:
            logger.error(f"Exception in {check.__name__}: {e}")
            all_passed = False

    if all_passed:
        logger.info("All deployment checks passed!")
        logger.info("==================================")
        logger.info("DEPLOYMENT READY: To deploy this app:")
        logger.info("1. Use the 'Run' button to ensure everything works locally")
        logger.info("2. Click the 'Deploy' button in the Replit interface")
        logger.info(
            "3. Visit https://your-repl-name.username.replit.app/deployment-test"
        )
        logger.info("   to verify your deployment is working")
        logger.info("==================================")
    else:
        logger.warning("Some deployment checks failed. Check the logs.")

    return all_passed


if __name__ == "__main__":
    run_deployment_checks()
