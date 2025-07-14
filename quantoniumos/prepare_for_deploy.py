#!/usr/bin/env python3
"""
QuantoniumOS Deployment Preparation Script

This script prepares the application for deployment by checking for common issues
and ensuring all settings are optimized for production.

Usage:
    python prepare_for_deploy.py
"""

import subprocess
import sys
import os
import time

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def run_command(command):
    """Run a shell command and return whether it succeeded."""
    print(f"Running: {command}")
    try:
        subprocess.check_call(command, shell=True)
        print("✅ Command succeeded")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False

def main():
    print_header("QuantoniumOS Deployment Preparation")
    
    print("Checking system requirements...")
    
    # Run deployment helper
    print("\n📋 Running deployment helper checks...")
    import deployment_helper
    if not deployment_helper.run_deployment_checks():
        print("❌ Deployment checks failed. Please fix the issues before deploying.")
        return False
    
    # Check .replit file
    print("\n📋 Verifying .replit configuration...")
    if not os.path.exists(".replit"):
        print("❌ .replit file is missing. Run deployment_helper.py first.")
        return False
    
    # Check if we can access the app
    print("\n📋 Verifying local app access...")
    print("Note: This will attempt to curl the locally running app.")
    print("Make sure the app is running in another terminal before continuing.")
    
    input("Press Enter to continue, or Ctrl+C to cancel...")
    
    success = run_command("curl -s http://localhost:5000/health | grep -q 'ok'")
    if not success:
        print("⚠️ Could not verify local app is running. Is it started in another terminal?")
        retry = input("Continue anyway? (y/n): ")
        if retry.lower() != 'y':
            return False
    
    print("\n📋 Checking for absolute URL paths in static files...")
    if run_command("grep -r 'src=\"/' static/ | grep -v '/static/' | grep -q html"):
        print("⚠️ Found absolute URL paths in HTML files.")
        print("   These should be changed to relative paths (e.g., src=\"./path\" instead of src=\"/path\").")
        fix = input("Attempt to automatically fix these issues? (y/n): ")
        if fix.lower() == 'y':
            run_command("find static -name '*.html' -type f -exec sed -i 's/src=\"\\//src=\".\\//' {} \\;")
            run_command("find static -name '*.html' -type f -exec sed -i 's/href=\"\\//href=\".\\//' {} \\;")
            print("✅ Fixed absolute paths in HTML files.")
    
    # Deployment summary
    print("\n" + "=" * 80)
    print("📋 DEPLOYMENT PREPARATION SUMMARY")
    print("=" * 80)
    print("Your QuantoniumOS application is ready for deployment!")
    print("\nTo deploy this app:")
    print("1. Run the application locally to verify all features")
    print("2. Click the 'Deploy' button in the Replit interface")
    print("3. Visit your deployed app at: https://<your-repl-name>.<username>.replit.app")
    print("4. Test the deployment using: https://<your-repl-name>.<username>.replit.app/deployment-test")
    print("\nNote: If you encounter any 'refused to connect' errors after deployment,")
    print("check that your .replit file has the correct deployment configuration.")
    
    print("\n" + "=" * 80)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)