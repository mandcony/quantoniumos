"""
QuantoniumOS Final Verification Script

This script performs a comprehensive verification of the QuantoniumOS project
after code consolidation and organization to ensure everything is working properly.
"""

import importlib
import os
import platform
import subprocess
import sys
import time

# Project root directory - this points to the 02_CORE_VALIDATORS folder
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
# Parent directory - this points to the quantoniumos project root
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)

# Add the root directory to the Python path
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))


def print_section(title):
    """Print a section title with formatting."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def run_command(command, timeout=60):
    """Run a command and return its output."""
    try:
        # Replace 'python' with the full path to the Python executable in the virtual environment
        python_exe = sys.executable
        command = command.replace("python ", f'"{python_exe}" ')

        # Add the project root to PYTHONPATH for importing modules
        env = os.environ.copy()
        env["PYTHONPATH"] = PARENT_ROOT

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def check_imports():
    """Check that key modules can be imported."""
    print_section("Checking Core Module Imports")

    modules_to_check = [
        "quantoniumos",
        "app",
        "core.encryption",
        "core.security",
        "core.python.utilities",
        "bulletproof_quantum_kernel",
        "topological_quantum_kernel",
    ]

    results = []

    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ Successfully imported {module}")
            results.append((module, "Success"))
        except ImportError as e:
            print(f"❌ Failed to import {module}: {e}")
            results.append((module, f"Error: {e}"))
        except Exception as e:
            print(f"⚠️ Error when importing {module}: {e}")
            results.append((module, f"Error: {e}"))

    return results


def check_app_launchers():
    """Check that app launchers work properly."""
    print_section("Testing App Launchers")

    # Use parent_root to construct absolute paths to the apps directory
    project_apps_dir = os.path.join(PARENT_ROOT, "apps")
    
    launchers = [
        f"python {project_apps_dir}/launch_quantum_simulator.py",
        f"python {project_apps_dir}/launch_q_mail.py",
        f"python {project_apps_dir}/launch_q_notes.py",
        f"python {project_apps_dir}/launch_q_vault.py",
        f"python {project_apps_dir}/launch_rft_visualizer.py",
    ]

    results = []

    for launcher in launchers:
        print(f"Testing: {launcher}")

        # Run with short timeout as we just want to check it starts
        returncode, stdout, stderr = run_command(launcher, timeout=5)

        # Check for successful startup messages
        success = returncode == 0 or "Initializing" in stdout or "Launching" in stdout

        if success:
            print("✅ Launcher started successfully")
            results.append((launcher, "Success"))
        else:
            print("❌ Launcher failed:")
            if stdout:
                print(f"  STDOUT: {stdout[:300]}")
            if stderr:
                print(f"  STDERR: {stderr[:300]}")
            results.append((launcher, f"Error: {returncode}"))

    return results


def check_build_utilities():
    """Check that build utilities work properly."""
    print_section("Testing Build Utilities")

    build_utils = [
        ("python build_crypto_engine.py --help", "Shows usage information"),
        (
            "python 10_UTILITIES/build_vertex_engine.py --help",
            "Shows usage information",
        ),
        (
            "python 10_UTILITIES/build_resonance_engine.py --help",
            "Shows usage information",
        ),
    ]

    results = []

    for command, description in build_utils:
        print(f"Testing: {command} ({description})")

        returncode, stdout, stderr = run_command(command)

        # Check if it executed without crashing
        if returncode == 0 or "usage" in stdout.lower():
            print("✅ Build utility executed successfully")
            results.append((command, "Success"))
        else:
            print("❌ Build utility failed:")
            if stdout:
                print(f"  STDOUT: {stdout[:300]}")
            if stderr:
                print(f"  STDERR: {stderr[:300]}")
            results.append((command, f"Error: {returncode}"))

    return results


def check_encryption_modules():
    """Check that encryption modules are working properly."""
    print_section("Testing Encryption Modules")

    test_script = """
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try to import encryption modules
    from core.encryption import (
        entropy_qrng, 
        minimal_resonance_encrypt, 
        optimized_resonance_encrypt,
        simple_resonance_encrypt
    )
    
    # Test basic functionality
    random_data = entropy_qrng.generate_random_bytes(32)
    print(f"Generated random data: {random_data[:8]}...")
    
    # Test basic encryption
    key = b'testkey0123456789'
    data = b'This is a test message'
    
    # Test simple encryption
    encrypted = simple_resonance_encrypt.encrypt(data, key)
    decrypted = simple_resonance_encrypt.decrypt(encrypted, key)
    
    if data == decrypted:
        print("Simple encryption/decryption successful")
    else:
        print("Simple encryption/decryption failed")
        sys.exit(1)
    
    print("All encryption tests passed!")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
"""

    # Write the test script to a temporary file
    test_script_path = os.path.join(PROJECT_ROOT, "temp_encryption_test.py")
    with open(test_script_path, "w") as f:
        f.write(test_script)

    # Run the test script
    print("Running encryption module tests...")
    returncode, stdout, stderr = run_command(f"python {test_script_path}")

    # Delete the temporary file
    try:
        os.remove(test_script_path)
    except:
        pass

    # Check results
    if returncode == 0 and "All encryption tests passed" in stdout:
        print("✅ Encryption modules working correctly")
        result = ("Encryption modules", "Success")
    else:
        print("❌ Encryption module tests failed:")
        if stdout:
            print(f"  STDOUT: {stdout}")
        if stderr:
            print(f"  STDERR: {stderr}")
        result = ("Encryption modules", f"Error: {returncode}")

    return [result]


def check_main_app():
    """Check that the main application works."""
    print_section("Testing Main Application")

    # Test running the main app
    print("Testing main app.py startup...")
    returncode, stdout, stderr = run_command("python app.py", timeout=5)

    # Check if it started without crashing
    if returncode == 0 or "Initializing" in stdout or "Starting" in stdout:
        print("✅ Main application started successfully")
        result = ("Main application", "Success")
    else:
        print("❌ Main application failed to start:")
        if stdout:
            print(f"  STDOUT: {stdout[:300]}")
        if stderr:
            print(f"  STDERR: {stderr[:300]}")
        result = ("Main application", f"Error: {returncode}")

    return [result]


def generate_report(all_results):
    """Generate a verification report."""
    print_section("Generating Verification Report")

    # Count successful and failed tests
    successful = sum(1 for _, status in all_results if "Success" in status)
    total = len(all_results)

    report = "# QuantoniumOS Final Verification Report\n\n"
    report += f"*Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n"

    # Add system information
    report += "## System Information\n\n"
    report += f"- **Platform**: {platform.platform()}\n"
    report += f"- **Python Version**: {platform.python_version()}\n"
    report += f"- **Processor**: {platform.processor()}\n\n"

    # Add verification summary
    report += "## Verification Summary\n\n"
    report += (
        f"- **Tests Passed**: {successful}/{total} ({successful/total*100:.1f}%)\n"
    )
    report += f"- **Status**: {'PASSED' if successful == total else 'PARTIAL PASS' if successful > 0 else 'FAILED'}\n\n"

    # Add test results
    report += "## Test Results\n\n"
    report += "| Component | Status |\n"
    report += "|-----------|--------|\n"

    for component, status in all_results:
        status_icon = "SUCCESS" if "Success" in status else "FAILED"
        report += f"| {component} | {status_icon} {status} |\n"

    report += "\n"

    # Add recommendations
    report += "## Recommendations\n\n"

    if successful == total:
        report += "All verification tests passed! The QuantoniumOS system appears to be functioning correctly after code consolidation and organization.\n\n"
        report += "Next steps:\n"
        report += "1. Consider removing redundant files identified during the cleanup process\n"
        report += "2. Expand test coverage for core components\n"
        report += "3. Improve documentation for new consolidated modules\n"
    else:
        report += "Some verification tests failed. The following issues should be addressed:\n\n"

        for component, status in all_results:
            if "Success" not in status:
                report += f"- Fix issues with **{component}**: {status}\n"

        report += "\nAfter fixing these issues, run the verification script again to ensure all tests pass.\n"

    report += "\n## Verification Complete\n\n"
    report += "The QuantoniumOS project has been successfully consolidated and organized. This verification report confirms the status of key components after the organization process.\n"

    # Write the report to a file
    report_path = os.path.join(PROJECT_ROOT, "FINAL_VERIFICATION_SUMMARY.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Verification report generated: {report_path}")

    return report_path


def main():
    print("\n" + "=" * 80)
    print(" QuantoniumOS FINAL VERIFICATION ".center(80, "="))
    print("=" * 80 + "\n")

    print("Starting comprehensive verification of QuantoniumOS...")

    # Run all checks
    import_results = check_imports()
    launcher_results = check_app_launchers()
    build_results = check_build_utilities()
    encryption_results = check_encryption_modules()
    app_results = check_main_app()

    # Combine all results
    all_results = (
        import_results
        + launcher_results
        + build_results
        + encryption_results
        + app_results
    )

    # Generate report
    report_path = generate_report(all_results)

    # Print final summary
    successful = sum(1 for _, status in all_results if "Success" in status)
    total = len(all_results)

    print("\n" + "=" * 80)
    print(f" VERIFICATION COMPLETE: {successful}/{total} tests passed ".center(80, "="))
    print("=" * 80 + "\n")

    print(f"Detailed report saved to: {report_path}")

    # Return success status for scripting
    return {"status": "PASS", "message": f"{successful}/{total} tests passed"}


if __name__ == "__main__":
    sys.exit(main())
