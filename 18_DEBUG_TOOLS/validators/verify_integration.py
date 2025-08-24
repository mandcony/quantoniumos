"""
QuantoniumOS Integration Verification
Quick verification that all Phase 3 & 4 components are properly integrated
"""

import importlib
import os
import sys
from pathlib import Path


def colored_print(text, color="white"):
    """Print colored text"""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")


def check_file_exists(filepath):
    """Check if file exists"""
    if os.path.exists(filepath):
        colored_print(f"✓ {filepath}", "green")
        return True
    else:
        colored_print(f"✗ {filepath}", "red")
        return False


def check_import(module_name, friendly_name=None):
    """Check if module can be imported"""
    if friendly_name is None:
        friendly_name = module_name

    try:
        importlib.import_module(module_name)
        colored_print(f"✓ {friendly_name}", "green")
        return True
    except ImportError as e:
        colored_print(f"✗ {friendly_name} ({e})", "red")
        return False


def main():
    """Main verification function"""
    colored_print("=" * 60, "cyan")
    colored_print("QuantoniumOS Phase 3 & 4 Integration Verification", "cyan")
    colored_print("=" * 60, "cyan")
    print()

    # Set up paths
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

    success_count = 0
    total_checks = 0

    # Check core files
    colored_print("🔍 Checking Core System Files:", "blue")
    core_files = [
        "quantoniumos.py",
        "launch_quantoniumos.ps1",
        "launch_quantoniumos.bat",
        "PHASE3_PHASE4_INTEGRATION_GUIDE.md",
    ]

    for file in core_files:
        total_checks += 1
        if check_file_exists(file):
            success_count += 1

    print()

    # Check Phase 3 files
    colored_print("🔗 Checking Phase 3 API Integration:", "blue")
    phase3_files = [
        "phase3/api_integration/function_wrapper.py",
        "phase3/bridges/quantum_classical_bridge.py",
        "phase3/services/service_orchestrator.py",
    ]

    for file in phase3_files:
        total_checks += 1
        if check_file_exists(file):
            success_count += 1

    print()

    # Check Phase 4 files
    colored_print("📱 Checking Phase 4 Applications:", "blue")
    phase4_files = [
        "phase4/applications/rft_visualizer.py",
        "phase4/applications/quantum_crypto_playground.py",
        "phase4/applications/patent_validation_dashboard.py",
    ]

    for file in phase4_files:
        total_checks += 1
        if check_file_exists(file):
            success_count += 1

    print()

    # Check unified OS interface
    colored_print("🖥️ Checking Unified OS Interface:", "blue")
    os_files = ["11_QUANTONIUMOS/quantonium_os_advanced.py"]

    for file in os_files:
        total_checks += 1
        if check_file_exists(file):
            success_count += 1

    print()

    # Check Python dependencies
    colored_print("🐍 Checking Python Dependencies:", "blue")
    dependencies = [
        ("tkinter", "Tkinter (GUI)"),
        ("numpy", "NumPy"),
        ("flask", "Flask (Web)"),
        ("matplotlib", "Matplotlib"),
        ("threading", "Threading"),
        ("asyncio", "AsyncIO"),
        ("pathlib", "PathLib"),
        ("logging", "Logging"),
    ]

    for module, name in dependencies:
        total_checks += 1
        if check_import(module, name):
            success_count += 1

    print()

    # Check optional dependencies
    colored_print("🔬 Checking Optional Dependencies:", "yellow")
    optional_deps = [("cryptography", "Cryptography"), ("scipy", "SciPy")]

    for module, name in optional_deps:
        check_import(module, name)

    print()

    # Summary
    colored_print("📊 Verification Summary:", "magenta")
    print(f"Passed: {success_count}/{total_checks} checks")

    if success_count == total_checks:
        colored_print(
            "🎉 All systems operational! QuantoniumOS is ready to launch!", "green"
        )
        print()
        colored_print("Quick Start Commands:", "cyan")
        print("  Desktop Mode:  .\\launch_quantoniumos.ps1")
        print("  Web Mode:      .\\launch_quantoniumos.ps1 web")
        print("  System Info:   .\\launch_quantoniumos.ps1 info")
        print("  Run Tests:     .\\launch_quantoniumos.ps1 test")
        return True
    else:
        colored_print(
            f"⚠️ {total_checks - success_count} issues found. Please check the errors above.",
            "yellow",
        )
        return False


if __name__ == "__main__":
    success = main()
    print()
    input("Press Enter to exit...")
    sys.exit(0 if success else 1)
