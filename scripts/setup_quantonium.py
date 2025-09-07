"""
QuantoniumOS Setup & Validation Script
=====================================
Validates project structure, fixes paths, and prepares for GitHub deployment.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from utils.paths import paths
    from utils.config import config
    from utils.imports import imports
except ImportError as e:
    print(f"Could not import utilities (expected during setup): {e}")
    
    # Minimal path setup for initial run
    class TempPaths:
        def __init__(self):
            self.project_root = project_root
            self.apps = project_root / "apps"
            self.ui = project_root / "ui"
            self.icons = self.ui / "icons"
            
    paths = TempPaths()


def check_python_environment():
    """Check Python environment and dependencies"""
    print("🐍 Checking Python Environment...")
    
    print(f"  Python version: {sys.version}")
    print(f"  Python executable: {sys.executable}")
    
    # Check required packages
    required_packages = [
        "PyQt5", "numpy", "scipy", "qtawesome", "pytz"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("✓ All required packages available")
        return True


def validate_project_structure():
    """Validate current project structure"""
    print("\n📁 Validating Project Structure...")
    
    required_dirs = [
        "apps", "ASSEMBLY", "core", "frontend", "ui", "tools"
    ]
    
    required_files = [
        "quantonium_os_main.py", "README.md", ".gitignore"
    ]
    
    issues = []
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = paths.project_root / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            issues.append(f"Missing directory: {dir_name}")
    
    # Check files
    for file_name in required_files:
        file_path = paths.project_root / file_name
        if file_path.exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} (missing)")
            issues.append(f"Missing file: {file_name}")
    
    return issues


def check_app_files():
    """Check application files"""
    print("\n🎯 Checking Application Files...")
    
    if not paths.apps.exists():
        print("  ✗ Apps directory not found")
        return ["Apps directory missing"]
    
    issues = []
    app_files = list(paths.apps.glob("*.py"))
    
    print(f"  Found {len(app_files)} Python files in apps/")
    
    for app_file in app_files:
        if app_file.name != "__init__.py":
            print(f"  ✓ {app_file.name}")
    
    # Check for launcher_base.py (important for app system)
    launcher_base = paths.apps / "launcher_base.py"
    if launcher_base.exists():
        print(f"  ✓ launcher_base.py (app framework)")
    else:
        print(f"  ⚠ launcher_base.py (missing, may cause issues)")
        issues.append("Missing launcher_base.py")
    
    return issues


def check_ui_resources():
    """Check UI resources"""
    print("\n🎨 Checking UI Resources...")
    
    issues = []
    
    # Check UI directory
    if paths.ui.exists():
        print(f"  ✓ ui/ directory")
        
        # Check for styles
        style_files = list(paths.ui.glob("*.qss"))
        print(f"  Found {len(style_files)} style files:")
        for style_file in style_files:
            print(f"    ✓ {style_file.name}")
            
        # Check icons
        if paths.icons.exists():
            icon_files = list(paths.icons.glob("*.svg"))
            print(f"  Found {len(icon_files)} icon files:")
            for icon_file in icon_files:
                print(f"    ✓ {icon_file.name}")
        else:
            print(f"  ⚠ icons/ directory missing")
            issues.append("Missing icons directory")
    else:
        print(f"  ✗ ui/ directory missing")
        issues.append("Missing ui directory")
        
    return issues


def check_kernel_system():
    """Check kernel/assembly system"""
    print("\n⚙️ Checking Kernel System...")
    
    issues = []
    assembly_dir = paths.project_root / "ASSEMBLY"
    
    if assembly_dir.exists():
        print(f"  ✓ ASSEMBLY/ directory")
        
        # Check for key subdirectories
        subdirs = ["kernel", "python_bindings", "build"]
        for subdir in subdirs:
            subdir_path = assembly_dir / subdir
            if subdir_path.exists():
                file_count = len(list(subdir_path.glob("*")))
                print(f"  ✓ {subdir}/ ({file_count} files)")
            else:
                print(f"  ⚠ {subdir}/ (missing)")
                issues.append(f"Missing ASSEMBLY/{subdir}")
                
        # Check for unitary_rft.py (key binding)
        unitary_rft = assembly_dir / "python_bindings" / "unitary_rft.py"
        if unitary_rft.exists():
            print(f"  ✓ unitary_rft.py (kernel binding)")
        else:
            print(f"  ⚠ unitary_rft.py (missing)")
            issues.append("Missing kernel binding")
    else:
        print(f"  ✗ ASSEMBLY/ directory missing")
        issues.append("Missing ASSEMBLY directory")
        
    return issues


def test_path_resolution():
    """Test the new path resolution system"""
    print("\n🔍 Testing Path Resolution...")
    
    try:
        from utils.paths import paths as new_paths
        
        print("  ✓ Path utilities imported successfully")
        
        # Test key paths
        test_paths = {
            "project_root": new_paths.project_root,
            "apps": new_paths.apps,
            "assembly": new_paths.assembly,
            "ui": new_paths.ui,
            "tools": new_paths.tools
        }
        
        for name, path in test_paths.items():
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {name}: {path}")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Path resolution failed: {e}")
        return False


def test_import_system():
    """Test the new import system"""
    print("\n📦 Testing Import System...")
    
    try:
        from utils.imports import imports
        
        print("  ✓ Import utilities loaded")
        
        # Test kernel import
        kernel = imports.import_kernel()
        if kernel:
            print("  ✓ Kernel import successful")
        else:
            print("  ⚠ Kernel import failed (may be expected)")
        
        # Test import status
        status = imports.get_import_status()
        print("  Import status:")
        for name, success in status.items():
            symbol = "✓" if success else "✗"
            print(f"    {symbol} {name}")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Import system failed: {e}")
        return False


def test_configuration_system():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        from utils.config import config
        
        print("  ✓ Configuration utilities loaded")
        
        # Test app registry
        apps = config.get_app_registry()
        print(f"  ✓ App registry loaded ({len(apps)} apps)")
        
        # Test build config
        build_config = config.get_build_config()
        print(f"  ✓ Build config loaded (compiler: {build_config.compiler})")
        
        # Validate configuration
        issues = config.validate_config()
        if issues:
            print("  Configuration issues:")
            for issue in issues:
                print(f"    ⚠ {issue}")
        else:
            print("  ✓ Configuration is valid")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration system failed: {e}")
        return False


def create_missing_directories():
    """Create any missing directories"""
    print("\n📁 Creating Missing Directories...")
    
    dirs_to_create = [
        "src/utils",
        "tests",
        "tools", 
        "docs"
    ]
    
    for dir_path in dirs_to_create:
        full_path = paths.project_root / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_path}/")
        else:
            print(f"  ✓ {dir_path}/ already exists")


def fix_gitignore():
    """Ensure .gitignore is properly configured"""
    print("\n🚫 Checking .gitignore...")
    
    gitignore_path = paths.project_root / ".gitignore"
    
    required_entries = [
        "personalAi/",
        "__pycache__/",
        "*.pyc",
        ".vs/",
        ".vscode/",
        "build/",
        "*.dll",
        "*.so",
        "*.log",
        ".env"
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
            
        missing = [entry for entry in required_entries if entry not in content]
        
        if missing:
            print(f"  ⚠ Missing entries: {missing}")
            with open(gitignore_path, 'a') as f:
                f.write("\n# Added by setup script\n")
                for entry in missing:
                    f.write(f"{entry}\n")
            print(f"  ✓ Added missing entries to .gitignore")
        else:
            print(f"  ✓ .gitignore is complete")
    else:
        print(f"  ✗ .gitignore missing")
        return False
        
    return True


def create_setup_py():
    """Create setup.py for package installation"""
    print("\n📦 Creating setup.py...")
    
    setup_py_content = '''"""
QuantoniumOS Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="quantoniumos",
    version="1.0.0",
    description="Quantum Operating System with RFT Kernel",
    author="QuantoniumOS Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "PyQt5>=5.15.0",
        "numpy>=1.20.0", 
        "scipy>=1.7.0",
        "qtawesome>=1.0.0",
        "pytz>=2021.1"
    ],
    entry_points={
        "console_scripts": [
            "quantonium=quantonium:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Scientists",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.8+",
    ]
)
'''
    
    setup_path = paths.project_root / "setup.py"
    if not setup_path.exists():
        with open(setup_path, 'w') as f:
            f.write(setup_py_content)
        print("  ✓ Created setup.py")
    else:
        print("  ✓ setup.py already exists")


def create_requirements_txt():
    """Create requirements.txt"""
    print("\n📋 Creating requirements.txt...")
    
    requirements = [
        "PyQt5>=5.15.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0", 
        "qtawesome>=1.0.0",
        "pytz>=2021.1"
    ]
    
    req_path = paths.project_root / "requirements.txt"
    if not req_path.exists():
        with open(req_path, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        print("  ✓ Created requirements.txt")
    else:
        print("  ✓ requirements.txt already exists")


def run_full_validation():
    """Run complete project validation"""
    print("🚀 QuantoniumOS Project Validation & Setup")
    print("=" * 50)
    
    all_issues = []
    
    # Basic checks
    env_ok = check_python_environment()
    struct_issues = validate_project_structure()
    app_issues = check_app_files()
    ui_issues = check_ui_resources()
    kernel_issues = check_kernel_system()
    
    all_issues.extend(struct_issues)
    all_issues.extend(app_issues)
    all_issues.extend(ui_issues)
    all_issues.extend(kernel_issues)
    
    # Test new systems
    path_ok = test_path_resolution()
    import_ok = test_import_system()
    config_ok = test_configuration_system()
    
    # Setup tasks
    create_missing_directories()
    gitignore_ok = fix_gitignore()
    create_setup_py()
    create_requirements_txt()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    print(f"Environment:     {'✓' if env_ok else '✗'}")
    print(f"Path System:     {'✓' if path_ok else '✗'}")
    print(f"Import System:   {'✓' if import_ok else '✗'}")
    print(f"Configuration:   {'✓' if config_ok else '✗'}")
    print(f"Git Configuration: {'✓' if gitignore_ok else '✗'}")
    
    if all_issues:
        print(f"\n⚠ Issues Found ({len(all_issues)}):")
        for issue in all_issues:
            print(f"  • {issue}")
    else:
        print(f"\n✓ No critical issues found!")
        
    print(f"\n🎯 GitHub Readiness:")
    github_ready = env_ok and path_ok and gitignore_ok and len(all_issues) == 0
    print(f"  Status: {'✓ READY' if github_ready else '⚠ NEEDS ATTENTION'}")
    
    if github_ready:
        print(f"\n🚀 QuantoniumOS is ready for GitHub deployment!")
        print(f"  • Run: python engine/launch_quantonium_os.py")
        print(f"  • Install: pip install -e .")
        print(f"  • Deploy: git add . && git commit && git push")
    else:
        print(f"\n🔧 Fix the issues above before deployment.")
        
    return github_ready


if __name__ == "__main__":
    run_full_validation()
