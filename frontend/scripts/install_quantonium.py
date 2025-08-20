"""
QuantoniumOS Installation and Setup Script
Automated installer for the complete frontend system
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import json
import urllib.request
import zipfile

class QuantoniumInstaller:
    """Complete installer for QuantoniumOS frontend system"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.project_root = Path(__file__).parent.parent
        self.frontend_root = self.project_root / "frontend"
        
        self.requirements = [
            "PyQt5>=5.15.0",
            "psutil>=5.9.0",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "pillow>=8.0.0",
            "pywin32>=227;platform_system=='Windows'",
            "requests>=2.25.0"
        ]
        
        print(f"🌌 QuantoniumOS Frontend Installer")
        print(f"System: {self.system}")
        print(f"Python: {self.python_version}")
        print(f"Project Root: {self.project_root}")
        print("=" * 50)
    
    def check_requirements(self):
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        
        print(f"✅ Python {self.python_version}")
        
        # Check pip
        try:
            import pip
            print(f"✅ pip available")
        except ImportError:
            print("❌ pip not available")
            return False
        
        # Check if we can create directories
        try:
            test_dir = self.project_root / ".quantonium_test"
            test_dir.mkdir(exist_ok=True)
            test_dir.rmdir()
            print("✅ Write permissions OK")
        except Exception as e:
            print(f"❌ Write permissions failed: {e}")
            return False
        
        return True
    
    def install_python_packages(self):
        """Install required Python packages"""
        print("📦 Installing Python packages...")
        
        for requirement in self.requirements:
            print(f"Installing {requirement}...")
            
            # Handle platform-specific requirements
            if "platform_system" in requirement and self.system not in requirement:
                print(f"⏭️ Skipping {requirement} (not needed on {self.system})")
                continue
            
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    requirement.split(';')[0]  # Remove platform specifier
                ], check=True, capture_output=True)
                print(f"✅ {requirement}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {requirement}: {e}")
                return False
        
        return True
    
    def setup_directory_structure(self):
        """Create the complete directory structure"""
        print("📁 Setting up directory structure...")
        
        directories = [
            "frontend",
            "frontend/styles",
            "frontend/ui",
            "frontend/components",
            "frontend/extensions",
            "frontend/resources",
            "frontend/themes",
            "frontend/templates",
            "frontend/scripts",
            ".quantonium",
            ".quantonium/sessions",
            ".quantonium/logs",
            ".quantonium/cache"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {directory}")
        
        return True
    
    def install_vscode_extension(self):
        """Install VS Code extension"""
        print("🔌 Installing VS Code extension...")
        
        try:
            # Check if VS Code is installed
            vscode_cmd = "code" if self.system != "Windows" else "code.cmd"
            
            # Test VS Code availability
            result = subprocess.run([vscode_cmd, "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("⚠️ VS Code not found - extension will be available for manual install")
                return True
            
            print(f"✅ VS Code detected: {result.stdout.split()[0]}")
            
            # Install extension from local directory
            extension_path = self.frontend_root / "extensions"
            if extension_path.exists():
                result = subprocess.run([
                    vscode_cmd, "--install-extension", str(extension_path)
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ VS Code extension installed")
                else:
                    print(f"⚠️ Extension install warning: {result.stderr}")
            else:
                print("⚠️ Extension directory not found")
            
            return True
            
        except Exception as e:
            print(f"⚠️ VS Code extension setup failed: {e}")
            return True  # Non-critical
    
    def create_launch_scripts(self):
        """Create platform-specific launch scripts"""
        print("🚀 Creating launch scripts...")
        
        scripts_dir = self.frontend_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Windows PowerShell script
        if self.system == "Windows":
            ps_script = scripts_dir / "launch_quantonium.ps1"
            ps_content = f'''# QuantoniumOS Launch Script
Write-Host "🌌 Starting QuantoniumOS..." -ForegroundColor Cyan

$ProjectRoot = "{self.project_root}"
$PythonScript = Join-Path $ProjectRoot "frontend\\ui\\quantum_app_controller.py"

# Set environment
$env:PYTHONPATH = $ProjectRoot

# Check if script exists
if (Test-Path $PythonScript) {{
    Write-Host "✅ Found QuantoniumOS script" -ForegroundColor Green
    
    # Launch QuantoniumOS
    try {{
        & python "$PythonScript"
        Write-Host "✅ QuantoniumOS launched successfully!" -ForegroundColor Green
    }}
    catch {{
        Write-Host "❌ Error launching QuantoniumOS: $($_.Exception.Message)" -ForegroundColor Red
        Read-Host "Press Enter to continue..."
    }}
}} else {{
    Write-Host "❌ QuantoniumOS script not found: $PythonScript" -ForegroundColor Red
    Read-Host "Press Enter to continue..."
}}
'''
            with open(ps_script, 'w') as f:
                f.write(ps_content)
            print("✅ PowerShell script created")
            
            # Windows batch script
            bat_script = scripts_dir / "launch_quantonium.bat"
            bat_content = f'''@echo off
echo 🌌 Starting QuantoniumOS...

set PROJECT_ROOT={self.project_root}
set PYTHON_SCRIPT=%PROJECT_ROOT%\\frontend\\ui\\quantum_app_controller.py
set PYTHONPATH=%PROJECT_ROOT%

if exist "%PYTHON_SCRIPT%" (
    echo ✅ Found QuantoniumOS script
    python "%PYTHON_SCRIPT%"
    echo ✅ QuantoniumOS launched successfully!
) else (
    echo ❌ QuantoniumOS script not found: %PYTHON_SCRIPT%
    pause
)
'''
            with open(bat_script, 'w') as f:
                f.write(bat_content)
            print("✅ Batch script created")
        
        # Unix shell script
        else:
            sh_script = scripts_dir / "launch_quantonium.sh"
            sh_content = f'''#!/bin/bash
echo "🌌 Starting QuantoniumOS..."

PROJECT_ROOT="{self.project_root}"
PYTHON_SCRIPT="$PROJECT_ROOT/frontend/ui/quantum_app_controller.py"
export PYTHONPATH="$PROJECT_ROOT"

if [ -f "$PYTHON_SCRIPT" ]; then
    echo "✅ Found QuantoniumOS script"
    python3 "$PYTHON_SCRIPT"
    echo "✅ QuantoniumOS launched successfully!"
else
    echo "❌ QuantoniumOS script not found: $PYTHON_SCRIPT"
    read -p "Press Enter to continue..."
fi
'''
            with open(sh_script, 'w') as f:
                f.write(sh_content)
            
            # Make executable
            os.chmod(sh_script, 0o755)
            print("✅ Shell script created")
        
        return True
    
    def create_desktop_shortcuts(self):
        """Create desktop shortcuts (Windows)"""
        if self.system != "Windows":
            return True
        
        print("🖥️ Creating desktop shortcuts...")
        
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            shortcut_path = os.path.join(desktop, "QuantoniumOS.lnk")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = str(self.frontend_root / "scripts" / "launch_quantonium.bat")
            shortcut.WorkingDirectory = str(self.project_root)
            shortcut.IconLocation = str(self.frontend_root / "resources" / "quantum-icon.ico")
            shortcut.Description = "🌌 QuantoniumOS - Advanced Quantum Operating System"
            shortcut.save()
            
            print("✅ Desktop shortcut created")
            return True
            
        except ImportError:
            print("⚠️ winshell not available - desktop shortcut skipped")
            return True
        except Exception as e:
            print(f"⚠️ Desktop shortcut failed: {e}")
            return True
    
    def create_config_files(self):
        """Create configuration files"""
        print("⚙️ Creating configuration files...")
        
        config_dir = self.project_root / ".quantonium"
        
        # Main configuration
        main_config = {
            "version": "2.0.0",
            "frontend": {
                "theme": "quantum-dark",
                "window_management": True,
                "animations": True,
                "auto_save_session": True
            },
            "backend": {
                "auto_start": True,
                "quantum_validation": True,
                "crypto_validation": True
            },
            "vscode": {
                "integration": True,
                "auto_analyze": True,
                "status_bar": True
            },
            "paths": {
                "project_root": str(self.project_root),
                "frontend_root": str(self.frontend_root),
                "python_executable": sys.executable
            }
        }
        
        config_file = config_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(main_config, f, indent=2)
        
        print("✅ Main configuration created")
        
        # Window management configuration
        window_config = {
            "default_arrangements": ["cascade", "tile_horizontal", "tile_vertical"],
            "animation_duration": 300,
            "window_spacing": 10,
            "minimum_window_size": [400, 300],
            "auto_arrange": False
        }
        
        window_config_file = config_dir / "window_config.json"
        with open(window_config_file, 'w') as f:
            json.dump(window_config, f, indent=2)
        
        print("✅ Window configuration created")
        
        return True
    
    def run_tests(self):
        """Run basic functionality tests"""
        print("🧪 Running installation tests...")
        
        # Test Python imports
        test_imports = ["PyQt5", "psutil", "numpy", "matplotlib"]
        for module in test_imports:
            try:
                __import__(module)
                print(f"✅ {module}")
            except ImportError:
                print(f"❌ {module} not available")
                return False
        
        # Test file structure
        required_files = [
            "frontend/styles/quantonium_master.qss",
            "frontend/components/window_manager.py",
            "frontend/ui/quantum_app_controller.py",
            "frontend/extensions/quantonium_vscode.js",
            "frontend/extensions/package.json"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} missing")
                return False
        
        print("🎉 All tests passed!")
        return True
    
    def install(self):
        """Run complete installation"""
        print("🚀 Starting QuantoniumOS Frontend Installation...")
        print("=" * 50)
        
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Installing Python packages", self.install_python_packages),
            ("Setting up directories", self.setup_directory_structure),
            ("Installing VS Code extension", self.install_vscode_extension),
            ("Creating launch scripts", self.create_launch_scripts),
            ("Creating desktop shortcuts", self.create_desktop_shortcuts),
            ("Creating configuration", self.create_config_files),
            ("Running tests", self.run_tests)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            if not step_func():
                print(f"❌ Installation failed at: {step_name}")
                return False
            print(f"✅ {step_name} completed")
        
        print("\n" + "=" * 50)
        print("🎉 QuantoniumOS Frontend Installation Complete!")
        print("\n📋 Next steps:")
        print("1. 🚀 Launch QuantoniumOS from desktop shortcut or scripts/")
        print("2. 🔌 Use Ctrl+Shift+Q in VS Code to launch from extension")
        print("3. 📖 Check QUANTONIUM_FRONTEND_DESIGN_MANUAL.md for details")
        print("4. ⚙️ Customize settings in .quantonium/config.json")
        
        return True

def main():
    """Main installer entry point"""
    installer = QuantoniumInstaller()
    
    try:
        success = installer.install()
        if success:
            print("\n🌌 Welcome to QuantoniumOS! 🌌")
            sys.exit(0)
        else:
            print("\n❌ Installation failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n⚠️ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
