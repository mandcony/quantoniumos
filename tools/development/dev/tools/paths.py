"""
QuantoniumOS Path Management System
==================================
Centralized path management for all file operations.
Ensures consistent, portable paths across all modules.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List


class QuantoniumPaths:
    """
    Centralized path management for QuantoniumOS.
    All paths are computed dynamically from project root.
    """
    
    def __init__(self):
        # Find project root by looking for marker files
        self._project_root = self._find_project_root()
        
    def _find_project_root(self) -> Path:
        """Find project root by looking for characteristic files"""
        current = Path(__file__).parent
        
        # Look for characteristic QuantoniumOS files
        markers = [
            'quantonium_os_main.py', 
            'COMPREHENSIVE_CODEBASE_ANALYSIS.md',
            'PROJECT_CANONICAL_CONTEXT.md',
            '.git'
        ]
        
        # Search up the directory tree
        for parent in [current] + list(current.parents):
            if any((parent / marker).exists() for marker in markers):
                return parent.absolute()
                
        # Fallback: assume we're in src/utils and go up 2 levels
        return current.parent.parent.absolute()
    
    @property
    def project_root(self) -> Path:
        """Root directory of QuantoniumOS project"""
        return self._project_root
    
    @property
    def src(self) -> Path:
        """Source code directory"""
        return self.project_root / "src"
    
    @property
    def core(self) -> Path:
        """Core quantum modules directory"""
        return self.project_root / "core"
    
    @property
    def assembly(self) -> Path:
        """Assembly/kernel directory"""
        return self.project_root / "ASSEMBLY"
    
    @property
    def kernel(self) -> Path:
        """Kernel source directory"""
        return self.assembly / "kernel"
    
    @property
    def python_bindings(self) -> Path:
        """Python bindings directory"""
        return self.assembly / "python_bindings"
    
    @property
    def apps(self) -> Path:
        """Applications directory"""
        return self.project_root / "apps"
    
    @property
    def frontend(self) -> Path:
        """Frontend/UI directory"""
        return self.project_root / "frontend"
    
    @property
    def ui(self) -> Path:
        """UI resources directory"""
        return self.project_root / "ui"
    
    @property
    def icons(self) -> Path:
        """Icons directory"""
        return self.ui / "icons"
    
    @property
    def styles(self) -> Path:
        """Styles directory"""
        return self.ui
    
    @property
    def tools(self) -> Path:
        """Development tools directory"""
        return self.project_root / "tools"
    
    @property
    def tests(self) -> Path:
        """Tests directory"""
        return self.project_root / "tests"
    
    @property
    def docs(self) -> Path:
        """Documentation directory"""
        return self.project_root / "docs"
    
    @property
    def build(self) -> Path:
        """Build output directory"""
        return self.assembly / "build"
    
    @property
    def compiled(self) -> Path:
        """Compiled libraries directory"""
        return self.assembly / "compiled"
    
    def get_app_path(self, app_name: str) -> Path:
        """Get path to specific app"""
        return self.apps / f"{app_name}.py"
    
    def get_icon_path(self, icon_name: str) -> Path:
        """Get path to specific icon"""
        return self.icons / icon_name
    
    def get_style_path(self, style_name: str) -> Path:
        """Get path to specific style file"""
        return self.styles / style_name
    
    def ensure_path_exists(self, path: Path) -> Path:
        """Ensure directory exists, create if needed"""
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_relative_to_root(self, path: Path) -> Path:
        """Get path relative to project root"""
        return path.relative_to(self.project_root)
    
    def find_library(self, lib_name: str) -> Optional[Path]:
        """Find compiled library in various possible locations"""
        possible_locations = [
            self.compiled / f"{lib_name}.dll",
            self.compiled / f"{lib_name}.so", 
            self.compiled / f"lib{lib_name}.dll",
            self.compiled / f"lib{lib_name}.so",
            self.build / f"{lib_name}.dll",
            self.build / f"{lib_name}.so",
            self.build / f"lib{lib_name}.dll", 
            self.build / f"lib{lib_name}.so",
            self.python_bindings / f"{lib_name}.dll",
            self.python_bindings / f"{lib_name}.so"
        ]
        
        for location in possible_locations:
            if location.exists():
                return location
                
        return None
    
    def get_all_paths(self) -> dict:
        """Get all paths as dictionary for debugging"""
        return {
            'project_root': str(self.project_root),
            'src': str(self.src),
            'core': str(self.core),
            'assembly': str(self.assembly),
            'kernel': str(self.kernel),
            'python_bindings': str(self.python_bindings),
            'apps': str(self.apps),
            'frontend': str(self.frontend),
            'ui': str(self.ui),
            'icons': str(self.icons),
            'tools': str(self.tools),
            'tests': str(self.tests),
            'build': str(self.build),
            'compiled': str(self.compiled)
        }


# Global instance for easy access
paths = QuantoniumPaths()


def get_project_root() -> Path:
    """Get project root directory"""
    return paths.project_root


def setup_python_path() -> None:
    """Add all necessary directories to Python path"""
    required_paths = [
        str(paths.project_root),
        str(paths.core),
        str(paths.python_bindings),
        str(paths.apps),
        str(paths.frontend),
        str(paths.tools)
    ]
    
    for path in required_paths:
        if path not in sys.path:
            sys.path.insert(0, path)


if __name__ == "__main__":
    # Debug: print all paths
    print("QuantoniumOS Path Configuration:")
    print("=" * 40)
    for name, path in paths.get_all_paths().items():
        exists = "✓" if Path(path).exists() else "✗"
        print(f"{exists} {name:<15}: {path}")
