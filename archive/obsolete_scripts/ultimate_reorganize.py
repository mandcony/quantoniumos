#!/usr/bin/env python3
"""
🚀 ULTIMATE PROJECT REORGANIZATION
Create the most logical and professional structure possible
"""

import os
import shutil
from pathlib import Path

def ultimate_reorganization():
    """Create the ultimate project structure"""
    
    print("🚀 ULTIMATE QUANTONIUMOS REORGANIZATION")
    print("=" * 60)
    
    # Define the ideal structure
    reorganization_plan = {
        # Move all assembly-related to one place
        'src/assembly/': [
            'ASSEMBLY/*',
        ],
        
        # Create proper source structure
        'src/core/': [
            'core/*'
        ],
        
        'src/engine/': [
            'engine/*'
        ],
        
        'src/apps/': [
            'apps/*'
        ],
        
        'src/frontend/': [
            'frontend/*'
        ],
        
        # Consolidate all testing and validation
        'tests/': [
            'validation/*'
        ],
        
        # Better documentation structure
        'docs/technical/': [
            'docs/_agent_audit/*'
        ],
        
        # Development tools
        'dev/tools/': [
            'tools/*'
        ],
        
        'dev/scripts/': [
            'scripts/*'
        ],
        
        'dev/examples/': [
            'examples/*'
        ],
        
        # Configuration and data
        'data/config/': [
            'config/*'
        ],
        
        'data/weights/': [
            'weights/*'
        ],
        
        'data/logs/': [
            'logs/*'
        ],
        
        # Archive old stuff
        'archive/': [
            'archive/*',
            'QVault/*',
            'ui/*'
        ]
    }
    
    # Create new structure
    print("\n📁 Creating new directory structure...")
    for target_dir in reorganization_plan.keys():
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {target_dir}")
    
    # Move files according to plan
    print("\n🔄 Moving files to new structure...")
    for target_dir, sources in reorganization_plan.items():
        print(f"\n📂 Moving to {target_dir}:")
        
        for source_pattern in sources:
            source_path = Path(source_pattern.replace('/*', ''))
            
            if source_path.exists():
                if source_pattern.endswith('/*'):
                    # Move contents of directory
                    if source_path.is_dir():
                        for item in source_path.iterdir():
                            if item.name not in ['.git', '.gitignore']:
                                dest = Path(target_dir) / item.name
                                if dest.exists():
                                    if dest.is_dir():
                                        shutil.rmtree(dest)
                                    else:
                                        dest.unlink()
                                shutil.move(str(item), str(dest))
                                print(f"   ✅ {item} -> {dest}")
                        
                        # Remove empty source directory
                        if source_path.exists() and not any(source_path.iterdir()):
                            source_path.rmdir()
                            print(f"   🗑️ Removed empty: {source_path}")
                else:
                    # Move entire directory/file
                    dest = Path(target_dir) / source_path.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(source_path), str(dest))
                    print(f"   ✅ {source_path} -> {dest}")
    
    # Create modern project files
    create_modern_project_files()
    
    print("\n🎯 ULTIMATE REORGANIZATION COMPLETE!")
    print("\n📊 NEW STRUCTURE:")
    print("quantoniumos/")
    print("├── src/                    # All source code")
    print("│   ├── core/              # Core quantum algorithms")
    print("│   ├── engine/            # Main QuantoniumOS engine")
    print("│   ├── assembly/          # Assembly-optimized components")
    print("│   ├── apps/              # Applications")
    print("│   └── frontend/          # User interfaces")
    print("├── tests/                 # All testing and validation")
    print("├── docs/                  # Documentation")
    print("├── dev/                   # Development tools")
    print("│   ├── tools/             # Development utilities")
    print("│   ├── scripts/           # Build and analysis scripts")
    print("│   └── examples/          # Example code")
    print("├── data/                  # Data and configuration")
    print("│   ├── config/            # Configuration files")
    print("│   ├── weights/           # Model weights")
    print("│   └── logs/              # Log files")
    print("└── archive/               # Legacy and backup files")

def create_modern_project_files():
    """Create modern project configuration files"""
    
    # Create pyproject.toml
    pyproject_content = '''[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "quantoniumos"
description = "Advanced Quantum Computing Operating System"
authors = [{name = "QuantoniumOS Team"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Physics",
]
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.4.0",
    "qiskit>=0.30.0",
    "cryptography>=3.4.0",
]
dynamic = ["version"]

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black>=21.0",
    "flake8>=3.9",
    "mypy>=0.910",
    "sphinx>=4.0",
]

[project.urls]
Homepage = "https://github.com/mandcony/quantoniumos"
Documentation = "https://quantoniumos.readthedocs.io"
Repository = "https://github.com/mandcony/quantoniumos.git"

[tool.setuptools]
packages = ["src"]

[tool.setuptools_scm]
write_to = "src/quantoniumos/_version.py"

[tool.black]
line-length = 88
target-version = ['py38']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
'''
    
    with open('pyproject.toml', 'w') as f:
        f.write(pyproject_content)
    print("   ✅ Created: pyproject.toml")
    
    # Create .github/workflows/ci.yml
    Path('.github/workflows').mkdir(parents=True, exist_ok=True)
    
    ci_content = '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Run linting
      run: |
        flake8 src/
        black --check src/
    
    - name: Type checking
      run: |
        mypy src/
'''
    
    with open('.github/workflows/ci.yml', 'w') as f:
        f.write(ci_content)
    print("   ✅ Created: .github/workflows/ci.yml")
    
    # Create modern .gitignore
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
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
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
data/logs/*.log
data/weights/*.bin
*.dll
*.so
cmake_install.cmake
CMakeCache.txt
CMakeFiles/

# Temporary files
*.tmp
*.temp
backup_*/
'''
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("   ✅ Updated: .gitignore")

if __name__ == "__main__":
    ultimate_reorganization()
