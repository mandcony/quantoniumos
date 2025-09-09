#!/usr/bin/env python3
"""
🧹 ORGANIZE DIRECTORY STRUCTURE
Move misplaced directories to better locations
"""

import os
import shutil
from pathlib import Path

def reorganize_directories():
    """Reorganize misplaced directories"""
    
    print("🧹 REORGANIZING DIRECTORY STRUCTURE")
    print("=" * 50)
    
    # Create archive directory for backups
    archive_dir = Path('archive')
    archive_dir.mkdir(exist_ok=True)
    
    # Move backup to archive
    if Path('backup_20250908_231415').exists():
        shutil.move('backup_20250908_231415', 'archive/backup_20250908_231415')
        print("✅ Moved backup_20250908_231415 -> archive/backup_20250908_231415")
    
    # Move crypto_validation to validation directory
    if Path('crypto_validation').exists() and Path('validation').exists():
        # Move contents of crypto_validation to validation/crypto
        crypto_dest = Path('validation/crypto')
        crypto_dest.mkdir(exist_ok=True)
        
        for item in Path('crypto_validation').iterdir():
            dest_path = crypto_dest / item.name
            if dest_path.exists():
                shutil.rmtree(dest_path) if item.is_dir() else dest_path.unlink()
            shutil.move(str(item), str(dest_path))
            print(f"✅ Moved crypto_validation/{item.name} -> validation/crypto/{item.name}")
        
        # Remove empty crypto_validation directory
        Path('crypto_validation').rmdir()
        print("✅ Removed empty crypto_validation directory")
    
    # Move UNIFIED_ASSEMBLY to ASSEMBLY directory
    if Path('UNIFIED_ASSEMBLY').exists() and Path('ASSEMBLY').exists():
        unified_dest = Path('ASSEMBLY/unified')
        if unified_dest.exists():
            shutil.rmtree(unified_dest)
        
        shutil.move('UNIFIED_ASSEMBLY', str(unified_dest))
        print("✅ Moved UNIFIED_ASSEMBLY -> ASSEMBLY/unified")
    
    # Create an organized structure info file
    with open('DIRECTORY_STRUCTURE.md', 'w') as f:
        f.write("""# QuantoniumOS Directory Structure

## Core Components
- `core/` - Core quantum computing algorithms
- `engine/` - Main QuantoniumOS engine components
- `apps/` - Application layer
- `ASSEMBLY/` - Assembly optimized components
  - `unified/` - Unified assembly components
  - `engines/` - Specialized engines
  - `python_bindings/` - Python interfaces

## Development & Validation
- `validation/` - Complete testing and validation suite
  - `crypto/` - Cryptographic validation (moved from crypto_validation/)
  - `tests/` - Unit tests
  - `benchmarks/` - Performance benchmarks
  - `analysis/` - Technical analysis
  - `results/` - Test results
  - `reports/` - Validation reports

## Documentation & Support
- `docs/` - Documentation
  - `audits/` - Audit reports
  - `safety/` - AI safety documentation
  - `reports/` - Technical reports
- `scripts/` - Utility scripts
  - `analysis/` - Analysis tools
  - `security/` - Security tools
- `tools/` - Development tools
- `examples/` - Example code

## Configuration & Assets
- `config/` - Configuration files
- `frontend/` - User interfaces
- `ui/` - UI components
- `weights/` - Model weights and data

## Archive
- `archive/` - Backups and historical data
""")
    
    print("\n✅ DIRECTORY REORGANIZATION COMPLETE!")
    print("📁 Created DIRECTORY_STRUCTURE.md documentation")

if __name__ == "__main__":
    reorganize_directories()
