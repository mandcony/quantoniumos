#!/usr/bin/env python3
"""
Safe cleanup script for QuantoniumOS - archives obsolete files without breaking the project.
Moves files to archive/ folder instead of deleting them.
"""

import os
import shutil
import sys
from pathlib import Path

# Essential files that must be kept in root
ESSENTIAL_FILES = {
    # Core v2 Implementation (CRITICAL - DO NOT MOVE)
    'enhanced_rft_crypto.cpp',
    'enhanced_rft_crypto_bindings_v2.cpp', 
    'enhanced_rft_crypto.cpython-312-x86_64-linux-gnu.so',
    
    # Active test files
    'test_v2_comprehensive.py',
    'test_final_v2.py',
    
    # Essential documentation
    'README.md',
    'LICENSE',
    'LICENSE_COMMERCIAL.md',
    
    # Build/setup files
    'requirements.txt',
    'requirements-dev.txt',
    'setup.py',
    'pyproject.toml',
    '.gitignore',
    
    # Research core (might be imported by other code)
    'canonical_true_rft.py',
    'publication_ready_validation.py',
    
    # Main application (if still used)
    'app.py',
    'main.py',
}

# Essential directories that should be kept as-is
ESSENTIAL_DIRS = {
    'wrappers',           # Contains enhanced_v2_wrapper.py
    'core',              # Might contain dependencies
    '.git',
    '.github',
    '.vscode', 
    '.venv',
    '__pycache__',
    'third_party',       # External dependencies
}

# Files to archive (known obsolete patterns)
ARCHIVE_PATTERNS = {
    # Debug files
    'debug_*.py',
    
    # Old test files (except v2)
    'test_enhanced_cpp_rft.py',
    'test_basic_v2.py',
    'test_wrapper_v2.py', 
    'test_final_performance.py',
    'test_fixed_crypto.py',
    'test_current_rft.py',
    'test_minimal_rft_demo.py',
    'test_bell_state.py',
    'test_branch_cut_continuity.py',
    'test_choi_channel.py',
    'test_claim*.py',
    'test_conservative_capacity.py',
    'test_direct_capacity.py',
    'test_dft_cleanup.py',
    'test_epsilon_reproducibility.py',
    'test_focused_capacity.py',
    'test_hamiltonian_recovery.py',
    'test_lie_closure.py',
    'test_mixer_validation.py',
    'test_patent_*.py',
    'test_path_continuity_core.py',
    'test_qubit_capacity.py',
    'test_randomized_benchmarking.py',
    'test_rft_*.py',
    'test_spectral_locality.py',
    'test_state_evolution_benchmarks.py',
    'test_surgical_fix.py',
    'test_time_evolution.py',
    'test_trotter_error.py',
    'test_unitarity.py',
    
    # Old comprehensive files
    'comprehensive_*.py',
    'advanced_*.py',
    
    # Old crypto files  
    'crypto_benchmark*.py',
    'crypto_test*.py',
    'corrected_rft_*.py',
    
    # RFT variants (keep canonical_true_rft.py)
    'rft_*.py',
    'simple_rft_*.py',
    'reversible_rft_*.py',
    'enhanced_rft_feistel_cipher.py',
    
    # Benchmark files
    '*benchmark*.py',
    'rigorous_*.py',
    'honest_*.py',
    'credible_*.py',
    'realistic_*.py',
    'external_wins_*.py',
    
    # Build/fix files
    'apply_*.py',
    'build_*.py',
    'fix_*.py',
    'final_*.py',
    'check_*.py',
    'verify_*.py',
    'remove_*.py',
    'unicode_*.py',
    
    # Validation files (except publication_ready)
    '*_validation*.py',
    'definitive_*.py',
    'symbiotic_*.py',
    'symbolic_*.py',
    
    # Documentation (archive most, keep essential)
    'ARCHITECTURE_*.md',
    'BEGINNERS_GUIDE.md',
    'COMPLETE_*.md', 
    'COMPREHENSIVE_*.md',
    'CPP_ENGINE_*.md',
    'CRYPTO_TEST_*.md',
    'DFT_CLEANUP_*.md',
    'FORMAL_*.md',
    'HONEST_*.md',
    'MODULAR_*.md',
    'PACKAGE_*.md',
    'PATENT_*.md',
    'PATH_*.md',
    'QUANTUM_*.md',
    'REPRODUCIBILITY_*.md',
    'REVIEWER_*.md',
    'REVISED_*.md',
    'RFT_*.md',
    'SAFE_*.md',
    'SURGICAL_*.md',
    'SYMBOLIC_*.md',
    'TRUE_RFT_*.md',
    'WINDOWED_*.md',
    
    # Result files
    '*.json',
    '*.log',
    '*.txt',
    '*.html',
    'demo_output.txt',
    'hash_metrics.txt',
    'known_answer_tests.txt',
    'metrics_output.txt',
    'validation_output.txt',
    
    # Old bindings
    'enhanced_rft_crypto_bindings.cpp',  # Keep v2 version only
    
    # Misc
    'image_resonance_analyzer.py',
    'env_loader.py',
    'models.py',
    'routes.py',
    'security.py',
    '*_delegate.py',
    '*_diagnostic.py',
    'minimal_*.py',
    'simple_*.py',
    'ultra_*.py',
    'quantum_*.py',
    'quick_*.py',
    'optimization_*.py',
    'pattern_*.py',
    'standalone_*.py',
    'simplified_*.py',
    'mathematical_*.py',
    'deep_*.py',
    'high_performance_*.py',
    'improved_*.py',
    'impulse_*.py',
    'spec_*.py',
}

def is_essential_file(filepath):
    """Check if a file is essential and should not be archived."""
    filename = os.path.basename(filepath)
    return filename in ESSENTIAL_FILES

def is_essential_dir(dirpath):
    """Check if a directory is essential and should not be archived."""
    dirname = os.path.basename(dirpath)
    return dirname in ESSENTIAL_DIRS

def should_archive_file(filepath):
    """Check if a file matches archive patterns."""
    filename = os.path.basename(filepath)
    
    # Never archive essential files
    if is_essential_file(filepath):
        return False
        
    # Check against archive patterns
    for pattern in ARCHIVE_PATTERNS:
        if '*' in pattern:
            # Simple glob matching
            prefix, suffix = pattern.split('*', 1)
            if filename.startswith(prefix) and filename.endswith(suffix):
                return True
        else:
            # Exact match
            if filename == pattern:
                return True
    
    return False

def safe_cleanup():
    """Safely archive obsolete files without breaking the project."""
    root_dir = Path('/workspaces/quantoniumos')
    archive_dir = root_dir / 'archive'
    
    # Create archive directory
    archive_dir.mkdir(exist_ok=True)
    
    archived_files = []
    kept_files = []
    
    print("🧹 QuantoniumOS Safe Cleanup")
    print("=" * 50)
    
    # Process files in root directory only (don't recurse into essential dirs)
    for item in root_dir.iterdir():
        if item.name.startswith('.'):
            continue  # Skip hidden files/dirs
            
        if item.is_dir():
            if is_essential_dir(str(item)):
                print(f"📁 KEEP DIR:  {item.name}")
                kept_files.append(str(item))
            else:
                # Archive non-essential directories
                if item.name not in ['archive']:  # Don't archive the archive!
                    print(f"📦 ARCHIVE DIR: {item.name}")
                    shutil.move(str(item), str(archive_dir / item.name))
                    archived_files.append(str(item))
        else:
            # Process files
            if should_archive_file(str(item)):
                print(f"📦 ARCHIVE: {item.name}")
                shutil.move(str(item), str(archive_dir / item.name))
                archived_files.append(str(item))
            else:
                print(f"✅ KEEP:    {item.name}")
                kept_files.append(str(item))
    
    print("\n" + "=" * 50)
    print(f"📊 SUMMARY:")
    print(f"   ✅ Files kept:     {len(kept_files)}")
    print(f"   📦 Files archived: {len(archived_files)}")
    print(f"   📁 Archive location: {archive_dir}")
    
    print(f"\n🔍 ESSENTIAL FILES VERIFIED:")
    for essential in sorted(ESSENTIAL_FILES):
        if (root_dir / essential).exists():
            print(f"   ✅ {essential}")
        else:
            print(f"   ⚠️  {essential} (not found)")
    
    print(f"\n✨ Cleanup complete! Project should still work perfectly.")
    print(f"   If you need any archived file, check: {archive_dir}")
    
    return len(archived_files), len(kept_files)

if __name__ == "__main__":
    try:
        archived, kept = safe_cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)
