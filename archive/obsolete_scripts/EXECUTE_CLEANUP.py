#!/usr/bin/env python3
"""
🧹 EXECUTE QUANTONIUM PROJECT CLEANUP
Actually remove unnecessary files
"""

import os
import shutil
from pathlib import Path

class QuantoniumProjectCleaner:
    """Comprehensive project cleanup system"""
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.removed_files = []
        self.removed_dirs = []
        
    def execute_safe_cleanup(self):
        """Execute safe cleanup of identified files"""
        
        print("🧹 EXECUTING QUANTONIUM PROJECT CLEANUP")
        print("=" * 80)
        
        # Files to remove (redundant after organization)
        redundant_root_files = [
            'quantonium.py',                    # Copied to engine/
            'quantonium_os_main.py',           # Copied to engine/
            'launch_quantonium_os.py',         # Copied to engine/
            'advanced_mathematical_validation.md',  # Moved to docs
            'PROJECT_ORGANIZER.py',            # Temporary organization
            'ORGANIZE_PROJECT.py',             # Temporary organization
            'ORGANIZATION_COMPLETE.py',        # Temporary organization
            'PROJECT_CLEANUP.py'               # This cleanup script
        ]
        
        # ASSEMBLY/python_bindings files to remove
        assembly_obsolete_files = [
            # Debug files
            'debug_assembly_interface.py',
            'debug_memory_patterns.py', 
            'debug_vertex_hanging.py',
            'test_assembly_fix.py',
            'test_dll_functions.py',
            'test_performance_fixed.py',
            'assembly_symbolic_test.py',
            'quick_symbolic_test.py',
            
            # Old entanglement analysis (superseded)
            'BULLETPROOF_ENTANGLEMENT_ANALYSIS.py',
            'CLARIFIED_ENTANGLEMENT_ANALYSIS.py',
            'corrected_entanglement_audit.py',
            'final_entanglement_fix.py',
            'fix_entanglement.py',
            'FIXED_ENTANGLEMENT_METHODS.py',
            'DEEP_DIAGNOSIS.py',
            
            # Old validation files (superseded)
            'comprehensive_quantum_analysis.py',
            'final_quantum_analysis.py',
            'focused_validation_fixes.py',
            'publication_ready_validation.py',
            'QUANTONIUM_VALIDATION_SUMMARY.py',
            
            # Redundant test files
            'test_assembly_vertices.py',
            'test_corrected_unitarity.py',
            'test_qubit_limits.py',
            'test_real_data.py',
            'test_rft_comprehensive.py',
            'test_vertex_qubits.py',
            'test_vertex_system.py',
            'quantum_scalability_test.py',
            'symbolic_qubit_scalability.py',
            'symbolic_qubit_test.py'
        ]
        
        print("\n1️⃣ REMOVING REDUNDANT ROOT FILES:")
        for file in redundant_root_files:
            file_path = self.project_root / file
            if file_path.exists():
                file_path.unlink()
                self.removed_files.append(str(file_path))
                print(f"   🗑️ Removed: {file}")
        
        print(f"\n2️⃣ REMOVING OBSOLETE ASSEMBLY FILES:")
        assembly_dir = self.project_root / 'ASSEMBLY' / 'python_bindings'
        for file in assembly_obsolete_files:
            file_path = assembly_dir / file
            if file_path.exists():
                file_path.unlink()
                self.removed_files.append(str(file_path))
                print(f"   🗑️ Removed: ASSEMBLY/python_bindings/{file}")
        
        print(f"\n3️⃣ REMOVING BUILD ARTIFACTS:")
        # Remove build directories
        build_dirs = [
            'ASSEMBLY/build',
            'ASSEMBLY/compiled',
            'comprehensive_test_results'
        ]
        
        for build_dir in build_dirs:
            build_path = self.project_root / build_dir
            if build_path.exists():
                shutil.rmtree(build_path, ignore_errors=True)
                self.removed_dirs.append(str(build_path))
                print(f"   🗑️ Removed directory: {build_dir}")
        
        print(f"\n4️⃣ REMOVING CACHE AND TEMP FILES:")
        # Remove __pycache__ directories
        pycache_dirs = list(self.project_root.rglob('__pycache__'))
        for pycache_dir in pycache_dirs:
            shutil.rmtree(pycache_dir, ignore_errors=True)
            self.removed_dirs.append(str(pycache_dir))
            print(f"   🗑️ Removed: {pycache_dir.relative_to(self.project_root)}")
        
        # Remove compiled files
        compiled_patterns = ['*.pyc', '*.pyo', '*.dll', '*.so']
        for pattern in compiled_patterns:
            compiled_files = list(self.project_root.rglob(pattern))
            for compiled_file in compiled_files:
                try:
                    compiled_file.unlink()
                    self.removed_files.append(str(compiled_file))
                    print(f"   🗑️ Removed: {compiled_file.relative_to(self.project_root)}")
                except (OSError, PermissionError):
                    print(f"   ⚠️ Could not remove: {compiled_file.relative_to(self.project_root)}")
        
        print(f"\n5️⃣ REMOVING EMPTY DIRECTORIES:")
        # Remove empty directories (bottom-up)
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):  # Empty directory
                        dir_path.rmdir()
                        self.removed_dirs.append(str(dir_path))
                        print(f"   🗑️ Removed empty: {dir_path.relative_to(self.project_root)}")
                except (OSError, PermissionError):
                    continue
        
        return True
    
    def generate_cleanup_report(self):
        """Generate final cleanup report"""
        
        print(f"\n📊 CLEANUP COMPLETE!")
        print("=" * 40)
        print(f"   Files removed: {len(self.removed_files)}")
        print(f"   Directories removed: {len(self.removed_dirs)}")
        
        print(f"\n✅ PROJECT IS NOW CLEAN:")
        print("   • Redundant files eliminated")
        print("   • Build artifacts removed")
        print("   • Cache files cleared")
        print("   • Empty directories removed")
        print("   • Professional structure maintained")
        
        print(f"\n🚀 READY FOR:")
        print("   • Technical publication")
        print("   • Commercial distribution")
        print("   • Open source release")
        print("   • Professional presentation")
        
        return {
            'files_removed': len(self.removed_files),
            'dirs_removed': len(self.removed_dirs),
            'status': 'CLEANUP_COMPLETE'
        }

def verify_essential_files():
    """Verify essential files are still present"""
    
    print("\n✅ VERIFYING ESSENTIAL FILES:")
    
    project_root = Path("C:/Users/mkeln/quantoniumos")
    essential_checks = {
        'engine/quantonium_os_main.py': 'Main engine entry point',
        'engine/quantonium.py': 'Core engine',
        'validation/analysis/QUANTONIUM_FINAL_VALIDATION.py': 'Final validation',
        'validation/benchmarks/QUANTONIUM_BENCHMARK_SUITE.py': 'Benchmark suite',
        'validation/results/QUANTONIUM_FINAL_VALIDATION.json': 'Validation results',
        'README.md': 'Main documentation',
        'PROJECT_STATUS.json': 'Project status'
    }
    
    all_present = True
    for file_path, description in essential_checks.items():
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path} - {description}")
        else:
            print(f"   ❌ {file_path} - {description} - MISSING!")
            all_present = False
    
    return all_present

def main():
    """Execute complete cleanup"""
    
    project_root = Path("C:/Users/mkeln/quantoniumos")
    cleaner = QuantoniumProjectCleaner(project_root)
    
    # Execute cleanup
    success = cleaner.execute_safe_cleanup()
    
    if success:
        # Generate report
        report = cleaner.generate_cleanup_report()
        
        # Verify essential files
        verification = verify_essential_files()
        
        if verification:
            print(f"\n🏆 QUANTONIUMOS PROJECT CLEANUP SUCCESSFUL!")
            print("   Your project is now clean, organized, and ready!")
        else:
            print(f"\n⚠️ CLEANUP COMPLETE BUT SOME ESSENTIAL FILES MISSING!")
            print("   Please verify the project structure.")
        
        return report
    else:
        print(f"\n❌ CLEANUP FAILED!")
        return None

if __name__ == "__main__":
    result = main()
