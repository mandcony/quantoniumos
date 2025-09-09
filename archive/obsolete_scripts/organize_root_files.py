#!/usr/bin/env python3
"""
🧹 ORGANIZE ROOT FILES SCRIPT
Move scattered root files to appropriate directories
"""

import os
import shutil
from pathlib import Path

def organize_root_files():
    """Organize files scattered in root directory"""
    
    root = Path('.')
    
    # Create directories if they don't exist
    dirs_to_create = [
        'docs/reports',
        'docs/audits',
        'docs/safety',
        'validation/reports',
        'validation/results',
        'scripts/analysis',
        'scripts/security'
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Define file organization rules
    moves = {
        # AI Safety and Audit Reports
        'docs/audits': [
            'ai_audit_report_*.json',
            'ASSEMBLY_COMPREHENSIVE_AUDIT.md',
            'COMPREHENSIVE_REAL_PROJECT_SCAN.md',
            'CONFIG_COMPREHENSIVE_AUDIT.md',
            'CORE_COMPREHENSIVE_AUDIT.md',
            'CRYPTO_VALIDATION_COMPREHENSIVE_AUDIT.md',
            'QUANTONIUMOS_FINAL_COMPREHENSIVE_AUDIT.md'
        ],
        
        'docs/safety': [
            'ai_safety_report_*.json',
            'complete_ai_safety_report.json',
            'AI_SAFETY_CERTIFICATION.md'
        ],
        
        'docs/reports': [
            'GREEN_STATUS_FINAL_REPORT.md',
            'ENHANCED_CONVERSATIONAL_AI_GUIDE.md',
            'SECURITY_MAINTENANCE_GUIDE.md'
        ],
        
        'validation/reports': [
            'differential_analysis_report_*.json',
            'post_quantum_analysis_report_*.json',
            'smart_engine_validation_*.json',
            'engine_distribution_proof_*.json',
            'security_maintenance_report.json'
        ],
        
        'validation/results': [
            'quantonium_benchmark_results.json',
            'quantonium_benchmark_results.png'
        ],
        
        'scripts/analysis': [
            'quantonium_ai_intelligence_analysis.py',
            'quantonium_assembly_security_analysis.py',
            'quantonium_complete_ai_safety_analysis.py',
            'quantonium_security_analysis.py'
        ],
        
        'scripts/security': [
            'ai_safety_safeguards.py',
            'ai_safety_testing_framework.py',
            'security_maintenance_toolkit.py'
        ]
    }
    
    print("🧹 ORGANIZING ROOT FILES")
    print("=" * 50)
    
    for target_dir, patterns in moves.items():
        print(f"\n📁 Moving files to {target_dir}:")
        
        for pattern in patterns:
            if '*' in pattern:
                # Handle wildcards
                import glob
                matches = glob.glob(pattern)
                for match in matches:
                    if os.path.exists(match):
                        dest = Path(target_dir) / Path(match).name
                        shutil.move(match, dest)
                        print(f"   ✅ {match} -> {dest}")
            else:
                # Handle exact matches
                if os.path.exists(pattern):
                    dest = Path(target_dir) / pattern
                    shutil.move(pattern, dest)
                    print(f"   ✅ {pattern} -> {dest}")
    
    # Handle remaining root files that should stay
    keep_in_root = [
        'README.md',
        'requirements.txt',
        'LICENSE.md',
        'PATENT-NOTICE.md',
        'QUICK_START.md',
        'PROJECT_STATUS.json',
        'PROJECT_SUMMARY.json',
        'quantonium_boot.py',
        'security_operations.sh',
        'EXECUTE_CLEANUP.py',
        'FINAL_PROJECT_STATUS.py',
        'UPDATED_STATUS.py',
        'test_conversation_response.py',
        'ai_safe_test_demo.py',
        'Author'
    ]
    
    print(f"\n📌 Files kept in root:")
    for file in keep_in_root:
        if os.path.exists(file):
            print(f"   ✅ {file}")
    
    print(f"\n✅ ROOT ORGANIZATION COMPLETE!")

if __name__ == "__main__":
    organize_root_files()
