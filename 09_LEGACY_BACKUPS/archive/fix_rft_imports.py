||#!/usr/bin/env python3
"""
RFT Import Cleanup Script - Replace All Duplicate RFT Implementations This script systematically replaces all duplicate/legacy RFT implementations with imports from the canonical source, addressing the major reviewer concern. PROBLEM: 50+ files with conflicting RFT implementations confuse reviewers SOLUTION: Replace all with canonical_true_rft.py imports
"""
"""

import os
import re
import sys from pathlib
import Path
def find_rft_files():
"""
"""
        Find all Python files that
import or define RFT functions.
"""
        """ rft_files = [] root = Path("/workspaces/quantoniumos")
        for py_file in root.rglob("*.py"):
        if py_file.name in ["canonical_true_rft.py", "minimal_true_rft.py", "fix_rft_imports.py"]: continue

        # Skip canonical implementations and this script
        try: with open(py_file, 'r', encoding='utf-8') as f: content = f.read()

        # Check for RFT-related patterns patterns = [ r'from.*true_rft.*import', r'import.*true_rft', r'from.*resonance_fourier.*import', r'import.*resonance_fourier', r'def.*forward_true_rft', r'def.*inverse_true_rft', r'def.*resonance_fourier_transform', r'class.*RFT', r'ResonanceFourierTransform' ]
        if any(re.search(pattern, content, re.IGNORECASE)
        for pattern in patterns): rft_files.append(py_file) except (UnicodeDecodeError, PermissionError): continue
        return rft_files
def fix_file(filepath): """
        Fix RFT imports in a single file.
"""
"""
        print(f"Fixing {filepath}...")
        try: with open(filepath, 'r', encoding='utf-8') as f: content = f.read() original_content = content # 1. Replace imports from core.true_rft content = re.sub( r'from core\.true_rft import.*', 'from canonical_true_rft
import forward_true_rft, inverse_true_rft', content, flags=re.MULTILINE ) # 2. Replace direct imports of resonance_fourier functions content = re.sub( r'from.*resonance_fourier.*import\s+(.*)', lambda m: f'from canonical_true_rft
import forward_true_rft, inverse_true_rft||n

        # Legacy wrapper maintained for: {m.group(1)}', content, flags=re.MULTILINE ) # 3. Replace inline RFT definitions with canonical calls

        # This is more complex and file-specific, so we'll add a comment instead
        if re.search(r'def (forward_true_rft|inverse_true_rft|||resonance_fourier_transform)', content): if "

        # LEGACY RFT IMPLEMENTATION - REPLACE WITH CANONICAL" not in content: content = "

        # LEGACY RFT IMPLEMENTATION - REPLACE WITH CANONICAL\n# from canonical_true_rft
import forward_true_rft, inverse_true_rft\n||n" + content # 4. Add canonical
        import
        if RFT functions are used but not imported
        if re.search(r'(forward_true_rft|||inverse_true_rft)\s*\(', content) and '
from canonical_true_rft import' not in content:

        # Add canonical
import at the top import_lines = [] other_lines = [] in_imports = True
        for line in content.split('\n'):
        if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')): import_lines.append(line)
        else: in_imports = False other_lines.append(line) import_lines.append('from canonical_true_rft
import forward_true_rft, inverse_true_rft') content = '\n'.join(import_lines + other_lines)

        # Only write
        if changes were made
        if content != original_content: with open(filepath, 'w', encoding='utf-8') as f: f.write(content)
        print(f" ✅ Updated {filepath}")
        return True
        else:
        print(f" ⏭️ No changes needed for {filepath}")
        return False except Exception as e:
        print(f" ❌ Error fixing {filepath}: {e}")
        return False
def main():
        print("🔧 RFT Import Cleanup - Fixing Reviewer Concerns")
        print("="*50)

        # Find all files with RFT implementations rft_files = find_rft_files()
        print(f"Found {len(rft_files)} files with RFT-related code")
        if not rft_files:
        print("No files to fix!")
        return

        # Show files to be processed
        print("\nFiles to be processed:")
        for f in sorted(rft_files):
        print(f" - {f}") input("\nPress Enter to proceed with fixes...")

        # Fix each file
        print("\nApplying fixes...") fixed_count = 0
        for filepath in rft_files:
        if fix_file(filepath): fixed_count += 1
        print(f"\n🎉 Cleanup complete!")
        print(f"Fixed {fixed_count} out of {len(rft_files)} files")
        print("||n📋 Next steps:")
        print("1. Test critical functions still work")
        print("2. Remove duplicate RFT definition files")
        print("3. Run validation suite")

if __name__ == "__main__": main()