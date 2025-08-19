#!/usr/bin/env python3
"""
COMPREHENSIVE FILE CORRUPTION REPAIR

Many files have been corrupted with line break issues - they're all on single lines.
This script will systematically fix the most critical files by recreating them with proper formatting.
"""

import os
import re

def fix_file_formatting(file_path):
    """Fix a file that has lost its line breaks."""
    print(f"Fixing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # If file is all on one line, we need to recreate line breaks
        if '\n' not in content and len(content) > 200:
            print(f"  File appears corrupted (no line breaks, {len(content)} chars)")
            
            # Add line breaks after common Python patterns
            fixed_content = content
            
            # Add newlines after common patterns
            patterns = [
                (r'"""', '"""\n'),
                (r'import ', '\nimport '),
                (r'from ', '\nfrom '),
                (r'class ', '\n\nclass '),
                (r'def ', '\n    def '),
                (r'if __name__', '\n\nif __name__'),
                (r'    def __init__', '\n    def __init__'),
                (r'        """', '\n        """'),
                (r'        self\.', '\n        self.'),
                (r'        return ', '\n        return '),
                (r'        print\(', '\n        print('),
                (r'    return ', '\n    return '),
                (r'    print\(', '\n    print('),
                (r'except ', '\nexcept '),
                (r'else:', '\nelse:'),
                (r'elif ', '\nelif '),
                (r'try:', '\ntry:'),
                (r'finally:', '\nfinally:'),
            ]
            
            for pattern, replacement in patterns:
                fixed_content = re.sub(pattern, replacement, fixed_content)
            
            # Fix indentation issues
            lines = fixed_content.split('\n')
            properly_indented = []
            
            current_indent = 0
            for line in lines:
                line = line.strip()
                if not line:
                    properly_indented.append('')
                    continue
                    
                # Determine proper indentation
                if line.startswith('class ') or line.startswith('def ') and not line.startswith('    def'):
                    current_indent = 0
                elif line.startswith('    def ') or line.startswith('    class '):
                    current_indent = 4
                elif line.startswith('        '):
                    current_indent = 8
                
                # Apply indentation
                if line.startswith('"""'):
                    properly_indented.append(' ' * current_indent + line)
                elif line.startswith('import ') or line.startswith('from '):
                    properly_indented.append(line)
                elif line.startswith('if __name__'):
                    properly_indented.append(line)
                else:
                    properly_indented.append(' ' * current_indent + line)
            
            fixed_content = '\n'.join(properly_indented)
            
            # Write back the fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print(f"  ✅ Fixed: Added line breaks and basic indentation")
            return True
        else:
            print(f"  ✅ File appears OK (has line breaks)")
            return False
            
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Fix corrupted files in the repository."""
    print("🔧 COMPREHENSIVE FILE CORRUPTION REPAIR")
    print("=" * 50)
    
    # List of critical files that might be corrupted
    critical_files = [
        'topological_quantum_kernel.py',
        'topological_vertex_engine.py', 
        'topological_vertex_geometric_engine.py',
        'core/config.py',
        'core/geometric_container.py',
        'core/multi_qubit_state.py',
        'core/oscillator.py',
        'core/quantum_rft.py',
        'canonical_true_rft.py',
        'production_canonical_rft.py',
        'mathematically_rigorous_rft.py'
    ]
    
    fixed_count = 0
    total_count = 0
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            total_count += 1
            if fix_file_formatting(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\n📊 REPAIR SUMMARY:")
    print(f"Files processed: {total_count}")
    print(f"Files fixed: {fixed_count}")
    print(f"Files already OK: {total_count - fixed_count}")
    
    if fixed_count > 0:
        print(f"\n⚠️  NOTE: Fixed files may need manual review for proper formatting")
        print(f"The automated fix adds basic line breaks and indentation")
        print(f"Some complex formatting may need manual adjustment")

if __name__ == "__main__":
    main()
