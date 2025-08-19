#!/usr/bin/env python3
"""
COMPREHENSIVE REPOSITORY CORRUPTION REPAIR

Scan entire repository for corrupted Python files and fix them systematically.
A file is considered corrupted if it's large (>1000 chars) but has no newlines.
"""

import os
import glob
import re

def fix_corrupted_python_file(file_path):
    """Fix a Python file that has lost its line breaks."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if corrupted (large file with no newlines)
        if '\n' not in content and len(content) > 1000:
            print(f"🔧 Fixing corrupted file: {file_path} ({len(content)} chars)")
            
            # Intelligent line break insertion for Python
            fixed_content = content
            
            # Critical patterns that MUST have newlines
            critical_patterns = [
                (r'#!/usr/bin/env python3', '#!/usr/bin/env python3\n'),
                (r'"""([^"]+)"""', r'"""\n\1\n"""\n'),
                (r'import ([a-zA-Z_][a-zA-Z0-9_]*)', r'\nimport \1'),
                (r'from ([a-zA-Z_][a-zA-Z0-9_.]*) import', r'\nfrom \1 import'),
                (r'class ([A-Z][a-zA-Z0-9_]*)', r'\n\nclass \1'),
                (r'def ([a-zA-Z_][a-zA-Z0-9_]*)\(', r'\n    def \1('),
                (r'if __name__ == "__main__":', r'\n\nif __name__ == "__main__":'),
                (r'try:', r'\ntry:'),
                (r'except ([A-Za-z]+):', r'\nexcept \1:'),
                (r'except:', r'\nexcept:'),
                (r'else:', r'\nelse:'),
                (r'elif ', r'\nelif '),
                (r'finally:', r'\nfinally:'),
                (r'for ([a-zA-Z_][a-zA-Z0-9_]*) in', r'\n        for \1 in'),
                (r'while ', r'\n        while '),
                (r'if ([a-zA-Z_])', r'\n        if \1'),
                (r'return ', r'\n        return '),
                (r'yield ', r'\n        yield '),
                (r'raise ', r'\n        raise '),
                (r'print\(', r'\n        print('),
                (r'self\.([a-zA-Z_])', r'\n        self.\1'),
                (r'# ([A-Z])', r'\n\n        # \1'),
                (r'"""$', r'"""\n'),
            ]
            
            # Apply pattern fixes
            for pattern, replacement in critical_patterns:
                fixed_content = re.sub(pattern, replacement, fixed_content)
            
            # Split into lines and fix indentation
            lines = fixed_content.split('\n')
            properly_formatted = []
            
            indent_level = 0
            in_class = False
            in_function = False
            
            for line in lines:
                stripped = line.strip()
                
                if not stripped:
                    properly_formatted.append('')
                    continue
                
                # Determine indentation based on context
                if stripped.startswith('#!/'):
                    properly_formatted.append(stripped)
                elif stripped.startswith('"""') and len(stripped) == 3:
                    properly_formatted.append('"""')
                elif stripped.startswith('import ') or stripped.startswith('from '):
                    properly_formatted.append(stripped)
                elif stripped.startswith('class '):
                    properly_formatted.append(stripped)
                    in_class = True
                    indent_level = 0
                elif stripped.startswith('def ') and not in_class:
                    properly_formatted.append(stripped)
                    in_function = True
                    indent_level = 0
                elif stripped.startswith('def ') and in_class:
                    properly_formatted.append('    ' + stripped)
                    in_function = True
                    indent_level = 4
                elif stripped.startswith('if __name__'):
                    properly_formatted.append(stripped)
                    in_class = False
                    in_function = False
                    indent_level = 0
                elif stripped.startswith('return ') or stripped.startswith('yield '):
                    if in_function:
                        properly_formatted.append('        ' + stripped)
                    else:
                        properly_formatted.append('    ' + stripped)
                elif stripped.startswith('self.'):
                    if in_function and in_class:
                        properly_formatted.append('        ' + stripped)
                    else:
                        properly_formatted.append('    ' + stripped)
                elif stripped.startswith('print(') or stripped.startswith('try:') or stripped.startswith('for ') or stripped.startswith('while ') or stripped.startswith('if '):
                    if in_function:
                        properly_formatted.append('        ' + stripped)
                    elif in_class:
                        properly_formatted.append('    ' + stripped)
                    else:
                        properly_formatted.append(stripped)
                else:
                    # Default indentation
                    if in_function:
                        properly_formatted.append('        ' + stripped)
                    elif in_class:
                        properly_formatted.append('    ' + stripped)
                    else:
                        properly_formatted.append(stripped)
            
            # Join back and clean up excess newlines
            final_content = '\n'.join(properly_formatted)
            final_content = re.sub(r'\n{3,}', '\n\n', final_content)  # Max 2 consecutive newlines
            
            # Write the fixed file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(final_content)
            
            print(f"  ✅ Fixed: Added proper line breaks and indentation")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False

def main():
    """Scan and fix all corrupted Python files."""
    print("🔧 COMPREHENSIVE REPOSITORY CORRUPTION REPAIR")
    print("=" * 60)
    
    # Find all Python files
    python_files = glob.glob("**/*.py", recursive=True)
    
    print(f"Found {len(python_files)} Python files to check...")
    
    corrupted_files = []
    fixed_files = []
    
    # Check each file for corruption
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if corrupted (large file with no newlines)
            if '\n' not in content and len(content) > 1000:
                corrupted_files.append(file_path)
        except Exception as e:
            print(f"⚠️  Could not read {file_path}: {e}")
    
    print(f"\n🚨 Found {len(corrupted_files)} corrupted files:")
    for file_path in corrupted_files:
        print(f"  {file_path}")
    
    if corrupted_files:
        print(f"\n🔧 Fixing corrupted files...")
        for file_path in corrupted_files:
            if fix_corrupted_python_file(file_path):
                fixed_files.append(file_path)
    
    print(f"\n📊 REPAIR SUMMARY:")
    print(f"Total Python files: {len(python_files)}")
    print(f"Corrupted files found: {len(corrupted_files)}")
    print(f"Files successfully fixed: {len(fixed_files)}")
    
    if fixed_files:
        print(f"\n✅ Fixed files:")
        for file_path in fixed_files:
            print(f"  {file_path}")
        
        print(f"\n⚠️  IMPORTANT:")
        print(f"Fixed files use automated formatting and may need manual review.")
        print(f"Check syntax and indentation for any remaining issues.")
    
    return len(fixed_files)

if __name__ == "__main__":
    main()
