#!/usr/bin/env python3
"""
Fix Unicode/Emoji issues in QuantoniumOS files
"""

import os
import re


def fix_unicode_in_file(filepath):
    """Remove problematic Unicode characters and replace with ASCII alternatives"""
    try:
        # Read file with UTF-8 encoding
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace common problematic emoji/Unicode with ASCII alternatives
        replacements = {
            "✅": "[OK]",
            "❌": "[ERROR]",
            "⚠️": "[WARNING]",
            "🌌": "[QUANTUM]",
            "🚀": "[LAUNCH]",
            "🔹": "-",
            "📍": "*",
            "📊": "[STATS]",
            "🔄": "[REFRESH]",
            "📈": "[CHART]",
            "🌊": "[WAVE]",
            "✨": "[SPARKLE]",
            "📥": "[DOWNLOAD]",
            "🎯": "[TARGET]",
            "�": "[?]",
            "🗑️": "[DELETE]",
        }

        # Apply replacements
        for old, new in replacements.items():
            content = content.replace(old, new)

        # Remove any remaining non-ASCII characters that could cause issues
        # Keep only printable ASCII and common characters
        clean_content = ""
        for char in content:
            if ord(char) < 128 or char in "\n\t":
                clean_content += char
            else:
                # Replace with a safe alternative
                clean_content += "?"

        # Write back with UTF-8 encoding
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(clean_content)

        print(f"✓ Fixed Unicode issues in {filepath}")
        return True

    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}")
        return False


def main():
    """Fix Unicode issues in main QuantoniumOS files"""
    files_to_fix = [
        "quantonium_os_unified.py",
        "apps/rft_visualizer.py",
        "frontend/ui/quantum_app_controller_clean.py",
    ]

    for filepath in files_to_fix:
        if os.path.exists(filepath):
            fix_unicode_in_file(filepath)
        else:
            print(f"File not found: {filepath}")


if __name__ == "__main__":
    main()
