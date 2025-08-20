#!/usr/bin/env python3
"""
Fix the main OS file by removing corrupted code after patent dashboard
"""

import re

def fix_main_os():
    """Fix the main OS file"""
    print("🔧 Fixing QuantoniumOS main file...")
    
    # Read the current file
    with open("quantonium_os_unified.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the end of the patent dashboard
    patent_end = "        status_text.insert(tk.END, status_info)\n        status_text.config(state=tk.DISABLED)"
    
    # Find the proper main function
    main_func = '''

def main():
    """Main entry point for QuantoniumOS Unified"""
    try:
        os_instance = QuantoniumOSUnified()
        os_instance.run()
    except Exception as e:
        print(f"Failed to start QuantoniumOS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''
    
    # Find where the patent dashboard ends
    patent_end_pos = content.find(patent_end)
    if patent_end_pos != -1:
        # Find the end of that line
        end_line_pos = content.find('\n', patent_end_pos + len(patent_end))
        if end_line_pos != -1:
            # Keep everything up to the end of the patent dashboard
            clean_content = content[:end_line_pos + 1]
            # Add the clean main function
            clean_content += main_func
            
            # Write the clean file
            with open("quantonium_os_unified.py", "w", encoding="utf-8") as f:
                f.write(clean_content)
            
            print("✅ Fixed QuantoniumOS main file")
            return True
        else:
            print("❌ Could not find end of patent dashboard line")
            return False
    else:
        print("❌ Could not find patent dashboard end marker")
        return False

if __name__ == "__main__":
    success = fix_main_os()
    if success:
        print("🎯 QuantoniumOS ready to launch!")
    else:
        print("💥 Fix failed")
