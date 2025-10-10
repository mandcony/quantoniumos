#!/usr/bin/env python3
"""
Direct RFT Visualizer Launcher (.pyw for silent Windows launch)
This bypasses all subprocess complexity and launches directly
"""
import os
import sys

# Change to the correct directory
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
os.chdir(root_dir)

# Add the root directory to Python path
sys.path.insert(0, root_dir)

# Import and run the RFT visualizer directly
try:
    # Import the main function from the visualizer
    sys.path.insert(0, os.path.join(root_dir, 'apps'))
    
    # Import the RFT visualizer module
    import rft_visualizer
    
    # Run it directly
    if __name__ == "__main__":
        rft_visualizer.main()
        
except Exception as e:
    print(f"❌ Error launching RFT Visualizer: {e}")
    import traceback
    traceback.print_exc()
