#!/usr/bin/env python3
"""
RFT Validation Test Driver - QuantoniumOS
=========================================
This script runs a quick validation of the RFT implementation
and displays the results.
"""

import os
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="RFT Validation Test Driver")
    parser.add_argument('--gui', action='store_true', help='Launch the GUI visualizer')
    parser.add_argument('--quick', action='store_true', help='Run a quick validation with smaller sizes')
    parser.add_argument('--math-only', action='store_true', help='Run only mathematical validity tests')
    parser.add_argument('--perf-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--crypto-only', action='store_true', help='Run only cryptography tests')
    parser.add_argument('--report', type=str, help='Output file for validation report')
    args = parser.parse_args()
    
    # Find the root directory (this file's directory)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.gui:
        # Launch the GUI visualizer
        try:
            sys.path.append(os.path.join(root_dir, 'apps'))
            from rft_validation_visualizer import main as run_gui
            run_gui()
        except ImportError as e:
            print(f"Error: Failed to launch GUI visualizer: {e}")
            return 1
    else:
        # Command-line validation
        start_time = time.time()
        
        try:
            # Import validation module
            sys.path.append(root_dir)
            from rft_scientific_validation import RFTValidation
            
            validator = RFTValidation()
            
            # Build command-line args for validation module
            cmd_args = []
            if args.quick:
                cmd_args.append('--quick')
            if args.math_only:
                cmd_args.append('--math-only')
            if args.perf_only:
                cmd_args.append('--perf-only')
            if args.crypto_only:
                cmd_args.append('--crypto-only')
            if args.report:
                cmd_args.extend(['--report', args.report])
            
            print("Starting RFT validation...")
            
            if args.math_only:
                result = validator.math_suite.run_all_tests()
            elif args.perf_only:
                result = validator.perf_suite.run_all_tests()
            elif args.crypto_only:
                result = validator.crypto_suite.run_all_tests()
            else:
                result = validator.run_all_validations()
            
            if args.report:
                validator.generate_report(args.report)
                print(f"Report saved to {args.report}")
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\nValidation completed in {duration:.2f} seconds")
            print(f"Result: {'PASSED' if result else 'FAILED'}")
            
            return 0 if result else 1
            
        except ImportError as e:
            print(f"Error: Failed to import validation module: {e}")
            return 1
        except Exception as e:
            print(f"Error during validation: {e}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
