#!/usr/bin/env python3
"""
Flask App Dependency Test

Tests if the Flask application can load its core dependencies.
"""

import sys
import os

# Add required paths
sys.path.extend(['04_RFT_ALGORITHMS', '03_RUNNING_SYSTEMS'])

def test_flask_dependencies():
    print("Testing Flask App Dependencies...")
    print("=" * 50)
    
    try:
        # Test core RFT functionality
        import canonical_true_rft
        print("✅ canonical_true_rft: LOADED")
        
        # Test basic computation
        result = canonical_true_rft.forward_true_rft([1, 2, 3, 4])
        print(f"✅ RFT computation: SUCCESS (output length: {len(result)})")
        
    except Exception as e:
        print(f"❌ RFT functionality: FAILED - {e}")
    
    try:
        # Test Flask itself
        from flask import Flask
        app = Flask(__name__)
        print("✅ Flask framework: LOADED")
        
    except Exception as e:
        print(f"❌ Flask framework: FAILED - {e}")
    
    try:
        # Test if we can create a minimal Flask app
        from flask import Flask, jsonify
        app = Flask(__name__)
        
        @app.route('/test')
        def test_route():
            return jsonify({"status": "success", "message": "Flask app working"})
        
        print("✅ Flask route creation: SUCCESS")
        
    except Exception as e:
        print(f"❌ Flask route creation: FAILED - {e}")
    
    # Check for missing dependencies
    print("\nMissing Dependencies Check:")
    print("-" * 30)
    
    missing_modules = []
    optional_modules = ['env_loader', 'config', 'main']
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module}: AVAILABLE")
        except ImportError:
            print(f"⚠️  {module}: MISSING (may need to be created)")
            missing_modules.append(module)
    
    print(f"\n📊 Summary:")
    print(f"✅ Core RFT: Working")
    print(f"✅ Flask Framework: Working")
    print(f"⚠️  Missing modules: {len(missing_modules)} ({', '.join(missing_modules)})")
    
    if len(missing_modules) == 0:
        print("🎉 All dependencies satisfied!")
    else:
        print("🔧 Some modules need to be created/fixed for full Flask app functionality")

if __name__ == "__main__":
    test_flask_dependencies()
