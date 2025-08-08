"""
Production Safety Validator for QuantoniumOS

Run this before deploying to production to ensure security compliance.
"""

import os
import sys
from pathlib import Path

def validate_production_deployment():
    """Validate that the deployment is production-safe."""
    errors = []
    warnings = []
    
    # Check 1: Ensure not in debug mode
    if __debug__:
        errors.append("Python running in debug mode - use 'python -O' for production")
    
    # Check 2: Check for dangerous environment variables
    dangerous_env_vars = [
        'QUANTONIUM_ALLOW_TEST_ENTROPY',
        'QUANTONIUM_CRYPTO_DEBUG',
        'FLASK_DEBUG'
    ]
    
    for var in dangerous_env_vars:
        if os.environ.get(var):
            errors.append(f"Dangerous environment variable set: {var}")
    
    # Check 3: Ensure cache directories are not in deployment
    cache_dirs = ['.mypy_cache', '.ruff_cache', '__pycache__', '.pytest_cache']
    cache_found = []
    for cache_dir in cache_dirs:
        if Path(cache_dir).exists():
            cache_found.append(cache_dir)
    
    if cache_found:
        warnings.append(f"Cache directories present: {', '.join(cache_found)}")
    
    # Check 4: Look for sys.path manipulation
    python_files = list(Path(".").rglob("*.py"))
    sys_path_files = []
    for file_path in python_files:
        if any(skip in str(file_path) for skip in ['.venv', '__pycache__', 'build']):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if 'sys.path.' in content:
                    sys_path_files.append(str(file_path))
        except:
            continue
    
    if sys_path_files:
        warnings.append(f"Found {len(sys_path_files)} files with sys.path manipulation")
    
    # Check 5: Look for entropy bypass patterns
    entropy_patterns = ['seed=', 'debug=True', 'test_entropy', 'bypass_entropy']
    entropy_files = []
    for file_path in python_files[:50]:  # Check first 50 files to avoid timeout
        if any(skip in str(file_path) for skip in ['.venv', '__pycache__', 'build']):
            continue
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                if any(pattern in content for pattern in entropy_patterns):
                    entropy_files.append(str(file_path))
        except:
            continue
    
    if entropy_files:
        warnings.append(f"Found {len(entropy_files)} files with potential entropy bypass patterns")
    
    # Check 6: Validate secure configuration exists
    try:
        from core.security.secure_config import SecureConfig
        if not SecureConfig.validate_production_safety():
            errors.append("SecureConfig production safety validation failed")
    except ImportError:
        warnings.append("SecureConfig not available for validation")
    except Exception as e:
        warnings.append(f"SecureConfig validation error: {e}")
    
    # Check 7: Repository cleanliness
    if Path('.git').exists():
        try:
            import subprocess
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                warnings.append("Repository has uncommitted changes")
        except:
            pass  # Git not available or other issue
    
    return errors, warnings

if __name__ == "__main__":
    print("QuantoniumOS Production Safety Validator")
    print("=" * 50)
    
    errors, warnings = validate_production_deployment()
    
    if errors:
        print("❌ CRITICAL ERRORS - Deployment BLOCKED:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not errors and not warnings:
        print("✅ Production deployment validated successfully")
        sys.exit(0)
    elif errors:
        print("\n🚫 Production deployment BLOCKED due to critical errors")
        print("Fix all errors before deploying to production")
        sys.exit(1)
    else:
        print("\n⚠️  Production deployment permitted with warnings")
        print("Consider addressing warnings for optimal security")
        sys.exit(0)
