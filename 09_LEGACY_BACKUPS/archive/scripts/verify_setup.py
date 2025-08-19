#!/usr/bin/env python3
"""
QuantoniumOS Setup Verification Script
Verifies all dependencies and configurations are correctly installed.
Run: python scripts/verify_setup.py
"""

import sys
import subprocess
import importlib
import os
from pathlib import Path

# Load environment variables from .env file if it exists
if Path('.env').exists():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not installed, skip loading

def check_python_version():
    """Check Python version requirement"""
    min_version = (3, 9)
    current = sys.version_info[:2]

    print(f"🐍 Python Version: {sys.version}")
    if current >= min_version:
        print(f" ✅ Python {current[0]}.{current[1]} meets requirement (>={min_version[0]}.{min_version[1]})")
        return True
    else:
        print(f" ❌ Python {current[0]}.{current[1]} is too old (need >={min_version[0]}.{min_version[1]})")
        return False

def check_package(package_name, import_name=None):
    """Check if a Python package is installed"""
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        print(f" ✅ {package_name}")
        return True
    except ImportError:
        print(f" ❌ {package_name} (pip install {package_name})")
        return False

def check_system_command(command, name=None):
    """Check if a system command is available"""
    name = name or command
    try:
        subprocess.run([command, '--version'],
                      capture_output=True, check=True, timeout=10)
        print(f" ✅ {name}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print(f" ❌ {name} (not found in PATH)")
        return False

def check_database_connection():
    """Check database connection (SQLite or PostgreSQL)"""
    try:
        db_url = os.getenv('DATABASE_URL', 'sqlite:///quantonium_dev.db')

        if db_url.startswith('sqlite:'):
            # SQLite database - just check if we can create a connection
            import sqlite3
            db_path = db_url.replace('sqlite:///', '').replace('sqlite://', '')
            conn = sqlite3.connect(db_path)
            conn.close()
            print(f" ✅ SQLite database connection successful ({db_path})")
            return True
        else:
            # PostgreSQL database
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.close()
            print(" ✅ PostgreSQL connection successful")
            return True
    except Exception as e:
        print(f" ❌ Database connection failed: {e}")
        print(" Check DATABASE_URL environment variable or run setup_local_env.py")
        return False

def check_redis_connection():
    """Check Redis connection (optional for development)"""
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        r = redis.from_url(redis_url)
        r.ping()
        print(" ✅ Redis connection successful")
        return True
    except Exception as e:
        print(f" ⚠️ Redis connection failed: {e}")
        print(" Redis is optional for development (will use memory fallback)")
        return True  # Not critical for development

def check_file_structure():
    """Check essential file structure"""
    essential_files = [
        'quantoniumos/main.py',
        'quantoniumos/app.py',
        'quantoniumos/models.py',
        'requirements.txt',
        'CMakeLists.txt'
    ]

    missing_files = []
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f" ✅ {file_path}")
        else:
            print(f" ❌ {file_path}")
            missing_files.append(file_path)

    return len(missing_files) == 0

def check_environment_variables():
    """Check important environment variables"""
    env_vars = {
        'DATABASE_URL': 'Database connection string',
        'REDIS_URL': 'Redis connection string',
        'FLASK_SECRET_KEY': 'Flask secret key for sessions'
    }

    missing_vars = []
    for var, description in env_vars.items():
        value = os.getenv(var)
        if value:
            print(f" ✅ {var} (set)")
        else:
            print(f" ⚠️ {var} (not set) - {description}")
            missing_vars.append(var)

    return len(missing_vars) == 0

def main():
    """Run all verification checks"""
    print("🔍 QuantoniumOS Setup Verification")
    print("=" * 50)

    checks = []

    # Python version
    print("\n📋 Python Environment:")
    checks.append(check_python_version())

    # Core Python packages
    print("\n📦 Core Python Packages:")
    core_packages = [
        ('flask', 'flask'),
        ('sqlalchemy', 'sqlalchemy'),
        ('psycopg2-binary', 'psycopg2'),
        ('redis', 'redis'),
        ('cryptography', 'cryptography'),
        ('pyjwt', 'jwt'),
        ('numpy', 'numpy'),
        ('requests', 'requests')
    ]

    for package, import_name in core_packages:
        checks.append(check_package(package, import_name))

    # Optional packages
    print("\n📦 Optional Packages:")
    optional_packages = [
        ('pyqt5', 'PyQt5'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('pytest', 'pytest')
    ]

    for package, import_name in optional_packages:
        check_package(package, import_name)  # Don't add to required checks # System commands print("\n🛠️ System Tools:") checks.append(check_system_command('git')) checks.append(check_system_command('cmake'))

    # Database connections
    print("\n🗄️ Database Connections:")
    checks.append(check_database_connection())
    check_redis_connection()  # Not added to required checks

    # File structure
    print("\n📁 File Structure:")
    checks.append(check_file_structure())

    # Environment variables
    print("\n🌍 Environment Variables:")
    env_ok = check_environment_variables()

    # Summary
    print("\n" + "=" * 50)
    passed_checks = sum(checks)
    total_checks = len(checks)

    if passed_checks == total_checks:
        print("🎉 All critical checks passed! QuantoniumOS is ready to run.")
        if not env_ok:
            print("⚠️ Some environment variables are missing but loaded from .env file.")

        print("\n🚀 Next steps:")
        print(" python start_dev.py")
        print(" Open http://localhost:5000")
        return 0
    else:
        print(f"❌ {total_checks - passed_checks} critical checks failed.")
        print("\n🔧 Fix the issues above and run this script again.")
        print(" Try running: python setup_local_env.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
