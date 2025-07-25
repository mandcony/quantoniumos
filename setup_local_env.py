#!/usr/bin/env python3
"""
QuantoniumOS Local Development Setup
Quick setup script for development environment
Run: python setup_local_env.py
"""

import os
import sys
import secrets
import subprocess
from pathlib import Path

def create_env_file():
    """Create .env file with development settings"""
    env_content = f"""# QuantoniumOS Development Environment
# Generated by setup_local_env.py

# Database Configuration (SQLite for development)
DATABASE_URL=sqlite:///quantonium_dev.db

# Redis Configuration (optional - fallback to memory)
REDIS_URL=redis://localhost:6379/0

# Flask Configuration
FLASK_SECRET_KEY={secrets.token_hex(32)}
FLASK_ENV=development
FLASK_DEBUG=True

# QuantoniumOS Specific
QUANTONIUM_LOG_LEVEL=INFO
QUANTONIUM_MODE=development

# Security Settings (development only)
DISABLE_HTTPS_REDIRECT=true
DISABLE_RATE_LIMITING=false
"""
    
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(env_content)
    
    print("✅ Created .env file with development settings")
    print("   🔐 Generated secure Flask secret key")
    print("   💾 Using SQLite database for development")

def install_dependencies():
    """Install Python dependencies"""
    try:
        print("📦 Installing Python dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("   Try running manually: pip install -r requirements.txt")
        return False

def setup_database():
    """Setup SQLite database for development"""
    try:
        print("🗄️  Setting up development database...")
        
        # Create a simple database initialization
        setup_code = """
import os
import sys
sys.path.append('quantoniumos')

try:
    from quantoniumos.models import db
    from quantoniumos.app import create_app
    
    app = create_app()
    with app.app_context():
        db.create_all()
        print("✅ Database tables created")
except Exception as e:
    print(f"⚠️  Database setup skipped: {e}")
    print("   Database will be created on first run")
"""
        
        with open('temp_db_setup.py', 'w', encoding='utf-8') as f:
            f.write(setup_code)
        
        result = subprocess.run([sys.executable, 'temp_db_setup.py'], 
                              capture_output=True, text=True)
        
        # Clean up temp file
        if os.path.exists('temp_db_setup.py'):
            os.remove('temp_db_setup.py')
        
        if result.returncode == 0:
            print("✅ Development database setup complete")
        else:
            print("⚠️  Database will be created on first application start")
        
        return True
    except Exception as e:
        print(f"⚠️  Database setup skipped: {e}")
        return True  # Not critical for setup

def check_optional_services():
    """Check if optional services are running"""
    print("\n🔍 Checking optional services:")
    
    # Check Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("   ✅ Redis is running (rate limiting enabled)")
    except:
        print("   ⚠️  Redis not running (will use memory fallback)")
        print("      To install Redis: https://redis.io/download")
    
    # Check PostgreSQL (optional for development)
    try:
        import psycopg2
        conn = psycopg2.connect("postgresql://postgres@localhost:5432/postgres")
        conn.close()
        print("   ✅ PostgreSQL available (optional)")
    except:
        print("   ⚠️  PostgreSQL not available (using SQLite)")

def create_start_script():
    """Create a simple start script"""
    start_script = """#!/usr/bin/env python3
# QuantoniumOS Development Server
# Auto-generated by setup_local_env.py

import os
import sys
from pathlib import Path

# Load environment variables
if Path('.env').exists():
    from dotenv import load_dotenv
    load_dotenv()

# Add quantoniumos to path
sys.path.insert(0, 'quantoniumos')

try:
    from quantoniumos.main import main
    if __name__ == '__main__':
        print("Starting QuantoniumOS Development Server...")
        print("   Web: http://localhost:5000")
        print("   API: http://localhost:5000/docs")
        print("   Press Ctrl+C to stop")
        main()
except ImportError as e:
    print(f"Import error: {e}")
    print("   Make sure you're in the project root directory")
    sys.exit(1)
"""
    
    with open('start_dev.py', 'w', encoding='utf-8') as f:
        f.write(start_script)
    
    print("✅ Created start_dev.py script")

def main():
    """Main setup function"""
    print("🔧 QuantoniumOS Local Development Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('quantoniumos').exists():
        print("❌ Error: quantoniumos directory not found")
        print("   Make sure you're running this from the project root")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing setup despite dependency issues...")
    
    # Setup database
    setup_database()
    
    # Check optional services
    check_optional_services()
    
    # Create start script
    create_start_script()
    
    print("\n" + "=" * 50)
    print("QuantoniumOS development environment setup complete!")
    print("\nQuick start:")
    print("   python start_dev.py")
    print("\nFull documentation:")
    print("   See QUANTONIUM_DEVELOPER_GUIDE.md")
    print("\nVerify setup:")
    print("   python scripts/verify_setup.py")

if __name__ == "__main__":
    main()
