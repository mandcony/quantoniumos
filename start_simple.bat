@echo off
echo ======================================
echo Starting QuantoniumOS - Direct Mode
echo ======================================

:: Set environment variables directly
set QUANTONIUM_MASTER_KEY=simple-dev-key-for-testing-only
set DATABASE_ENCRYPTION_KEY=dev-db-key-for-testing-only
set JWT_SECRET_KEY=dev-jwt-key-for-testing-only
set FLASK_ENV=development
set REDIS_DISABLED=true
set PORT=5000
set SQLALCHEMY_DATABASE_URI=sqlite:///instance/quantonium.db

:: Change to the app directory
cd quantoniumos

:: Create instance directory if it doesn't exist
if not exist instance mkdir instance

:: Create a simple start.py file that bypasses the complex env loading
echo import os > simple_start.py
echo from flask import Flask >> simple_start.py
echo app = Flask(__name__) >> simple_start.py
echo @app.route('/') >> simple_start.py
echo def home(): >> simple_start.py
echo     return 'QuantoniumOS is running in simplified mode' >> simple_start.py
echo if __name__ == '__main__': >> simple_start.py
echo     app.run(debug=True, port=5000) >> simple_start.py

:: Start the application
echo Starting simplified Flask app...
python simple_start.py

echo ======================================
echo QuantoniumOS has been stopped
echo ======================================
