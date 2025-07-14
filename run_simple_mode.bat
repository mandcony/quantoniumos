@echo off
echo ======================================
echo Starting QuantoniumOS - Simple Mode
echo ======================================

:: Change to the app directory
cd quantoniumos

:: Create instance directory if it doesn't exist
if not exist instance mkdir instance

:: Start the simplified application
python simple_app.py

echo ======================================
echo QuantoniumOS has been stopped
echo ======================================
