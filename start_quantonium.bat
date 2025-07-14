@echo off
echo ======================================
echo Starting QuantoniumOS Simplified Mode
echo ======================================

:: Run the PowerShell setup script first
powershell -ExecutionPolicy Bypass -File setup_local_env.ps1

:: Start the simplified application
call run_simple_mode.bat

echo ======================================
echo QuantoniumOS has been stopped
echo ======================================
