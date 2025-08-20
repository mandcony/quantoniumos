@echo off
REM QuantoniumOS Windows Launcher
REM Complete system startup script for Windows

echo ================================================
echo QuantoniumOS - Quantum Operating System
echo ================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "quantoniumos.py" (
    echo ERROR: quantoniumos.py not found
    echo Please run this script from the QuantoniumOS root directory
    pause
    exit /b 1
)

REM Set environment variables
set PYTHONPATH=%CD%;%CD%\kernel;%CD%\gui;%CD%\web;%CD%\filesystem;%CD%\apps;%CD%\phase3;%CD%\phase4;%CD%\11_QUANTONIUMOS
set QUANTONIUMOS_ROOT=%CD%

echo Setting up environment...
echo PYTHONPATH=%PYTHONPATH%
echo.

REM Install dependencies if needed
if not exist "requirements.txt" (
    echo Creating basic requirements.txt...
    echo numpy^>^=1.21.0 > requirements.txt
    echo flask^>^=2.0.0 >> requirements.txt
    echo cryptography^>^=3.4.0 >> requirements.txt
    echo matplotlib^>^=3.5.0 >> requirements.txt
    echo scipy^>^=1.7.0 >> requirements.txt
)

echo Checking dependencies...
python -c "import numpy, flask, cryptography" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Parse command line arguments
set MODE=desktop
if "%1"=="web" set MODE=web
if "%1"=="headless" set MODE=headless
if "%1"=="test" set MODE=test
if "%1"=="info" set MODE=info

echo Starting QuantoniumOS in %MODE% mode...
echo.

REM Launch based on mode
if "%MODE%"=="desktop" (
    echo Launching Desktop Interface...
    python quantoniumos.py desktop
) else if "%MODE%"=="web" (
    echo Launching Web Interface...
    python quantoniumos.py web --port 5000
) else if "%MODE%"=="headless" (
    echo Launching Headless Mode...
    python quantoniumos.py headless
) else if "%MODE%"=="test" (
    echo Running Test Suite...
    python quantoniumos.py test
) else if "%MODE%"=="info" (
    echo Displaying System Information...
    python quantoniumos.py info
) else (
    echo Unknown mode: %MODE%
    echo Available modes: desktop, web, headless, test, info
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo ERROR: QuantoniumOS failed to start
    echo Check the logs above for details
    pause
    exit /b 1
)

echo.
echo QuantoniumOS shutdown complete
pause
