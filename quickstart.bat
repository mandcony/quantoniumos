@echo off
echo ===================================================
echo QuantoniumOS - Quick Start
echo ===================================================
echo.
echo This script will help you quickly set up and validate
echo the core scientific components of QuantoniumOS.
echo.
echo Steps:
echo 1. Set up the environment
echo 2. Build and test C++ components
echo 3. Run the simplified API server
echo 4. Test API endpoints
echo.
echo ===================================================

echo.
echo Step 1: Setting up the environment...
powershell -ExecutionPolicy Bypass -File setup_local_env.ps1
if %ERRORLEVEL% NEQ 0 (
    echo Environment setup failed. Exiting.
    exit /b %ERRORLEVEL%
)
echo Environment setup completed successfully.
echo.

echo Step 2: Building and testing C++ components...
call run_simple_test.bat
if %ERRORLEVEL% NEQ 0 (
    echo C++ component tests failed. Exiting.
    exit /b %ERRORLEVEL%
)
echo C++ component validation completed successfully.
echo.

echo Step 3: Starting the simplified API server...
echo The server will start in a new window. Please keep it running.
start cmd /k run_simple_mode.bat
echo.
echo Waiting for server to initialize...
timeout /t 5 /nobreak > nul

echo Step 4: Testing API endpoints...
python test_api_simple.py
if %ERRORLEVEL% NEQ 0 (
    echo API tests failed.
    echo Please check the API server window for errors.
    exit /b %ERRORLEVEL%
)
echo API endpoints validated successfully.
echo.

echo ===================================================
echo QuantoniumOS setup and validation completed successfully!
echo.
echo The API server is running in a separate window.
echo You can access the API at http://localhost:5000/
echo.
echo Press any key to exit this setup guide...
echo ===================================================
pause > nul
