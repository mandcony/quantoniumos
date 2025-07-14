@echo off
REM QuantoniumOS C++ Build Wrapper for Windows
echo Building QuantoniumOS C++ components...

REM Check if PowerShell is available
powershell -Command "Write-Host 'PowerShell available'" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: PowerShell is required but not available
    exit /b 1
)

REM Run the PowerShell build script
powershell -ExecutionPolicy Bypass -File "%~dp0build_cpp.ps1" %*

REM Check if build was successful
if %errorlevel% equ 0 (
    echo.
    echo ✅ C++ build completed successfully!
    echo    You can now run: build\Release\robust_test_symbolic.exe
) else (
    echo.
    echo ❌ C++ build failed
    exit /b 1
)

pause
