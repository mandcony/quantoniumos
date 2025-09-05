@echo off
REM RFT Scientific Validation Runner
REM This script runs the comprehensive RFT validation suite

echo QuantoniumOS - RFT Scientific Validation Suite
echo ==============================================
echo.

python "%~dp0\rft_scientific_validation.py" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Validation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo.
echo Validation complete!
