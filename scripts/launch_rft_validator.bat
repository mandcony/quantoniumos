@echo off
REM RFT Scientific Validation Visualizer
REM This script launches the GUI visualizer for RFT validation

echo QuantoniumOS - RFT Scientific Validation Visualizer
echo ==================================================
echo.

python "%~dp0\apps\rft_validation_visualizer.py"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Failed to launch visualizer with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
