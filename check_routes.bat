@echo off
echo ======================================
echo QuantoniumOS Route Analysis and Testing
echo ======================================

:: Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found. Please install Python 3.6+ and try again.
    exit /b 1
)

:: Install required packages
echo Installing required packages...
pip install requests flask graphviz

:: Generate route documentation and visualization
echo Analyzing routes...
python analyze_routes.py

:: Validate all routes
echo Testing routes...
python validate_routes.py

echo All operations completed. Check the generated reports:
echo - routes_documentation.md : Documentation of all routes
echo - routes_visualization.png : Visual map of routes
echo - routes_data.json : JSON data of all routes
echo - route_test_results.json : Results of route tests
echo - quantonium_routes_validation.log : Detailed test log
