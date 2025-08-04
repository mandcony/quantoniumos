@echo off
REM Run all comparative benchmarks and scientific validation
REM This script demonstrates the novelty and advantages of QuantoniumOS

echo ===================================================================
echo QuantoniumOS Scientific Demonstration and Comparative Analysis
echo ===================================================================
echo.

echo Step 1: Running comparative benchmarks vs. standard algorithms...
python benchmarks\comparative_benchmarks.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Benchmarks failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.

echo Step 2: Quantum-inspired scheduler demonstration...
python quantoniumos\secure_core\quantum_scheduler.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Quantum scheduler demo failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.

echo Step 3: Quantum entanglement simulation...
python quantoniumos\secure_core\quantum_entanglement.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Quantum entanglement demo failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.

echo Step 4: Full system demonstration with all features...
python demo_showcase.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Demo showcase failed with code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.

echo Step 5: Running scientific validation tests...
call run_scientific_tests.bat
echo.

echo ===================================================================
echo All demonstrations and benchmarks complete!
echo Results are available in the 'benchmark_results' and 'demonstration_results' folders.
echo See SCIENTIFIC_NOVELTY.md for a detailed analysis of advantages.
echo ===================================================================
