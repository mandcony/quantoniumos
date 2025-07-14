@echo off
echo Building QuantoniumOS robust symbolic eigenvector tests...

set INCLUDE_PATH=quantoniumos\secure_core\include
set SOURCE_FILE=quantoniumos\core\symbolic_eigenvector.cpp
set TEST_FILE=quantoniumos\robust_test_symbolic.cpp
set EIGEN_PATH=quantoniumos\Eigen\eigen-3.4.0

g++ -std=c++17 -I%INCLUDE_PATH% -I%EIGEN_PATH% -DBUILDING_DLL -fopenmp %SOURCE_FILE% %TEST_FILE% -o robust_test_symbolic.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful. Running tests...
    robust_test_symbolic.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
)

if %ERRORLEVEL% EQU 0 (
    echo Build successful. Running tests...
    robust_test_symbolic.exe
    
    if %ERRORLEVEL% EQU 0 (
        echo All tests passed successfully!
    ) else (
        echo Some tests failed. See test output for details.
    )
) else (
    echo Build failed with error code %ERRORLEVEL%
)
