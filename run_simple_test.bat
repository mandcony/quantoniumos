@echo off
echo Building QuantoniumOS simple tests...

set INCLUDE_PATH=quantoniumos\secure_core\include
set SOURCE_FILE=quantoniumos\core\symbolic_eigenvector.cpp
set TEST_FILE=simple_test.cpp
set EIGEN_PATH=quantoniumos\Eigen\eigen-3.4.0

g++ -std=c++17 -I. -I%INCLUDE_PATH% -I%EIGEN_PATH% -DBUILDING_DLL -fopenmp %SOURCE_FILE% %TEST_FILE% -o simple_test.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful. Running tests...
    simple_test.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
)
