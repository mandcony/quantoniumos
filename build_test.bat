@echo off
echo Building QuantoniumOS symbolic eigenvector tests...

set INCLUDE_PATH=secure_core\include
set SOURCE_FILE=core\symbolic_eigenvector.cpp
set TEST_FILE=test_symbolic.cpp

g++ -std=c++17 -I%INCLUDE_PATH% -I"Eigen" -DBUILDING_DLL -fopenmp %SOURCE_FILE% %TEST_FILE% -o test_symbolic.exe

if %ERRORLEVEL% EQU 0 (
    echo Build successful. Running tests...
    test_symbolic.exe
) else (
    echo Build failed with error code %ERRORLEVEL%
)
