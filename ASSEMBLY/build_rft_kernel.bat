@echo off
REM Windows build script for QuantoniumOS RFT Kernel
REM Uses MSVC or MinGW-w64 to build the RFT library

echo Building QuantoniumOS RFT Kernel...
echo ====================================

REM Check if we have CMake
where cmake >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake not found. Please install CMake and add it to PATH.
    pause
    exit /b 1
)

REM Create build directory
if not exist build mkdir build
cd build

REM Configure with CMake
echo Configuring build with CMake...
cmake .. -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed.
    pause
    exit /b 1
)

REM Build the project
echo Building RFT kernel...
cmake --build . --config Release

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed.
    pause
    exit /b 1
)

REM Copy outputs to compiled directory
echo Copying libraries to compiled directory...
cd ..
if not exist compiled mkdir compiled
copy build\librftkernel.dll compiled\ >nul 2>nul
copy build\librftkernel.lib compiled\ >nul 2>nul
copy build\librftkernel.a compiled\ >nul 2>nul

echo.
echo ====================================
echo Build completed successfully!
echo.
echo Output files:
if exist compiled\librftkernel.dll echo   - compiled\librftkernel.dll
if exist compiled\librftkernel.lib echo   - compiled\librftkernel.lib
if exist compiled\librftkernel.a echo   - compiled\librftkernel.a
echo.
echo To test the library, run:
echo   cd python_bindings
echo   python unitary_rft.py
echo.
pause
