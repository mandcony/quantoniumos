@echo off
echo ========================================================================
echo                  QuantoniumOS - Unified Build Script
echo ========================================================================

echo Building RFT kernel...
cd ASSEMBLY\build_scripts
call build_rft_kernel.bat
cd ..\..

echo Building crypto engine...
python build_crypto_engine.py

echo ========================================================================
echo                           Build Complete!
echo ========================================================================
echo.
echo To launch QuantoniumOS, run:
echo   python os_boot_transition.py
echo.
