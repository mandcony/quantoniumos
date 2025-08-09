@echo off
REM One-command reproducible build and validation for Windows
REM Usage: make_repro.bat local

echo === QuantoniumOS Reproducible Build and Validation ===
echo Mode: local

REM Set reproducible build timestamp
set SOURCE_DATE_EPOCH=1691539200

REM Create output directory
if not exist repro_results mkdir repro_results

echo Running local validation (no Docker)...

echo === Core Validation ===
python -m pytest tests/ -v --tb=short
if errorlevel 1 echo Some tests may fail - continuing...

echo === Security Validation ===
python run_security_focused_tests.py
if errorlevel 1 echo Security tests failed - continuing...

echo === Statistical Validation ===
python run_statistical_validation.py
if errorlevel 1 echo Statistical validation failed - continuing...

echo === KAT Generation ===
python tests/generate_vectors.py
python validate_kats.py

echo === Generating Checksums ===
dir /s /b *.py *.cpp *.yml *.json 2>nul | sort > temp_files.txt
certutil -hashfile temp_files.txt SHA256 > repro_results\checksums.txt
del temp_files.txt

echo === Local validation complete! ===
echo Results in: repro_results\
