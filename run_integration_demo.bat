@echo off
REM Run QuantoniumOS Integration Demo with Third-party Replication
REM This batch file executes the end-to-end integration demo with replication

echo ===== QuantoniumOS Integration Demo with Third-party Replication =====

echo.
echo Building Docker verification image...
docker build -t quantoniumos/verification:latest -f docker\Dockerfile .

echo.
echo Running integration demo...
python integration_demo.py

echo.
echo Demo complete. Check verification_results\reports for the verification report.

pause
