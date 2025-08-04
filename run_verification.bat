@echo off
REM Run QuantoniumOS Verification Suite
REM This batch file executes the verification process for QuantoniumOS

echo ===== QuantoniumOS Independent Replication Framework =====

echo.
echo Building Docker verification image...
docker build -t quantoniumos/verification:latest -f docker\Dockerfile .

echo.
echo Running verification suite...
python verification\run_verification.py --test_suite=all --iterations=100

echo.
echo Verification complete. Results are in verification_results directory.
echo.
echo To verify a specific result:
echo python verification\verify_result.py --result=verification_results\[RESULT_FILE] --public-key=verification\keys\public_key.pem
echo.
echo To compare results:
echo python verification\compare_results.py --result1=[FIRST_RESULT] --result2=[SECOND_RESULT]

pause
