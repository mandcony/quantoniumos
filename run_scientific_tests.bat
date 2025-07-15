@echo off
echo Running QuantoniumOS Scientific Validation Suite
echo ===============================================
echo.
echo Starting QuantoniumOS services...
start "QuantoniumOS" /b python main.py

echo Waiting for services to initialize...
timeout /t 10 /nobreak > nul

echo.
echo Running scientific tests...
echo.
echo Test 1: Resonance Information Preservation
python tests\resonance_information_test.py
echo.
echo Test 2: Symbolic Avalanche Effect
python tests\symbolic_avalanche_test.py
echo.
echo Test 3: Pattern Detection
python tests\pattern_detection_test.py
echo.
echo Test 4: Quantum Simulation
python tests\quantum_simulation_test.py

echo.
echo All tests completed. Review the output above to see the scientific properties
echo demonstrated by your QuantoniumOS system.
pause
