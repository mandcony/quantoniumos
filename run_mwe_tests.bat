@echo off
echo QuantoniumOS Scientific Tests (Minimum Working Examples)
echo ===================================================
echo.
echo These are simplified MWEs that demonstrate the core concepts
echo without requiring the full QuantoniumOS environment to be running.
echo.

echo Test 1: Resonance Information Preservation
python tests\resonance_information_mwe.py
echo.

echo Test 2: Symbolic Avalanche Effect
python tests\symbolic_avalanche_mwe.py
echo.

echo Test 3: Pattern Detection
python tests\pattern_detection_mwe.py
echo.

echo Test 4: Quantum Simulation
python tests\quantum_simulation_mwe.py
echo.

echo All MWE tests completed. These demonstrate the core scientific
echo concepts behind QuantoniumOS without requiring the full system.
echo.
echo For full tests with the actual QuantoniumOS implementation, start
echo the main services first, then run the non-MWE versions of these tests.
pause
