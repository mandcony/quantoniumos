@echo off
echo Running QuantoniumOS Demo Suite...
echo.

echo Step 1: Running main demo showcase...
python demo_showcase.py

echo.
echo Step 2: Running system benchmark...
python benchmarks\system_benchmark.py

echo.
echo Step 3: Testing symbolic collision and avalanche effect...
python tests\test_symbolic_collision.py

echo.
echo Step 4: Running quantum link test...
python core\monitor_main_system.py

echo.
echo Demo suite complete!
echo All results have been saved to the demonstration_results directory.
echo.
