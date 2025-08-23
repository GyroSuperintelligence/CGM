@echo off
REM CGM Experimental Framework Launcher
REM This script activates the virtual environment and runs the experiments

echo Activating CGM virtual environment...
call .venv\Scripts\activate.bat

if %errorlevel% neq 0 (
    echo Failed to activate virtual environment
    pause
    exit /b 1
)

echo Running CGM experiments...
python run_experiments.py

echo.
echo Experiments completed!
echo Press any key to exit...
pause > nul
