# CGM Experimental Framework Launcher (PowerShell)
# This script activates the virtual environment and runs the experiments

Write-Host "Activating CGM virtual environment..." -ForegroundColor Green
& .venv\Scripts\Activate.ps1

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Running CGM experiments..." -ForegroundColor Green
python run_experiments.py

Write-Host "Experiments completed!" -ForegroundColor Green
Read-Host "Press Enter to exit"
