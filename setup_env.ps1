# Setup script for Windows PowerShell
# Creates virtual environment and installs dependencies

Write-Host "Setting up LLM Inference Performance Analysis environment..." -ForegroundColor Green

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python not found. Please install Python 3.10+ first." -ForegroundColor Red
    exit 1
}

# Check Python version
$pythonVersion = python --version
Write-Host "Found: $pythonVersion" -ForegroundColor Cyan

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSetup complete! " -ForegroundColor Green
    Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "`nTo run the web application:" -ForegroundColor Cyan
    Write-Host "  python web_app.py" -ForegroundColor White
    Write-Host "`nTo run tests:" -ForegroundColor Cyan
    Write-Host "  pytest tests/ -v" -ForegroundColor White
} else {
    Write-Host "Error: Failed to install dependencies." -ForegroundColor Red
    exit 1
}
