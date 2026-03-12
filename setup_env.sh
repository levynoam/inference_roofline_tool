#!/bin/bash
# Setup script for Linux/Mac
# Creates virtual environment and installs dependencies

echo "Setting up LLM Inference Performance Analysis environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 --version)
echo "Found: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup complete!"
    echo "To activate the environment in the future, run:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run the web application:"
    echo "  python web_app.py"
    echo ""
    echo "To run tests:"
    echo "  pytest tests/ -v"
else
    echo "Error: Failed to install dependencies."
    exit 1
fi
