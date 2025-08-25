#!/bin/bash

echo "====================================="
echo "Machine Learning Environment Setup"
echo "====================================="
echo

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed!"
    echo "Please install Python from https://www.python.org/downloads/"
    echo "Or use Homebrew: brew install python"
    exit 1
fi

python3 --version
echo

echo "Python found! Upgrading pip..."
python3 -m pip install --upgrade pip

echo
echo "Installing machine learning packages..."
python3 -m pip install -r requirements.txt

echo
echo "Setting up Jupyter kernel..."
python3 -m ipykernel install --user --name ml-env --display-name "Python (ML Environment)"

echo
echo "====================================="
echo "Installation Complete!"
echo "====================================="
echo
echo "You can now run:"
echo "jupyter lab"
echo "or"
echo "jupyter notebook"
echo
echo "To start working with your notebooks!"
