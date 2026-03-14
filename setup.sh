#!/bin/bash

# JASPER Setup Script
# Automated setup for the textile analysis pipeline

echo "================================================"
echo "JASPER: Japanese x Sri Lankan Textile Analysis"
echo "Setup Script"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create directories
echo ""
echo "Creating directory structure..."
mkdir -p data
mkdir -p results
mkdir -p results/figures

echo ""
echo "================================================"
echo "✓ Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Download dataset from Kaggle:"
echo "     kaggle datasets download -d ronithrr/jasper-japanese-x-sri-lankan-textile-image-dataset"
echo "     unzip jasper-japanese-x-sri-lankan-textile-image-dataset.zip -d data/"
echo ""
echo "  2. Run analysis:"
echo "     python main_analysis.py --dataset data/ --output results --verbose"
echo ""
echo "  3. Or use interactive notebook:"
echo "     jupyter notebook notebooks/interactive_analysis.ipynb"
echo ""
