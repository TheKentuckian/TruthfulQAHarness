#!/bin/bash
# Setup script for TruthfulQA Harness on Chromebook/Crostini

echo "========================================="
echo "TruthfulQA Harness - Chromebook Setup"
echo "========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "✓ Python version: $python_version"
echo ""

# Update package lists
echo "Updating system packages (this may take a moment)..."
sudo apt-get update -qq

# Install build essentials (lightweight, in case user wants to try sklearn later)
echo "Installing basic build tools..."
sudo apt-get install -y python3-pip python3-venv build-essential 2>&1 | grep -v "^Reading\|^Building\|^0 upgraded"

echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

echo ""
echo "========================================="
echo "Installing dependencies (minimal set)..."
echo "========================================="
echo ""
echo "This will install only the essential packages without sklearn."
echo "The Simple Text Verifier will be used (no sklearn required)."
echo ""

# Install minimal requirements
pip install -r requirements-minimal.txt

echo ""
echo "✓ Installation complete!"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY"
    echo ""
else
    echo "✓ .env file already exists"
fi

echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "✓ Using Simple Text Verifier (Word Overlap)"
echo "  This is a lightweight verifier that works without sklearn"
echo ""
echo "📝 Next steps:"
echo "  1. Edit .env and add your Anthropic API key:"
echo "     nano .env"
echo ""
echo "  2. Run the server:"
echo "     ./run.sh"
echo ""
echo "  3. Open your browser to:"
echo "     http://localhost:8000"
echo ""
echo "  4. In the UI, make sure 'Simple Text (Word Overlap)' is selected"
echo ""
echo "Optional: If you want to try installing sklearn (may be slow):"
echo "  source venv/bin/activate"
echo "  pip install scikit-learn numpy scipy"
echo ""
