#!/bin/bash
# Run script for TruthfulQA Harness

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ".env file not found. Please create it from .env.example and add your API key."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run the server
echo "Starting TruthfulQA Evaluation Harness..."
echo "Server will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo ""

python -m backend.app
