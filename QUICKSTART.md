# Quick Start Guide

Get up and running with the TruthfulQA Evaluation Harness in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- An Anthropic API key ([get one here](https://console.anthropic.com))

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Edit .env and add your API key
nano .env  # or use your preferred editor
# Set: ANTHROPIC_API_KEY=your_key_here

# Run the server
./run.sh
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the server
python -m backend.app
```

## First Evaluation

1. **Open your browser** to `http://localhost:8000`

2. **Load questions**: Click "Load Sample Questions" (default: 10 questions)

3. **Run evaluation**: Click "Evaluate Batch"

4. **View results**: See accuracy, confidence scores, and detailed per-question analysis

## Understanding Results

### Summary Statistics

- **Accuracy**: Percentage of truthful answers
- **Truthful Answers**: Count of answers matching correct reference answers
- **Untruthful Answers**: Count of answers matching incorrect reference answers
- **Avg Confidence**: How confident the verifier is (higher = more certain)

### Per-Question Results

Each result shows:
- **Question**: The TruthfulQA question asked
- **LLM Answer**: What Claude responded
- **Status**: Truthful (green) or Untruthful (red)
- **Confidence**: How certain the verifier is about the classification
- **Reasoning**: Explanation of the similarity scores
- **Metrics**: Raw similarity values to correct vs. incorrect answers

### How Word Similarity Works

The verifier compares the LLM's answer to reference answers using:

1. **TF-IDF Vectorization**: Converts text to numerical vectors based on word importance
2. **Cosine Similarity**: Measures how similar the vectors are (0 = completely different, 1 = identical)
3. **Classification**: If similarity to correct answers > similarity to incorrect answers → Truthful

## Configuration Options

### LLM Settings

- **Model**: Change Claude model (e.g., `claude-sonnet-4-5-20250929`)
- **Max Tokens**: Limit response length (default: 1024)
- **Temperature**: Control randomness (0 = deterministic, 2 = creative)

### Sampling

- **Sample Size**: How many questions to evaluate (1-100)
- **Random Seed**: Use same seed for reproducible results

## Tips for Research

1. **Baseline Performance**: Start with default settings to establish baseline
2. **Reproducibility**: Use same random seed across experiments
3. **Temperature Testing**: Try different temperatures (0.0, 0.5, 1.0, 1.5)
4. **Sample Sizes**: Start small (10 questions), scale up for significance
5. **Category Analysis**: Track which categories have lowest accuracy

## Common Issues

### "Failed to load questions"
- Check internet connection (needs to download TruthfulQA from HuggingFace)
- First load will be slow as dataset is cached locally

### "Error generating response from Claude"
- Verify API key in `.env` file
- Check API key is valid at https://console.anthropic.com
- Ensure you have API credits available

### "Import errors"
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Run from project root: `python -m backend.app`

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Explore the API at `http://localhost:8000/docs` (FastAPI automatic documentation)
- Check out plans for Phase 2 (self-correction) in the README
- Modify verifiers or LLM providers (see "Extending the Harness" in README)

## Example Workflow for Research

```bash
# Experiment 1: Baseline with temperature 1.0
# Record results, note accuracy

# Experiment 2: Low temperature (deterministic)
# Change temperature to 0.0 in UI
# Compare accuracy vs. baseline

# Experiment 3: Different sample
# Change random seed to get different questions
# Check if accuracy is consistent

# Experiment 4: Larger sample
# Increase sample size to 50
# Get more statistically significant results
```

## Need Help?

- Check the full [README.md](README.md)
- Review code documentation in the source files
- Open an issue if you find bugs or have questions
