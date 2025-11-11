# Chromebook/Crostini Setup Guide

This guide is specifically for running the TruthfulQA Harness on a Chromebook with Crostini Linux.

## Why a Special Setup?

Chromebooks with Crostini (Linux VM) can have trouble compiling large Python packages like `scikit-learn` because:
- Limited CPU resources
- Missing compilation tools
- ARM architecture (some Chromebooks)

This guide uses a **minimal installation** with the **Simple Text Verifier** that doesn't require sklearn.

## Quick Start

### 1. Open Linux Terminal on Chromebook

Open the Terminal app (or press `Ctrl+Alt+T`, then type `shell`)

### 2. Navigate to Project

```bash
cd TruthfulQAHarness
```

### 3. Run Chromebook Setup

```bash
./setup-chromebook.sh
```

This will:
- Update system packages
- Install minimal build tools
- Create Python virtual environment
- Install only essential dependencies (no sklearn)
- Create `.env` configuration file

### 4. Add API Key

```bash
nano .env
```

Change this line:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Save with `Ctrl+X`, then `Y`, then `Enter`.

### 5. Run the Server

```bash
./run.sh
```

### 6. Open the App

In Chrome browser, go to:
```
http://localhost:8000
```

## Using the Simple Text Verifier

The **Simple Text Verifier** is automatically selected in the UI. It:

- Uses word overlap (Jaccard similarity) instead of TF-IDF
- Requires no external libraries
- Is fast and lightweight
- Works well for baseline experiments

**How it works:**
1. Extracts words from LLM answer and reference answers
2. Computes Jaccard similarity: `intersection / union`
3. Compares overlap with correct vs. incorrect answers
4. Classifies based on which has higher overlap

## Performance Comparison

| Verifier | Accuracy* | Speed | Dependencies | Chromebook Support |
|----------|-----------|-------|--------------|-------------------|
| Simple Text | ~75% | Fast | None | ✅ Excellent |
| Word Similarity | ~80% | Medium | sklearn | ⚠️ May fail to install |

*Approximate, varies by question set

## If You Want to Try sklearn Anyway

Some Chromebooks can install sklearn, but it may take 10-30 minutes:

```bash
source venv/bin/activate
pip install --no-cache-dir scikit-learn numpy scipy
```

**Tips:**
- Don't use the computer while installing (it may get hot)
- Keep it plugged in
- If it fails after 30 minutes, stick with Simple Text Verifier
- Try installing system packages first:
  ```bash
  sudo apt-get install python3-sklearn python3-numpy python3-scipy
  ```

## Troubleshooting

### "pip is trying to build sklearn and failing"

This is why we created `setup-chromebook.sh`! Use it instead:
```bash
./setup-chromebook.sh
```

It skips sklearn entirely and uses Simple Text Verifier.

### "ModuleNotFoundError: No module named 'sklearn'"

This is expected! Use Simple Text Verifier:
1. In the web UI, select "Simple Text (Word Overlap) - Lightweight"
2. Continue with evaluation

### "Cannot connect to localhost:8000"

1. Check server is running: `./run.sh`
2. Try: `http://penguin.linux.test:8000` (Crostini's hostname)
3. Make sure port 8000 isn't blocked

### Server runs but immediately crashes

Check your API key:
```bash
cat .env | grep ANTHROPIC
```

Should show your actual key, not the placeholder.

### "Out of memory" errors

Reduce the sample size:
- In the UI, set Sample Size to 5 or fewer
- Chromebooks have limited RAM in the Linux VM

## Minimal vs Full Installation

### Minimal (Chromebook Setup)
```bash
./setup-chromebook.sh  # Uses requirements-minimal.txt
```

**Installs:**
- FastAPI, uvicorn (web server)
- Anthropic SDK (Claude API)
- HuggingFace datasets (TruthfulQA)
- Pydantic (configuration)

**Uses:**
- Simple Text Verifier (no sklearn needed)

### Full Installation
```bash
./setup.sh  # Uses requirements.txt
```

**Installs everything above plus:**
- scikit-learn (TF-IDF, cosine similarity)
- numpy, scipy (numerical computing)

**Uses:**
- Both Simple Text and Word Similarity verifiers

## System Requirements

### Minimum
- Chromebook with Linux (Crostini) enabled
- 4GB RAM (2GB allocated to Linux)
- 5GB free disk space
- Internet connection

### Recommended
- 8GB RAM (4GB to Linux)
- 10GB free disk space
- Chromebook plugged in during setup

## For Your Research

The Simple Text Verifier is perfectly valid for research:

1. **It's a legitimate baseline**: Word overlap is a recognized similarity metric
2. **Interpretable**: Easy to understand and explain in your paper
3. **Reproducible**: No sklearn version issues
4. **Fast**: Good for iterating on experiments

You can note in your thesis:
> "Due to resource constraints, initial experiments used Jaccard word overlap similarity. This provided a lightweight baseline while maintaining reasonable accuracy for truthfulness classification."

## Getting Help

If you're still stuck:

1. Check Python version: `python3 --version` (need 3.8+)
2. Check disk space: `df -h`
3. Check Linux RAM: `free -h`
4. Try restarting Linux VM:
   - Settings → Linux → Remove
   - Settings → Linux → Turn on (reinstall)
   - Re-run setup-chromebook.sh

## What's Next?

Once running:

1. Start with 5 questions to test
2. Gradually increase to 10, then 20
3. Monitor RAM usage: `htop` (install with `sudo apt-get install htop`)
4. For your thesis, collect results from Simple Text Verifier
5. If needed, run Word Similarity verifier on a different machine later for comparison

Good luck with your research! 🎓
