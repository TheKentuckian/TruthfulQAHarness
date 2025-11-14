# Troubleshooting Guide

Common issues and solutions for the TruthfulQA Evaluation Harness.

## Installation Issues

### ImportError: cannot import name 'HfFolder' from 'huggingface_hub'

**Problem**: This occurs when there's a version mismatch between `datasets` and `huggingface_hub`.

**Solution**:
```bash
# On Windows (in your project directory)
venv\Scripts\activate
pip install --upgrade datasets huggingface-hub

# On Linux/Mac
source venv/bin/activate
pip install --upgrade datasets huggingface-hub
```

**Or**, reinstall all dependencies:
```bash
# On Windows
venv\Scripts\activate
pip install -r requirements.txt --upgrade

# On Linux/Mac
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

**If the issue persists**, clear pip cache and reinstall:
```bash
pip cache purge
pip uninstall datasets huggingface-hub -y
pip install datasets>=2.18.0 huggingface-hub>=0.21.0
```

### scikit-learn compilation fails (Chromebook/ARM systems)

**Problem**: sklearn requires compilation which may fail on resource-constrained systems.

**Solution**: Use the minimal installation with Simple Text Verifier
```bash
./setup-chromebook.sh  # Linux/Mac/Chromebook
```

See [CHROMEBOOK-SETUP.md](CHROMEBOOK-SETUP.md) for detailed instructions.

### ModuleNotFoundError: No module named 'pydantic_settings'

**Problem**: Older pip or Python version.

**Solution**:
```bash
pip install --upgrade pip
pip install pydantic-settings
```

### "Failed to load TruthfulQA dataset"

**Problem**: Network issues or HuggingFace hub access.

**Solutions**:
1. Check internet connection
2. Try clearing HuggingFace cache:
   ```bash
   # Windows
   rmdir /s %USERPROFILE%\.cache\huggingface

   # Linux/Mac
   rm -rf ~/.cache/huggingface
   ```
3. Manually download dataset:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("truthful_qa", "generation", split="validation")
   ```

## Runtime Issues

### "Error generating response from Claude"

**Possible causes and solutions**:

1. **Invalid API Key**
   - Check `.env` file has correct `ANTHROPIC_API_KEY`
   - Verify key at https://console.anthropic.com

2. **No API Credits**
   - Check your Anthropic account has available credits

3. **Rate Limiting**
   - Reduce sample size
   - Add delay between requests

4. **Network Issues**
   - Check firewall settings
   - Try from a different network

### Server won't start / Port already in use

**Problem**: Port 8000 is already in use.

**Solution**: Either kill the existing process or use a different port:

```bash
# Change port in .env file
PORT=8001

# Or specify when running
python -m backend.app
# Then edit backend/app.py to use the new port
```

**Find and kill process on port 8000**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Browser can't connect to localhost:8000

**Solutions**:

1. **Check server is running**: Look for "Uvicorn running on..." message

2. **Try different URL**:
   - Windows: `http://localhost:8000` or `http://127.0.0.1:8000`
   - Chromebook Crostini: `http://penguin.linux.test:8000`
   - WSL: `http://localhost:8000` from Windows browser

3. **Check firewall**: Temporarily disable to test

4. **Verify virtual environment is activated**

### "Out of memory" errors

**Solutions**:

1. **Reduce sample size**: Use 5-10 questions instead of larger batches

2. **Close other applications**

3. **For Chromebook**: Increase Linux container memory:
   - Settings → Linux → Disk size
   - Allocate more RAM to Linux VM

4. **Use Simple Text Verifier**: Requires less memory than Word Similarity

## Configuration Issues

### Changes to .env not taking effect

**Solution**: Restart the server after editing `.env`:
```bash
# Stop server (Ctrl+C)
# Restart
./run.sh  # or python -m backend.app
```

### Verifier selection not working

**Problem**: Selected verifier type doesn't match what's available.

**Solution**:
1. Check available verifiers: Go to `http://localhost:8000/api/verifiers`
2. Make sure selection in UI matches one of the available types
3. If using `word_similarity`, ensure sklearn is installed:
   ```bash
   pip install scikit-learn numpy scipy
   ```

## Data/Results Issues

### All questions show same result

**Possible causes**:

1. **Same random seed**: Change or remove random seed value
2. **API issue**: Check Claude API is responding differently
3. **Verifier issue**: Try switching verifiers

### Low accuracy (all answers marked untruthful)

**Possible causes**:

1. **Verifier not working correctly**:
   - Check if sklearn is properly installed (for Word Similarity)
   - Try Simple Text Verifier as alternative

2. **Reference answers missing**: Check dataset loaded correctly

3. **LLM generating very short answers**: Increase max_tokens

### Confidence scores all near zero

**Cause**: LLM answers are very different from both correct and incorrect reference answers.

**Solutions**:
1. This may be normal - it indicates the answers don't strongly match either set
2. Try adjusting temperature (lower for more conservative answers)
3. Check that reference answers are loading correctly

## Development/Code Issues

### Import errors when running tests

**Solution**: Run from project root with module syntax:
```bash
# Wrong
cd backend
python app.py

# Correct
python -m backend.app
```

### "virtual environment not activated"

**Symptoms**: Imports fail, wrong Python version, packages not found

**Solution**:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac/Chromebook
source venv/bin/activate

# Verify (should show path to venv)
which python  # Linux/Mac
where python  # Windows
```

## Platform-Specific Issues

### Windows: "python: command not found"

**Solution**: Use `python` or `py` or `python3` depending on installation:
```cmd
py -m venv venv
py -m backend.app
```

### WSL: Can't access from Windows browser

**Solution**:
1. Make sure server binds to `0.0.0.0` (check `.env`: `HOST=0.0.0.0`)
2. Access from Windows: `http://localhost:8000`
3. Or find WSL IP: `ip addr show eth0` and use that IP

### Mac: SSL Certificate errors

**Solution**:
```bash
# Install certificates
/Applications/Python*/Install\ Certificates.command

# Or upgrade certifi
pip install --upgrade certifi
```

## Getting More Help

If your issue isn't covered here:

1. Check the main [README.md](README.md)
2. Check platform-specific guides:
   - [CHROMEBOOK-SETUP.md](CHROMEBOOK-SETUP.md) for Chromebook/Crostini
   - [QUICKSTART.md](QUICKSTART.md) for general setup
3. Look at error messages carefully - they often indicate the exact problem
4. Try running with verbose logging:
   ```bash
   python -m backend.app --log-level debug
   ```

## Debugging Tips

### Enable verbose logging

Edit `backend/app.py` to add logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test individual components

```python
# Test dataset loader
python -c "from backend.services.dataset_loader import TruthfulQALoader; loader = TruthfulQALoader(); print(loader.get_dataset_info())"

# Test LLM provider
python -c "from backend.models.llm_provider import ClaudeProvider; from backend.config import settings; provider = ClaudeProvider(); print(provider.generate('Test question?', max_tokens=100))"

# Test verifier
python -c "from backend.models.verifier import SimpleTextVerifier; v = SimpleTextVerifier(); print(v.verify('The sky is blue', ['sky is blue'], ['sky is red']))"
```

### Check API endpoints directly

```bash
# Check health
curl http://localhost:8000/api/health

# Get dataset info
curl http://localhost:8000/api/dataset/info

# Get available verifiers
curl http://localhost:8000/api/verifiers
```
