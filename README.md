# TruthfulQA Evaluation Harness

A simplified evaluation harness for assessing the truthfulness of Large Language Models (LLMs) using the TruthfulQA dataset.

## Overview

This tool provides a straightforward interface to:
- Load questions from the TruthfulQA dataset
- Generate responses using configurable LLMs (Claude, LM Studio)
- Apply self-correction techniques (Chain of Thought, Critique, Reward/Feedback)
- Validate answers using multiple verification methods
- Track and analyze results through a session-based workflow

## Features

- **4-Phase Evaluation Workflow**:
  1. **Gather**: Load questions from TruthfulQA dataset
  2. **Generate**: Generate LLM responses
  3. **Correct**: Apply self-correction techniques (optional)
  4. **Validate**: Verify truthfulness of answers

- **Multiple LLM Providers**:
  - Claude (via Anthropic API)
  - LM Studio (local models)

- **Self-Correction Methods**:
  - Chain of Thought prompting
  - Critique-based self-correction
  - Reward/Feedback-based correction

- **Verification Methods**:
  - Simple text comparison (word overlap)
  - Word similarity (TF-IDF + cosine similarity)
  - LLM-based judge

- **Session Management**: Track multiple evaluation runs with persistent storage

## Repository Structure

```
TruthfulQAHarness/
├── backend/
│   ├── app.py                           # FastAPI server (optional)
│   ├── config.py                        # Configuration management
│   ├── models/
│   │   ├── llm_provider.py             # LLM provider abstraction
│   │   ├── verifier.py                 # Verifier abstraction
│   │   ├── reward_model.py             # Reward model for self-correction
│   │   └── self_corrector.py           # Self-correction logic
│   └── services/
│       ├── dataset_loader.py           # TruthfulQA loader
│       ├── evaluator.py                # Evaluation pipeline
│       ├── session_service.py          # Session management
│       ├── database.py                 # SQLite storage
│       └── prompt_strategies.py        # Prompting strategies
├── console.py                          # Interactive console application
├── analysis.ipynb                      # Jupyter notebook for analysis
├── requirements.txt                    # Python dependencies
└── .env.example                        # Environment variables template
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (optional, for Claude)
- LM Studio (optional, for local models)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TruthfulQAHarness
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Anthropic API key (if using Claude):
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Console Application (Recommended)

Run the interactive console app:

```bash
python console.py
```

The console app provides a menu-driven interface to:
- Create and manage evaluation sessions
- Run the complete 4-phase workflow
- View session results
- Manage the dataset

### Jupyter Notebook

For data analysis and visualization:

```bash
jupyter notebook analysis.ipynb
```

### FastAPI Backend (Optional)

If you need the REST API:

```bash
python -m backend.app
```

The API will be available at `http://localhost:8000`

API documentation: `http://localhost:8000/docs`

## Quick Start Example

1. Start the console app:
   ```bash
   python console.py
   ```

2. Select "Run Session Workflow"

3. Follow the prompts to:
   - Create a new session or select existing
   - Choose number of questions to evaluate
   - Select LLM provider (Claude or LM Studio)
   - Choose self-correction method (optional)
   - Select verification method

4. View results at the end of the workflow

## Configuration Options

### Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required for Claude)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEFAULT_MODEL`: Default Claude model
- `DEFAULT_MAX_TOKENS`: Default max tokens (default: 1024)
- `DEFAULT_TEMPERATURE`: Default temperature (default: 1.0)

### LLM Providers

**Claude**:
- Requires `ANTHROPIC_API_KEY` in `.env`
- Default model: `claude-sonnet-4-5-20250929`

**LM Studio**:
- No API key required
- Runs models locally
- Default URL: `http://localhost:1234/v1`
- Supports Qwen thinking mode

### Verification Methods

1. **Simple Text**: Fast word overlap comparison
2. **Word Similarity**: TF-IDF vectorization with cosine similarity
3. **LLM Judge**: Uses an LLM to judge truthfulness (most accurate)

## About TruthfulQA

TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance, and politics.

**Citation**:
```
@article{lin2021truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal={arXiv preprint arXiv:2109.07958},
  year={2021}
}
```

## Data Storage

Session data is stored in a SQLite database (`truthfulqa_harness.db`) which includes:
- Session metadata
- Questions sampled for each session
- Generated responses (initial and corrected)
- Validation results
- Phase configurations and results

## Extending the Harness

### Adding a New LLM Provider

Create a class in `backend/models/llm_provider.py`:

```python
class MyLLMProvider(LLMProvider):
    def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
        # Implementation
        pass

    def get_provider_name(self):
        return "My LLM"
```

Register it in the factory:
```python
LLMProviderFactory._providers["myllm"] = MyLLMProvider
```

### Adding a New Verifier

Create a class in `backend/models/verifier.py`:

```python
class MyVerifier(Verifier):
    def verify(self, llm_answer, correct_answers, incorrect_answers, **kwargs):
        # Implementation
        return {
            "is_truthful": True/False,
            "confidence": 0.0-1.0,
            "reasoning": "...",
            "metrics": {}
        }

    def get_verifier_name(self):
        return "My Verifier"
```

Register it in the factory:
```python
VerifierFactory._verifiers["myverifier"] = MyVerifier
```

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

## Acknowledgments

- TruthfulQA dataset by Lin et al.
- Anthropic for the Claude API
- HuggingFace for the datasets library
