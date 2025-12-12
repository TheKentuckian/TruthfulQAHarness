# TruthfulQA Evaluation Harness

A streamlined evaluation harness for assessing LLM truthfulness using the TruthfulQA dataset.

## Overview

- Load questions from the TruthfulQA dataset
- Generate responses using Claude or local LLMs (via LM Studio)
- Apply self-correction techniques (Chain of Thought, Critique, Reward/Feedback)
- Validate answers with multiple verification methods
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
│   ├── config.py                        # Configuration
│   ├── models/                          # LLM providers, verifiers, reward models
│   └── services/                        # Dataset, evaluation, session management
├── console.py                           # Interactive console application
├── analysis.ipynb                       # Jupyter notebook
├── requirements.txt                     # Dependencies
└── .env.example                         # Environment template
```

## Installation

**Prerequisites:** Python 3.8+, Anthropic API key (for Claude) or LM Studio (for local models)

```bash
git clone <repository-url>
cd TruthfulQAHarness
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

## Usage

**Console App:**
```bash
python console.py
```

**Jupyter Notebook:**
```bash
jupyter notebook analysis.ipynb
```

## Quick Start

```bash
python console.py
```

1. Select "Run Session Workflow"
2. Create a new session or select existing
3. Configure: questions, LLM provider, self-correction method, verifier
4. View results

## Configuration

**Environment Variables** (`.env`):
- `ANTHROPIC_API_KEY` - Required for Claude
- `DEFAULT_MODEL` - Default: `claude-sonnet-4-5-20250929`
- `DEFAULT_MAX_TOKENS` - Default: 1024
- `DEFAULT_TEMPERATURE` - Default: 1.0

**LLM Providers:**
- **Claude**: Requires API key in `.env`
- **LM Studio**: No API key needed, runs locally at `http://localhost:1234/v1`

**Verification Methods:**
- **Simple Text**: Word overlap
- **Word Similarity**: TF-IDF + cosine similarity
- **LLM Judge**: Most accurate, uses LLM to judge truthfulness

## About TruthfulQA

817 questions across 38 categories designed to test whether LLMs generate truthful answers. Questions target common human falsehoods and misconceptions.

**Citation:** Lin et al., "TruthfulQA: Measuring How Models Mimic Human Falsehoods", arXiv:2109.07958, 2021

**Data Storage:** SQLite database (`truthfulqa_harness.db`) stores session metadata, questions, responses, and validation results.

## Extending

**Add LLM Provider:** Create a class in `backend/models/llm_provider.py` inheriting from `LLMProvider` and register it in the factory.

**Add Verifier:** Create a class in `backend/models/verifier.py` inheriting from `Verifier` and register it in the factory.

## License

Academic research purposes. Please cite appropriately if used in publications.
