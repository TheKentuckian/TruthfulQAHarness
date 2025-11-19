# TruthfulQA Evaluation Harness

A web-based evaluation harness for assessing the truthfulness of Large Language Models (LLMs) using the TruthfulQA dataset. Built for research on self-correction techniques in LLMs.

## Overview

This harness provides a simple, configurable interface to:
- Load questions from the TruthfulQA dataset
- Prompt configurable LLMs (starting with Claude Sonnet)
- Evaluate answers using configurable verifiers (starting with word similarity)
- View detailed results and statistics in a web interface

## Features

### Phase 1 (Current Implementation)

- **Dataset Loading**: Automatic loading and sampling from the TruthfulQA dataset via HuggingFace
- **Configurable LLM Provider**: Extensible architecture supporting multiple LLMs
  - Currently implemented: Claude Sonnet 4.5
  - Easy to add: OpenAI GPT, other Claude models, etc.
- **Configurable Verifier**: Pluggable verification system
  - Currently implemented:
    - Simple Text Verifier (Word Overlap - lightweight, no sklearn required)
    - Word Similarity (TF-IDF + Cosine Similarity - requires sklearn)
  - Framework ready for: LLM-based verifiers, semantic similarity, etc.
- **Web Interface**: Clean, responsive UI for configuration and results visualization
- **Batch Evaluation**: Evaluate multiple questions in a single run
- **Detailed Metrics**: Per-question and aggregate statistics

### Phase 2 (Self-Correction - **NEW!**)

- **Reward/Feedback Self-Correction**: External reward model scores answers and provides feedback
  - LLM-based reward model with multi-criteria scoring (truthfulness, coherence, completeness, etc.)
  - Detailed feedback and actionable suggestions
  - Automatic self-correction based on reward scores
  - Comparison metrics (initial vs. corrected answers)
  - API endpoints for self-correcting evaluation
  - See [SELF_CORRECTION.md](SELF_CORRECTION.md) for detailed documentation

### Future Phases

- Phase 3: Advanced verifiers (LLM-as-judge, semantic similarity)
- Phase 4: Result persistence and experiment tracking

## Architecture

```
TruthfulQAHarness/
├── backend/
│   ├── app.py                  # FastAPI server
│   ├── config.py               # Configuration management
│   ├── models/
│   │   ├── llm_provider.py    # LLM provider abstraction
│   │   └── verifier.py        # Verifier abstraction
│   └── services/
│       ├── dataset_loader.py  # TruthfulQA loader
│       └── evaluator.py       # Evaluation pipeline
├── frontend/
│   ├── index.html             # Main UI
│   ├── styles.css             # Styling
│   └── app.js                 # Frontend logic
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Anthropic API key (for Claude)

**🔵 Chromebook/Crostini Users**: See [CHROMEBOOK-SETUP.md](CHROMEBOOK-SETUP.md) for a lightweight installation that skips sklearn.

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

   Edit `.env` and add your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Starting the Server

```bash
python -m backend.app
```

The server will start on `http://localhost:8000` by default.

### Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:8000`

2. **Configure the evaluation**:
   - **LLM Provider**: Select Claude (currently the only option)
   - **Model**: Specify Claude model (default: claude-sonnet-4-5-20250929)
   - **Max Tokens**: Set maximum tokens for generation (default: 1024)
   - **Temperature**: Set sampling temperature (default: 1.0)
   - **Verifier**: Select Word Similarity (currently the only option)
   - **Sample Size**: Number of questions to evaluate (default: 10)
   - **Random Seed**: Optional seed for reproducibility

3. **Load Sample Questions**: Click "Load Sample Questions" to fetch random questions from TruthfulQA

4. **Evaluate**: Click "Evaluate Batch" to run the evaluation

5. **View Results**: See detailed results including:
   - Overall accuracy and statistics
   - Per-question truthfulness assessment
   - Confidence scores
   - Similarity metrics
   - LLM responses

### API Endpoints

The harness provides a comprehensive REST API for programmatic access:

#### Self-Correction Endpoints (**NEW!**)

- `POST /api/evaluate/self-correct/single` - Evaluate single question with self-correction
  ```json
  {
    "question_index": 0,
    "config": {
      "llm_provider": "claude",
      "verifier_type": "simple_text",
      "reward_model_type": "llm_reward",
      "enable_correction": true,
      "score_threshold": 0.7,
      "max_iterations": 1
    }
  }
  ```

- `POST /api/evaluate/self-correct/batch` - Evaluate batch with self-correction
  ```json
  {
    "sample_size": 5,
    "seed": 42,
    "config": {
      "llm_provider": "claude",
      "verifier_type": "simple_text",
      "reward_model_type": "llm_reward",
      "enable_correction": true
    }
  }
  ```

#### Dataset Endpoints

- `GET /api/dataset/info` - Get dataset information
- `GET /api/dataset/sample?sample_size=10&seed=42` - Get sample questions
- `GET /api/dataset/question/{index}` - Get specific question by index

#### Evaluation Endpoints

- `POST /api/evaluate/single` - Evaluate a single question
  ```json
  {
    "question_index": 0,
    "config": {
      "llm_provider": "claude",
      "llm_config": {"model": "claude-sonnet-4-5-20250929"},
      "verifier_type": "word_similarity",
      "max_tokens": 1024,
      "temperature": 1.0
    }
  }
  ```

- `POST /api/evaluate/batch` - Evaluate multiple questions
  ```json
  {
    "sample_size": 10,
    "seed": 42,
    "config": {
      "llm_provider": "claude",
      "verifier_type": "word_similarity"
    }
  }
  ```

#### Configuration Endpoints

- `GET /api/providers` - List available LLM providers
- `GET /api/verifiers` - List available verifiers
- `GET /api/reward-models` - List available reward models

## Quick Start: Self-Correction Demo

To see the reward/feedback self-correction technique in action:

```bash
python demo_self_correction.py
```

This demonstration shows:
1. Initial answer generation (baseline)
2. Reward model scoring on multiple criteria
3. Detailed feedback and suggestions
4. Self-corrected answer generation
5. Comparison of improvements

For comprehensive documentation on self-correction, see [SELF_CORRECTION.md](SELF_CORRECTION.md).

## How It Works

### 1. Dataset Loading

The harness uses the HuggingFace `datasets` library to load the TruthfulQA dataset. Questions include:
- The question text
- List of correct (truthful) answers
- List of incorrect (untruthful) answers
- Category information

### 2. LLM Prompting

Questions are formatted into prompts and sent to the configured LLM. The harness uses a simple prompt format:
```
Q: {question}
A:
```

### 3. Answer Verification

The Word Similarity verifier:
1. Preprocesses all text (lowercase, remove extra whitespace)
2. Vectorizes using TF-IDF (Term Frequency-Inverse Document Frequency)
3. Computes cosine similarity between LLM answer and reference answers
4. Determines truthfulness based on whether similarity to correct answers exceeds similarity to incorrect answers
5. Provides confidence score based on the difference in similarities

### 4. Results Presentation

Results include:
- **Binary truthfulness classification**: Is the answer truthful?
- **Confidence score**: How confident is the verifier? (0-1)
- **Detailed metrics**: Similarity scores, reasoning
- **Aggregate statistics**: Overall accuracy, average confidence

## Extending the Harness

### Adding a New LLM Provider

1. Create a new class in `backend/models/llm_provider.py`:
   ```python
   class MyLLMProvider(LLMProvider):
       def generate(self, prompt, max_tokens=None, temperature=None, **kwargs):
           # Implementation
           pass

       def get_provider_name(self):
           return "My LLM"
   ```

2. Register it in the factory:
   ```python
   LLMProviderFactory._providers["myllm"] = MyLLMProvider
   ```

### Adding a New Verifier

1. Create a new class in `backend/models/verifier.py`:
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

2. Register it in the factory:
   ```python
   VerifierFactory._verifiers["myverifier"] = MyVerifier
   ```

## Configuration Options

### Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEFAULT_MODEL`: Default Claude model
- `DEFAULT_MAX_TOKENS`: Default max tokens (default: 1024)
- `DEFAULT_TEMPERATURE`: Default temperature (default: 1.0)
- `TRUTHFULQA_SAMPLE_SIZE`: Default sample size (default: 10)

## Research Notes

### About TruthfulQA

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

### Word Similarity Verifier

The word similarity approach is a simple baseline that:
- **Advantages**: Fast, interpretable, no additional API calls
- **Limitations**: Surface-level comparison, doesn't understand semantics
- **Best for**: Initial exploration and baseline performance

### Self-Correction Research

**Implemented - Reward/Feedback Technique**

The harness now includes a complete implementation of reward/feedback self-correction:

- **Reward Model**: LLM-based scoring on 5 criteria (truthfulness, coherence, completeness, relevance, safety)
- **Feedback Loop**: Detailed feedback and suggestions provided to the LLM
- **Automatic Correction**: LLM self-corrects based on reward scores and feedback
- **Metrics**: Comprehensive comparison of initial vs. corrected answers
- **Configurable**: Adjustable score thresholds, iteration counts, and criteria weights

See [SELF_CORRECTION.md](SELF_CORRECTION.md) for complete documentation.

**Research Questions You Can Explore:**
- Does self-correction improve truthfulness on TruthfulQA?
- What score threshold maximizes improvement?
- How many iterations are optimal?
- Which criteria correlate with truthfulness improvements?
- When does correction make things worse?

### Future Directions

Additional self-correction techniques to implement:
1. **Multi-model Correction**: Use different LLMs for answering vs. correction
2. **Iterative Refinement**: Continue correction until convergence
3. **Chain-of-Thought**: Prompt LLM to explain reasoning before answering
4. **Verification-based Re-prompting**: Use verifier results directly for correction
5. **Ensemble Methods**: Combine multiple correction strategies

## Troubleshooting

For detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

### Common Issues - Quick Fixes

**ImportError: cannot import name 'HfFolder'**
```bash
pip install --upgrade datasets huggingface-hub
```

**Client.__init__() got unexpected keyword 'proxies'**
```bash
pip install --upgrade anthropic
```

**scikit-learn won't compile (Chromebook)**
```bash
./setup-chromebook.sh  # Use Simple Text Verifier instead
```

**Server won't start**
- Check port 8000 isn't in use
- Verify virtual environment is activated
- Check `.env` file exists with valid API key

**Low accuracy / All results untruthful**
- Try different verifier (Simple Text vs Word Similarity)
- Check dataset loaded correctly
- Verify API key is working

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for comprehensive solutions to all common issues.

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

## Acknowledgments

- TruthfulQA dataset by Lin et al.
- Anthropic for the Claude API
- HuggingFace for the datasets library
