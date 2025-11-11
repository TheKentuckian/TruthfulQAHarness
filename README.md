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

### Future Phases

- Phase 2: Self-correction capabilities with conditional re-prompting
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

The harness also provides a REST API for programmatic access:

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

### Future Directions

For your master's research on self-correction:
1. **Phase 2**: Implement conditional re-prompting based on verification results
2. **LLM-as-Judge**: Use another LLM to verify answers
3. **Semantic Similarity**: Use embeddings for deeper semantic comparison
4. **Chain-of-Thought**: Prompt LLM to explain reasoning
5. **Iterative Refinement**: Multiple correction attempts with different strategies

## Troubleshooting

### Dataset Loading Issues

If the TruthfulQA dataset fails to load:
- Check internet connection
- Verify HuggingFace datasets is installed: `pip install datasets`
- Try clearing cache: `rm -rf ~/.cache/huggingface`

### API Key Issues

If you get authentication errors:
- Verify your `.env` file has the correct API key
- Ensure the environment variable is loaded: `echo $ANTHROPIC_API_KEY`
- Check API key is valid at https://console.anthropic.com

### Import Errors

If you get import errors when running the server:
- Ensure you're in the project root directory
- Run with module syntax: `python -m backend.app`
- Verify virtual environment is activated

## License

This project is for academic research purposes. Please cite appropriately if used in publications.

## Acknowledgments

- TruthfulQA dataset by Lin et al.
- Anthropic for the Claude API
- HuggingFace for the datasets library
