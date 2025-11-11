"""FastAPI application for TruthfulQA Harness."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os

from backend.config import settings
from backend.models.llm_provider import LLMProviderFactory
from backend.models.verifier import VerifierFactory
from backend.services.dataset_loader import TruthfulQALoader
from backend.services.evaluator import Evaluator

# Initialize FastAPI app
app = FastAPI(
    title="TruthfulQA Evaluation Harness",
    description="A harness for evaluating LLM truthfulness using TruthfulQA dataset",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dataset loader
dataset_loader = TruthfulQALoader()


# Pydantic models for request/response
class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    llm_provider: str = Field(default="claude", description="LLM provider type")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration")
    verifier_type: str = Field(default="word_similarity", description="Verifier type")
    verifier_config: Dict[str, Any] = Field(default_factory=dict, description="Verifier configuration")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for generation")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation")


class SingleEvaluationRequest(BaseModel):
    """Request for evaluating a single question."""
    question_index: int = Field(description="Index of the question to evaluate")
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)


class BatchEvaluationRequest(BaseModel):
    """Request for evaluating a batch of questions."""
    sample_size: Optional[int] = Field(default=None, description="Number of questions to sample")
    seed: Optional[int] = Field(default=None, description="Random seed for sampling")
    question_indices: Optional[List[int]] = Field(default=None, description="Specific question indices")
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - serve frontend."""
    return FileResponse("frontend/index.html")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
    }


@app.get("/api/dataset/info")
async def get_dataset_info():
    """Get information about the TruthfulQA dataset."""
    try:
        info = dataset_loader.get_dataset_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/sample")
async def get_sample_questions(
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Get a random sample of questions from the dataset.

    Args:
        sample_size: Number of questions to sample
        seed: Random seed for reproducibility
    """
    try:
        questions = dataset_loader.get_sample(sample_size=sample_size, seed=seed)
        return {
            "questions": questions,
            "count": len(questions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/question/{index}")
async def get_question_by_index(index: int):
    """Get a specific question by index."""
    try:
        question = dataset_loader.get_question_by_index(index)
        return question
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/providers")
async def get_available_providers():
    """Get list of available LLM providers."""
    return {
        "providers": LLMProviderFactory.get_available_providers(),
    }


@app.get("/api/verifiers")
async def get_available_verifiers():
    """Get list of available verifiers."""
    return {
        "verifiers": VerifierFactory.get_available_verifiers(),
    }


@app.post("/api/evaluate/single")
async def evaluate_single_question(request: SingleEvaluationRequest):
    """
    Evaluate a single question.

    Args:
        request: Single evaluation request with question index and config
    """
    try:
        # Get the question
        question = dataset_loader.get_question_by_index(request.question_index)

        # Create LLM provider and verifier
        llm_provider = LLMProviderFactory.create(
            request.config.llm_provider,
            **request.config.llm_config
        )
        verifier = VerifierFactory.create(
            request.config.verifier_type,
            **request.config.verifier_config
        )

        # Create evaluator
        evaluator = Evaluator(llm_provider=llm_provider, verifier=verifier)

        # Evaluate
        result = evaluator.evaluate_single(
            question_data=question,
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/batch")
async def evaluate_batch_questions(request: BatchEvaluationRequest):
    """
    Evaluate a batch of questions.

    Args:
        request: Batch evaluation request with sampling config
    """
    try:
        # Get questions
        if request.question_indices:
            # Use specific indices
            questions = [
                dataset_loader.get_question_by_index(idx)
                for idx in request.question_indices
            ]
        else:
            # Use random sample
            questions = dataset_loader.get_sample(
                sample_size=request.sample_size,
                seed=request.seed,
            )

        # Create LLM provider and verifier
        llm_provider = LLMProviderFactory.create(
            request.config.llm_provider,
            **request.config.llm_config
        )
        verifier = VerifierFactory.create(
            request.config.verifier_type,
            **request.config.verifier_config
        )

        # Create evaluator
        evaluator = Evaluator(llm_provider=llm_provider, verifier=verifier)

        # Evaluate batch
        results = evaluator.evaluate_batch(
            questions=questions,
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
