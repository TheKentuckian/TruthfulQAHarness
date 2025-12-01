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
from backend.models.reward_model import RewardModelFactory
from backend.services.dataset_loader import TruthfulQALoader
from backend.services.evaluator import Evaluator
from backend.services.self_correcting_evaluator import SelfCorrectingEvaluator
from backend.services.database import get_database

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


def create_verifier(verifier_type: str, verifier_config: Dict[str, Any]):
    """
    Create a verifier instance, handling special cases like LLM judge.

    Args:
        verifier_type: Type of verifier
        verifier_config: Configuration for the verifier

    Returns:
        Verifier instance
    """
    if verifier_type == "llm_judge":
        # Extract judge LLM configuration from verifier_config
        judge_provider_type = verifier_config.get("judge_provider", "lm_studio")
        judge_llm_config = verifier_config.get("judge_llm_config", {})

        # Create a separate LLM provider for the judge
        judge_llm_provider = LLMProviderFactory.create(
            judge_provider_type,
            **judge_llm_config
        )

        # Create verifier with the judge LLM provider
        verifier_kwargs = {k: v for k, v in verifier_config.items()
                         if k not in ["judge_provider", "judge_llm_config"]}
        return VerifierFactory.create(
            verifier_type,
            llm_provider=judge_llm_provider,
            **verifier_kwargs
        )
    else:
        return VerifierFactory.create(
            verifier_type,
            **verifier_config
        )


# Pydantic models for request/response
class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    llm_provider: str = Field(default="claude", description="LLM provider type")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration")
    verifier_type: str = Field(default="word_similarity", description="Verifier type")
    verifier_config: Dict[str, Any] = Field(default_factory=dict, description="Verifier configuration")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for generation")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation")


class SelfCorrectionConfig(BaseModel):
    """Configuration for self-correction evaluation."""
    llm_provider: str = Field(default="claude", description="LLM provider type")
    llm_config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration")
    verifier_type: str = Field(default="word_similarity", description="Verifier type")
    verifier_config: Dict[str, Any] = Field(default_factory=dict, description="Verifier configuration")
    reward_model_type: str = Field(default="llm_reward", description="Reward model type")
    reward_model_config: Dict[str, Any] = Field(default_factory=dict, description="Reward model configuration")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for generation")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation")
    enable_correction: bool = Field(default=True, description="Enable self-correction")
    score_threshold: float = Field(default=0.7, description="Score threshold for correction")
    max_iterations: int = Field(default=1, description="Maximum correction iterations")


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


class SelfCorrectionRequest(BaseModel):
    """Request for self-correction evaluation."""
    question_index: Optional[int] = Field(default=None, description="Index of single question")
    sample_size: Optional[int] = Field(default=None, description="Number of questions to sample")
    seed: Optional[int] = Field(default=None, description="Random seed for sampling")
    question_indices: Optional[List[int]] = Field(default=None, description="Specific question indices")
    config: SelfCorrectionConfig = Field(default_factory=SelfCorrectionConfig)


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


@app.get("/api/reward-models")
async def get_available_reward_models():
    """Get list of available reward models."""
    return {
        "reward_models": RewardModelFactory.get_available_models(),
    }


@app.get("/api/results")
async def list_evaluation_results(
    limit: int = 50,
    offset: int = 0,
):
    """
    List past evaluation results.

    Args:
        limit: Maximum number of evaluations to return (default: 50)
        offset: Number of evaluations to skip (default: 0)
    """
    try:
        db = get_database()
        evaluations = db.list_evaluations(limit=limit, offset=offset)
        total_count = db.get_evaluation_count()

        return {
            "evaluations": evaluations,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{evaluation_id}")
async def get_evaluation_result(evaluation_id: int):
    """
    Get a specific evaluation result.

    Args:
        evaluation_id: The evaluation ID
    """
    try:
        db = get_database()
        evaluation = db.get_evaluation(evaluation_id)

        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return evaluation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/results/{evaluation_id}/questions")
async def get_evaluation_question_results(evaluation_id: int):
    """
    Get question-level results for an evaluation.

    Args:
        evaluation_id: The evaluation ID
    """
    try:
        db = get_database()

        # Check if evaluation exists
        evaluation = db.get_evaluation(evaluation_id)
        if not evaluation:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        # Get question results
        question_results = db.get_question_results(evaluation_id)

        return {
            "evaluation_id": evaluation_id,
            "question_results": question_results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/results/{evaluation_id}")
async def delete_evaluation_result(evaluation_id: int):
    """
    Delete an evaluation and its results.

    Args:
        evaluation_id: The evaluation ID
    """
    try:
        db = get_database()
        deleted = db.delete_evaluation(evaluation_id)

        if not deleted:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return {"success": True, "message": f"Evaluation {evaluation_id} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        verifier = create_verifier(
            request.config.verifier_type,
            request.config.verifier_config
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
        verifier = create_verifier(
            request.config.verifier_type,
            request.config.verifier_config
        )

        # Create evaluator
        evaluator = Evaluator(llm_provider=llm_provider, verifier=verifier)

        # Evaluate batch
        results = evaluator.evaluate_batch(
            questions=questions,
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
        )

        # Save results to database
        try:
            db = get_database()
            evaluation_id = db.save_evaluation(
                summary=results['summary'],
                results=results['results'],
                config=request.config.dict()
            )
            results['evaluation_id'] = evaluation_id
            print(f"Saved evaluation to database with ID: {evaluation_id}")
        except Exception as e:
            print(f"Warning: Failed to save evaluation to database: {str(e)}")
            # Don't fail the request if database save fails

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/self-correct/single")
async def evaluate_single_with_self_correction(request: SelfCorrectionRequest):
    """
    Evaluate a single question with self-correction.

    Args:
        request: Self-correction request with question index and config
    """
    try:
        if request.question_index is None:
            raise ValueError("question_index is required for single evaluation")

        # Get the question
        question = dataset_loader.get_question_by_index(request.question_index)

        # Create LLM provider
        llm_provider = LLMProviderFactory.create(
            request.config.llm_provider,
            **request.config.llm_config
        )

        # Create verifier
        verifier = create_verifier(
            request.config.verifier_type,
            request.config.verifier_config
        )

        # Create reward model
        reward_model = RewardModelFactory.create(
            request.config.reward_model_type,
            llm_provider=llm_provider,
            **request.config.reward_model_config
        )

        # Create self-correcting evaluator
        evaluator = SelfCorrectingEvaluator(
            llm_provider=llm_provider,
            verifier=verifier,
            reward_model=reward_model,
            score_threshold=request.config.score_threshold,
            max_iterations=request.config.max_iterations,
        )

        # Evaluate with self-correction
        result = evaluator.evaluate_single_with_correction(
            question_data=question,
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
            enable_correction=request.config.enable_correction,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/self-correct/batch")
async def evaluate_batch_with_self_correction(request: SelfCorrectionRequest):
    """
    Evaluate a batch of questions with self-correction.

    Args:
        request: Self-correction batch request
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

        # Create LLM provider
        llm_provider = LLMProviderFactory.create(
            request.config.llm_provider,
            **request.config.llm_config
        )

        # Create verifier
        verifier = create_verifier(
            request.config.verifier_type,
            request.config.verifier_config
        )

        # Create reward model
        reward_model = RewardModelFactory.create(
            request.config.reward_model_type,
            llm_provider=llm_provider,
            **request.config.reward_model_config
        )

        # Create self-correcting evaluator
        evaluator = SelfCorrectingEvaluator(
            llm_provider=llm_provider,
            verifier=verifier,
            reward_model=reward_model,
            score_threshold=request.config.score_threshold,
            max_iterations=request.config.max_iterations,
        )

        # Evaluate batch with self-correction
        results = evaluator.evaluate_batch_with_correction(
            questions=questions,
            max_tokens=request.config.max_tokens,
            temperature=request.config.temperature,
            enable_correction=request.config.enable_correction,
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
