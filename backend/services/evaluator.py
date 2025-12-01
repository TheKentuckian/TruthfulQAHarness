"""Evaluation pipeline for TruthfulQA."""
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.models.llm_provider import LLMProvider
from backend.models.verifier import Verifier


class Evaluator:
    """
    Evaluation pipeline that coordinates LLM generation and answer verification.
    """

    def __init__(self, llm_provider: LLMProvider, verifier: Verifier):
        """
        Initialize the evaluator.

        Args:
            llm_provider: The LLM provider to use for generation
            verifier: The verifier to use for answer verification
        """
        self.llm_provider = llm_provider
        self.verifier = verifier

    def _create_prompt(self, question: str) -> str:
        """
        Create a prompt for the LLM from a question.

        Args:
            question: The question to ask

        Returns:
            Formatted prompt string
        """
        # Simple direct prompt - can be made more sophisticated later
        return f"Q: {question}\nA:"

    def evaluate_single(
        self,
        question_data: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question.

        Args:
            question_data: Dictionary containing question and answers
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation

        Returns:
            Dictionary containing evaluation results
        """
        start_time = datetime.now()

        question = question_data["question"]
        correct_answers = question_data.get("correct_answers", [])
        incorrect_answers = question_data.get("incorrect_answers", [])

        # Create prompt and generate answer
        prompt = self._create_prompt(question)

        try:
            llm_answer = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            import traceback
            error_details = f"LLM generation failed: {str(e)}"
            print(f"\n=== ERROR in evaluate_single ===")
            print(f"Question: {question}")
            print(f"Error: {error_details}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("================================\n")
            return {
                "question": question,
                "llm_answer": None,
                "error": error_details,
                "verification": None,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        # Verify the answer
        try:
            verification_result = self.verifier.verify(
                llm_answer=llm_answer,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
                question=question,  # Pass question for LLM judge verifiers
            )
        except Exception as e:
            verification_result = {
                "is_truthful": None,
                "confidence": 0.0,
                "reasoning": f"Verification failed: {str(e)}",
                "metrics": {"error": str(e)}
            }

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "question": question,
            "question_index": question_data.get("index"),
            "category": question_data.get("category", "Unknown"),
            "llm_answer": llm_answer,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "verification": verification_result,
            "llm_provider": self.llm_provider.get_provider_name(),
            "verifier": self.verifier.get_verifier_name(),
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
        }

    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of questions.

        Args:
            questions: List of question dictionaries
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation

        Returns:
            Dictionary containing batch evaluation results and summary
        """
        results = []
        start_time = datetime.now()

        for question_data in questions:
            result = self.evaluate_single(
                question_data=question_data,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Compute summary statistics
        total_questions = len(results)
        successful_evaluations = sum(
            1 for r in results
            if r.get("verification") and r["verification"].get("is_truthful") is not None
        )

        truthful_count = sum(
            1 for r in results
            if r.get("verification") and r["verification"].get("is_truthful") is True
        )

        # Calculate accuracy (truthful answers / successful evaluations)
        accuracy = truthful_count / successful_evaluations if successful_evaluations > 0 else 0.0

        # Calculate average confidence
        confidences = [
            r["verification"]["confidence"]
            for r in results
            if r.get("verification") and "confidence" in r["verification"]
        ]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        summary = {
            "total_questions": total_questions,
            "successful_evaluations": successful_evaluations,
            "truthful_count": truthful_count,
            "untruthful_count": successful_evaluations - truthful_count,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_duration_seconds": total_duration,
            "average_duration_per_question": total_duration / total_questions if total_questions > 0 else 0.0,
            "llm_provider": self.llm_provider.get_provider_name(),
            "verifier": self.verifier.get_verifier_name(),
            "timestamp": start_time.isoformat(),
        }

        return {
            "results": results,
            "summary": summary,
        }
