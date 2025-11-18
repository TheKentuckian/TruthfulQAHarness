"""Self-correction evaluation pipeline using chain of thought."""
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.models.llm_provider import LLMProvider
from backend.models.verifier import Verifier
from backend.services.evaluator import Evaluator
from backend.services.prompt_strategies import (
    PromptStrategy,
    PromptStrategyFactory,
    DirectPromptStrategy,
    ChainOfThoughtStrategy,
    SelfCorrectionStrategy,
)


class SelfCorrectionEvaluator(Evaluator):
    """
    Enhanced evaluator that supports self-correction through chain of thought.

    This evaluator can:
    1. Generate initial answers with or without CoT
    2. Apply self-correction when answers are deemed untruthful
    3. Track the correction process and measure improvement
    4. Use different prompting strategies for different stages
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        verifier: Verifier,
        initial_strategy: Optional[PromptStrategy] = None,
        correction_strategy: Optional[PromptStrategy] = None,
    ):
        """
        Initialize the self-correction evaluator.

        Args:
            llm_provider: The LLM provider to use
            verifier: The verifier to use
            initial_strategy: Strategy for initial answer (default: Direct)
            correction_strategy: Strategy for correction (default: SelfCorrection)
        """
        super().__init__(llm_provider, verifier)
        self.initial_strategy = initial_strategy or DirectPromptStrategy()
        self.correction_strategy = correction_strategy or SelfCorrectionStrategy()

    def _create_prompt(self, question: str, strategy: Optional[PromptStrategy] = None) -> str:
        """
        Create a prompt using a specific strategy.

        Args:
            question: The question to ask
            strategy: The prompt strategy to use (default: initial_strategy)

        Returns:
            Formatted prompt string
        """
        strategy = strategy or self.initial_strategy
        return strategy.create_prompt(question)

    def evaluate_with_correction(
        self,
        question_data: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_correction_attempts: int = 1,
        apply_correction_only_if_wrong: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a question with optional self-correction.

        Args:
            question_data: Dictionary containing question and answers
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation
            max_correction_attempts: Maximum number of correction attempts
            apply_correction_only_if_wrong: Only attempt correction if initial answer is wrong

        Returns:
            Dictionary containing evaluation results with correction history
        """
        start_time = datetime.now()

        question = question_data["question"]
        correct_answers = question_data.get("correct_answers", [])
        incorrect_answers = question_data.get("incorrect_answers", [])

        # Track correction history
        correction_history = []

        # Step 1: Get initial answer
        initial_prompt = self.initial_strategy.create_prompt(question)

        try:
            initial_answer = self.llm_provider.generate(
                prompt=initial_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            import traceback
            error_details = f"LLM generation failed: {str(e)}"
            print(f"\n=== ERROR in evaluate_with_correction ===")
            print(f"Question: {question}")
            print(f"Error: {error_details}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("==========================================\n")
            return {
                "question": question,
                "error": error_details,
                "correction_history": [],
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        # Step 2: Verify initial answer
        try:
            initial_verification = self.verifier.verify(
                llm_answer=initial_answer,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
            )
        except Exception as e:
            initial_verification = {
                "is_truthful": None,
                "confidence": 0.0,
                "reasoning": f"Verification failed: {str(e)}",
                "metrics": {"error": str(e)}
            }

        # Record initial attempt
        correction_history.append({
            "attempt": 0,
            "strategy": self.initial_strategy.get_strategy_name(),
            "prompt": initial_prompt,
            "answer": initial_answer,
            "verification": initial_verification,
            "was_truthful": initial_verification.get("is_truthful"),
        })

        # Step 3: Apply self-correction if needed
        current_answer = initial_answer
        current_verification = initial_verification
        correction_applied = False

        # Determine if we should attempt correction
        should_correct = True
        if apply_correction_only_if_wrong:
            should_correct = not initial_verification.get("is_truthful", False)

        if should_correct and max_correction_attempts > 0:
            for attempt in range(1, max_correction_attempts + 1):
                correction_applied = True

                # Create correction prompt with context
                context = {
                    "previous_answer": current_answer,
                    "verification_feedback": current_verification.get("reasoning", ""),
                    "correction_history": correction_history,
                }

                correction_prompt = self.correction_strategy.create_prompt(
                    question, context
                )

                # Generate corrected answer
                try:
                    corrected_answer = self.llm_provider.generate(
                        prompt=correction_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except Exception as e:
                    # If correction fails, keep previous answer
                    correction_history.append({
                        "attempt": attempt,
                        "strategy": self.correction_strategy.get_strategy_name(),
                        "error": f"Generation failed: {str(e)}",
                    })
                    break

                # Verify corrected answer
                try:
                    corrected_verification = self.verifier.verify(
                        llm_answer=corrected_answer,
                        correct_answers=correct_answers,
                        incorrect_answers=incorrect_answers,
                    )
                except Exception as e:
                    corrected_verification = {
                        "is_truthful": None,
                        "confidence": 0.0,
                        "reasoning": f"Verification failed: {str(e)}",
                        "metrics": {"error": str(e)}
                    }

                # Record correction attempt
                correction_history.append({
                    "attempt": attempt,
                    "strategy": self.correction_strategy.get_strategy_name(),
                    "prompt": correction_prompt,
                    "answer": corrected_answer,
                    "verification": corrected_verification,
                    "was_truthful": corrected_verification.get("is_truthful"),
                })

                # Update current answer
                current_answer = corrected_answer
                current_verification = corrected_verification

                # Check if we've achieved truthfulness
                if corrected_verification.get("is_truthful", False):
                    break

        # Step 4: Compile results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Determine if correction was successful
        initial_truthful = initial_verification.get("is_truthful", False)
        final_truthful = current_verification.get("is_truthful", False)
        correction_successful = correction_applied and final_truthful and not initial_truthful

        result = {
            "question": question,
            "question_index": question_data.get("index"),
            "category": question_data.get("category", "Unknown"),
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            # Initial attempt
            "initial_answer": initial_answer,
            "initial_verification": initial_verification,
            "initial_strategy": self.initial_strategy.get_strategy_name(),
            # Final result
            "final_answer": current_answer,
            "final_verification": current_verification,
            # Correction tracking
            "correction_applied": correction_applied,
            "correction_successful": correction_successful,
            "total_attempts": len(correction_history),
            "correction_history": correction_history,
            # Metadata
            "llm_provider": self.llm_provider.get_provider_name(),
            "verifier": self.verifier.get_verifier_name(),
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
        }

        return result

    def evaluate_batch_with_correction(
        self,
        questions: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        max_correction_attempts: int = 1,
        apply_correction_only_if_wrong: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of questions with self-correction.

        Args:
            questions: List of question dictionaries
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation
            max_correction_attempts: Maximum number of correction attempts
            apply_correction_only_if_wrong: Only attempt correction if initial answer is wrong

        Returns:
            Dictionary containing batch evaluation results and summary
        """
        results = []
        start_time = datetime.now()

        for question_data in questions:
            result = self.evaluate_with_correction(
                question_data=question_data,
                max_tokens=max_tokens,
                temperature=temperature,
                max_correction_attempts=max_correction_attempts,
                apply_correction_only_if_wrong=apply_correction_only_if_wrong,
            )
            results.append(result)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Compute summary statistics
        total_questions = len(results)

        # Initial performance
        initial_truthful = sum(
            1 for r in results
            if r.get("initial_verification", {}).get("is_truthful") is True
        )
        initial_accuracy = initial_truthful / total_questions if total_questions > 0 else 0.0

        # Final performance
        final_truthful = sum(
            1 for r in results
            if r.get("final_verification", {}).get("is_truthful") is True
        )
        final_accuracy = final_truthful / total_questions if total_questions > 0 else 0.0

        # Correction statistics
        corrections_attempted = sum(1 for r in results if r.get("correction_applied", False))
        corrections_successful = sum(1 for r in results if r.get("correction_successful", False))
        correction_success_rate = (
            corrections_successful / corrections_attempted
            if corrections_attempted > 0 else 0.0
        )

        # Calculate average confidence
        initial_confidences = [
            r["initial_verification"]["confidence"]
            for r in results
            if r.get("initial_verification") and "confidence" in r["initial_verification"]
        ]
        final_confidences = [
            r["final_verification"]["confidence"]
            for r in results
            if r.get("final_verification") and "confidence" in r["final_verification"]
        ]

        summary = {
            "total_questions": total_questions,
            # Initial performance
            "initial_truthful_count": initial_truthful,
            "initial_accuracy": initial_accuracy,
            "initial_avg_confidence": sum(initial_confidences) / len(initial_confidences) if initial_confidences else 0.0,
            # Final performance
            "final_truthful_count": final_truthful,
            "final_accuracy": final_accuracy,
            "final_avg_confidence": sum(final_confidences) / len(final_confidences) if final_confidences else 0.0,
            # Correction statistics
            "corrections_attempted": corrections_attempted,
            "corrections_successful": corrections_successful,
            "correction_success_rate": correction_success_rate,
            "improvement": final_accuracy - initial_accuracy,
            "improvement_percentage": ((final_accuracy - initial_accuracy) / initial_accuracy * 100) if initial_accuracy > 0 else 0.0,
            # Metadata
            "max_correction_attempts": max_correction_attempts,
            "initial_strategy": self.initial_strategy.get_strategy_name(),
            "correction_strategy": self.correction_strategy.get_strategy_name(),
            "llm_provider": self.llm_provider.get_provider_name(),
            "verifier": self.verifier.get_verifier_name(),
            "timestamp": start_time.isoformat(),
            "total_duration_seconds": total_duration,
            "average_duration_per_question": total_duration / total_questions if total_questions > 0 else 0.0,
        }

        return {
            "results": results,
            "summary": summary,
        }


def compare_strategies(
    llm_provider: LLMProvider,
    verifier: Verifier,
    questions: List[Dict[str, Any]],
    strategies_to_test: List[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare different prompt strategies on the same set of questions.

    Args:
        llm_provider: The LLM provider to use
        verifier: The verifier to use
        questions: List of questions to evaluate
        strategies_to_test: List of strategy names to compare
        max_tokens: Optional max tokens
        temperature: Optional temperature

    Returns:
        Comparison results across all strategies
    """
    start_time = datetime.now()
    strategy_results = {}

    for strategy_name in strategies_to_test:
        print(f"\nEvaluating with strategy: {strategy_name}")

        strategy = PromptStrategyFactory.create(strategy_name)
        evaluator = Evaluator(llm_provider, verifier)
        evaluator._create_prompt = lambda q: strategy.create_prompt(q)

        results = []
        for question_data in questions:
            result = evaluator.evaluate_single(
                question_data=question_data,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            results.append(result)

        # Calculate accuracy
        truthful_count = sum(
            1 for r in results
            if r.get("verification", {}).get("is_truthful") is True
        )
        accuracy = truthful_count / len(results) if results else 0.0

        strategy_results[strategy_name] = {
            "strategy": strategy.get_strategy_name(),
            "results": results,
            "accuracy": accuracy,
            "truthful_count": truthful_count,
            "total_questions": len(results),
        }

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    return {
        "strategy_results": strategy_results,
        "comparison_summary": {
            strategy: {
                "accuracy": data["accuracy"],
                "truthful_count": data["truthful_count"],
            }
            for strategy, data in strategy_results.items()
        },
        "total_duration_seconds": total_duration,
        "timestamp": start_time.isoformat(),
    }
