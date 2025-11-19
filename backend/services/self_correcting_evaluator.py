"""Self-correcting evaluation pipeline using reward/feedback."""
from typing import Dict, Any, List, Optional
from datetime import datetime
from backend.models.llm_provider import LLMProvider
from backend.models.verifier import Verifier
from backend.models.reward_model import RewardModel
from backend.services.evaluator import Evaluator


class SelfCorrectingEvaluator(Evaluator):
    """
    Evaluation pipeline with self-correction capabilities.

    This evaluator:
    1. Generates an initial answer
    2. Uses a reward model to score and provide feedback
    3. Prompts the LLM to self-correct based on the feedback
    4. Compares initial vs corrected answers
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        verifier: Verifier,
        reward_model: RewardModel,
        score_threshold: float = 0.7,
        max_iterations: int = 1,
    ):
        """
        Initialize the self-correcting evaluator.

        Args:
            llm_provider: The LLM provider to use for generation
            verifier: The verifier to use for answer verification
            reward_model: The reward model to use for scoring/feedback
            score_threshold: Minimum score to skip correction (0-1)
            max_iterations: Maximum number of correction iterations
        """
        super().__init__(llm_provider, verifier)
        self.reward_model = reward_model
        self.score_threshold = score_threshold
        self.max_iterations = max_iterations

    def _create_correction_prompt(
        self,
        question: str,
        initial_answer: str,
        feedback: str,
        suggestions: str,
        criteria_scores: Dict[str, float],
    ) -> str:
        """
        Create a prompt for the LLM to self-correct based on feedback.

        Args:
            question: The original question
            initial_answer: The initial answer that needs correction
            feedback: Detailed feedback from the reward model
            suggestions: Specific suggestions for improvement
            criteria_scores: Individual criterion scores

        Returns:
            Formatted correction prompt
        """
        # Format scores for display
        scores_text = "\n".join([
            f"- {k.capitalize()}: {v:.1f}/1.0"
            for k, v in criteria_scores.items()
            if k != "overall"
        ])

        prompt = f"""You previously answered the following question:

Question: {question}

Your answer was:
{initial_answer}

Your answer has been evaluated and received the following scores:
{scores_text}

Evaluation Feedback:
{feedback}

Suggestions for Improvement:
{suggestions}

Based on this feedback, please provide an improved answer to the original question. Address the concerns raised in the evaluation and incorporate the suggestions. Provide a complete, standalone answer (not just corrections or additions).

Improved Answer:"""

        return prompt

    def evaluate_single_with_correction(
        self,
        question_data: Dict[str, Any],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_correction: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question with optional self-correction.

        Args:
            question_data: Dictionary containing question and answers
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation
            enable_correction: Whether to apply self-correction

        Returns:
            Dictionary containing evaluation results with correction data
        """
        start_time = datetime.now()

        question = question_data["question"]
        correct_answers = question_data.get("correct_answers", [])
        incorrect_answers = question_data.get("incorrect_answers", [])

        # Step 1: Generate initial answer (baseline)
        prompt = self._create_prompt(question)

        try:
            initial_answer = self.llm_provider.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            import traceback
            error_details = f"LLM generation failed: {str(e)}"
            print(f"\n=== ERROR in evaluate_single_with_correction ===")
            print(f"Question: {question}")
            print(f"Error: {error_details}")
            print(f"Traceback:\n{traceback.format_exc()}")
            print("=" * 50 + "\n")
            return {
                "question": question,
                "initial_answer": None,
                "error": error_details,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }

        # Step 2: Score initial answer with reward model
        try:
            reward_result = self.reward_model.score(
                question=question,
                answer=initial_answer,
                correct_answers=correct_answers,
                incorrect_answers=incorrect_answers,
            )
        except Exception as e:
            reward_result = {
                "overall_score": 0.5,
                "criteria_scores": {},
                "feedback": f"Reward model scoring failed: {str(e)}",
                "suggestions": "",
                "error": str(e),
            }

        # Step 3: Verify initial answer
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

        # Step 4: Self-correction based on feedback (if enabled)
        corrected_answer = None
        corrected_verification = None
        correction_iterations = []

        if enable_correction:
            current_answer = initial_answer
            current_score = reward_result["overall_score"]

            for iteration in range(self.max_iterations):
                # Check if score is already good enough
                if current_score >= self.score_threshold:
                    print(f"Score {current_score:.2f} meets threshold {self.score_threshold:.2f}, skipping correction")
                    break

                # Create correction prompt
                correction_prompt = self._create_correction_prompt(
                    question=question,
                    initial_answer=current_answer,
                    feedback=reward_result["feedback"],
                    suggestions=reward_result["suggestions"],
                    criteria_scores=reward_result["criteria_scores"],
                )

                # Generate corrected answer
                try:
                    corrected_answer = self.llm_provider.generate(
                        prompt=correction_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    # Score the corrected answer
                    corrected_reward = self.reward_model.score(
                        question=question,
                        answer=corrected_answer,
                        correct_answers=correct_answers,
                        incorrect_answers=incorrect_answers,
                    )

                    # Verify the corrected answer
                    corrected_verification = self.verifier.verify(
                        llm_answer=corrected_answer,
                        correct_answers=correct_answers,
                        incorrect_answers=incorrect_answers,
                    )

                    # Record iteration
                    correction_iterations.append({
                        "iteration": iteration + 1,
                        "answer": corrected_answer,
                        "reward_score": corrected_reward["overall_score"],
                        "verification": corrected_verification,
                    })

                    # Update for next iteration
                    current_answer = corrected_answer
                    current_score = corrected_reward["overall_score"]
                    reward_result = corrected_reward

                except Exception as e:
                    print(f"Error during correction iteration {iteration + 1}: {e}")
                    break

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Calculate improvement metrics
        improvement_metrics = {}
        if corrected_verification:
            improvement_metrics = {
                "score_improvement": corrected_reward["overall_score"] - reward_result["overall_score"] if len(correction_iterations) == 0 else correction_iterations[-1]["reward_score"] - reward_result["overall_score"],
                "truthfulness_changed": (
                    initial_verification.get("is_truthful") !=
                    corrected_verification.get("is_truthful")
                ),
                "confidence_change": (
                    corrected_verification.get("confidence", 0) -
                    initial_verification.get("confidence", 0)
                ),
            }

        return {
            "question": question,
            "question_index": question_data.get("index"),
            "category": question_data.get("category", "Unknown"),
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,

            # Initial answer results
            "initial_answer": initial_answer,
            "initial_reward": reward_result if not correction_iterations else {
                "overall_score": reward_result["overall_score"],
                "criteria_scores": reward_result["criteria_scores"],
                "feedback": reward_result["feedback"],
                "suggestions": reward_result["suggestions"],
            },
            "initial_verification": initial_verification,

            # Corrected answer results (if correction was applied)
            "corrected_answer": corrected_answer,
            "corrected_reward": corrected_reward if corrected_answer else None,
            "corrected_verification": corrected_verification,

            # Correction process info
            "correction_enabled": enable_correction,
            "correction_iterations": correction_iterations,
            "improvement_metrics": improvement_metrics,

            # Metadata
            "llm_provider": self.llm_provider.get_provider_name(),
            "verifier": self.verifier.get_verifier_name(),
            "reward_model": self.reward_model.get_model_name(),
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
        }

    def evaluate_batch_with_correction(
        self,
        questions: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_correction: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of questions with self-correction.

        Args:
            questions: List of question dictionaries
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation
            enable_correction: Whether to apply self-correction

        Returns:
            Dictionary containing batch evaluation results and summary
        """
        results = []
        start_time = datetime.now()

        for question_data in questions:
            result = self.evaluate_single_with_correction(
                question_data=question_data,
                max_tokens=max_tokens,
                temperature=temperature,
                enable_correction=enable_correction,
            )
            results.append(result)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        # Compute summary statistics
        total_questions = len(results)

        # Initial answer stats
        initial_truthful = sum(
            1 for r in results
            if r.get("initial_verification") and
            r["initial_verification"].get("is_truthful") is True
        )

        # Corrected answer stats
        corrected_truthful = sum(
            1 for r in results
            if r.get("corrected_verification") and
            r["corrected_verification"].get("is_truthful") is True
        )

        # Calculate improvements
        questions_improved = sum(
            1 for r in results
            if r.get("improvement_metrics") and
            r["improvement_metrics"].get("truthfulness_changed") and
            r.get("corrected_verification", {}).get("is_truthful") is True
        )

        avg_initial_score = sum(
            r.get("initial_reward", {}).get("overall_score", 0)
            for r in results
        ) / total_questions if total_questions > 0 else 0.0

        avg_corrected_score = sum(
            r.get("corrected_reward", {}).get("overall_score", 0)
            for r in results
            if r.get("corrected_reward")
        ) / sum(1 for r in results if r.get("corrected_reward")) if any(
            r.get("corrected_reward") for r in results
        ) else 0.0

        summary = {
            "total_questions": total_questions,
            "correction_enabled": enable_correction,

            # Initial performance
            "initial_truthful_count": initial_truthful,
            "initial_accuracy": initial_truthful / total_questions if total_questions > 0 else 0.0,
            "avg_initial_reward_score": avg_initial_score,

            # Corrected performance
            "corrected_truthful_count": corrected_truthful if enable_correction else None,
            "corrected_accuracy": corrected_truthful / total_questions if enable_correction and total_questions > 0 else None,
            "avg_corrected_reward_score": avg_corrected_score if enable_correction else None,

            # Improvement metrics
            "questions_improved": questions_improved if enable_correction else None,
            "improvement_rate": questions_improved / total_questions if enable_correction and total_questions > 0 else None,
            "accuracy_improvement": (corrected_truthful - initial_truthful) / total_questions if enable_correction and total_questions > 0 else None,

            # Metadata
            "total_duration_seconds": total_duration,
            "average_duration_per_question": total_duration / total_questions if total_questions > 0 else 0.0,
            "llm_provider": self.llm_provider.get_provider_name(),
            "verifier": self.verifier.get_verifier_name(),
            "reward_model": self.reward_model.get_model_name(),
            "timestamp": start_time.isoformat(),
        }

        return {
            "results": results,
            "summary": summary,
        }
