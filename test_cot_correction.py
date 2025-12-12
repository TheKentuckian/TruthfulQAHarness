#!/usr/bin/env python3
"""
Unit tests for Chain of Thought Self-Correction implementation.

These tests verify the structure and logic without making actual API calls.
"""
import sys
from typing import Dict, Any, Optional, List


# Mock LLM Provider for testing
class MockLLMProvider:
    """Mock LLM provider that returns predetermined responses."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Return a mock response based on the prompt."""
        self.call_count += 1

        # Check for correction prompts
        if "previous answer" in prompt.lower() or "reconsider" in prompt.lower():
            return self.responses.get("correction", "This is a corrected answer after careful reconsideration.")

        # Check for CoT prompts
        if "step by step" in prompt.lower() or "think through" in prompt.lower():
            return self.responses.get("cot", "Let me think step by step: This is a reasoned answer.")

        # Default direct response
        return self.responses.get("direct", "This is a direct answer.")

    def get_provider_name(self) -> str:
        return "Mock LLM Provider"


# Mock Verifier for testing
class MockVerifier:
    """Mock verifier that returns predetermined results."""

    def __init__(self, truthfulness_pattern: Optional[List[bool]] = None):
        """
        Initialize mock verifier.

        Args:
            truthfulness_pattern: List of truthfulness values to cycle through
        """
        self.truthfulness_pattern = truthfulness_pattern or [False, True]  # First wrong, then correct
        self.call_count = 0

    def verify(
        self,
        llm_answer: str,
        correct_answers: List[str],
        incorrect_answers: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Return a mock verification result."""
        # Cycle through truthfulness pattern
        is_truthful = self.truthfulness_pattern[self.call_count % len(self.truthfulness_pattern)]
        confidence = 0.8 if is_truthful else 0.3

        self.call_count += 1

        return {
            "is_truthful": is_truthful,
            "confidence": confidence,
            "reasoning": f"Mock verification: {'truthful' if is_truthful else 'not truthful'}",
            "metrics": {
                "mock": True,
            }
        }

    def get_verifier_name(self) -> str:
        return "Mock Verifier"


def test_prompt_strategies():
    """Test that prompt strategies are correctly defined."""
    print("\n=== Testing Prompt Strategies ===")

    from backend.services.prompt_strategies import (
        DirectPromptStrategy,
        ChainOfThoughtStrategy,
        SelfCorrectionStrategy,
        ReflectivePromptStrategy,
        PromptStrategyFactory,
    )

    question = "What is the capital of France?"

    # Test Direct Strategy
    direct = DirectPromptStrategy()
    direct_prompt = direct.create_prompt(question)
    assert "Q:" in direct_prompt, "Direct prompt should contain 'Q:'"
    assert "A:" in direct_prompt, "Direct prompt should contain 'A:'"
    print(f"[OK] Direct Strategy: {direct.get_strategy_name()}")

    # Test CoT Strategy
    cot = ChainOfThoughtStrategy()
    cot_prompt = cot.create_prompt(question)
    assert "step by step" in cot_prompt.lower(), "CoT prompt should mention step by step"
    print(f"[OK] Chain of Thought Strategy: {cot.get_strategy_name()}")

    # Test Self-Correction Strategy
    correction = SelfCorrectionStrategy()
    context = {"previous_answer": "London"}
    correction_prompt = correction.create_prompt(question, context)
    assert "previous answer" in correction_prompt.lower(), "Should reference previous answer"
    assert "London" in correction_prompt, "Should include the previous answer"
    print(f"[OK] Self-Correction Strategy: {correction.get_strategy_name()}")

    # Test Reflective Strategy
    reflective = ReflectivePromptStrategy()
    reflective_prompt = reflective.create_prompt(question)
    assert "verify" in reflective_prompt.lower() or "verification" in reflective_prompt.lower()
    print(f"[OK] Reflective Strategy: {reflective.get_strategy_name()}")

    # Test Factory
    strategies = PromptStrategyFactory.get_available_strategies()
    assert len(strategies) > 0, "Factory should return available strategies"
    print(f"[OK] Strategy Factory: {len(strategies)} strategies available")

    print("\nAll prompt strategy tests passed! [OK]")
    return True


def test_self_correction_evaluator():
    """Test the self-correction evaluator logic."""
    print("\n=== Testing Self-Correction Evaluator ===")

    from backend.services.self_correction_evaluator import SelfCorrectionEvaluator
    from backend.services.prompt_strategies import DirectPromptStrategy, SelfCorrectionStrategy

    # Create mock components
    mock_llm = MockLLMProvider({
        "direct": "Monaco is the smallest country.",
        "correction": "Actually, after reconsidering, Vatican City is smaller than Monaco.",
    })

    # Pattern: first answer is wrong, correction is right
    mock_verifier = MockVerifier(truthfulness_pattern=[False, True])

    # Create evaluator
    evaluator = SelfCorrectionEvaluator(
        llm_provider=mock_llm,
        verifier=mock_verifier,
        initial_strategy=DirectPromptStrategy(),
        correction_strategy=SelfCorrectionStrategy(),
    )

    # Test question
    question_data = {
        "question": "What is the smallest country in the world?",
        "index": 0,
        "category": "Geography",
        "correct_answers": ["Vatican City"],
        "incorrect_answers": ["Monaco", "San Marino"],
    }

    # Evaluate with correction
    result = evaluator.evaluate_with_correction(
        question_data=question_data,
        max_correction_attempts=1,
        apply_correction_only_if_wrong=True,
    )

    # Verify structure
    assert "initial_answer" in result, "Should have initial answer"
    assert "final_answer" in result, "Should have final answer"
    assert "correction_applied" in result, "Should track if correction was applied"
    assert "correction_history" in result, "Should have correction history"
    print("[OK] Result structure is correct")

    # Verify correction was applied (since initial was wrong)
    assert result["correction_applied"] == True, "Correction should be applied when initial is wrong"
    print("[OK] Correction was applied for wrong answer")

    # Verify correction history
    assert len(result["correction_history"]) == 2, "Should have 2 attempts (initial + 1 correction)"
    print(f"[OK] Correction history has {len(result['correction_history'])} attempts")

    # Verify improvement tracking
    initial_truthful = result["initial_verification"]["is_truthful"]
    final_truthful = result["final_verification"]["is_truthful"]
    assert initial_truthful == False, "Initial should be wrong (per mock)"
    assert final_truthful == True, "Final should be correct (per mock)"
    assert result["correction_successful"] == True, "Should be marked as successful correction"
    print("[OK] Correction successfully improved the answer")

    print("\nAll self-correction evaluator tests passed! [OK]")
    return True


def test_batch_evaluation():
    """Test batch evaluation with self-correction."""
    print("\n=== Testing Batch Evaluation ===")

    from backend.services.self_correction_evaluator import SelfCorrectionEvaluator
    from backend.services.prompt_strategies import DirectPromptStrategy, SelfCorrectionStrategy

    # Create mocks
    mock_llm = MockLLMProvider()
    # Pattern: alternating wrong/right, so some corrections succeed
    mock_verifier = MockVerifier(truthfulness_pattern=[False, True, True, False])

    # Create evaluator
    evaluator = SelfCorrectionEvaluator(
        llm_provider=mock_llm,
        verifier=mock_verifier,
        initial_strategy=DirectPromptStrategy(),
        correction_strategy=SelfCorrectionStrategy(),
    )

    # Test questions
    questions = [
        {
            "question": "Question 1?",
            "index": 0,
            "category": "Test",
            "correct_answers": ["Answer 1"],
            "incorrect_answers": ["Wrong 1"],
        },
        {
            "question": "Question 2?",
            "index": 1,
            "category": "Test",
            "correct_answers": ["Answer 2"],
            "incorrect_answers": ["Wrong 2"],
        },
        {
            "question": "Question 3?",
            "index": 2,
            "category": "Test",
            "correct_answers": ["Answer 3"],
            "incorrect_answers": ["Wrong 3"],
        },
    ]

    # Evaluate batch
    results = evaluator.evaluate_batch_with_correction(
        questions=questions,
        max_correction_attempts=1,
        apply_correction_only_if_wrong=True,
    )

    # Verify structure
    assert "results" in results, "Should have results"
    assert "summary" in results, "Should have summary"
    assert len(results["results"]) == 3, "Should have 3 results"
    print("[OK] Batch results structure is correct")

    # Verify summary metrics
    summary = results["summary"]
    required_metrics = [
        "total_questions",
        "initial_accuracy",
        "final_accuracy",
        "corrections_attempted",
        "corrections_successful",
        "improvement",
    ]

    for metric in required_metrics:
        assert metric in summary, f"Summary should include {metric}"
    print(f"[OK] Summary includes all {len(required_metrics)} required metrics")

    # Verify improvement tracking
    assert summary["total_questions"] == 3, "Should have 3 questions"
    print(f"[OK] Processed {summary['total_questions']} questions")

    # With our mock pattern [False, True, True, False]:
    # Q1: initial=False (idx 0), correction=True (idx 1) -> correction successful
    # Q2: initial=True (idx 2), no correction needed
    # Q3: initial=False (idx 3), correction=True (idx 4... wraps to 0) -> correction successful
    # So we expect 2 corrections attempted and likely 2 successful

    print(f"[OK] Initial accuracy: {summary['initial_accuracy']:.1%}")
    print(f"[OK] Final accuracy: {summary['final_accuracy']:.1%}")
    print(f"[OK] Improvement: {summary['improvement']:+.1%}")
    print(f"[OK] Corrections attempted: {summary['corrections_attempted']}")
    print(f"[OK] Corrections successful: {summary['corrections_successful']}")

    print("\nAll batch evaluation tests passed! [OK]")
    return True


def test_strategy_comparison():
    """Test strategy comparison functionality."""
    print("\n=== Testing Strategy Comparison ===")

    from backend.services.self_correction_evaluator import compare_strategies
    from backend.services.evaluator import Evaluator

    # Create mocks
    mock_llm = MockLLMProvider()
    mock_verifier = MockVerifier(truthfulness_pattern=[True, False, True])

    # Test questions
    questions = [
        {
            "question": "Question 1?",
            "index": 0,
            "category": "Test",
            "correct_answers": ["Answer 1"],
            "incorrect_answers": ["Wrong 1"],
        },
        {
            "question": "Question 2?",
            "index": 1,
            "category": "Test",
            "correct_answers": ["Answer 2"],
            "incorrect_answers": ["Wrong 2"],
        },
    ]

    # Compare strategies
    results = compare_strategies(
        llm_provider=mock_llm,
        verifier=mock_verifier,
        questions=questions,
        strategies_to_test=["direct", "chain_of_thought"],
    )

    # Verify structure
    assert "strategy_results" in results, "Should have strategy results"
    assert "comparison_summary" in results, "Should have comparison summary"
    print("[OK] Comparison results structure is correct")

    # Verify each strategy was tested
    assert "direct" in results["strategy_results"], "Should have direct results"
    assert "chain_of_thought" in results["strategy_results"], "Should have CoT results"
    print("[OK] Both strategies were tested")

    # Verify metrics
    for strategy, data in results["strategy_results"].items():
        assert "accuracy" in data, f"{strategy} should have accuracy"
        assert "results" in data, f"{strategy} should have results"
        print(f"[OK] {strategy}: {data['accuracy']:.1%} accuracy")

    print("\nAll strategy comparison tests passed! [OK]")
    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("Chain of Thought Self-Correction - Unit Tests")
    print("=" * 80)

    all_passed = True

    try:
        # Test each component
        all_passed &= test_prompt_strategies()
        all_passed &= test_self_correction_evaluator()
        all_passed &= test_batch_evaluation()
        all_passed &= test_strategy_comparison()

        if all_passed:
            print("\n" + "=" * 80)
            print("ALL TESTS PASSED! [OK]")
            print("=" * 80)
            print("\nThe Chain of Thought Self-Correction implementation is working correctly.")
            print("\nTo run a live demo with actual LLM API calls:")
            print("  1. Set up your .env file with ANTHROPIC_API_KEY")
            print("  2. Run: python demo_cot_correction.py --mode correction --questions 5")
            print("\nFor more information, see COT_SELF_CORRECTION.md")
            return 0
        else:
            print("\n" + "=" * 80)
            print("SOME TESTS FAILED [FAILED]")
            print("=" * 80)
            return 1

    except Exception as e:
        print(f"\n\nTest failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
