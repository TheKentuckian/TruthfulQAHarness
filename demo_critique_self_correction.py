#!/usr/bin/env python3
"""
Demonstration of Critique-based Self-Correction for TruthfulQA.

This script demonstrates how critiquing can be used as a self-correction
technique to improve the truthfulness of LLM answers.

The technique works in three steps:
1. Generate an initial answer to a question
2. Ask the LLM to critique its own answer
3. Generate a revised answer based on the critique

We then compare the baseline answer (without correction) against the
corrected answer using the TruthfulQA verifier.
"""

import argparse
import sys
from typing import Dict, Any
from backend.models.llm_provider import LLMProviderFactory
from backend.models.verifier import VerifierFactory
from backend.models.self_corrector import CritiqueSelfCorrector
from backend.services.dataset_loader import TruthfulQALoader
from backend.config import settings


def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


def print_subsection(title: str, width: int = 80):
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width + "\n")


def format_verification_result(result: Dict[str, Any]) -> str:
    """Format verification result for display."""
    is_truthful = result.get("is_truthful")
    confidence = result.get("confidence", 0.0)

    if is_truthful is None:
        status = "UNKNOWN"
    elif is_truthful:
        status = "TRUTHFUL ✓"
    else:
        status = "UNTRUTHFUL ✗"

    return f"{status} (confidence: {confidence:.2f})"


def demonstrate_critique_correction(
    question_data: Dict[str, Any],
    llm_provider,
    verifier,
    max_tokens: int = 1024,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Demonstrate critique-based self-correction on a single question.

    Args:
        question_data: Question data from TruthfulQA
        llm_provider: LLM provider instance
        verifier: Verifier instance
        max_tokens: Max tokens for generation
        temperature: Temperature for generation

    Returns:
        Dictionary containing all results
    """
    question = question_data["question"]
    correct_answers = question_data.get("correct_answers", [])
    incorrect_answers = question_data.get("incorrect_answers", [])

    print_section("DEMONSTRATION: Critique-based Self-Correction")

    # Display the question
    print_subsection("Question")
    print(f"Category: {question_data.get('category', 'Unknown')}")
    print(f"Question: {question}")
    print(f"\nExpected correct answers ({len(correct_answers)}):")
    for i, ans in enumerate(correct_answers[:3], 1):  # Show first 3
        print(f"  {i}. {ans}")
    if len(correct_answers) > 3:
        print(f"  ... and {len(correct_answers) - 3} more")

    # Step 1: Generate baseline answer (no correction)
    print_subsection("Step 1: Generate Baseline Answer (No Correction)")
    baseline_prompt = f"Q: {question}\nA:"

    try:
        print("Generating baseline answer...")
        baseline_answer = llm_provider.generate(
            prompt=baseline_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(f"Baseline answer: {baseline_answer}")
    except Exception as e:
        print(f"ERROR: Failed to generate baseline answer: {e}")
        return {"error": str(e)}

    # Step 2: Apply critique-based self-correction
    print_subsection("Step 2: Apply Critique-based Self-Correction")

    corrector = CritiqueSelfCorrector(
        llm_provider=llm_provider,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    print("Generating critique and corrected answer...")
    correction_result = corrector.correct(
        question=question,
        initial_answer=baseline_answer,
    )

    if not correction_result["correction_metadata"]["success"]:
        print(f"ERROR: {correction_result['correction_metadata'].get('error')}")
        return {"error": correction_result["correction_metadata"].get("error")}

    critique = correction_result["correction_metadata"]["critique"]
    corrected_answer = correction_result["corrected_answer"]

    print_subsection("Generated Critique")
    print(critique)

    print_subsection("Corrected Answer")
    print(corrected_answer)

    # Step 3: Verify both answers
    print_subsection("Step 3: Verification Results")

    print("Verifying baseline answer...")
    baseline_verification = verifier.verify(
        llm_answer=baseline_answer,
        correct_answers=correct_answers,
        incorrect_answers=incorrect_answers,
    )

    print("Verifying corrected answer...")
    corrected_verification = verifier.verify(
        llm_answer=corrected_answer,
        correct_answers=correct_answers,
        incorrect_answers=incorrect_answers,
    )

    # Display comparison
    print_subsection("Comparison")
    print(f"Baseline Answer:  {format_verification_result(baseline_verification)}")
    print(f"Corrected Answer: {format_verification_result(corrected_verification)}")

    # Determine improvement
    baseline_truthful = baseline_verification.get("is_truthful", False)
    corrected_truthful = corrected_verification.get("is_truthful", False)

    if not baseline_truthful and corrected_truthful:
        print("\n✓ IMPROVEMENT: Answer became truthful after correction!")
    elif baseline_truthful and not corrected_truthful:
        print("\n✗ REGRESSION: Answer became untruthful after correction")
    elif baseline_truthful and corrected_truthful:
        print("\n→ MAINTAINED: Answer remained truthful")
    else:
        print("\n→ NO CHANGE: Answer remained untruthful")

    # Show confidence change
    baseline_conf = baseline_verification.get("confidence", 0.0)
    corrected_conf = corrected_verification.get("confidence", 0.0)
    conf_change = corrected_conf - baseline_conf

    print(f"Confidence change: {conf_change:+.3f}")

    return {
        "question": question,
        "category": question_data.get("category"),
        "baseline_answer": baseline_answer,
        "baseline_verification": baseline_verification,
        "critique": critique,
        "corrected_answer": corrected_answer,
        "corrected_verification": corrected_verification,
        "improved": not baseline_truthful and corrected_truthful,
        "regressed": baseline_truthful and not corrected_truthful,
    }


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate critique-based self-correction for TruthfulQA"
    )
    parser.add_argument(
        "--question-index",
        type=int,
        help="Specific question index to test (default: random sample)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of questions to test (default: 5)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.default_model,
        help=f"Claude model to use (default: {settings.default_model})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for generation (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation (default: 1.0)",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="simple_text",
        choices=["simple_text", "word_similarity"],
        help="Verifier to use (default: simple_text)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Initialize components
    print("Initializing components...")

    try:
        llm_provider = LLMProviderFactory.create("claude", model=args.model)
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Please ensure ANTHROPIC_API_KEY is set in your .env file")
        sys.exit(1)

    try:
        verifier = VerifierFactory.create(args.verifier)
    except Exception as e:
        print(f"ERROR: Failed to create verifier: {e}")
        sys.exit(1)

    # Load dataset
    print("Loading TruthfulQA dataset...")
    loader = TruthfulQALoader()

    if args.question_index is not None:
        # Test specific question
        question_data = loader.get_question(args.question_index)
        if not question_data:
            print(f"ERROR: Question index {args.question_index} not found")
            sys.exit(1)

        demonstrate_critique_correction(
            question_data=question_data,
            llm_provider=llm_provider,
            verifier=verifier,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        # Test multiple random questions
        questions = loader.get_sample(
            sample_size=args.num_questions,
            seed=args.seed,
        )

        results = []
        for i, question_data in enumerate(questions, 1):
            print(f"\n\n{'#' * 80}")
            print(f"# Question {i} of {args.num_questions}")
            print(f"{'#' * 80}\n")

            result = demonstrate_critique_correction(
                question_data=question_data,
                llm_provider=llm_provider,
                verifier=verifier,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            results.append(result)

        # Summary statistics
        print_section("SUMMARY STATISTICS")

        total = len(results)
        improved = sum(1 for r in results if r.get("improved", False))
        regressed = sum(1 for r in results if r.get("regressed", False))

        baseline_truthful = sum(
            1 for r in results
            if r.get("baseline_verification", {}).get("is_truthful", False)
        )
        corrected_truthful = sum(
            1 for r in results
            if r.get("corrected_verification", {}).get("is_truthful", False)
        )

        print(f"Total questions tested: {total}")
        print(f"\nBaseline truthful: {baseline_truthful}/{total} ({baseline_truthful/total*100:.1f}%)")
        print(f"Corrected truthful: {corrected_truthful}/{total} ({corrected_truthful/total*100:.1f}%)")
        print(f"\nImproved (false → true): {improved}")
        print(f"Regressed (true → false): {regressed}")
        print(f"Net improvement: {improved - regressed}")

        if corrected_truthful > baseline_truthful:
            print(f"\n✓ Overall improvement: +{corrected_truthful - baseline_truthful} more truthful answers")
        elif corrected_truthful < baseline_truthful:
            print(f"\n✗ Overall regression: {baseline_truthful - corrected_truthful} fewer truthful answers")
        else:
            print("\n→ No overall change in truthfulness")


if __name__ == "__main__":
    main()
