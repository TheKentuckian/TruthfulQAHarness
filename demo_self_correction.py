#!/usr/bin/env python
"""
Demonstration of Reward/Feedback Self-Correction Technique

This script demonstrates how an external reward model can score LLM-generated
answers and provide feedback for self-correction, improving answer quality.

Self-Correction Flow:
1. Generate initial answer to a TruthfulQA question
2. Reward model scores the answer on multiple criteria (truthfulness, coherence, etc.)
3. Reward model provides detailed feedback and suggestions
4. LLM receives feedback and generates a corrected answer
5. Compare initial vs corrected answers
"""

import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from backend.models.llm_provider import LLMProviderFactory
from backend.models.verifier import VerifierFactory
from backend.models.reward_model import RewardModelFactory
from backend.services.dataset_loader import TruthfulQALoader
from backend.services.self_correcting_evaluator import SelfCorrectingEvaluator


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def format_score(score: float) -> str:
    """Format a score as a percentage with color."""
    percentage = score * 100
    if score >= 0.7:
        return f"{percentage:.1f}% ✓"
    elif score >= 0.5:
        return f"{percentage:.1f}% ~"
    else:
        return f"{percentage:.1f}% ✗"


def main():
    print_section("TruthfulQA Self-Correction Demonstration")
    print("This demo shows how reward/feedback enables LLM self-correction.")
    print("\nTechnique: Reward/Feedback")
    print("Description: Generated text is scored by an external reward model")
    print("             based on criteria like truthfulness, coherence, safety, etc.")
    print("             Feedback is provided to the LLM to self-correct.\n")

    # Initialize components
    print("Initializing components...")

    try:
        # Load dataset
        dataset_loader = TruthfulQALoader()
        print("✓ Dataset loaded")

        # Create LLM provider
        llm_provider = LLMProviderFactory.create("claude")
        print("✓ LLM provider created (Claude)")

        # Create verifier
        verifier = VerifierFactory.create("simple_text")  # Use simple text for speed
        print("✓ Verifier created (Simple Text)")

        # Create reward model
        reward_model = RewardModelFactory.create("llm_reward", llm_provider=llm_provider)
        print("✓ Reward model created (LLM-based)")

        # Create self-correcting evaluator
        evaluator = SelfCorrectingEvaluator(
            llm_provider=llm_provider,
            verifier=verifier,
            reward_model=reward_model,
            score_threshold=0.7,  # Only correct if score < 0.7
            max_iterations=1,
        )
        print("✓ Self-correcting evaluator created")

    except Exception as e:
        print(f"\n✗ Error initializing components: {e}")
        print("\nMake sure:")
        print("  1. Your .env file contains a valid ANTHROPIC_API_KEY")
        print("  2. You have an active internet connection")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")
        return

    # Get a sample question
    print_section("Loading Sample Question")

    try:
        # Get a specific interesting question (or random if you prefer)
        questions = dataset_loader.get_sample(sample_size=1, seed=42)
        question_data = questions[0]

        print(f"Question: {question_data['question']}")
        print(f"Category: {question_data.get('category', 'Unknown')}")
        print(f"\nCorrect answers ({len(question_data['correct_answers'])}):")
        for i, ans in enumerate(question_data['correct_answers'][:3], 1):
            print(f"  {i}. {ans}")
        if len(question_data['correct_answers']) > 3:
            print(f"  ... and {len(question_data['correct_answers']) - 3} more")

    except Exception as e:
        print(f"\n✗ Error loading question: {e}")
        return

    # Run evaluation WITHOUT self-correction (baseline)
    print_section("Step 1: Baseline Evaluation (No Self-Correction)")
    print("Generating initial answer without any feedback or correction...")

    try:
        baseline_result = evaluator.evaluate_single_with_correction(
            question_data=question_data,
            max_tokens=512,
            temperature=1.0,
            enable_correction=False,  # Disable correction for baseline
        )

        print_subsection("Initial Answer")
        print(baseline_result['initial_answer'])

        print_subsection("Reward Model Scoring")
        initial_reward = baseline_result['initial_reward']
        print(f"\nOverall Score: {format_score(initial_reward['overall_score'])}")
        print("\nCriteria Scores:")
        for criterion, score in initial_reward['criteria_scores'].items():
            if criterion != 'overall':
                print(f"  • {criterion.capitalize():15} {format_score(score)}")

        print(f"\nFeedback:\n{initial_reward['feedback']}")
        print(f"\nSuggestions:\n{initial_reward['suggestions']}")

        print_subsection("Truthfulness Verification")
        initial_verification = baseline_result['initial_verification']
        truthful_status = "✓ TRUTHFUL" if initial_verification['is_truthful'] else "✗ UNTRUTHFUL"
        print(f"Assessment: {truthful_status}")
        print(f"Confidence: {initial_verification['confidence']:.2%}")
        print(f"Reasoning: {initial_verification['reasoning']}")

    except Exception as e:
        print(f"\n✗ Error during baseline evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Run evaluation WITH self-correction
    print_section("Step 2: Self-Correction Evaluation")
    print("Now applying reward/feedback self-correction...")
    print("The LLM will receive the scores, feedback, and suggestions,")
    print("then generate an improved answer.\n")

    try:
        corrected_result = evaluator.evaluate_single_with_correction(
            question_data=question_data,
            max_tokens=512,
            temperature=1.0,
            enable_correction=True,  # Enable correction
        )

        if corrected_result.get('corrected_answer'):
            print_subsection("Corrected Answer")
            print(corrected_result['corrected_answer'])

            print_subsection("Reward Model Scoring (After Correction)")
            corrected_reward = corrected_result['corrected_reward']
            print(f"\nOverall Score: {format_score(corrected_reward['overall_score'])}")
            print("\nCriteria Scores:")
            for criterion, score in corrected_reward['criteria_scores'].items():
                if criterion != 'overall':
                    print(f"  • {criterion.capitalize():15} {format_score(score)}")

            print(f"\nFeedback:\n{corrected_reward['feedback']}")

            print_subsection("Truthfulness Verification (After Correction)")
            corrected_verification = corrected_result['corrected_verification']
            truthful_status = "✓ TRUTHFUL" if corrected_verification['is_truthful'] else "✗ UNTRUTHFUL"
            print(f"Assessment: {truthful_status}")
            print(f"Confidence: {corrected_verification['confidence']:.2%}")
            print(f"Reasoning: {corrected_verification['reasoning']}")

        else:
            print("No correction was applied (score already met threshold)")

    except Exception as e:
        print(f"\n✗ Error during self-correction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare results
    print_section("Step 3: Comparison & Analysis")

    if corrected_result.get('improvement_metrics'):
        metrics = corrected_result['improvement_metrics']

        print_subsection("Score Improvements")
        print(f"Overall Score Change: {metrics.get('score_improvement', 0):+.2%}")

        initial_score = baseline_result['initial_reward']['overall_score']
        corrected_score = corrected_reward['overall_score']
        print(f"  Initial:   {format_score(initial_score)}")
        print(f"  Corrected: {format_score(corrected_score)}")

        print_subsection("Truthfulness Improvements")
        initial_truthful = baseline_result['initial_verification']['is_truthful']
        corrected_truthful = corrected_result['corrected_verification']['is_truthful']

        print(f"  Initial:   {'✓ TRUTHFUL' if initial_truthful else '✗ UNTRUTHFUL'}")
        print(f"  Corrected: {'✓ TRUTHFUL' if corrected_truthful else '✗ UNTRUTHFUL'}")

        if metrics['truthfulness_changed']:
            if corrected_truthful:
                print("\n  🎉 Self-correction IMPROVED truthfulness!")
            else:
                print("\n  ⚠️  Self-correction changed assessment (worse)")
        else:
            print("\n  → Truthfulness assessment unchanged")

        print_subsection("Confidence Improvements")
        confidence_change = metrics.get('confidence_change', 0)
        print(f"Confidence Change: {confidence_change:+.2%}")
        print(f"  Initial:   {baseline_result['initial_verification']['confidence']:.2%}")
        print(f"  Corrected: {corrected_result['corrected_verification']['confidence']:.2%}")

    print_section("Summary")
    print("Self-correction technique demonstrated successfully!")
    print("\nKey Insights:")
    print("  • Reward model provides multi-dimensional scoring (truthfulness, coherence, etc.)")
    print("  • Detailed feedback highlights specific strengths and weaknesses")
    print("  • Actionable suggestions guide the LLM toward improvement")
    print("  • Self-correction can improve both quality scores and truthfulness")
    print("\nThis technique is valuable for:")
    print("  - Improving answer quality without human intervention")
    print("  - Identifying specific areas of weakness in LLM responses")
    print("  - Iteratively refining answers based on objective criteria")
    print("  - Research on self-correction mechanisms in LLMs")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
