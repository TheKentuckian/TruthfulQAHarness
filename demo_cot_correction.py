#!/usr/bin/env python3
"""
Demo script showcasing Chain of Thought (CoT) as a Self-Correction mechanism.

This script demonstrates how chain of thought prompting can help LLMs
self-correct their answers when initial responses are untruthful.

Usage:
    python demo_cot_correction.py [--questions N] [--mode MODE]

Modes:
    - basic: Simple direct vs CoT comparison
    - correction: Full self-correction pipeline
    - comparison: Compare multiple strategies
"""
import argparse
import json
from datetime import datetime
from backend.config import settings
from backend.models.llm_provider import LLMProviderFactory
from backend.models.verifier import VerifierFactory
from backend.services.dataset_loader import TruthfulQALoader
from backend.services.self_correction_evaluator import (
    SelfCorrectionEvaluator,
    compare_strategies,
)
from backend.services.prompt_strategies import PromptStrategyFactory


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def print_section(text):
    """Print a formatted section header."""
    print("\n" + "-" * 80)
    print(f"  {text}")
    print("-" * 80 + "\n")


def print_result(result):
    """Print a single evaluation result."""
    print(f"Question: {result['question']}")
    print(f"Category: {result.get('category', 'Unknown')}")

    if 'initial_answer' in result:
        print(f"\n  Initial Answer ({result.get('initial_strategy', 'Direct')}):")
        print(f"  {result['initial_answer'][:200]}...")
        initial_ver = result.get('initial_verification', {})
        print(f"  Truthful: {initial_ver.get('is_truthful')} (Confidence: {initial_ver.get('confidence', 0):.2f})")

        if result.get('correction_applied'):
            print(f"\n  Final Answer (After Correction):")
            print(f"  {result['final_answer'][:200]}...")
            final_ver = result.get('final_verification', {})
            print(f"  Truthful: {final_ver.get('is_truthful')} (Confidence: {final_ver.get('confidence', 0):.2f})")

            if result.get('correction_successful'):
                print("\n  ✓ Self-correction SUCCESSFUL - Answer improved from false to true!")
            elif not result['initial_verification'].get('is_truthful') and not final_ver.get('is_truthful'):
                print("\n  ✗ Self-correction attempted but answer still untruthful")
            else:
                print("\n  → No correction needed (initial answer was truthful)")
    else:
        print(f"\nAnswer: {result.get('llm_answer', 'N/A')[:200]}...")
        ver = result.get('verification', {})
        print(f"Truthful: {ver.get('is_truthful')} (Confidence: {ver.get('confidence', 0):.2f})")

    print()


def demo_basic_comparison(num_questions=5):
    """
    Demo 1: Basic comparison between direct prompting and CoT prompting.

    This demonstrates that CoT alone can improve truthfulness.
    """
    print_header("Demo 1: Direct Prompting vs Chain of Thought")

    print("Loading dataset and models...")
    loader = TruthfulQALoader()
    questions = loader.get_sample(sample_size=num_questions, seed=42)

    llm_provider = LLMProviderFactory.create("claude", model=settings.default_model)
    verifier = VerifierFactory.create("simple_text")

    print(f"Testing {num_questions} questions with two strategies:")
    print("  1. Direct prompting (baseline)")
    print("  2. Chain of Thought prompting")

    # Compare strategies
    results = compare_strategies(
        llm_provider=llm_provider,
        verifier=verifier,
        questions=questions,
        strategies_to_test=["direct", "chain_of_thought"],
        max_tokens=1024,
        temperature=1.0,
    )

    # Print comparison summary
    print_section("Results Summary")

    for strategy, summary in results["comparison_summary"].items():
        print(f"{strategy}:")
        print(f"  Accuracy: {summary['accuracy']:.1%}")
        print(f"  Truthful: {summary['truthful_count']}/{num_questions}")
        print()

    # Calculate improvement
    direct_acc = results["comparison_summary"]["direct"]["accuracy"]
    cot_acc = results["comparison_summary"]["chain_of_thought"]["accuracy"]
    improvement = cot_acc - direct_acc

    print(f"Improvement with CoT: {improvement:+.1%}")

    if improvement > 0:
        print("\n✓ Chain of Thought improved answer truthfulness!")
    elif improvement < 0:
        print("\n✗ Chain of Thought decreased truthfulness in this sample")
    else:
        print("\n→ No change in truthfulness")

    return results


def demo_self_correction(num_questions=5):
    """
    Demo 2: Self-correction pipeline.

    This demonstrates how an LLM can correct its own mistakes when prompted
    to reconsider using chain of thought.
    """
    print_header("Demo 2: Self-Correction with Chain of Thought")

    print("Loading dataset and models...")
    loader = TruthfulQALoader()
    questions = loader.get_sample(sample_size=num_questions, seed=42)

    llm_provider = LLMProviderFactory.create("claude", model=settings.default_model)
    verifier = VerifierFactory.create("simple_text")

    # Create evaluator with direct initial prompting and CoT correction
    initial_strategy = PromptStrategyFactory.create("direct")
    correction_strategy = PromptStrategyFactory.create("self_correction")

    evaluator = SelfCorrectionEvaluator(
        llm_provider=llm_provider,
        verifier=verifier,
        initial_strategy=initial_strategy,
        correction_strategy=correction_strategy,
    )

    print(f"\nEvaluating {num_questions} questions with self-correction:")
    print("  1. Initial answer using direct prompting")
    print("  2. If wrong, apply self-correction using CoT")
    print("  3. Compare initial vs corrected answers\n")

    # Evaluate with correction
    results = evaluator.evaluate_batch_with_correction(
        questions=questions,
        max_tokens=1024,
        temperature=1.0,
        max_correction_attempts=1,
        apply_correction_only_if_wrong=True,
    )

    # Print detailed results
    print_section("Detailed Results")

    for i, result in enumerate(results["results"], 1):
        print(f"\n[Question {i}/{num_questions}]")
        print_result(result)

    # Print summary
    print_section("Summary Statistics")

    summary = results["summary"]
    print(f"Total Questions: {summary['total_questions']}")
    print(f"\nInitial Performance:")
    print(f"  Truthful: {summary['initial_truthful_count']}/{summary['total_questions']}")
    print(f"  Accuracy: {summary['initial_accuracy']:.1%}")
    print(f"  Avg Confidence: {summary['initial_avg_confidence']:.2f}")

    print(f"\nFinal Performance (after correction):")
    print(f"  Truthful: {summary['final_truthful_count']}/{summary['total_questions']}")
    print(f"  Accuracy: {summary['final_accuracy']:.1%}")
    print(f"  Avg Confidence: {summary['final_avg_confidence']:.2f}")

    print(f"\nCorrection Statistics:")
    print(f"  Corrections Attempted: {summary['corrections_attempted']}")
    print(f"  Corrections Successful: {summary['corrections_successful']}")
    if summary['corrections_attempted'] > 0:
        print(f"  Success Rate: {summary['correction_success_rate']:.1%}")

    print(f"\nOverall Improvement:")
    print(f"  Accuracy Change: {summary['improvement']:+.1%}")
    if summary['initial_accuracy'] > 0:
        print(f"  Relative Improvement: {summary['improvement_percentage']:+.1f}%")

    if summary['improvement'] > 0:
        print("\n✓ Self-correction with CoT improved overall truthfulness!")
    elif summary['improvement'] < 0:
        print("\n✗ Performance decreased after correction")
    else:
        print("\n→ No change in overall accuracy")

    return results


def demo_strategy_comparison(num_questions=5):
    """
    Demo 3: Compare multiple prompting strategies.

    This demonstrates the effectiveness of different CoT approaches.
    """
    print_header("Demo 3: Comparison of Multiple Prompting Strategies")

    print("Loading dataset and models...")
    loader = TruthfulQALoader()
    questions = loader.get_sample(sample_size=num_questions, seed=42)

    llm_provider = LLMProviderFactory.create("claude", model=settings.default_model)
    verifier = VerifierFactory.create("simple_text")

    strategies = ["direct", "chain_of_thought", "reflective"]

    print(f"\nComparing {len(strategies)} prompting strategies on {num_questions} questions:")
    for i, s in enumerate(strategies, 1):
        strategy = PromptStrategyFactory.create(s)
        print(f"  {i}. {strategy.get_strategy_name()}")

    # Compare strategies
    results = compare_strategies(
        llm_provider=llm_provider,
        verifier=verifier,
        questions=questions,
        strategies_to_test=strategies,
        max_tokens=1024,
        temperature=1.0,
    )

    # Print comparison
    print_section("Strategy Comparison Results")

    # Create comparison table
    print(f"{'Strategy':<40} {'Accuracy':<12} {'Truthful':<10}")
    print("-" * 65)

    for strategy, summary in results["comparison_summary"].items():
        strategy_obj = PromptStrategyFactory.create(strategy)
        name = strategy_obj.get_strategy_name()
        acc = summary['accuracy']
        count = summary['truthful_count']
        total = num_questions
        print(f"{name:<40} {acc:>6.1%}      {count:>2}/{total:<2}")

    # Find best strategy
    best_strategy = max(
        results["comparison_summary"].items(),
        key=lambda x: x[1]["accuracy"]
    )
    best_name = PromptStrategyFactory.create(best_strategy[0]).get_strategy_name()
    best_acc = best_strategy[1]["accuracy"]

    print(f"\n✓ Best performing strategy: {best_name} ({best_acc:.1%})")

    return results


def save_results(results, filename):
    """Save results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filename}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Chain of Thought for Self-Correction"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=5,
        help="Number of questions to test (default: 5)"
    )
    parser.add_argument(
        "--mode",
        choices=["basic", "correction", "comparison", "all"],
        default="correction",
        help="Demo mode to run (default: correction)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    print_header("Chain of Thought Self-Correction Demo")
    print(f"Model: {settings.default_model}")
    print(f"Questions: {args.questions}")
    print(f"Mode: {args.mode}")

    results = None

    try:
        if args.mode == "basic" or args.mode == "all":
            results = demo_basic_comparison(args.questions)

        if args.mode == "correction" or args.mode == "all":
            results = demo_self_correction(args.questions)

        if args.mode == "comparison" or args.mode == "all":
            results = demo_strategy_comparison(args.questions)

        if args.save and results:
            save_results(results, f"cot_demo_{args.mode}")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    print_header("Demo Complete")
    print("\nKey Takeaways:")
    print("  1. Chain of Thought prompting encourages step-by-step reasoning")
    print("  2. Self-correction leverages CoT to reconsider initial answers")
    print("  3. LLMs can identify and correct their own mistakes when prompted appropriately")
    print("  4. Different CoT strategies may work better for different types of questions")
    print("\nFor more details, see the generated JSON results file (if --save was used)")

    return 0


if __name__ == "__main__":
    exit(main())
