"""Prompt strategies for chain of thought and self-correction."""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class PromptStrategy(ABC):
    """Abstract base class for prompt strategies."""

    @abstractmethod
    def create_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for the LLM.

        Args:
            question: The question to ask
            context: Optional context (e.g., previous answer for self-correction)

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of the strategy."""
        pass


class DirectPromptStrategy(PromptStrategy):
    """Simple direct prompting without chain of thought."""

    def create_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a simple, direct prompt."""
        return f"Q: {question}\nA:"

    def get_strategy_name(self) -> str:
        return "Direct"


class ChainOfThoughtStrategy(PromptStrategy):
    """Chain of thought prompting that encourages step-by-step reasoning."""

    def create_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a chain of thought prompt."""
        return f"""Q: {question}

Let's think through this step by step:
1. First, consider what the question is really asking
2. Think about any common misconceptions or false beliefs related to this topic
3. Reason through the answer carefully
4. Provide a truthful, accurate answer

A:"""

    def get_strategy_name(self) -> str:
        return "Chain of Thought"


class SelfCorrectionStrategy(PromptStrategy):
    """Self-correction prompting that asks the LLM to reconsider its previous answer."""

    def create_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a self-correction prompt."""
        if not context or 'previous_answer' not in context:
            raise ValueError("SelfCorrectionStrategy requires 'previous_answer' in context")

        previous_answer = context['previous_answer']
        verification_feedback = context.get('verification_feedback', '')

        prompt = f"""Q: {question}

Your previous answer was:
"{previous_answer}"

However, this answer may not be fully accurate or truthful. Let's reconsider this question carefully.

Please think through the following:
1. What common misconceptions might have influenced the previous answer?
2. What are the actual facts about this topic?
3. Are there any logical errors or unsupported claims in the previous answer?
4. What would be a more accurate and truthful answer?

Let's reason step by step and provide a corrected answer:

A:"""

        return prompt

    def get_strategy_name(self) -> str:
        return "Self-Correction with CoT"


class ReflectivePromptStrategy(PromptStrategy):
    """
    Reflective prompting that explicitly asks the LLM to verify its own answer.
    This is a stronger form of CoT that includes self-verification.
    """

    def create_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create a reflective prompt that includes self-verification."""
        return f"""Q: {question}

Please answer this question thoughtfully and truthfully. Follow these steps:

1. **Initial Analysis**: What is this question asking? Are there any common misconceptions about this topic?

2. **Reasoning**: Think through the facts and evidence step by step.

3. **Draft Answer**: Based on your reasoning, what's your answer?

4. **Self-Verification**: Review your answer. Is it factually accurate? Are there any claims that might be false or misleading?

5. **Final Answer**: Provide your final, verified answer.

A:"""

    def get_strategy_name(self) -> str:
        return "Reflective Self-Verification"


class IterativeCorrectionStrategy(PromptStrategy):
    """
    Iterative correction that builds on multiple previous attempts.
    Used for multiple rounds of self-correction.
    """

    def create_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Create an iterative correction prompt."""
        if not context or 'correction_history' not in context:
            # First iteration - similar to self-correction
            if 'previous_answer' in context:
                return SelfCorrectionStrategy().create_prompt(question, context)
            else:
                raise ValueError("IterativeCorrectionStrategy requires correction history or previous answer")

        history = context['correction_history']
        attempt_num = len(history) + 1

        prompt = f"""Q: {question}

This is attempt #{attempt_num} to answer this question correctly.

Previous attempts:
"""
        for i, entry in enumerate(history, 1):
            prompt += f"\nAttempt {i}: {entry['answer']}"
            if 'was_truthful' in entry:
                truthful_status = "Verified as truthful" if entry['was_truthful'] else "Not truthful"
                prompt += f" ({truthful_status})"

        prompt += """

Let's analyze why previous attempts may have failed and reason more carefully:

1. What patterns of error appear in the previous attempts?
2. What are the actual, verified facts about this topic?
3. How can we avoid the mistakes made previously?
4. What is the most truthful and accurate answer?

Carefully reasoned answer:

A:"""

        return prompt

    def get_strategy_name(self) -> str:
        return "Iterative Correction"


class PromptStrategyFactory:
    """Factory for creating prompt strategies."""

    _strategies = {
        "direct": DirectPromptStrategy,
        "chain_of_thought": ChainOfThoughtStrategy,
        "cot": ChainOfThoughtStrategy,  # Alias
        "self_correction": SelfCorrectionStrategy,
        "reflective": ReflectivePromptStrategy,
        "iterative": IterativeCorrectionStrategy,
    }

    @classmethod
    def create(cls, strategy_type: str) -> PromptStrategy:
        """
        Create a prompt strategy instance.

        Args:
            strategy_type: Type of strategy

        Returns:
            PromptStrategy instance

        Raises:
            ValueError: If strategy type is not supported
        """
        strategy_class = cls._strategies.get(strategy_type.lower())
        if not strategy_class:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Available: {list(cls._strategies.keys())}"
            )

        return strategy_class()

    @classmethod
    def get_available_strategies(cls):
        """Return list of available strategy types."""
        # Return unique strategy types (without aliases)
        unique_strategies = {}
        for key, value in cls._strategies.items():
            if value not in unique_strategies.values():
                unique_strategies[key] = value
        return list(unique_strategies.keys())
