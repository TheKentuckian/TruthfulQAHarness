"""Self-correction techniques for improving LLM answer quality."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from backend.models.llm_provider import LLMProvider


class SelfCorrector(ABC):
    """Abstract base class for self-correction techniques."""

    @abstractmethod
    def correct(
        self,
        question: str,
        initial_answer: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply self-correction to improve an initial answer.

        Args:
            question: The original question
            initial_answer: The initial answer to improve
            **kwargs: Additional correction-specific parameters

        Returns:
            Dictionary containing:
                - corrected_answer: The improved answer
                - correction_metadata: Details about the correction process
        """
        pass

    @abstractmethod
    def get_corrector_name(self) -> str:
        """Return the name of the corrector."""
        pass


class CritiqueSelfCorrector(SelfCorrector):
    """
    Self-correction technique using critique.

    This technique:
    1. Generates an initial answer
    2. Prompts the LLM to critique its own answer
    3. Uses the critique to generate an improved answer
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the critique-based self-corrector.

        Args:
            llm_provider: The LLM provider to use for correction
            max_tokens: Optional max tokens for generation
            temperature: Optional temperature for generation
        """
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _create_critique_prompt(self, question: str, initial_answer: str) -> str:
        """
        Create a prompt asking the LLM to critique its own answer.

        Args:
            question: The original question
            initial_answer: The initial answer to critique

        Returns:
            Formatted critique prompt
        """
        return f"""You previously answered the following question:

Question: {question}

Your answer: {initial_answer}

Now, critically analyze your answer. Consider:
1. Is there any misinformation or factual errors?
2. Are there common misconceptions you might be perpetuating?
3. Is the answer potentially misleading or incomplete?
4. What assumptions did you make that might be wrong?
5. What are the potential flaws in your reasoning?

Provide a detailed critique of your answer, identifying any problems or areas for improvement:"""

    def _create_revision_prompt(
        self,
        question: str,
        initial_answer: str,
        critique: str
    ) -> str:
        """
        Create a prompt asking the LLM to revise based on the critique.

        Args:
            question: The original question
            initial_answer: The initial answer
            critique: The critique of the initial answer

        Returns:
            Formatted revision prompt
        """
        return f"""You previously answered this question:

Question: {question}

Your initial answer: {initial_answer}

You then critiqued your answer as follows:
{critique}

Based on this critique, provide an improved, more accurate answer to the original question. Focus on addressing the issues you identified:

Improved answer:"""

    def correct(
        self,
        question: str,
        initial_answer: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Apply critique-based self-correction.

        Args:
            question: The original question
            initial_answer: The initial answer to improve
            **kwargs: Additional parameters

        Returns:
            Dictionary containing corrected answer and metadata
        """
        # Step 1: Generate critique
        critique_prompt = self._create_critique_prompt(question, initial_answer)

        try:
            critique = self.llm_provider.generate(
                prompt=critique_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            return {
                "corrected_answer": initial_answer,  # Fall back to initial answer
                "correction_metadata": {
                    "critique": None,
                    "error": f"Failed to generate critique: {str(e)}",
                    "success": False,
                }
            }

        # Step 2: Generate revised answer based on critique
        revision_prompt = self._create_revision_prompt(
            question,
            initial_answer,
            critique
        )

        try:
            corrected_answer = self.llm_provider.generate(
                prompt=revision_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            return {
                "corrected_answer": initial_answer,  # Fall back to initial answer
                "correction_metadata": {
                    "critique": critique,
                    "error": f"Failed to generate revision: {str(e)}",
                    "success": False,
                }
            }

        return {
            "corrected_answer": corrected_answer,
            "correction_metadata": {
                "critique": critique,
                "initial_answer": initial_answer,
                "revision_prompt": revision_prompt,
                "success": True,
            }
        }

    def get_corrector_name(self) -> str:
        """Return the name of the corrector."""
        return "Critique-based Self-Correction"


class SelfCorrectorFactory:
    """Factory for creating self-corrector instances."""

    _correctors = {
        "critique": CritiqueSelfCorrector,
    }

    @classmethod
    def create(
        cls,
        corrector_type: str,
        llm_provider: LLMProvider,
        **config
    ) -> SelfCorrector:
        """
        Create a self-corrector instance.

        Args:
            corrector_type: Type of corrector ('critique', etc.)
            llm_provider: The LLM provider to use
            **config: Configuration for the corrector

        Returns:
            SelfCorrector instance

        Raises:
            ValueError: If corrector type is not supported
        """
        corrector_class = cls._correctors.get(corrector_type.lower())
        if not corrector_class:
            raise ValueError(
                f"Unknown corrector type: {corrector_type}. "
                f"Available: {list(cls._correctors.keys())}"
            )

        return corrector_class(llm_provider=llm_provider, **config)

    @classmethod
    def get_available_correctors(cls) -> list:
        """Return list of available corrector types."""
        return list(cls._correctors.keys())
