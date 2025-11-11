"""LLM provider abstraction and implementations."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import anthropic
from backend.config import settings


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters

        Returns:
            The generated text response
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass


class ClaudeProvider(LLMProvider):
    """Claude LLM provider using Anthropic API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the Claude provider.

        Args:
            api_key: Anthropic API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.api_key = api_key or settings.anthropic_api_key
        self.model = model or settings.default_model
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using Claude.

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            The generated text response
        """
        max_tokens = max_tokens or settings.default_max_tokens
        temperature = temperature or settings.default_temperature

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract text from response
            return message.content[0].text

        except Exception as e:
            raise RuntimeError(f"Error generating response from Claude: {str(e)}")

    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        return f"Claude ({self.model})"


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers = {
        "claude": ClaudeProvider,
    }

    @classmethod
    def create(cls, provider_type: str, **config) -> LLMProvider:
        """
        Create an LLM provider instance.

        Args:
            provider_type: Type of provider ('claude', etc.)
            **config: Configuration for the provider

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider type is not supported
        """
        provider_class = cls._providers.get(provider_type.lower())
        if not provider_class:
            raise ValueError(
                f"Unknown provider type: {provider_type}. "
                f"Available: {list(cls._providers.keys())}"
            )

        return provider_class(**config)

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Return list of available provider types."""
        return list(cls._providers.keys())
