"""TruthfulQA dataset loader."""
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from backend.config import settings


class TruthfulQALoader:
    """Loader for the TruthfulQA dataset."""

    def __init__(self):
        """Initialize the dataset loader."""
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        """Load the TruthfulQA dataset from HuggingFace."""
        try:
            # Load the TruthfulQA dataset
            # Using the 'generation' split which has questions and answers
            self.dataset = load_dataset("truthful_qa", "generation", split="validation")
            print(f"Loaded TruthfulQA dataset with {len(self.dataset)} questions")
        except Exception as e:
            raise RuntimeError(f"Failed to load TruthfulQA dataset: {str(e)}")

    def _format_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a dataset item into a standardized question format.

        Args:
            item: Raw dataset item

        Returns:
            Formatted question dictionary
        """
        # Extract fields from the TruthfulQA dataset
        question = item.get("question", "")

        # Get correct answers (can be in different fields)
        correct_answers = []
        if "correct_answers" in item and item["correct_answers"]:
            correct_answers = item["correct_answers"]
        elif "best_answer" in item and item["best_answer"]:
            correct_answers = [item["best_answer"]]

        # Get incorrect answers
        incorrect_answers = []
        if "incorrect_answers" in item and item["incorrect_answers"]:
            incorrect_answers = item["incorrect_answers"]

        # Get category and additional info
        category = item.get("category", "Unknown")

        return {
            "question": question,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
            "category": category,
            "metadata": {
                "source": item.get("source", ""),
            }
        }

    def get_sample(self, sample_size: Optional[int] = None, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get a random sample of questions from the dataset.

        Args:
            sample_size: Number of questions to sample (defaults to settings)
            seed: Random seed for reproducibility

        Returns:
            List of formatted question dictionaries
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        sample_size = sample_size or settings.truthfulqa_sample_size

        # Don't sample more than available
        sample_size = min(sample_size, len(self.dataset))

        # Set seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)

        # Get random indices
        indices = random.sample(range(len(self.dataset)), sample_size)

        # Format questions
        questions = []
        for idx in indices:
            item = self.dataset[idx]
            formatted = self._format_question(item)
            formatted["index"] = idx
            questions.append(formatted)

        return questions

    def get_question_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a specific question by index.

        Args:
            index: Index of the question in the dataset

        Returns:
            Formatted question dictionary
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        if index < 0 or index >= len(self.dataset):
            raise ValueError(f"Index {index} out of range [0, {len(self.dataset)})")

        item = self.dataset[index]
        formatted = self._format_question(item)
        formatted["index"] = index
        return formatted

    def get_all_questions(self) -> List[Dict[str, Any]]:
        """
        Get all questions from the dataset.

        Returns:
            List of all formatted question dictionaries
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        questions = []
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            formatted = self._format_question(item)
            formatted["index"] = idx
            questions.append(formatted)

        return questions

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.

        Returns:
            Dictionary with dataset information
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        return {
            "total_questions": len(self.dataset),
            "dataset_name": "TruthfulQA",
            "split": "validation",
        }
