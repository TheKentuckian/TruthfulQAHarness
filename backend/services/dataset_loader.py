"""TruthfulQA dataset loader."""
import random
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from backend.config import settings


class TruthfulQALoader:
    def __init__(self):
        self.dataset = None
        self._load_dataset()

    def _load_dataset(self):
        try:
            self.dataset = load_dataset("truthful_qa", "generation", split="validation")
            print(f"Loaded TruthfulQA dataset with {len(self.dataset)} questions")
        except Exception as e:
            raise RuntimeError(f"Failed to load TruthfulQA dataset: {str(e)}")

    def _format_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item.get("question", "")

        correct_answers = []
        if "correct_answers" in item and item["correct_answers"]:
            correct_answers = item["correct_answers"]
        elif "best_answer" in item and item["best_answer"]:
            correct_answers = [item["best_answer"]]

        incorrect_answers = []
        if "incorrect_answers" in item and item["incorrect_answers"]:
            incorrect_answers = item["incorrect_answers"]

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
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        sample_size = sample_size or settings.truthfulqa_sample_size
        sample_size = min(sample_size, len(self.dataset))

        if seed is not None:
            random.seed(seed)

        indices = random.sample(range(len(self.dataset)), sample_size)

        questions = []
        for idx in indices:
            item = self.dataset[idx]
            formatted = self._format_question(item)
            formatted["index"] = idx
            questions.append(formatted)

        return questions

    def get_question_by_index(self, index: int) -> Dict[str, Any]:
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        if index < 0 or index >= len(self.dataset):
            raise ValueError(f"Index {index} out of range [0, {len(self.dataset)})")

        item = self.dataset[index]
        formatted = self._format_question(item)
        formatted["index"] = index
        return formatted

    def get_questions_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        questions = []
        dataset_len = len(self.dataset)

        for idx_1based in indices:
            idx = idx_1based - 1

            if idx < 0 or idx >= dataset_len:
                raise ValueError(f"Index {idx_1based} out of range [1, {dataset_len}]")

            item = self.dataset[idx]
            formatted = self._format_question(item)
            formatted["index"] = idx
            questions.append(formatted)

        return questions

    def get_all_questions(self) -> List[Dict[str, Any]]:
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
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded")

        return {
            "total_questions": len(self.dataset),
            "dataset_name": "TruthfulQA",
            "split": "validation",
        }
