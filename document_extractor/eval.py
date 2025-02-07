import json
import re
import numpy as np
from tokenizers import Tokenizer


def evaluate(ground_truths: list[str], predictions: list[str]) -> float:
    """Evaluate predictions accuracy with relaxed conditions"""
    results = []
    for ground_truth, prediction in zip(ground_truths, predictions):
        ground_truth = _parse_value(ground_truth)
        prediction = _parse_value(prediction)
        result = _compare_values(prediction, ground_truth)
        results.append(result)
    return sum(results) / len(results) if results else 0.0


def _normalize_text(text: str) -> str:
    """Normalize text by removing extra spaces and lowercasing."""
    return re.sub(r"\s+", " ", text.strip()).lower()


def _parse_value(value: str) -> str | int | float | list | dict:
    """Try to parse a value as JSON (list/dict/number), otherwise return normalized string."""
    try:
        parsed = json.loads(value)
        if isinstance(parsed, float):
            return round(parsed, 2)  # Round floats to 2 decimals
        return parsed
    except (json.JSONDecodeError, TypeError, ValueError):
        return _normalize_text(value)


def _compare_values(
    pred: str | int | float | list | dict, truth: str | int | float | list | dict
) -> bool:
    """Compare values with relaxed conditions."""
    if isinstance(pred, float) and isinstance(truth, float):
        return (
            abs(pred - truth) < 0.01
        )  # Allow small differences in floating-point values
    if isinstance(pred, list) and isinstance(truth, list):
        return sorted(pred) == sorted(truth)  # Allow different order in lists
    if isinstance(pred, dict) and isinstance(truth, dict):
        return pred == truth  # Dict comparison allows different key orders
    return pred == truth  # Default strict equality check


def get_compute_metrics(tokenizer: Tokenizer):
    def compute_metrics(eval_preds: tuple) -> dict:
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the preds as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

        decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]
        # Compute accuracy
        accuracy = evaluate(ground_truths=decoded_labels, predictions=decoded_preds)
        # Extract the median scores
        return {"accuracy": accuracy}

    return compute_metrics
