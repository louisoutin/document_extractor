from document_extractor.eval import evaluate
from PIL import Image


def test_evaluate():
    ground_truths = ["Test answer"]
    predictions = ["Test answer"]

    assert evaluate(ground_truths, predictions) == 1.0

    predictions = ["Wrong answer"]
    assert evaluate(ground_truths, predictions) == 0.0

    predictions = [" test answer  "]  # Extra spaces and different casing
    assert (
        evaluate(ground_truths, predictions) == 1.0
    )  # Should pass normalization check

    predictions = ["1.2345"]
    ground_truths = ["1.23"]  # Expect rounding to 2 decimals
    assert evaluate(ground_truths, predictions) == 1.0

    predictions = ["[1, 2, 3]"]
    ground_truths = ["[3, 2, 1]"]  # Lists with different order
    assert evaluate(ground_truths, predictions) == 1.0

    predictions = ['{"a":1, "b":2}']
    ground_truths = ['{"b":2, "a":1}']  # Dicts with different key order
    assert evaluate(ground_truths, predictions) == 1.0
