import pytest
import torch
from document_extractor.utils import generate_answers_from_samples, get_collate_fn
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from PIL import Image

import torch


@pytest.fixture
def model():
    return Qwen2VLForConditionalGeneration.from_pretrained(
        "yujiepan/qwen2-vl-tiny-random",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )


@pytest.fixture
def processor():
    min_pixels = 10 * 28 * 28
    max_pixels = 20 * 28 * 28
    return Qwen2VLProcessor.from_pretrained(
        "yujiepan/qwen2-vl-tiny-random", min_pixels=min_pixels, max_pixels=max_pixels
    )


def test_single_sample_batch_size_1(model, processor):
    samples = [
        [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (300, 300))},
                    {"type": "text", "text": "What's in this image?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Test answer"}]},
        ]
    ]

    answers = generate_answers_from_samples(
        model=model,
        processor=processor,
        samples=samples,
        batch_size=1,
    )

    assert len(answers) == 1
    assert isinstance(answers[0], str)
    assert len(answers[0]) > 0


def test_multiple_samples_batch_processing(model, processor):
    samples = [
        [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (300, 300))},
                    {"type": "text", "text": "What's in this image?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Test answer"}]},
        ],
        [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (300, 300))},
                    {"type": "text", "text": "What's the content of this image?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Test answer"}]},
        ],
        [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (300, 300))},
                    {"type": "text", "text": "Describe the image"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Test answer"}]},
        ],
    ]

    answers = generate_answers_from_samples(
        model=model,
        processor=processor,
        samples=samples,
        batch_size=2,
    )

    assert len(answers) == 3
    assert isinstance(answers[0], str)
    assert len(answers[0]) > 0


def test_empty_samples(model, processor):
    answers = generate_answers_from_samples(
        model=model,
        processor=processor,
        samples=[],
    )
    assert answers == []


def test_collate_fn(processor):
    """Tests that collate_fn correctly masks input and only allows loss on output."""
    collate_fn = get_collate_fn(processor=processor)

    samples = [
        [
            {"role": "system", "content": [{"type": "text", "text": "System message"}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (300, 300))},
                    {"type": "text", "text": "What's in this image?"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Test answer"}]},
        ]
    ]

    # Run collate function
    batch = collate_fn(samples)

    # Extract labels
    labels = batch["labels"]

    # Ensure padding tokens are masked (-100)
    assert (
        labels[labels == processor.tokenizer.pad_token_id] == -100
    ).all(), "Padding should be masked"

    # Ensure image token IDs are masked (-100)
    assert (labels[labels == 999] == -100).all(), "Image tokens should be masked"

    # Ensure at least one token is NOT masked (for the output part)
    assert (labels != -100).any(), "Some tokens should be left for loss computation"
