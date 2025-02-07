import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm


def generate_answers_from_samples(
    model: Qwen2VLForConditionalGeneration,
    processor: Qwen2VLProcessor,
    samples: list[list[dict]],
    batch_size: int = 8,
    max_new_tokens=1024,
    device="cuda",
) -> list[str]:
    if len(samples) == 0:
        return []

    if not torch.cuda.is_available():
        device = "cpu"

    # Prepare the text input by applying the chat template
    text_inputs = [
        processor.apply_chat_template(
            sample[1:2],  # Use the sample without the system message
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in samples
    ]

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(samples)

    answers = []

    for i in tqdm(range(0, len(samples), batch_size)):
        text_batch = text_inputs[i : i + batch_size]
        image_batch = image_inputs[i : i + batch_size]

        # Prepare the inputs for the model
        model_inputs = processor(
            text=text_batch,
            images=image_batch,
            padding=True,
            return_tensors="pt",
        ).to(
            device
        )  # Move inputs to the specified device

        # Generate text with the model
        generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

        # Trim the generated ids to remove the input ids
        trimmed_generated_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the output text
        output_text = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        answers.extend(output_text)

    return answers


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# Create a data collator to encode text and image pairs
def get_collate_fn(processor: Qwen2VLProcessor):
    def collate_fn(examples: list[list[dict]]):
        # Get the texts and images, and apply the chat template
        texts = [
            processor.apply_chat_template(example, tokenize=False)
            for example in examples
        ]  # Prepare texts for processing
        image_inputs = [
            process_vision_info(example)[0] for example in examples
        ]  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = (
            -100
        )  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(
            processor, Qwen2VLProcessor
        ):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [
                151652,
                151653,
                151655,
            ]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        # Mask all text input (question + image tokens)**
        for i, example in enumerate(examples):
            text_input = processor.tokenizer.apply_chat_template(
                example[:-1], tokenize=False
            )
            image_input = image_inputs[i]
            tokenized_input = processor(
                text=text_input,
                images=image_input,
                return_tensors="pt",
                padding=False,
            )
            input_length = len(tokenized_input)
            labels[i, :input_length] = -100  # Mask all input tokens (image + question)

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch

    return collate_fn
