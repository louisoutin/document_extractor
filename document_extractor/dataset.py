import json
from typing import Any
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import hashlib


SYSTEM_MESSAGE = """You are a Vision Language Model specialized in interpreting visual data from documents.
Your task is to analyze the provided document image and respond to queries with concise answers, usually a single word, number, or short phrase.
The document include a variety of types (e.g., invoice, bank statements, payslips, etc) and contain tables, dates, amounts, and text.
Focus on delivering accurate, succinct answers based on the visual information."""


class DatasetConfig(BaseModel):
    base_query: str = """Extract the"""
    list_query: str = """Extract the list of"""
    json_query: str = """Extract (in json) the"""
    extract_sub_dict_as_dict: bool = True
    extract_sub_dict_as_single_values: bool = True


def format_data(sample: dict):
    pil_image = Image.open(sample["image_path"])
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["answer"]}],
        },
    ]


def _generate_samples_recursive(
    image_path: str,
    document_infos: dict[str, Any],
    key_prefix: str = "",
    config: DatasetConfig = DatasetConfig(),
):
    res = []
    keys = list(document_infos.keys())
    has_sub_dicts: bool = any(
        [isinstance(key, dict) or isinstance(key, list) for key in keys]
    )
    extract_sub_dicts = not has_sub_dicts and len(keys) > 1 and key_prefix != ""
    if extract_sub_dicts and config.extract_sub_dict_as_dict:
        sample = {
            "image_path": image_path,
            "query": f"{config.json_query} {key_prefix}",
            "answer": json.dumps(document_infos, indent=2),
        }
        res.append(sample)
    for key, values in document_infos.items():
        key = key.replace("_", " ")
        key = f"{key_prefix} {key}" if key_prefix else key
        if isinstance(values, dict):
            res += _generate_samples_recursive(
                image_path=image_path,
                document_infos=values,
                key_prefix=key,
                config=config,
            )
        elif isinstance(values, list):
            sample = {
                "image_path": image_path,
                "query": f"{config.list_query} {key}",
                "answer": json.dumps(values, indent=2),
            }
            res.append(sample)
        else:
            if (
                extract_sub_dicts and config.extract_sub_dict_as_single_values
            ) or not extract_sub_dicts:
                sample = {
                    "image_path": image_path,
                    "query": f"{config.base_query} {key}",
                    "answer": str(values),
                }
                res.append(sample)

    return res


def generate_image_samples(
    image_path: str, json_path: str, config: DatasetConfig = DatasetConfig()
) -> list[dict]:
    # res dict keys: image, query, answer
    with open(json_path) as json_file:
        document_infos: dict = json.load(json_file)
    samples = _generate_samples_recursive(
        image_path=image_path, document_infos=document_infos, config=config
    )
    return samples


def generate_all_samples(
    folder: str, config: DatasetConfig = DatasetConfig()
) -> list[dict]:
    samples = []

    def process_image_file(image_path: Path) -> None:
        json_path = image_path.with_suffix(".json")
        samples.extend(
            generate_image_samples(
                image_path=str(image_path), json_path=str(json_path), config=config
            )
        )

    for item in Path(folder).iterdir():
        if item.is_dir():
            for subitem in item.iterdir():
                if subitem.is_file() and subitem.suffix != ".json":
                    process_image_file(subitem)
        elif item.is_file() and item.suffix != ".json":
            process_image_file(item)

    return samples


def check_leakage(folder1: str, folder2: str):
    def get_files(folder_path: str) -> list[Path]:
        return [
            file_path
            for file_path in Path(folder_path).rglob("*")
            if file_path.is_file()
        ]

    def compute_file_hash(file_path: str) -> str:
        hasher = hashlib.md5()
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_hash_map(file_list: list[Path]) -> dict[str, Path]:
        hash_map = {}
        for file_path in file_list:
            file_hash = compute_file_hash(file_path)
            if file_hash not in hash_map:
                hash_map[file_hash] = []
            hash_map[file_hash].append(file_path)
        return hash_map

    folder1_files = get_files(folder1)
    folder2_files = get_files(folder2)

    hash_map1 = build_hash_map(folder1_files)
    hash_map2 = build_hash_map(folder2_files)

    common_hashes = hash_map1.keys() & hash_map2.keys()

    duplicates = []
    for hash_val in common_hashes:
        duplicates.append(
            {
                "hash": hash_val,
                "folder1_files": hash_map1[hash_val],
                "folder2_files": hash_map2[hash_val],
            }
        )

    error_msg = ""
    if duplicates:
        error_msg = "Image leakage detected between the folders:\n\n"
        for dup in duplicates:
            error_msg += f"Hash {dup['hash']} found in both folders:\n"
            error_msg += f"Files in {folder1}:\n"
            error_msg += (
                "\n".join(f"  - {path}" for path in dup["folder1_files"]) + "\n"
            )
            error_msg += f"Files in {folder2}:\n"
            error_msg += (
                "\n".join(f"  - {path}" for path in dup["folder2_files"]) + "\n\n"
            )

    assert len(duplicates) == 0, error_msg
