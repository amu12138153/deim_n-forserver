"""Paths and COCO category contract shared by the comparison scripts."""
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "comparison_models.yml"


def path_from_root(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_config(path=DEFAULT_CONFIG, data_root=None, output_root=None):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config["data_root"] = path_from_root(data_root or config["data_root"])
    config["output_root"] = path_from_root(output_root or config["output_root"])
    if int(config["image_size"]) != 640:
        raise ValueError("This comparison requires image_size=640")
    protocol = config.get("training_protocol")
    if protocol:
        if protocol.get("augmentation") != "resize640_horizontal_flip_p05":
            raise ValueError("Unknown shared augmentation protocol")
        for name, model in config["models"].items():
            if model["epochs"] != protocol["epochs"] or model["batch_size"] != protocol["batch_size"]:
                raise ValueError(f"{name} does not match the shared training budget")
    return config


def split_paths(config, split):
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    source_split = config.get("test_source_split", "test") if split == "test" else split
    if source_split not in ("train", "val", "test"):
        raise ValueError(f"Invalid source split: {source_split}")
    image_dir = config["data_root"] / source_split
    ann_file = image_dir / config["annotations"][split]
    for path in (image_dir, ann_file):
        if not path.exists():
            raise FileNotFoundError(path)
    return image_dir, ann_file


def read_coco(ann_file):
    data = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    categories = {str(c["name"]).lower(): int(c["id"]) for c in data["categories"]}
    if set(categories) != {"normal", "hypoxia"} or len(data["categories"]) != 2:
        raise ValueError(f"Expected exactly normal/hypoxia categories: {ann_file}")
    return data, {0: categories["normal"], 1: categories["hypoxia"]}


def assert_same_categories(config, splits=("train", "val", "test")):
    mappings = [read_coco(split_paths(config, split)[1])[1] for split in splits]
    if not mappings:
        raise ValueError("At least one dataset split is required")
    if any(mapping != mappings[0] for mapping in mappings[1:]):
        raise ValueError(f"COCO category IDs differ across splits: {mappings}")
    return mappings[0]


def evaluation_split(config):
    """Label a copied validation annotation honestly in downstream results."""
    _, val_ann = split_paths(config, "val")
    _, eval_ann = split_paths(config, "test")
    return "validation_reused_as_test" if sha256(val_ann) == sha256(eval_ann) else "test"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
