"""Convert the three existing COCO splits without sampling or re-splitting."""
import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

from common import assert_same_categories, read_coco, read_config, sha256, split_paths


def convert(config, destination, image_limit=None, splits=("train", "val", "test")):
    destination = Path(destination).resolve()
    splits = tuple(splits)
    mapping = assert_same_categories(config, splits)
    coco_to_yolo = {value: key for key, value in mapping.items()}
    if destination.exists() and any(destination.iterdir()):
        manifest = destination / "conversion_manifest.json"
        if manifest.is_file() and image_limit is None:
            old = json.loads(manifest.read_text(encoding="utf-8"))
            expected = {s: sha256(split_paths(config, s)[1]) for s in splits}
            if old.get("annotation_sha256") == expected and old.get("category_mapping") == {str(k): v for k, v in mapping.items()}:
                return destination / "fish.yaml"
        raise FileExistsError(f"Nonempty destination; refusing to overwrite: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    counts = {}
    for split in splits:
        image_dir, ann_file = split_paths(config, split)
        data, split_mapping = read_coco(ann_file)
        if split_mapping != mapping:
            raise ValueError(f"Category mismatch in {split}")
        checksums[split] = sha256(ann_file)
        images = data["images"][:image_limit] if image_limit else data["images"]
        image_ids = {int(image["id"]) for image in images}
        anns = {image_id: [] for image_id in image_ids}
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            if image_id in anns and not ann.get("iscrowd", 0):
                anns[image_id].append(ann)
        out_images = destination / "images" / split
        out_labels = destination / "labels" / split
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)
        labels_count = 0
        for image in images:
            source = image_dir / image["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            if Path(image["file_name"]).name != image["file_name"]:
                raise ValueError(f"Nested/unsafe image path: {image['file_name']}")
            target = out_images / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
            width, height = int(image["width"]), int(image["height"])
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid image size: {source}")
            lines = []
            for ann in anns[int(image["id"])]:
                x, y, w, h = map(float, ann["bbox"])
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(float(width), x + w), min(float(height), y + h)
                if x2 <= x1 or y2 <= y1:
                    continue
                cls = coco_to_yolo[int(ann["category_id"])]
                xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
                wn, hn = (x2 - x1) / width, (y2 - y1) / height
                if not all(0 <= v <= 1 for v in (xc, yc, wn, hn)):
                    raise ValueError(f"Out-of-range YOLO label: {ann}")
                lines.append(f"{cls} {xc:.10f} {yc:.10f} {wn:.10f} {hn:.10f}")
            (out_labels / f"{source.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            labels_count += len(lines)
        counts[split] = {"images": len(images), "labels": labels_count}
    fish_yaml = destination / "fish.yaml"
    dataset_yaml = {"path": str(destination), "train": "images/train", "val": "images/val",
                    "names": {0: "normal", 1: "hypoxia"}}
    if "test" in splits:
        dataset_yaml["test"] = "images/test"
    fish_yaml.write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    (destination / "conversion_manifest.json").write_text(json.dumps({"annotation_sha256": checksums, "category_mapping": mapping, "counts": counts, "image_limit": image_limit}, indent=2), encoding="utf-8")
    return fish_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root")
    parser.add_argument("--out", default="comparison_dataset")
    parser.add_argument("--image-limit", type=int, help="Smoke-test only; not for final experiments")
    parser.add_argument("--train-only", action="store_true", help="Convert train/val without requiring a test annotation")
    args = parser.parse_args()
    cfg = read_config(args.config or Path(__file__).parent / "configs" / "comparison_models.yml", args.data_root)
    splits = ("train", "val") if args.train_only else ("train", "val", "test")
    print(convert(cfg, Path(args.out), args.image_limit, splits=splits))
