"""Create the 0/1 COCO test annotation without changing the original 1/2 file.

Run from any directory: python tools/remap_fish_test_categories.py
"""

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "fish_dataset622"
SOURCE = DATA / "test" / "coco_detection_test.json"
OUTPUT = DATA / "test" / "coco_detection_test0.json"


def remap(source=SOURCE, output=OUTPUT):
    source = Path(source)
    output = Path(output)
    data = json.loads(source.read_text(encoding="utf-8"))
    categories = {int(item["id"]): item["name"].lower() for item in data["categories"]}
    if categories != {1: "normal", 2: "hypoxia"}:
        raise ValueError(f"Expected source 1=normal, 2=hypoxia; got {categories}")
    image_ids = {int(item["id"]) for item in data["images"]}
    if len(image_ids) != len(data["images"]):
        raise ValueError("Duplicate image IDs in test annotation")
    for image in data["images"]:
        if not (source.parent / image["file_name"]).is_file():
            raise FileNotFoundError(source.parent / image["file_name"])
    mapped = copy.deepcopy(data)
    for item in mapped["categories"]:
        item["id"] = int(item["id"]) - 1
    for item in mapped["annotations"]:
        if int(item["image_id"]) not in image_ids:
            raise ValueError(f"Unknown image_id in annotation: {item['image_id']}")
        if int(item["category_id"]) not in categories:
            raise ValueError(f"Unexpected category_id: {item['category_id']}")
        item["category_id"] = int(item["category_id"]) - 1
    expected = json.dumps(mapped, ensure_ascii=False, indent=2) + "\n"
    if output.exists():
        existing = output.read_text(encoding="utf-8")
        if existing != expected:
            raise ValueError(f"Existing mapped test annotation differs from {source}: {output}")
    else:
        output.write_text(expected, encoding="utf-8")
    print(f"Mapped {len(mapped['images'])} images and {len(mapped['annotations'])} annotations: {output}")
    return output


if __name__ == "__main__":
    remap()
