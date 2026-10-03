"""Use the repository's existing DEIM loader and prediction loop."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from common import ROOT, read_coco

sys.path.insert(0, str(ROOT))


def load(config_path, checkpoint, ann_file, image_dir, device):
    from eval_curves import load_everything
    args = SimpleNamespace(config=str(config_path), resume=str(checkpoint), split="test",
                           ann_file=str(ann_file), image_dir=str(image_dir),
                           sensor_csv=str(Path(image_dir) / "test.csv"), device=device)
    return load_everything(args)


def predict(config_path, checkpoint, ann_file, image_dir, output_json, device="cuda"):
    from eval_curves import collect_predictions
    (_, model, post, loader, _, category_ids, id2gt, _, resolved_ann) = load(
        config_path, checkpoint, ann_file, image_dir, device)
    if Path(resolved_ann).resolve() != Path(ann_file).resolve():
        raise ValueError("DEIM loader did not use the requested test annotation")
    all_preds, _ = collect_predictions(model, post, loader, torch.device(device), id2gt)
    coco_data, mapping = read_coco(ann_file)
    if list(category_ids) != [mapping[0], mapping[1]]:
        raise ValueError("DEIM postprocessor category order differs from the test annotation")
    expected_ids = {int(image["id"]) for image in coco_data["images"]}
    observed_ids = {int(pred["image_id"]) for pred in all_preds}
    if observed_ids != expected_ids or len(all_preds) != len(expected_ids):
        raise ValueError("DEIM did not predict each test image exactly once")
    results = []
    for pred in all_preds:
        for label, box, score in zip(pred["labels"], pred["boxes"], pred["scores"]):
            label = int(label)
            if label not in mapping:
                raise ValueError(f"Unexpected DEIM class index: {label}")
            x1, y1, x2, y2 = map(float, box)
            if x2 > x1 and y2 > y1:
                results.append({"image_id": int(pred["image_id"]), "category_id": mapping[label],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    Path(output_json).write_text(json.dumps(results), encoding="utf-8")
    return results
