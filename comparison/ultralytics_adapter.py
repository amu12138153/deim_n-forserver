"""Ultralytics predictions mapped back to the source COCO image/category IDs."""
import json
from pathlib import Path

from common import read_coco


def to_coco_detection(image_id, yolo_class, xyxy, score, mapping):
    x1, y1, x2, y2 = map(float, xyxy)
    if x2 <= x1 or y2 <= y1:
        return None
    return {"image_id": int(image_id), "category_id": mapping[int(yolo_class)],
            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)}


def predict(checkpoint, ann_file, image_dir, output_json, device=None, imgsz=640):
    from ultralytics import YOLO
    data, mapping = read_coco(ann_file)
    model = YOLO(str(checkpoint))
    model_names = {int(k): str(v).lower() for k, v in model.names.items()}
    if model_names != {0: "normal", 1: "hypoxia"}:
        raise ValueError(f"Checkpoint class names do not match comparison classes: {model_names}")
    predictions = []
    for image in data["images"]:
        source = Path(image_dir) / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        kwargs = {"source": str(source), "imgsz": imgsz, "conf": 0.001,
                  "iou": 0.7, "max_det": 300, "verbose": False}
        if device is not None:
            kwargs["device"] = device
        result = model.predict(**kwargs)[0]
        if result.boxes is None:
            continue
        for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            detection = to_coco_detection(image["id"], cls, box, score, mapping)
            if detection:
                predictions.append(detection)
    Path(output_json).write_text(json.dumps(predictions), encoding="utf-8")
    return predictions
