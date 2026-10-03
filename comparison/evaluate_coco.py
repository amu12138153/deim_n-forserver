"""One COCOeval and IoU=0.5 best-F1 evaluator for every model."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from common import read_coco, sha256


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    overlap = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def best_f1(gt_data, predictions, iou_threshold=0.5):
    """Greedy confidence-ordered, category-aware one-to-one matching.

    Scan every distinct prediction score. Tied scores enter together, so the
    result is independent of an arbitrary confidence-grid resolution.
    """
    gt = defaultdict(list)
    for ann in gt_data["annotations"]:
        if not ann.get("iscrowd", 0) and not ann.get("ignore", 0):
            gt[(int(ann["image_id"]), int(ann["category_id"]))].append(ann["bbox"])
    used = {key: set() for key in gt}
    ordered = sorted(predictions, key=lambda x: -float(x["score"]))
    total_gt = sum(map(len, gt.values()))
    tp = fp = 0
    best = {"Precision": 0.0, "Recall": 0.0, "F1": 0.0, "Best_Confidence": 1.0, "TP": 0, "FP": 0, "FN": total_gt}
    for index, pred in enumerate(ordered):
        key = (int(pred["image_id"]), int(pred["category_id"]))
        candidates = gt.get(key, [])
        matches = [(iou_xywh(pred["bbox"], box), j) for j, box in enumerate(candidates) if j not in used.get(key, set())]
        if matches and max(matches)[0] >= iou_threshold:
            used[key].add(max(matches)[1])
            tp += 1
        else:
            fp += 1
        if index + 1 < len(ordered) and float(ordered[index + 1]["score"]) == float(pred["score"]):
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total_gt if total_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best["F1"]:
            best = {"Precision": precision, "Recall": recall, "F1": f1,
                    "Best_Confidence": float(pred["score"]), "TP": tp, "FP": fp, "FN": total_gt - tp}
    return best


def evaluate(gt_path, pred_path, out_dir=None):
    gt_path, pred_path = Path(gt_path).resolve(), Path(pred_path).resolve()
    gt_data, mapping = read_coco(gt_path)
    predictions = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("COCO predictions must be a JSON list")
    image_ids = {int(image["id"]) for image in gt_data["images"]}
    categories = set(mapping.values())
    for index, pred in enumerate(predictions):
        if not {"image_id", "category_id", "bbox", "score"} <= pred.keys():
            raise ValueError(f"Prediction {index} missing required fields")
        if int(pred["image_id"]) not in image_ids or int(pred["category_id"]) not in categories:
            raise ValueError(f"Prediction {index} has image/category outside the test annotation")
        box = pred["bbox"]
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in box) or float(box[2]) <= 0 or float(box[3]) <= 0:
            raise ValueError(f"Prediction {index} has invalid xywh bbox")
        if not math.isfinite(float(pred["score"])) or not 0 <= float(pred["score"]) <= 1:
            raise ValueError(f"Prediction {index} has invalid confidence")
    import faster_coco_eval
    faster_coco_eval.init_as_pycocotools()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(str(gt_path))
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {"images": gt_data["images"], "categories": gt_data["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = sorted(categories)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    pr = best_f1(gt_data, predictions)
    metrics = {**pr, "mAP50": float(evaluator.stats[1]), "mAP50_95": float(evaluator.stats[0]),
               "AP75": float(evaluator.stats[2]), "test_annotation": str(gt_path),
               "test_annotation_sha256": sha256(gt_path), "images": len(image_ids),
               "predictions": len(predictions), "evaluator": "faster_coco_eval COCOeval bbox",
               "prf1_definition": "IoU=0.5, confidence sweep over unique scores, class-aware greedy one-to-one matching; ignore crowd/ignored GT"}
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.gt, args.pred, args.out), indent=2))
