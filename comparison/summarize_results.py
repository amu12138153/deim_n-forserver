"""Build one CSV only from completed experiment_summary.json files."""
import argparse
import csv
import json
from pathlib import Path

from common import read_config, sha256

FIELDS = ["Model", "Precision", "Recall", "F1", "mAP50", "mAP50_95", "AP75",
          "Params_M", "GFLOPs", "FPS", "Pretrained", "Epochs", "Batch_Size",
          "Best_Confidence", "Evaluation_Split", "Test_Annotation_SHA256", "Status"]


def summarize(output_root, out=None):
    output_root = Path(output_root).resolve()
    rows = []
    test_hashes = set()
    for path in sorted(output_root.glob("*/experiment_summary.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("Status") != "complete":
            continue
        checkpoint = Path(item.get("checkpoint", ""))
        metric_path, profile_path = path.parent / "metrics.json", path.parent / "profile.json"
        if not checkpoint.is_file() or not metric_path.is_file() or not profile_path.is_file():
            continue
        digest = sha256(checkpoint)
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if (item.get("checkpoint_sha256") != digest or
                metrics.get("checkpoint_sha256") != digest or
                profile.get("checkpoint_sha256") != digest or
                item.get("Test_Annotation_SHA256") != metrics.get("test_annotation_sha256")):
            continue
        test_hashes.add(item["Test_Annotation_SHA256"])
        rows.append({field: item.get(field) for field in FIELDS})
    if len(test_hashes) > 1:
        raise ValueError("Different test annotation hashes in comparison summaries; refusing mixed CSV")
    target = Path(out).resolve() if out else output_root / "comparison_results.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target, len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", help="Defaults to output_root in comparison_models.yml")
    parser.add_argument("--out")
    args = parser.parse_args()
    print(summarize(args.output_root or read_config()["output_root"], args.out))
