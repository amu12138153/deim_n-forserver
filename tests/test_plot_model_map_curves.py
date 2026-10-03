"""Parsing checks for the raw epoch-comparison figure."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from plot_model_map_curves import find_history, load_jsonl, load_yolo_csv, with_gaps


class ModelMapCurveTests(unittest.TestCase):
    def test_deim_and_detr_jsonl_uses_validation_coco_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.txt"
            rows = [
                {"epoch": 0, "test_coco_eval_bbox": [0.2, 0.4]},
                {"epoch": 2, "test_map_50_95": 0.3, "test_map_50": 0.5},
                {"epoch": 2, "test_map_50_95": 0.35, "test_map_50": 0.55},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            history = load_jsonl(path, "ours")
            self.assertEqual(history.points, ((1, 0.4, 0.2), (3, 0.55, 0.35)))
            x, y = with_gaps(history.points, 1)
            self.assertEqual(x, [1, 2, 3])
            self.assertTrue(y[1] != y[1])  # NaN: no invented value at epoch 2.

    def test_yolo_csv_uses_one_based_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"))
                writer.writerow((1, 0.6, 0.3))
            self.assertEqual(load_yolo_csv(path, "yolov8n").points, ((1, 0.6, 0.3),))

    def test_selected_model_must_have_unambiguous_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ours").mkdir()
            (root / "ours" / "log.txt").write_text(
                json.dumps({"epoch": 0, "test_coco_eval_bbox": [0.1, 0.2]}), encoding="utf-8"
            )
            self.assertEqual(find_history("ours", [root]).points, ((1, 0.2, 0.1),))
            with self.assertRaises(FileNotFoundError):
                find_history("missing", [root])


if __name__ == "__main__":
    unittest.main()
