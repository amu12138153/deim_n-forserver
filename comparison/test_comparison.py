"""Fast contract tests; run with python -m unittest discover -s comparison."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import assert_same_categories, evaluation_split, read_coco, read_config, sha256, split_paths
from convert_coco_to_yolo import convert
from external_adapter import _retime_final_phase, load_official_config, prepare_config, repo_workdir
from external_train_bootstrap import patch_standard_coco_ranges
from external_worker import detection
from evaluate_coco import best_f1, evaluate
from summarize_results import summarize
from ultralytics_adapter import to_coco_detection
from train_yolo import common_augmentation_kwargs


class ComparisonContracts(unittest.TestCase):
    def test_bundled_config_uses_project_relative_sources_and_test_directory(self):
        config = read_config()
        self.assertEqual(config["data_root"].name, "fish_dataset622")
        self.assertEqual(config["test_source_split"], "test")
        self.assertEqual(split_paths(config, "test")[0].name, "test")
        self.assertEqual(split_paths(config, "test")[1].name, "coco_detection_test0.json")
        self.assertEqual(config["training_protocol"]["epochs"], 200)
        self.assertEqual(config["training_protocol"]["batch_size"], 8)
        for name, model in config["models"].items():
            self.assertEqual(model["epochs"], 200, name)
            self.assertEqual(model["batch_size"], 8, name)
        for name in ("dfine_n", "dfine_s", "rtdetrv2_s", "rtdetr_r18"):
            with self.subTest(name=name):
                self.assertFalse(Path(config["models"][name]["repo"]).is_absolute())
                self.assertTrue(repo_workdir(config["models"][name]).is_dir())

    def test_shared_augmentation_and_cosine_schedule_are_materialized(self):
        config = read_config()
        yolo = common_augmentation_kwargs(8, 4)
        self.assertEqual(yolo["fliplr"], 0.5)
        self.assertTrue(yolo["cos_lr"])
        self.assertEqual(yolo["warmup_epochs"], 4.0)
        for key in ("mosaic", "mixup", "cutmix", "copy_paste", "hsv_h", "hsv_s",
                    "hsv_v", "translate", "scale", "degrees", "flipud"):
            self.assertEqual(yolo[key], 0.0, key)
        with tempfile.TemporaryDirectory() as temp:
            for name in ("dfine_n", "dfine_s", "rtdetrv2_s", "rtdetr_r18"):
                with self.subTest(name=name):
                    model = config["models"][name]
                    snapshot = prepare_config(model, config, repo_workdir(model),
                                              Path(temp) / name, 200, 8)
                    cfg = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
                    self.assertEqual(cfg["lr_scheduler"],
                                     {"type": "CosineAnnealingLR", "T_max": 197, "eta_min": 0.0})
                    self.assertEqual(cfg["lr_warmup_scheduler"]["warmup_duration"], 224)
                    train = cfg["train_dataloader"]
                    self.assertEqual(train["dataset"]["transforms"]["policy"], None)
                    ops = [item["type"] for item in train["dataset"]["transforms"]["ops"]]
                    self.assertEqual(ops.count("RandomHorizontalFlip"), 1)
                    self.assertTrue(set(ops) <= {"RandomHorizontalFlip", "Resize",
                                                 "SanitizeBoundingBoxes", "ConvertPILImage", "ConvertBoxes"})

    def test_extended_detr_schedules_preserve_final_phase(self):
        for family, original_epochs, transition_epoch, expected_epoch in (
                ("dfine", 160, 148, 188),
                ("dfine", 132, 120, 188),
                ("rtdetrv2", 120, 117, 197)):
            with self.subTest(family=family, original_epochs=original_epochs):
                key = "epochs" if family == "dfine" else "epoches"
                cfg = {key: original_epochs, "train_dataloader": {
                    "dataset": {"transforms": {"policy": {"epoch": transition_epoch}}},
                    "collate_fn": {"stop_epoch": transition_epoch}}}
                _retime_final_phase(cfg, family, 200)
                self.assertEqual(cfg["train_dataloader"]["dataset"]["transforms"]
                                 ["policy"]["epoch"], expected_epoch)
                self.assertEqual(cfg["train_dataloader"]["collate_fn"]["stop_epoch"],
                                 expected_epoch if family == "dfine" else transition_epoch)
                smoke = {key: original_epochs, "train_dataloader": {
                    "dataset": {"transforms": {"policy": {"epoch": transition_epoch}}},
                    "collate_fn": {"stop_epoch": transition_epoch}}}
                _retime_final_phase(smoke, family, 1)
                self.assertEqual(smoke["train_dataloader"]["collate_fn"]["stop_epoch"],
                                 transition_epoch)
                self.assertEqual(smoke["train_dataloader"]["dataset"]["transforms"]
                                 ["policy"]["epoch"], transition_epoch)

    def test_external_evaluator_accepts_standard_ranges_only(self):
        from faster_coco_eval import COCO
        from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
        from faster_coco_eval.utils.pytorch import coco_eval as pytorch_eval

        original = pytorch_eval.COCOeval_faster
        try:
            if not patch_standard_coco_ranges():
                self.skipTest("Installed faster-coco-eval already accepts ranges")
            coco = COCO()
            coco.dataset = {"images": [], "annotations": [],
                            "categories": [{"id": 0, "name": "normal"}]}
            coco.createIndex()
            evaluator = FasterCocoEvaluator(coco, ["bbox"])
            self.assertEqual(evaluator.coco_eval["bbox"].params.areaRngLbl,
                             ["all", "small", "medium", "large"])
            evaluator.cleanup()
            with self.assertRaises(ValueError):
                pytorch_eval.COCOeval_faster(coco, iouType="bbox", ranges={"small": [0, 1]})
        finally:
            pytorch_eval.COCOeval_faster = original

    def test_external_config_flatten_and_fixed_comparison_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.yml"
            source = root / "model.yml"
            base.write_text(yaml.safe_dump({
                "num_classes": 80, "remap_mscoco_category": True,
                "PResNet": {"pretrained": True},
                "train_dataloader": {"dataset": {"transforms": {"ops": [{"type": "Resize", "size": [800, 800]}]}},
                                     "collate_fn": {"scales": [480, 800]}, "total_batch_size": 16},
                "val_dataloader": {"dataset": {"transforms": {"ops": [{"type": "Resize", "size": [800, 800]}]}}}
            }), encoding="utf-8")
            source.write_text("__include__: [base.yml]\nmodel: RTDETR\n", encoding="utf-8")
            for split in ("train", "val"):
                folder = root / split
                folder.mkdir()
                (folder / f"{split}.json").write_text("{}", encoding="utf-8")
            config = {"data_root": root, "annotations": {"train": "train.json", "val": "val.json"}}
            model = {"config": "model.yml", "family": "rtdetr", "pretrained": False}
            snapshot = prepare_config(model, config, root, root / "output", 72, 2)
            merged = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
            self.assertEqual(merged["num_classes"], 2)
            self.assertFalse(merged["remap_mscoco_category"])
            self.assertFalse(merged["PResNet"]["pretrained"])
            self.assertEqual(merged["epoches"], 72)
            self.assertEqual(merged["train_dataloader"]["collate_fn"]["scales"], [640])
            self.assertEqual(merged["train_dataloader"]["total_batch_size"], 2)
            self.assertEqual(merged["val_dataloader"]["total_batch_size"], 2)
            fast_snapshot = prepare_config(model, config, root, root / "output8", 72, 8)
            fast = yaml.safe_load(fast_snapshot.read_text(encoding="utf-8"))
            self.assertEqual(fast["train_dataloader"]["total_batch_size"], 8)
            self.assertEqual(fast["val_dataloader"]["total_batch_size"], 8)
            self.assertEqual(merged["val_dataloader"]["dataset"]["transforms"]["ops"][0]["size"], [640, 640])
            self.assertEqual(load_official_config(source)["num_classes"], 80)
            self.assertIn("__include__", source.read_text(encoding="utf-8"))

    def test_external_detection_uses_original_category_id(self):
        row = detection(4, 1, [2, 3, 12, 13], 0.7, {0: 1, 1: 2})
        self.assertEqual(row["category_id"], 2)
        self.assertEqual(row["bbox"], [2.0, 3.0, 10.0, 10.0])
        with self.assertRaises(ValueError):
            detection(4, 9, [2, 3, 12, 13], 0.7, {0: 1, 1: 2})

    def test_dfine_multiscale_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {"num_classes": 80, "remap_mscoco_category": True,
                       "HGNetv2": {"pretrained": True},
                       "train_dataloader": {"dataset": {"transforms": {"ops": [{"type": "Resize", "size": [640, 640]}]}},
                                            "collate_fn": {"base_size_repeat": 3}},
                       "val_dataloader": {"dataset": {"transforms": {"ops": [{"type": "Resize", "size": [640, 640]}]}}}}
            (root / "model.yml").write_text(yaml.safe_dump(payload), encoding="utf-8")
            for split in ("train", "val"):
                folder = root / split
                folder.mkdir()
                (folder / f"{split}.json").write_text("{}", encoding="utf-8")
            config = {"data_root": root, "annotations": {"train": "train.json", "val": "val.json"}}
            model = {"config": "model.yml", "family": "dfine", "pretrained": True}
            snapshot = prepare_config(model, config, root, root / "output", 160, 2)
            merged = yaml.safe_load(snapshot.read_text(encoding="utf-8"))
            self.assertIsNone(merged["train_dataloader"]["collate_fn"]["base_size_repeat"])
            self.assertEqual(merged["train_dataloader"]["collate_fn"]["base_size"], 640)
            self.assertEqual(merged["epochs"], 160)

    def test_coco_category_id_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "annotations.json"
            for ids in ((0, 1), (1, 2)):
                path.write_text(json.dumps({"images": [], "annotations": [], "categories": [
                    {"id": ids[0], "name": "normal"}, {"id": ids[1], "name": "hypoxia"}]}), encoding="utf-8")
                self.assertEqual(read_coco(path)[1], {0: ids[0], 1: ids[1]})
                first = to_coco_detection(7, 0, [1, 2, 4, 6], 0.9, {0: ids[0], 1: ids[1]})
                second = to_coco_detection(7, 1, [1, 2, 4, 6], 0.8, {0: ids[0], 1: ids[1]})
                self.assertEqual((first["category_id"], second["category_id"]), ids)
                self.assertEqual(first["bbox"], [1, 2, 3, 4])

    def test_conversion_preserves_split_and_normalizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "val", "test"):
                directory = root / split
                directory.mkdir()
                (directory / "fish.jpg").write_bytes(b"image-placeholder")
                data = {"images": [{"id": 7, "file_name": "fish.jpg", "width": 100, "height": 200}],
                        "annotations": [{"id": 1, "image_id": 7, "category_id": 2,
                                         "bbox": [10, 20, 40, 60], "iscrowd": 0}],
                        "categories": [{"id": 1, "name": "normal"}, {"id": 2, "name": "hypoxia"}]}
                (directory / f"{split}.json").write_text(json.dumps(data), encoding="utf-8")
            config = {"data_root": root, "annotations": {s: f"{s}.json" for s in ("train", "val", "test")}}
            self.assertEqual(evaluation_split(config), "validation_reused_as_test")
            convert(config, root / "out")
            for split in ("train", "val", "test"):
                self.assertEqual(len(list((root / "out" / "images" / split).glob("*.jpg"))), 1)
                tokens = (root / "out" / "labels" / split / "fish.txt").read_text().split()
                self.assertEqual(int(tokens[0]), 1)
                self.assertEqual([float(x) for x in tokens[1:]], [0.3, 0.25, 0.4, 0.3])
            self.assertEqual(convert(config, root / "out"), root / "out" / "fish.yaml")
            other_root = root / "other_dataset"
            for split in ("train", "val", "test"):
                shutil.copytree(root / split, other_root / split)
            other_config = {**config, "data_root": other_root}
            with self.assertRaisesRegex(FileExistsError, "does not match the selected dataset"):
                convert(other_config, root / "out")
            self.assertTrue(convert(other_config, root / "new_out").is_file())

    def test_train_only_conversion_without_test_annotation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for split in ("train", "val"):
                directory = root / split
                directory.mkdir()
                (directory / "fish.jpg").write_bytes(b"image-placeholder")
                annotation = {"images": [{"id": 1, "file_name": "fish.jpg", "width": 100, "height": 100}],
                              "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                                               "bbox": [10, 10, 20, 20]}],
                              "categories": [{"id": 0, "name": "normal"},
                                             {"id": 1, "name": "hypoxia"}]}
                (directory / f"{split}.json").write_text(json.dumps(annotation), encoding="utf-8")
            config = {"data_root": root, "annotations": {s: f"{s}.json" for s in ("train", "val", "test")}}
            assert_same_categories(config, ("train", "val"))
            with self.assertRaises(FileNotFoundError):
                assert_same_categories(config)
            fish_yaml = convert(config, root / "out", splits=("train", "val"))
            self.assertNotIn("test", yaml.safe_load(fish_yaml.read_text(encoding="utf-8")))
            manifest = json.loads((root / "out" / "conversion_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["annotation_sha256"]), {"train", "val"})

    def test_evaluation_can_reuse_validation_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            val_dir = root / "val"
            val_dir.mkdir()
            annotation = {"images": [], "annotations": [],
                          "categories": [{"id": 0, "name": "normal"},
                                         {"id": 1, "name": "hypoxia"}]}
            (val_dir / "val.json").write_text(json.dumps(annotation), encoding="utf-8")
            config = {"data_root": root, "test_source_split": "val",
                      "annotations": {"val": "val.json", "test": "val.json"}}
            self.assertEqual(split_paths(config, "test"), split_paths(config, "val"))
            self.assertEqual(evaluation_split(config), "validation_reused_as_test")

    def test_best_f1_and_cocoeval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gt = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20],
                                   "area": 400, "iscrowd": 0}],
                  "categories": [{"id": 1, "name": "normal"}, {"id": 2, "name": "hypoxia"}]}
            pred = [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
                    {"image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "score": 0.1}]
            self.assertEqual(best_f1(gt, pred)["Best_Confidence"], 0.9)
            gt_path, pred_path = root / "gt.json", root / "pred.json"
            gt_path.write_text(json.dumps(gt), encoding="utf-8")
            pred_path.write_text(json.dumps(pred), encoding="utf-8")
            result = evaluate(gt_path, pred_path, root)
            self.assertEqual(result["F1"], 1.0)
            self.assertGreater(result["mAP50"], 0.0)
            self.assertEqual(result["AP75"], result["mAP50"])

    def test_summary_rejects_mixed_test_sets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for model, digest in (("a", "first"), ("b", "second")):
                folder = root / model
                folder.mkdir()
                checkpoint = folder / "best.pt"
                checkpoint.write_bytes(model.encode())
                checkpoint_digest = sha256(checkpoint)
                (folder / "metrics.json").write_text(json.dumps({
                    "checkpoint_sha256": checkpoint_digest, "test_annotation_sha256": digest}), encoding="utf-8")
                (folder / "profile.json").write_text(json.dumps({
                    "checkpoint_sha256": checkpoint_digest}), encoding="utf-8")
                (folder / "experiment_summary.json").write_text(json.dumps({
                    "Model": model, "Status": "complete", "Test_Annotation_SHA256": digest,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_digest}), encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
