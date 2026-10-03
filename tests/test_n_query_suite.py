"""Contracts for the HGNetv2-N fish ablation suite and independent test set."""

import copy
import csv
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core.yaml_utils import load_config
from engine.core.yaml_config import YAMLConfig
from run_ablation import GROUPS


class NQuerySuite(unittest.TestCase):
    def test_all_configs_share_the_n_backbone_data_and_schedule(self):
        configs = {
            name: load_config(str(ROOT / "configs/deimv2/ablation_n" / f"{name}.yml"), {})
            for name in sum(GROUPS.values(), [])
        }
        reference = copy.deepcopy(configs["M4_full"])
        for name, config in configs.items():
            with self.subTest(name=name):
                self.assertEqual(config["epoches"], 200)
                self.assertEqual(config["train_dataloader"]["total_batch_size"], 8)
                self.assertEqual(config["lr_scheduler"],
                                 {"type": "CosineAnnealingLR", "T_max": 197, "eta_min": 0.0})
                self.assertEqual(config["lr_warmup_scheduler"]["warmup_duration"], 224)
                self.assertEqual(config["num_classes"], 2)
                self.assertEqual(config["HGNetv2"]["name"], "B0")
                self.assertEqual(config["HGNetv2"]["return_idx"], [2, 3])
                self.assertFalse(config["HGNetv2"]["use_water_quality"])
                self.assertEqual(config["DEIMTransformer"]["num_layers"], 3)
                self.assertEqual(config["DEIMTransformer"]["feat_channels"], [128, 128])
                for split in ("train", "val", "test"):
                    self.assertEqual(config[f"{split}_dataloader"], reference[f"{split}_dataloader"])
                self.assertEqual(config["HybridEncoder"], reference["HybridEncoder"])
                self.assertEqual(config["HGNetv2"], reference["HGNetv2"])
                train = config["train_dataloader"]
                ops = [item["type"] for item in train["dataset"]["transforms"]["ops"]]
                self.assertNotIn("Mosaic", ops)
                self.assertEqual(ops, ["RandomHorizontalFlip", "Resize",
                                       "SanitizeBoundingBoxes", "ConvertPILImage", "ConvertBoxes"])
                self.assertIsNone(train["dataset"]["transforms"]["policy"])
                self.assertEqual(train["collate_fn"]["mixup_prob"], 0.0)
                self.assertEqual(train["collate_fn"]["copyblend_prob"], 0.0)
                self.assertEqual(train["collate_fn"]["stop_epoch"], 200)

    def test_main_and_component_ablations_exclude_turbidity(self):
        full = load_config(str(ROOT / "configs/deimv2/ablation_n/M4_full.yml"), {})
        self.assertEqual(full["DEIMTransformer"]["water_features"], ["temperature", "do", "ph"])
        self.assertEqual(full["DEIMTransformer"]["query_water_cross_attn"], True)
        for name in GROUPS["modules"][1:]:
            with self.subTest(name=name):
                config = load_config(str(ROOT / "configs/deimv2/ablation_n" / f"{name}.yml"), {})
                self.assertEqual(config["DEIMTransformer"]["water_features"],
                                 ["temperature", "do", "ph"])
        four = load_config(str(ROOT / "configs/deimv2/ablation_n/V1_T_DO_pH_Tur.yml"), {})
        self.assertEqual(four["DEIMTransformer"]["water_features"],
                         ["temperature", "do", "ph", "turbidity"])

    def test_test_annotations_and_sensors_are_aligned(self):
        datasets = {}
        for split in ("train", "val", "test"):
            folder = ROOT / "fish_dataset622" / split
            data = json.loads((folder / f"coco_detection_{split}0.json").read_text(encoding="utf-8"))
            self.assertEqual({int(x["id"]): x["name"] for x in data["categories"]},
                             {0: "normal", 1: "hypoxia"})
            self.assertEqual({int(x["category_id"]) for x in data["annotations"]}, {0, 1})
            images = {x["file_name"] for x in data["images"]}
            with (folder / f"{split}.csv").open(encoding="utf-8-sig", newline="") as stream:
                sensors = {row["file_name"] for row in csv.DictReader(stream)}
            self.assertEqual(images, sensors)
            self.assertTrue(all((folder / name).is_file() for name in images))
            datasets[split] = images
        self.assertEqual(len(datasets["test"]), 151)
        self.assertFalse(datasets["test"] & datasets["train"])
        self.assertFalse(datasets["test"] & datasets["val"])

    def test_augmented_image_keeps_its_own_sensor_reading(self):
        dataset = YAMLConfig(str(ROOT / "configs/deimv2/ablation_n/M4_full.yml")).train_dataloader.dataset
        expected_water = dataset.get_water_quality(0)
        for epoch in (0, 4, 100, 188, 199):
            with self.subTest(epoch=epoch):
                dataset.set_epoch(epoch)
                image, target = dataset[0]
                self.assertEqual(tuple(image.shape), (3, 640, 640))
                self.assertEqual(target["image_id"].numel(), 1)
                self.assertTrue(target["water_quality"].equal(expected_water))


if __name__ == "__main__":
    unittest.main()
