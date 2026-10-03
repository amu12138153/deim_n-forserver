"""Train isolated water-quality fusion at LiteEncoder or HGNetv2.

Examples (run from any working directory):
    python train_water_fusion.py --variant encoder --check
    python train_water_fusion.py --variant hgnet --check
    python train_water_fusion.py --variant both

Each experiment starts from scratch, uses the existing training/evaluator code,
and refuses to write into a nonempty result directory.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from engine.core import YAMLConfig


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "encoder": ROOT / "configs/deimv2/ablation/water_encoder_only.yml",
    "hgnet": ROOT / "configs/deimv2/ablation/water_hgnet_only.yml",
}


def check_variant(name: str, forward: bool = False) -> None:
    config = YAMLConfig(str(CONFIGS[name]))
    data = config.yaml_cfg
    expected = {"encoder": (False, True), "hgnet": (True, False)}[name]
    actual = (data["HGNetv2"]["use_water_quality"], data["LiteEncoder"]["use_water_quality"])
    if actual != expected or data["DEIMTransformer"]["use_water_quality"]:
        raise ValueError(f"{name}: fusion switches are not isolated: {actual}")

    for split in ("train", "val"):
        dataset = data[f"{split}_dataloader"]["dataset"]
        if not dataset.get("use_water_quality"):
            raise ValueError(f"{name}: {split} dataset does not provide water_quality")
        for key in ("img_folder", "ann_file", "sensor_csv"):
            path = Path(dataset[key])
            if not path.exists():
                raise FileNotFoundError(f"{name}: missing {split} {key}: {path}")

    # Build the actual model and optimizer, catching registration, shape and
    # parameter-group problems before a potentially long training run.
    model = config.model
    optimizer = config.optimizer
    location = model.encoder if name == "encoder" else model.backbone
    if not hasattr(location, "water_fusion"):
        raise RuntimeError(f"{name}: expected water_fusion module was not built")
    count = sum(p.numel() for p in location.water_fusion.parameters() if p.requires_grad)
    if not count:
        raise RuntimeError(f"{name}: water_fusion has no trainable parameters")
    print(f"{name}: OK; water-fusion parameters={count:,}; optimizer groups={len(optimizer.param_groups)}")

    if forward:
        image, target = config.val_dataloader.dataset[0]
        water = target["water_quality"].unsqueeze(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        with torch.inference_mode():
            model(image.unsqueeze(0).to(device), water=water.to(device))
        print(f"{name}: validation-sample forward OK on {device}; water shape={tuple(water.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("encoder", "hgnet", "both"), required=True)
    parser.add_argument("--check", action="store_true", help="validate config and model; do not train")
    parser.add_argument("--check-forward", action="store_true", help="also run one real validation sample; do not train")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/water_fusion_locations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="passed to train.py, e.g. cuda:0")
    args = parser.parse_args()

    variants = ("encoder", "hgnet") if args.variant == "both" else (args.variant,)
    for name in variants:
        check_variant(name, forward=args.check_forward)
    if args.check or args.check_forward:
        return

    output_root = args.output_root.resolve()
    if output_root == ROOT or ROOT not in output_root.parents:
        raise ValueError("--output-root must be a subdirectory of this project")
    destinations = {name: output_root / name for name in variants}
    for destination in destinations.values():
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError(f"Refusing to overwrite existing results: {destination}")

    for name in variants:
        command = [
            sys.executable, str(ROOT / "train.py"),
            "-c", str(CONFIGS[name]),
            "--output-dir", str(destinations[name]),
            "--seed", str(args.seed),
            "--use-amp",
        ]
        if args.device:
            command.extend(("--device", args.device))
        print("Starting:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
