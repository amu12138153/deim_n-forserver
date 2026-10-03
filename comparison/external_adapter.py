"""Official D-FINE / RT-DETR subprocess bridge; never changes the shared evaluator."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from common import path_from_root, split_paths


def _merge(target, source):
    for key, value in source.items():
        if isinstance(target.get(key), dict) and isinstance(value, dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def load_official_config(path, seen=None):
    """Flatten upstream __include__ files with upstream's recursive merge semantics."""
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Cyclic YAML include: {path}")
    seen.add(path)
    try:
        source = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(source, dict):
            raise ValueError(f"Expected YAML mapping: {path}")
        result = {}
        for item in source.pop("__include__", []):
            include = Path(item).expanduser()
            if not include.is_absolute():
                include = path.parent / include
            _merge(result, load_official_config(include, seen))
        return _merge(result, source)
    finally:
        seen.remove(path)


def repo_workdir(model, repo_override=None):
    repo = repo_override or model.get("repo")
    if not repo:
        return None
    root = path_from_root(repo)
    workdir = (root / model["repo_subdir"]).resolve()
    if not workdir.is_dir():
        raise FileNotFoundError(f"External repository workdir: {workdir}")
    for key in ("config", "train_entry"):
        if not (workdir / model[key]).is_file():
            raise FileNotFoundError(f"Official {key}: {workdir / model[key]}")
    if not (workdir / "src" / "core").is_dir():
        raise FileNotFoundError(f"Official src/core: {workdir}")
    return workdir


def _retime_final_phase(cfg, family, epochs):
    """Keep the upstream no-augmentation/final-stage duration when extending training."""
    source_epochs = cfg.get("epochs" if family == "dfine" else "epoches")
    if not isinstance(source_epochs, int) or epochs <= source_epochs:
        return
    extra_epochs = epochs - source_epochs
    loader = cfg["train_dataloader"]
    policy = loader["dataset"]["transforms"].get("policy", {})
    policy_epoch = policy.get("epoch")
    if isinstance(policy_epoch, int) and 0 < policy_epoch < source_epochs:
        policy["epoch"] = policy_epoch + extra_epochs
    if family == "dfine":
        collate = loader["collate_fn"]
        stop_epoch = collate.get("stop_epoch")
        if isinstance(stop_epoch, int) and 0 < stop_epoch < source_epochs:
            collate["stop_epoch"] = stop_epoch + extra_epochs


def prepare_config(model, config, workdir, model_dir, epochs, batch_size):
    source_path = workdir / model["config"]
    cfg = load_official_config(source_path)
    if cfg.get("num_classes") != 80 or "remap_mscoco_category" not in cfg:
        raise ValueError("Unexpected official COCO config; inspect upstream changes before training")
    cfg["num_classes"] = 2
    cfg["remap_mscoco_category"] = False
    backbone = "HGNetv2" if model["family"] == "dfine" else "PResNet"
    if backbone not in cfg or "pretrained" not in cfg[backbone]:
        raise ValueError(f"Official {backbone} pretrained setting not found")
    cfg[backbone]["pretrained"] = bool(model["pretrained"])
    cfg["output_dir"] = str(model_dir)
    cfg["eval_spatial_size"] = [640, 640]
    protocol = config.get("training_protocol")
    if protocol is None:
        _retime_final_phase(cfg, model["family"], epochs)
    cfg["epochs" if model["family"] == "dfine" else "epoches"] = epochs
    for split in ("train", "val"):
        loader = cfg[f"{split}_dataloader"]
        image_dir, ann = split_paths(config, split)
        loader["dataset"]["img_folder"] = str(image_dir)
        loader["dataset"]["ann_file"] = str(ann)
        loader["total_batch_size"] = batch_size if split == "train" else min(batch_size, 8)
        loader["num_workers"] = 0
        ops = loader["dataset"]["transforms"]["ops"]
        resizes = [op for op in ops if op.get("type") == "Resize"]
        if len(resizes) != 1:
            raise ValueError(f"Expected one official Resize in {split} transforms")
        resizes[0]["size"] = [640, 640]
        if split == "train" and protocol is not None:
            # Shared single-image policy across DEIM, D-FINE, RT-DETR and YOLO.
            keep = {"RandomHorizontalFlip", "Resize", "SanitizeBoundingBoxes",
                    "ConvertPILImage", "ConvertBoxes"}
            loader["dataset"]["transforms"]["ops"] = [op for op in ops if op["type"] in keep]
            flips = [op for op in loader["dataset"]["transforms"]["ops"]
                     if op["type"] == "RandomHorizontalFlip"]
            if len(flips) != 1:
                raise ValueError("Expected one official RandomHorizontalFlip")
            flips[0]["p"] = 0.5
            loader["dataset"]["transforms"]["policy"] = None
    collate = cfg["train_dataloader"]["collate_fn"]
    if model["family"] == "dfine":
        if "base_size_repeat" not in collate:
            raise ValueError("D-FINE collate interface changed")
        collate["base_size"] = 640
        collate["base_size_repeat"] = None
    else:
        if "scales" not in collate:
            raise ValueError("RT-DETR collate interface changed")
        collate["scales"] = [640]
    if protocol is not None:
        collate["stop_epoch"] = epochs
        warmup_epochs = min(int(protocol["warmup_epochs"]), max(epochs - 1, 0))
        steps_per_epoch = (len(json.loads(split_paths(config, "train")[1].read_text(encoding="utf-8"))["images"])
                           + batch_size - 1) // batch_size
        cfg["lr_scheduler"] = {"type": "CosineAnnealingLR",
                               "T_max": max(1, epochs - warmup_epochs + 1), "eta_min": 0.0}
        cfg["lr_warmup_scheduler"] = {"type": "LinearWarmup",
                                      "warmup_duration": warmup_epochs * steps_per_epoch}
        cfg["use_amp"] = True
        cfg["use_ema"] = True
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot = model_dir / "config_used.yml"
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return snapshot


def best_checkpoint(model, model_dir):
    candidates = (["best_stg2.pth", "best_stg1.pth"] if model["family"] == "dfine"
                  else [model["checkpoint_name"]])
    return next((model_dir / name for name in candidates if (model_dir / name).is_file()),
                model_dir / candidates[-1])


def train(model, config, workdir, model_dir, epochs, batch_size, device, python):
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing training run: {model_dir}")
    snapshot = prepare_config(model, config, workdir, model_dir, epochs, batch_size)
    command = [python, str(Path(__file__).with_name("external_train_bootstrap.py")),
               "--train-entry", str(workdir / model["train_entry"]), "-c", str(snapshot),
               "--output-dir", str(model_dir), "--seed", "0"]
    if model.get("use_amp"):
        command.append("--use-amp")
    if device:
        command += ["--device", device]
    subprocess.run(command, cwd=workdir, check=True)
    checkpoint = best_checkpoint(model, model_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official training produced no validation-best checkpoint: {checkpoint}")
    revision = (subprocess.run(["git", "rev-parse", "HEAD"], cwd=workdir,
                               capture_output=True, text=True, check=False)
                if shutil.which("git") else None)
    (model_dir / "training_metadata.json").write_text(json.dumps({
        "pretrained": model["pretrained"], "pretrained_scope": model["pretrained_scope"],
        "epochs": epochs, "batch_size": batch_size, "source_model": model["config"],
        "data": str(config["data_root"]), "image_size": 640,
        "repo_workdir": str(workdir),
        "repo_commit": revision.stdout.strip() if revision and revision.returncode == 0 else None,
        "checkpoint": str(checkpoint), "training_protocol": config.get("training_protocol")
    }, indent=2), encoding="utf-8")
    return checkpoint


def run_worker(action, workdir, snapshot, checkpoint, ann_file, image_dir, out, device, python):
    command = [python, str(Path(__file__).with_name("external_worker.py")),
               "--action", action, "--repo-workdir", str(workdir),
               "--config", str(snapshot), "--checkpoint", str(checkpoint),
               "--ann", str(ann_file), "--image-dir", str(image_dir),
               "--out", str(out), "--device", device or ("cpu" if action == "profile" else "cuda"),
               "--imgsz", "640"]
    subprocess.run(command, cwd=workdir, check=True)
