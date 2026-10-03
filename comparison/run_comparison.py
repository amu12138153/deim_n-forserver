"""Train, predict/evaluate and profile comparison models with one evaluator."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from common import ROOT, assert_same_categories, evaluation_split, path_from_root, read_config, sha256, split_paths
from convert_coco_to_yolo import convert
from evaluate_coco import evaluate
from summarize_results import summarize


def _resolved_checkpoint(model, model_dir):
    trained = model_dir / "weights" / "best.pt" if model["type"] == "ultralytics" else model_dir / "best_stg1.pth"
    if trained.is_file():
        return trained
    fallback = model.get("checkpoint")
    return path_from_root(fallback) if fallback else trained


def _deim_config(model, model_dir):
    snapshot = model_dir / "config_used.yml"
    if snapshot.is_file():
        return snapshot
    return path_from_root(model.get("checkpoint_config") or model["config"])


def train_deim(model, model_dir, config, epochs, batch_size, device):
    from engine.core.yaml_utils import load_config
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing training run: {model_dir}")
    model_dir.mkdir(parents=True)
    cfg = load_config(str(path_from_root(model["config"])), {})
    for split in ("train", "val"):
        image_dir, ann = split_paths(config, split)
        dataset = cfg[f"{split}_dataloader"]["dataset"]
        dataset["img_folder"] = str(image_dir)
        dataset["ann_file"] = str(ann)
        if dataset.get("use_water_quality"):
            sensor = image_dir / f"{split}.csv"
            if not sensor.is_file():
                raise FileNotFoundError(sensor)
            dataset["sensor_csv"] = str(sensor)
    cfg["epoches"] = epochs
    cfg["train_dataloader"]["total_batch_size"] = batch_size
    protocol = config.get("training_protocol")
    if protocol is not None:
        warmup_epochs = min(int(protocol["warmup_epochs"]), max(epochs - 1, 0))
        train_images = len(json.loads(split_paths(config, "train")[1].read_text(encoding="utf-8"))["images"])
        steps_per_epoch = (train_images + batch_size - 1) // batch_size
        cfg["lr_scheduler"] = {"type": "CosineAnnealingLR",
                               "T_max": max(1, epochs - warmup_epochs + 1), "eta_min": 0.0}
        cfg["lr_warmup_scheduler"] = {"type": "LinearWarmup",
                                      "warmup_duration": warmup_epochs * steps_per_epoch}
        cfg["train_dataloader"]["collate_fn"]["stop_epoch"] = epochs
        cfg["DEIMCriterion"]["matcher"]["matcher_change_epoch"] = max(0, epochs - 24)
        cfg["use_amp"] = True
        cfg["use_ema"] = True
    cfg["output_dir"] = str(model_dir)
    cfg.pop("__include__", None)
    snapshot = model_dir / "config_used.yml"
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    command = [sys.executable, str(ROOT / "train.py"), "-c", str(snapshot), "--output-dir", str(model_dir),
               "--seed", str(protocol["seed"] if protocol else 0)]
    if device:
        command += ["--device", device]
    subprocess.run(command, cwd=ROOT, check=True)
    checkpoint = model_dir / "best_stg1.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    (model_dir / "training_metadata.json").write_text(json.dumps({
        "pretrained": bool(cfg["HGNetv2"].get("pretrained", False)), "epochs": epochs,
        "batch_size": batch_size, "source_model": model["config"],
        "data": str(config["data_root"]), "image_size": config["image_size"],
        "training_protocol": protocol,
    }, indent=2), encoding="utf-8")
    return checkpoint


def complete_summary(name, model, model_dir, checkpoint, config, metrics, profile):
    _, ann_file = split_paths(config, "test")
    if metrics["test_annotation_sha256"] != sha256(ann_file):
        raise ValueError("Metrics annotation differs from current test annotation")
    current_checkpoint_hash = sha256(checkpoint)
    if metrics.get("checkpoint_sha256") != current_checkpoint_hash or profile.get("checkpoint_sha256") != current_checkpoint_hash:
        raise ValueError("Metrics/profile were produced from a different checkpoint; rerun --eval --profile")
    speed_path = model_dir / "speed.json"
    speed = json.loads(speed_path.read_text(encoding="utf-8")) if speed_path.is_file() else None
    if speed and speed.get("checkpoint_sha256") != current_checkpoint_hash:
        speed = None
    summary = {"Model": model["display_name"], "model_key": name, "Status": "complete",
               "Evaluation_Split": evaluation_split(config),
               "input_size": int(config["image_size"]), "Pretrained": model.get("pretrained"),
               "pretrained_scope": model.get("pretrained_scope"),
               "Epochs": model.get("epochs"), "Batch_Size": model.get("batch_size"),
               "checkpoint": str(checkpoint), "checkpoint_sha256": current_checkpoint_hash,
               "test_annotation": str(ann_file),
               "Test_Annotation_SHA256": metrics["test_annotation_sha256"],
               "Precision": metrics["Precision"], "Recall": metrics["Recall"], "F1": metrics["F1"],
               "Best_Confidence": metrics["Best_Confidence"], "mAP50": metrics["mAP50"],
               "mAP50_95": metrics["mAP50_95"], "AP75": metrics["AP75"],
               "Params_M": profile["Params_M"], "GFLOPs": profile["GFLOPs"],
               "FPS": speed["FPS"] if speed else None, "profiler": profile["profiler"],
               "metric_source": "comparison/evaluate_coco.py"}
    path = model_dir / "experiment_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def run_one(name, model, config, actions, args):
    model_dir = config["output_root"] / name
    if model["type"] == "external":
        from external_adapter import best_checkpoint, prepare_config, repo_workdir, run_worker, train
        workdir = repo_workdir(model, args.repo)
        if workdir is None:
            print(f"{model['display_name']} official repository not configured; skip (no metrics written)")
            return
        image_dir, ann_file = (split_paths(config, "test") if actions & {"eval", "profile", "speed"}
                               else (None, None))
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else best_checkpoint(model, model_dir)
        if not args.checkpoint and not checkpoint.is_file() and model.get("checkpoint"):
            checkpoint = path_from_root(model["checkpoint"])
        metadata_path = model_dir / "training_metadata.json"
        if ("train" not in actions and checkpoint.is_file() and not args.checkpoint
                and not metadata_path.is_file()):
            raise ValueError("External checkpoint provenance is unknown; pass --checkpoint with --pretrained, --epochs and --batch-size")
        if args.checkpoint:
            model.update(pretrained=args.pretrained == "true", epochs=args.epochs, batch_size=args.batch_size)
            if not model["pretrained"]:
                model["pretrained_scope"] = None
        elif metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for field in ("pretrained", "pretrained_scope", "epochs", "batch_size"):
                if field in metadata:
                    model[field] = metadata[field]
        epochs = args.epochs or model["epochs"]
        batch_size = args.batch_size or model["batch_size"]
        python = str(Path(args.external_python).resolve()) if args.external_python else sys.executable
        if "train" in actions:
            checkpoint = train(model, config, workdir, model_dir, epochs, batch_size, args.device, python)
        if actions & {"eval", "profile", "speed"} and not checkpoint.is_file():
            raise FileNotFoundError(f"Trained checkpoint required for {name}: {checkpoint}")
        snapshot = model_dir / "config_used.yml"
        if actions & {"eval", "profile", "speed"} and not snapshot.is_file():
            snapshot = prepare_config(model, config, workdir, model_dir, epochs, batch_size)
        if "eval" in actions:
            prediction_path = model_dir / "predictions.json"
            run_worker("eval", workdir, snapshot, checkpoint, ann_file, image_dir,
                       prediction_path, args.device, python)
            metrics = evaluate(ann_file, prediction_path, model_dir)
            metrics["checkpoint_sha256"] = sha256(checkpoint)
            metrics["evaluation_split"] = evaluation_split(config)
            (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if "profile" in actions:
            profile_path = model_dir / "profile.json"
            run_worker("profile", workdir, snapshot, checkpoint, ann_file, image_dir,
                       profile_path, "cpu", python)
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["checkpoint_sha256"] = sha256(checkpoint)
            profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        if "speed" in actions:
            speed_path = model_dir / "speed.json"
            run_worker("speed", workdir, snapshot, checkpoint, ann_file, image_dir,
                       speed_path, "cuda", python)
            speed = json.loads(speed_path.read_text(encoding="utf-8"))
            speed["checkpoint_sha256"] = sha256(checkpoint)
            speed_path.write_text(json.dumps(speed, indent=2), encoding="utf-8")
        metrics_path, profile_path = model_dir / "metrics.json", model_dir / "profile.json"
        if metrics_path.is_file() and profile_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            print(complete_summary(name, model, model_dir, checkpoint, config, metrics, profile))
        return
    image_dir, ann_file = (split_paths(config, "test") if actions & {"eval", "profile", "speed"}
                           else (None, None))
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else _resolved_checkpoint(model, model_dir)
    if args.checkpoint:
        model["pretrained"] = args.pretrained == "true"
        model["epochs"] = args.epochs
        model["batch_size"] = args.batch_size
    else:
        metadata_path = model_dir / "training_metadata.json"
        if metadata_path.is_file():
            training_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            for field, key in (("pretrained", "pretrained"), ("epochs", "epochs"), ("batch_size", "batch_size")):
                model[field] = training_meta[key]
    epochs = args.epochs or model["epochs"]
    batch_size = args.batch_size or model["batch_size"]
    if "train" in actions:
        if model["type"] == "ultralytics":
            from train_yolo import train
            splits = ("train", "val") if actions == {"train"} else ("train", "val", "test")
            fish_yaml = convert(config, config["output_root"] / "comparison_dataset", splits=splits)
            checkpoint = train(model["model"], fish_yaml, config["output_root"], name,
                               epochs, config["image_size"], batch_size, args.device,
                               model["pretrained"], config.get("training_protocol"))
        else:
            checkpoint = train_deim(model, model_dir, config, epochs, batch_size, args.device)
    if "eval" in actions or "profile" in actions or "speed" in actions:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Trained checkpoint required for {name}: {checkpoint}")
        model_dir.mkdir(parents=True, exist_ok=True)
    if "eval" in actions:
        prediction_path = model_dir / "predictions.json"
        if model["type"] == "ultralytics":
            from ultralytics_adapter import predict
            predict(checkpoint, ann_file, image_dir, prediction_path, args.device, config["image_size"])
        else:
            from deim_adapter import predict
            predict(_deim_config(model, model_dir), checkpoint, ann_file, image_dir,
                    prediction_path, args.device or "cuda")
        metrics = evaluate(ann_file, prediction_path, model_dir)
        metrics["checkpoint_sha256"] = sha256(checkpoint)
        metrics["evaluation_split"] = evaluation_split(config)
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if "profile" in actions:
        from model_profile import profile_deim, profile_yolo, write_profile
        profile = (profile_yolo(checkpoint, config["image_size"]) if model["type"] == "ultralytics"
                   else profile_deim(_deim_config(model, model_dir), checkpoint, ann_file, image_dir,
                                     config["image_size"]))
        profile["checkpoint_sha256"] = sha256(checkpoint)
        write_profile(profile, model_dir)
    if "speed" in actions:
        from model_profile import benchmark_deim, benchmark_yolo
        speed = (benchmark_yolo(checkpoint, config["image_size"]) if model["type"] == "ultralytics"
                 else benchmark_deim(_deim_config(model, model_dir), checkpoint, ann_file, image_dir,
                                     config["image_size"]))
        speed["checkpoint_sha256"] = sha256(checkpoint)
        (model_dir / "speed.json").write_text(json.dumps(speed, indent=2), encoding="utf-8")
    metrics_path, profile_path = model_dir / "metrics.json", model_dir / "profile.json"
    if metrics_path.is_file() and profile_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        print(complete_summary(name, model, model_dir, checkpoint, config, metrics, profile))


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model")
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--speed", action="store_true", help="Optional CUDA model-only benchmark (50 warmup + 200 timed)")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "comparison_models.yml"))
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--checkpoint", help="Single-model checkpoint override")
    parser.add_argument("--pretrained", choices=["true", "false"], help="Required with --checkpoint to document its provenance")
    parser.add_argument("--device")
    parser.add_argument("--ultralytics-python", help="Optional Python executable with Ultralytics installed, for mixed-environment --all")
    parser.add_argument("--repo", help="Official external repository root for a single DETR model")
    parser.add_argument("--external-python", help="Python executable with the official external repository dependencies")
    parser.add_argument("--epochs", type=int, help="Training override, saved in result summary")
    parser.add_argument("--batch-size", type=int, help="Training override, saved in result summary")
    args = parser.parse_args()
    config = read_config(args.config, args.data_root, args.output_root)
    names = list(config["models"]) if args.all else [args.model]
    if any(name not in config["models"] for name in names):
        raise ValueError(f"Unknown model; choose from {list(config['models'])}")
    if args.checkpoint and len(names) != 1:
        raise ValueError("--checkpoint is only valid with --model")
    if args.repo and (len(names) != 1 or config["models"][names[0]]["type"] != "external"):
        raise ValueError("--repo is only valid with a single external model")
    if args.checkpoint and not args.pretrained:
        raise ValueError("--checkpoint requires --pretrained true|false (do not guess weight provenance)")
    if args.checkpoint and (args.epochs is None or args.batch_size is None):
        raise ValueError("--checkpoint also requires --epochs and --batch-size to document its training run")
    if args.checkpoint and args.train:
        raise ValueError("--checkpoint cannot be combined with --train")
    if args.pretrained and not (args.checkpoint or args.train):
        raise ValueError("--pretrained without --checkpoint is only valid when training")
    actions = {action for action in ("train", "eval", "profile", "speed") if getattr(args, action)}
    if not actions:
        actions = {"train", "eval", "profile"}
    required_splits = ("train", "val") if actions == {"train"} else ("train", "val", "test")
    assert_same_categories(config, required_splits)
    if actions & {"eval", "profile", "speed"} and evaluation_split(config) == "validation_reused_as_test":
        print("WARNING: evaluation uses a copy of the validation annotations, not an independent test set", flush=True)
    failures = []
    for name in names:
        try:
            print(f"\n===== {name}: {', '.join(sorted(actions))} =====", flush=True)
            model = dict(config["models"][name])
            if args.pretrained and args.train and not args.checkpoint:
                if model["type"] == "deim":
                    raise ValueError("Set DEIM pretrained weights in its YAML config; a boolean alone is insufficient")
                if model["type"] == "ultralytics":
                    model["pretrained"] = args.pretrained == "true"
                if model["type"] == "external":
                    model["pretrained"] = args.pretrained == "true"
                    if not model["pretrained"]:
                        model["pretrained_scope"] = None
            if model["type"] == "ultralytics" and args.ultralytics_python and Path(args.ultralytics_python).resolve() != Path(sys.executable).resolve():
                child = [str(Path(args.ultralytics_python).resolve()), str(Path(__file__).resolve()),
                         "--model", name, "--config", str(Path(args.config).resolve()),
                         "--data-root", str(config["data_root"]), "--output-root", str(config["output_root"])]
                child += [f"--{action}" for action in sorted(actions)]
                for flag, value in (("--device", args.device), ("--epochs", args.epochs),
                                    ("--batch-size", args.batch_size), ("--checkpoint", args.checkpoint),
                                    ("--pretrained", args.pretrained)):
                    if value is not None:
                        child += [flag, str(value)]
                subprocess.run(child, cwd=ROOT, check=True)
                continue
            if args.epochs and "train" in actions:
                model["epochs"] = args.epochs
            if args.batch_size and "train" in actions:
                model["batch_size"] = args.batch_size
            run_one(name, model, config, actions, args)
        except Exception as exc:
            if not args.all:
                raise
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            failures.append(name)
    path, count = summarize(config["output_root"])
    print(f"Comparison CSV: {path} ({count} completed models)")
    if failures:
        raise SystemExit(f"Failed models: {', '.join(failures)}")


if __name__ == "__main__":
    main()
