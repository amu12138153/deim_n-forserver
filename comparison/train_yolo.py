"""The same Ultralytics training entrypoint for v8n, 11n and 26n."""
import argparse
import json
from pathlib import Path


def common_augmentation_kwargs(batch, warmup_epochs):
    """Match the DETR runs: one 640 image, horizontal flip p=0.5 only."""
    return dict(optimizer="auto", cos_lr=True, lrf=0.0, warmup_epochs=float(warmup_epochs),
                warmup_bias_lr=0.0, nbs=batch, multi_scale=False,
                patience=0, workers=0, deterministic=True,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                flipud=0.0, fliplr=0.5, bgr=0.0,
                mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                close_mosaic=0, erasing=0.0)


def train(model_name, data, output_root, run_name, epochs=200, imgsz=640, batch=8,
          device=None, pretrained=True, protocol=None):
    from ultralytics import YOLO
    if imgsz != 640:
        raise ValueError("Comparison experiments require imgsz=640")
    run_dir = Path(output_root).resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing training run: {run_dir}")
    source = model_name if pretrained else model_name.removesuffix(".pt") + ".yaml"
    model = YOLO(source)
    kwargs = dict(data=str(Path(data).resolve()), epochs=epochs, imgsz=imgsz, batch=batch,
                  pretrained=bool(pretrained),
                  project=str(Path(output_root).resolve()), name=run_name, exist_ok=False)
    if protocol is not None:
        kwargs.update(common_augmentation_kwargs(batch, min(protocol["warmup_epochs"], max(epochs - 1, 0))))
        kwargs.update(seed=int(protocol["seed"]), amp=True)
    if device is not None:
        kwargs["device"] = device
    if protocol is not None:
        # Ultralytics adds low-probability Albumentations by default even when
        # its named HSV/geometric options are zero. Supply an empty list here.
        from ultralytics.data import dataset as yolo_dataset
        original_v8_transforms = yolo_dataset.v8_transforms

        def transforms_without_albumentations(dataset, image_size, hyp):
            hyp.augmentations = []
            return original_v8_transforms(dataset, image_size, hyp)

        yolo_dataset.v8_transforms = transforms_without_albumentations
        try:
            model.train(**kwargs)
        finally:
            yolo_dataset.v8_transforms = original_v8_transforms
    else:
        model.train(**kwargs)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    trainer = model.trainer
    optimizer = getattr(trainer, "optimizer", None)
    (run_dir / "training_metadata.json").write_text(json.dumps({
        "pretrained": bool(pretrained), "epochs": int(epochs), "batch_size": int(batch),
        "source_model": model_name, "data": str(Path(data).resolve()), "image_size": imgsz,
        "training_protocol": protocol,
        "optimizer_rule": str(trainer.args.optimizer),
        "configured_lr0": float(trainer.args.lr0),
        "selected_optimizer": type(optimizer).__name__ if optimizer is not None else None,
        "selected_initial_lrs": [float(group["initial_lr"]) if "initial_lr" in group else None
                                 for group in optimizer.param_groups]
        if optimizer is not None else None,
    }, indent=2), encoding="utf-8")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default="comparison_outputs")
    parser.add_argument("--device")
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    print(train(args.model, args.data, args.output_root, args.name, args.epochs, args.imgsz,
                args.batch, args.device, not args.from_scratch))
