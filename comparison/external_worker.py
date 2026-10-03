"""Run upstream DETR models in their own Python process, emitting shared COCO JSON."""
import argparse
import json
import sys
import time
from pathlib import Path


def detection(image_id, label, box, score, category_map):
    label = int(label)
    if label not in category_map:
        raise ValueError(f"Unexpected predicted class index: {label}")
    x1, y1, x2, y2 = map(float, box)
    return {"image_id": int(image_id), "category_id": category_map[label],
            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "score": float(score)}


def load_model(args):
    sys.path.insert(0, str(Path(args.repo_workdir).resolve()))
    import torch
    from src.core import YAMLConfig

    cfg = YAMLConfig(args.config, resume=args.checkpoint)
    for backbone in ("HGNetv2", "PResNet"):
        if backbone in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone]["pretrained"] = False
    state_file = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state_file["ema"]["module"] if state_file.get("ema") else state_file["model"]
    model = cfg.model
    model.load_state_dict(state)
    model.eval()
    return torch, model, cfg.postprocessor


def predict(args, torch, model, postprocessor):
    from PIL import Image
    from torchvision import transforms as T

    from common import read_coco
    annotation, category_map = read_coco(args.ann)
    device = torch.device(args.device)
    model = model.deploy().to(device).eval()
    postprocessor = postprocessor.deploy().to(device).eval()
    transform = T.Compose([T.Resize((args.imgsz, args.imgsz)), T.ToTensor()])
    rows = []
    with torch.inference_mode():
        for image in annotation["images"]:
            path = Path(args.image_dir) / image["file_name"]
            with Image.open(path) as source:
                source = source.convert("RGB")
                w, h = source.size
                pixels = transform(source).unsqueeze(0).to(device)
            sizes = torch.tensor([[w, h]], device=device)
            labels, boxes, scores = postprocessor(model(pixels), sizes)
            for label, box, score in zip(labels[0].tolist(), boxes[0].tolist(), scores[0].tolist()):
                rows.append(detection(image["id"], label, box, score, category_map))
    return rows


def profile(args, torch, model):
    from calflops.calculate_pipline import CalFlopsPipline

    model = model.cpu().eval()
    pipeline = CalFlopsPipline(model, include_backPropagation=False, compute_bp_factor=2.0)
    try:
        pipeline.start_flops_calculate()
        with torch.no_grad():
            model(torch.zeros(1, 3, args.imgsz, args.imgsz))
        macs = int(pipeline.get_total_macs())
    finally:
        pipeline.end_flops_calculate()
    params = sum(param.numel() for param in model.parameters())
    return {"Params": params, "Params_M": params / 1e6, "MACs": macs,
            "GMACs": macs / 1e9, "FLOPs": 2 * macs, "GFLOPs": 2 * macs / 1e9,
            "input_size": [1, 3, args.imgsz, args.imgsz],
            "profiler": "calflops local PyTorch forward", "FLOPs_convention": "2 x MACs",
            "limitations": "Operator-based estimate; unsupported ops may not be counted. Model forward excludes postprocessor."}


def speed(args, torch, model):
    if not torch.cuda.is_available():
        raise RuntimeError("Model-only FPS benchmark requires a CUDA GPU")
    model = model.deploy().cuda().eval()
    pixels = torch.zeros(1, 3, args.imgsz, args.imgsz, device="cuda")
    with torch.inference_mode():
        for _ in range(50):
            model(pixels)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            model(pixels)
            torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 5
    return {"Latency_ms": latency, "FPS": 1000 / latency, "warmup": 50,
            "iterations": 200, "batch_size": 1, "image_size": args.imgsz,
            "scope": "PyTorch model forward only; excludes image I/O, transform, postprocessor and file writing",
            "device": torch.cuda.get_device_name()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["eval", "profile", "speed"], required=True)
    for name in ("repo-workdir", "config", "checkpoint", "ann", "image-dir", "out", "device"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    args = parser.parse_args()
    torch, model, postprocessor = load_model(args)
    result = (predict(args, torch, model, postprocessor) if args.action == "eval" else
              profile(args, torch, model) if args.action == "profile" else
              speed(args, torch, model))
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
