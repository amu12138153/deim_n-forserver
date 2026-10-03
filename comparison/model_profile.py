"""Local operator profiling. All reported FLOPs are exactly 2 x MACs."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

from common import ROOT

sys.path.insert(0, str(ROOT))


def profile_deim(config_path, checkpoint, ann_file, image_dir, imgsz=640):
    from deim_adapter import load
    from get_info_param_and_flops import profile_model
    _, model, _, _, _, _, _, _, _ = load(config_path, checkpoint, ann_file, image_dir, "cpu")
    raw = profile_model(model, imgsz, "cpu")
    macs = int(raw["MACs"])
    return {"Params": int(raw["Parameters"]), "Params_M": raw["Parameters_M"],
            "MACs": macs, "GMACs": macs / 1e9, "FLOPs": 2 * macs,
            "GFLOPs": 2 * macs / 1e9, "input_size": [1, 3, imgsz, imgsz],
            "profiler": raw["profiler"], "FLOPs_convention": "2 x MACs",
            "water_shape": raw["water_shape"], "active_fusion_stages": raw["active_fusion_stages"],
            "limitations": raw["limitations"]}


def profile_yolo(checkpoint, imgsz=640):
    from calflops.calculate_pipline import CalFlopsPipline
    from ultralytics import YOLO
    model = YOLO(str(checkpoint)).model.cpu().eval()
    pipeline = CalFlopsPipline(model, include_backPropagation=False, compute_bp_factor=2.0)
    try:
        pipeline.start_flops_calculate()
        with torch.no_grad():
            model(torch.zeros(1, 3, imgsz, imgsz))
        macs = int(pipeline.get_total_macs())
    finally:
        pipeline.end_flops_calculate()
    params = sum(param.numel() for param in model.parameters())
    return {"Params": params, "Params_M": params / 1e6, "MACs": macs,
            "GMACs": macs / 1e9, "FLOPs": 2 * macs, "GFLOPs": 2 * macs / 1e9,
            "input_size": [1, 3, imgsz, imgsz], "profiler": "calflops local PyTorch forward",
            "FLOPs_convention": "2 x MACs",
            "limitations": "Operator-based estimate; unsupported ops may not be counted. Raw model forward excludes NMS."}


def write_profile(profile, out_dir):
    out = Path(out_dir) / "profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return out


def _benchmark(module, imgsz, warmup=50, iterations=200):
    if not torch.cuda.is_available():
        raise RuntimeError("Model-only FPS benchmark requires a CUDA GPU")
    module = module.cuda().eval()
    image = torch.zeros(1, 3, imgsz, imgsz, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            module(image)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            module(image)
            torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000 / iterations
    return {"Latency_ms": latency, "FPS": 1000 / latency, "warmup": warmup,
            "iterations": iterations, "batch_size": 1, "image_size": imgsz,
            "scope": "PyTorch model forward only; excludes image I/O, transform, NMS and file writing",
            "device": torch.cuda.get_device_name()}


def benchmark_deim(config_path, checkpoint, ann_file, image_dir, imgsz=640):
    from deim_adapter import load
    from get_info_param_and_flops import ModelForFlops
    _, model, _, _, _, _, _, _, _ = load(config_path, checkpoint, ann_file, image_dir, "cuda")
    if hasattr(model, "deploy"):
        model.deploy()
    return _benchmark(ModelForFlops(model), imgsz)


def benchmark_yolo(checkpoint, imgsz=640):
    from ultralytics import YOLO
    return _benchmark(YOLO(str(checkpoint)).model, imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["ultralytics", "deim"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--gt")
    parser.add_argument("--image-dir")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = (profile_yolo(args.checkpoint, args.imgsz) if args.type == "ultralytics"
              else profile_deim(args.config, args.checkpoint, args.gt, args.image_dir, args.imgsz))
    print(write_profile(result, args.out))
