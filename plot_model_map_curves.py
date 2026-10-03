"""Compare raw per-epoch validation mAP curves from ablation and comparison runs.

Examples (run from the project root)::

    python plot_model_map_curves.py --ablation-root outputs/ablation_200e_4090 \
        --comparison-root comparison_outputs/formal_200e_4090 \
        --models V0_visual M4_full yolov8n dfine_n rtdetrv2_s \
        --out figures/selected_models_map

The training log prefix ``test_`` in DETR/DEIM means evaluation during training,
which uses the validation loader. Final test-set metrics are not epoch curves.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Keep figure text editable when the SVG/PDF is opened for manuscript layout.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

PROJECT_ROOT = Path(__file__).resolve().parent
COLORS = (
    "#0F4D92", "#B64342", "#42949E", "#9A4D8E", "#B57925",
    "#496E45", "#5279AD", "#8D638C", "#636363", "#B16650",
)
LINESTYLES = ("-", "--", "-.")


@dataclass(frozen=True)
class History:
    name: str
    source: Path
    points: tuple[tuple[int, float, float], ...]  # epoch, mAP50, mAP50:95


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def metric(value: object, source: Path, epoch: int, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: epoch {epoch} has no valid {label}") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{source}: epoch {epoch} {label}={number} is outside [0, 1]")
    return number


def load_jsonl(path: Path, name: str) -> History:
    points: dict[int, tuple[int, float, float]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_number}") from exc
            if "epoch" not in row:
                continue
            epoch = int(row["epoch"]) + 1  # DEIM/DETR logs use zero-based epochs.
            if epoch < 1:
                continue
            bbox = row.get("test_coco_eval_bbox")
            if not isinstance(bbox, list):
                bbox = []
            map50 = row.get("test_map_50", bbox[1] if len(bbox) > 1 else None)
            map5095 = row.get("test_map_50_95", bbox[0] if bbox else None)
            if map50 is None and map5095 is None:
                continue  # e.g. a training-only log row without validation.
            points[epoch] = (
                epoch,
                metric(map50, path, epoch, "mAP50"),
                metric(map5095, path, epoch, "mAP50:95"),
            )
    return _history(name, path, points)


def load_yolo_csv(path: Path, name: str) -> History:
    points: dict[int, tuple[int, float, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        reader.fieldnames = [field.strip() for field in reader.fieldnames]
        needed = ("epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
        missing = [field for field in needed if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for row in reader:
            if not row["epoch"] or not row["epoch"].strip():
                continue
            epoch = int(row["epoch"])
            if epoch < 1:
                raise ValueError(f"{path}: YOLO epoch must be one-based")
            points[epoch] = (
                epoch,
                metric(row["metrics/mAP50(B)"], path, epoch, "mAP50"),
                metric(row["metrics/mAP50-95(B)"], path, epoch, "mAP50:95"),
            )
    return _history(name, path, points)


def _history(name: str, path: Path, points: dict[int, tuple[int, float, float]]) -> History:
    if not points:
        raise ValueError(f"{path}: no per-epoch validation mAP records")
    # A resumed run can repeat an epoch; the last logged value is authoritative.
    return History(name, path, tuple(points[epoch] for epoch in sorted(points)))


def find_history(name: str, roots: list[Path]) -> History:
    candidates = [root / name / filename for root in roots for filename in ("log.txt", "results.csv")]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No per-epoch log for {name!r} under {searched}")
    if len(existing) != 1:
        raise ValueError(f"Ambiguous model {name!r}; found: {', '.join(map(str, existing))}. Use one root.")
    path = existing[0]
    return load_yolo_csv(path, name) if path.suffix == ".csv" else load_jsonl(path, name)


def list_models(roots: list[Path]) -> None:
    for root in roots:
        print(f"{root}:")
        names = sorted({path.parent.name for filename in ("log.txt", "results.csv")
                        for path in root.glob(f"*/{filename}")})
        for name in names:
            print(f"  {name}")
        if not names:
            print("  (no per-epoch logs yet)")


def with_gaps(points: tuple[tuple[int, float, float], ...], metric_index: int) -> tuple[list[int], list[float]]:
    """Insert NaNs across missing epochs rather than interpolating them."""
    values = {point[0]: point[metric_index] for point in points}
    epochs = list(range(points[0][0], points[-1][0] + 1))
    return epochs, [values.get(epoch, math.nan) for epoch in epochs]


def plot(histories: list[History], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharex=True, sharey=True)
    for index, history in enumerate(histories):
        color = COLORS[index % len(COLORS)]
        linestyle = LINESTYLES[index // len(COLORS) % len(LINESTYLES)]
        for ax, field in zip(axes, (1, 2)):
            x, y = with_gaps(history.points, field)
            ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.15,
                    label=history.name, solid_capstyle="round")
    for ax, title in zip(axes, ("a  mAP@0.50", "b  mAP@0.50:0.95")):
        ax.set_title(title, loc="left", fontweight="bold", fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
        ax.tick_params(direction="out", length=2.5, width=0.7)
    axes[0].set_ylabel("Validation AP (fraction)")
    fig.suptitle("Model comparison — raw per-epoch validation metrics", y=0.98, fontsize=9)
    rows = math.ceil(len(histories) / min(4, len(histories)))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83,
                        bottom=min(0.46, 0.15 + 0.055 * rows), wspace=0.17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(histories)),
               bbox_to_anchor=(0.5, 0.02), fontsize=7, columnspacing=1.2)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("svg", {}), ("pdf", {}), ("png", {"dpi": 300})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, facecolor="white", **options)
        print(f"Saved {path}")
    plt.close(fig)


def save_source_data(histories: list[History], stem: Path) -> None:
    path = stem.with_name(stem.name + "_source_data.csv")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "source", "epoch", "map50", "map50_95"))
        for history in histories:
            for epoch, map50, map5095 in history.points:
                writer.writerow((history.name, str(history.source), epoch, map50, map5095))
    print(f"Saved {path}")
    manifest = stem.with_name(stem.name + "_sources.json")
    manifest.write_text(json.dumps([
        {"model": history.name, "source": str(history.source),
         "sha256": hashlib.sha256(history.source.read_bytes()).hexdigest(),
         "points": len(history.points), "first_epoch": history.points[0][0],
         "last_epoch": history.points[-1][0]}
        for history in histories
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", help="Directory containing V*/M* model folders")
    parser.add_argument("--comparison-root", help="Directory containing YOLO/DEIM/DETR model folders")
    parser.add_argument("--models", nargs="+", help="Exact model folder names to overlay")
    parser.add_argument("--list-models", action="store_true", help="Show models with per-epoch logs and exit")
    parser.add_argument("--out", default="figures/model_map_comparison", help="Output path stem")
    args = parser.parse_args()
    roots = [project_path(value) for value in (args.ablation_root, args.comparison_root) if value]
    if not roots:
        parser.error("Provide --ablation-root and/or --comparison-root")
    for root in roots:
        if not root.is_dir():
            parser.error(f"Not a directory: {root}")
    if args.list_models:
        list_models(roots)
        return
    if not args.models:
        parser.error("Provide --models NAME [NAME ...] or use --list-models")
    if len(args.models) != len(set(args.models)):
        parser.error("Duplicate names in --models")
    histories = [find_history(name, roots) for name in args.models]
    for history in histories:
        print(f"{history.name}: epochs {history.points[0][0]}–{history.points[-1][0]}, "
              f"{len(history.points)} logged points, source={history.source}")
    stem = project_path(args.out).with_suffix("")
    plot(histories, stem)
    save_source_data(histories, stem)


if __name__ == "__main__":
    main()
