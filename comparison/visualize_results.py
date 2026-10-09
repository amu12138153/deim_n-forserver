"""Draw matched COCO detections and prediction-score heatmaps for comparison models.

The heatmap shows where high-confidence predicted boxes are concentrated. It is
derived from saved predictions, not from model activations or Grad-CAM.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MODEL_KEYS = (
    'deimv2_n', 'dfine_n', 'dfine_s', 'ours', 'rtdetr_r18', 'rtdetrv2_s',
    'yolo11n', 'yolo26n', 'yolov8n',
)
LABELS = {
    'deimv2_n': 'DEIMv2-HGNetv2-N', 'dfine_n': 'D-FINE-N',
    'dfine_s': 'D-FINE-S', 'ours': 'Ours', 'rtdetr_r18': 'RT-DETR-R18',
    'rtdetrv2_s': 'RT-DETRv2-S', 'yolo11n': 'YOLO11n',
    'yolo26n': 'YOLO26n', 'yolov8n': 'YOLOv8n',
}
CLASS_COLORS = ('#19A7B8', '#EC6946', '#A45CE0', '#D4A52F')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_images(annotation, requested_ids, all_images, max_images):
    images = {int(item['id']): item for item in annotation['images']}
    if requested_ids:
        unknown = sorted(set(requested_ids) - set(images))
        if unknown:
            raise ValueError(f'Image IDs absent from annotation: {unknown}')
        chosen = requested_ids
    elif all_images:
        chosen = sorted(images)
    else:
        chosen = sorted(images)[:max_images]
    return [(image_id, images[image_id]) for image_id in chosen]


def load_model_predictions(results_root, model_key, annotation_hash, image_ids, category_ids):
    model_dir = results_root / model_key
    metrics_file = model_dir / 'metrics.json'
    predictions_file = model_dir / 'predictions.json'
    for path in (metrics_file, predictions_file):
        if not path.is_file():
            raise FileNotFoundError(f'{model_key}: required comparison result is missing: {path}')
    metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
    if metrics.get('test_annotation_sha256') != annotation_hash:
        raise ValueError(
            f'{model_key}: predictions belong to a different annotation. '
            f'Expected SHA256 {annotation_hash}; metrics report '
            f'{metrics.get("test_annotation_sha256")}. Use the images and annotation from this run.'
        )
    predictions = json.loads(predictions_file.read_text(encoding='utf-8'))
    if not isinstance(predictions, list):
        raise ValueError(f'{predictions_file}: expected a COCO prediction list')
    grouped = defaultdict(list)
    for index, pred in enumerate(predictions):
        image_id = int(pred['image_id'])
        category_id = int(pred['category_id'])
        box = [float(value) for value in pred['bbox']]
        score = float(pred['score'])
        if (image_id not in image_ids or category_id not in category_ids or
                len(box) != 4 or not all(math.isfinite(v) for v in box) or
                box[2] <= 0 or box[3] <= 0 or not math.isfinite(score) or
                not 0 <= score <= 1):
            raise ValueError(f'{predictions_file}: invalid prediction at index {index}')
        grouped[image_id].append((category_id, box, score))
    return grouped, predictions_file


def filtered_detections(predictions, threshold):
    return sorted((pred for pred in predictions if pred[2] >= threshold),
                  key=lambda pred: pred[2], reverse=True)


def draw_boxes(image, detections, category_names):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    width, height = result.size
    for category_id, (x, y, w, h), score in detections:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width - 1, x + w), min(height - 1, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        color = CLASS_COLORS[category_id % len(CLASS_COLORS)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = f'{category_names[category_id]} {score:.2f}'
        tx, ty = int(x1), max(0, int(y1) - 16)
        bounds = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bounds, fill=color)
        draw.text((tx, ty), label, fill='white', font=font)
    return result


def prediction_density(image, detections):
    """Elliptical score kernels combined by maximum on a fixed [0, 1] scale."""
    width, height = image.size
    heat = np.zeros((height, width), dtype=np.float32)
    for _, (x, y, w, h), score in detections:
        cx, cy = x + w / 2, y + h / 2
        sx, sy = max(w / 2, 4), max(h / 2, 4)
        x0, x1 = max(0, int(cx - 3 * sx)), min(width, int(cx + 3 * sx) + 1)
        y0, y1 = max(0, int(cy - 3 * sy)), min(height, int(cy + 3 * sy) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        grid_x = (np.arange(x0, x1, dtype=np.float32) - cx) / sx
        grid_y = (np.arange(y0, y1, dtype=np.float32) - cy) / sy
        kernel = score * np.exp(-0.5 * (grid_y[:, None] ** 2 + grid_x[None, :] ** 2))
        np.maximum(heat[y0:y1, x0:x1], kernel, out=heat[y0:y1, x0:x1])

    # Fixed color mapping and alpha keep values comparable across models.
    base = np.asarray(image.convert('RGB'), dtype=np.float32)
    red = np.clip(2.2 * heat, 0, 1)
    green = np.clip(2.2 * heat - 0.65, 0, 1)
    blue = np.clip(2.2 * heat - 1.55, 0, 1)
    colors = np.stack((red, green, blue), axis=-1) * 255
    alpha = (0.75 * heat)[..., None]
    blended = np.clip(base * (1 - alpha) + colors * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(blended, 'RGB')


def make_pair(boxes, heatmap, title):
    width, height = boxes.size
    pair = Image.new('RGB', (width * 2, height + 30), 'white')
    pair.paste(boxes, (0, 30))
    pair.paste(heatmap, (width, 30))
    draw = ImageDraw.Draw(pair)
    draw.text((8, 8), f'{title} | detections', fill='black')
    draw.text((width + 8, 8), 'Prediction-score density (not Grad-CAM)', fill='black')
    return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, required=True,
                        help='Parent directory containing each model/predictions.json and metrics.json')
    parser.add_argument('--models', nargs='+', choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument('--image-ids', nargs='+', type=int)
    parser.add_argument('--all-images', action='store_true')
    parser.add_argument('--max-images', type=int, default=3,
                        help='Number of first annotated images when --image-ids is omitted')
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--out', type=Path, default=Path('comparison_figures'))
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1 or args.max_images < 1:
        parser.error('--score-threshold must be in [0,1] and --max-images must be positive')
    if len(args.models) != len(set(args.models)):
        parser.error('Duplicate model keys')

    annotation = json.loads(args.ann_file.read_text(encoding='utf-8-sig'))
    category_names = {int(c['id']): c['name'] for c in annotation['categories']}
    image_ids = {int(item['id']) for item in annotation['images']}
    selected = select_images(annotation, args.image_ids, args.all_images, args.max_images)
    annotation_hash = sha256(args.ann_file)
    model_data = {}
    for key in args.models:
        model_data[key] = load_model_predictions(
            args.results_root, key, annotation_hash, image_ids, set(category_names))

    args.out.mkdir(parents=True, exist_ok=True)
    for image_id, info in selected:
        source = args.image_dir / info['file_name']
        if not source.is_file():
            raise FileNotFoundError(f'Image in annotation is missing: {source}')
        image = Image.open(source).convert('RGB')
        montage_rows = []
        for key in args.models:
            grouped, _ = model_data[key]
            detections = filtered_detections(grouped.get(image_id, []), args.score_threshold)
            boxes = draw_boxes(image, detections, category_names)
            heatmap = prediction_density(image, detections)
            model_dir = args.out / key
            model_dir.mkdir(exist_ok=True)
            stem = Path(info['file_name']).stem
            boxes.save(model_dir / f'{stem}_detections.png')
            heatmap.save(model_dir / f'{stem}_score_heatmap.png')
            pair = make_pair(boxes, heatmap, LABELS[key])
            pair.save(model_dir / f'{stem}_pair.png')
            montage_rows.append(pair)
        thumb_width = 960
        scaled = [row.resize((thumb_width, round(row.height * thumb_width / row.width)))
                  for row in montage_rows]
        montage = Image.new('RGB', (thumb_width, sum(row.height for row in scaled)), 'white')
        offset = 0
        for row in scaled:
            montage.paste(row, (0, offset))
            offset += row.height
        montage.save(args.out / f'{Path(info["file_name"]).stem}_comparison.png')
        print(f'Saved image {image_id}: {info["file_name"]}')

    manifest = {
        'kind': 'saved_detection_visualization',
        'heatmap_kind': 'prediction_score_density_not_gradcam',
        'heatmap_method': 'Maximum of score-weighted elliptical Gaussian kernels centered on predicted boxes; fixed 0–1 scale',
        'score_threshold': args.score_threshold,
        'annotation': str(args.ann_file.resolve()),
        'annotation_sha256': annotation_hash,
        'models': {key: {'prediction_file': str(path.resolve()), 'prediction_sha256': sha256(path)}
                   for key, (_, path) in model_data.items()},
        'image_ids': [image_id for image_id, _ in selected],
    }
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Saved visualizations to {args.out.resolve()}')


if __name__ == '__main__':
    main()
