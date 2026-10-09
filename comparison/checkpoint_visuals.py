"""Matching and drawing for checkpoint-based detection and Grad-CAM figures."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def match_predictions(annotation, predictions, threshold, iou_threshold=0.5):
    """Use the same confidence ordering and category-aware greedy rule as best_f1."""
    from evaluate_coco import iou_xywh
    ground_truth = defaultdict(list)
    for ann in annotation['annotations']:
        if not ann.get('iscrowd', 0) and not ann.get('ignore', 0):
            ground_truth[(int(ann['image_id']), int(ann['category_id']))].append(ann)
    used = defaultdict(set)
    results = defaultdict(lambda: {'TP': [], 'FP': [], 'FN': []})
    for pred in sorted(predictions, key=lambda row: -float(row['score'])):
        if float(pred['score']) < threshold:
            continue
        key = (int(pred['image_id']), int(pred['category_id']))
        candidates = [(iou_xywh(pred['bbox'], ann['bbox']), index)
                      for index, ann in enumerate(ground_truth.get(key, [])) if index not in used[key]]
        if candidates and max(candidates)[0] >= iou_threshold:
            used[key].add(max(candidates)[1])
            results[key[0]]['TP'].append(pred)
        else:
            results[key[0]]['FP'].append(pred)
    for key, annotations in ground_truth.items():
        for index, ann in enumerate(annotations):
            if index not in used[key]:
                results[key[0]]['FN'].append(ann)
    return results


def local_metrics(matches):
    tp, fp, fn = (len(matches[key]) for key in ('TP', 'FP', 'FN'))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall, 'F1': f1}


def draw_detections(image, predictions, names, show_labels=True):
    """Clean class-colored predicted boxes for a compact multi-model figure."""
    result = image.copy()
    draw = ImageDraw.Draw(result)
    colors = ('#19A7B8', '#EC6946', '#A45CE0', '#D4A52F')
    for row in predictions:
        x, y, width, height = row['bbox']
        box = (max(0, x), max(0, y), min(image.width-1, x+width), min(image.height-1, y+height))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        category = int(row['category_id'])
        color = colors[category % len(colors)]
        draw.rectangle(box, outline=color, width=2)
        if show_labels:
            label = f"{names[category]} {row['score']:.2f}"
            position = (int(box[0]), max(0, int(box[1])-14))
            draw.rectangle(draw.textbbox(position, label), fill=color)
            draw.text(position, label, fill='white')
    return result


def cam_overlay(image, cam):
    """Render a normalized Grad-CAM; each map expresses relative activation."""
    from matplotlib import colormaps
    color = colormaps['jet'](np.clip(cam, 0, 1))[..., :3]
    original = np.asarray(image, dtype=np.float32) / 255
    return Image.fromarray(np.uint8(np.clip(0.55 * original + 0.45 * color, 0, 1) * 255))


def header(image, text, height=44):
    result = Image.new('RGB', (image.width, image.height + height), 'white')
    result.paste(image, (0, height))
    try:
        font = ImageFont.truetype('arial.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(result).text((8, 5), text, fill='black', font=font)
    return result


def assemble_grids(annotation, image_dir, model_specs, out_dir, image_ids):
    """One column per model, one row per selected image, original at left."""
    images = {int(row['id']): row for row in annotation['images']}
    thumb_width = 320
    for kind, filename in [('detections', 'detections_comparison.png'), ('gradcam', 'gradcam_comparison.png')]:
        rows = []
        for image_id in image_ids:
            source = Image.open(Path(image_dir) / images[image_id]['file_name']).convert('RGB')
            cells = [source]
            for spec in model_specs:
                cell = Image.open(Path(out_dir) / spec['name'] / 'figures' / f'{image_id}_{kind}.png').convert('RGB')
                # Worker files have a 30px title; draw readable grid titles after resizing.
                cells.append(cell.crop((0, 30, cell.width, cell.height)))
            height = max(round(cell.height * thumb_width / cell.width) for cell in cells)
            row = Image.new('RGB', (thumb_width * len(cells), height), 'white')
            for index, cell in enumerate(cells):
                cell = cell.resize((thumb_width, round(cell.height * thumb_width / cell.width)), Image.Resampling.LANCZOS)
                row.paste(cell, (index * thumb_width, 0))
            rows.append(row)
        grid = Image.new('RGB', (max(row.width for row in rows), 30 + sum(row.height for row in rows)), 'white')
        titles = ['Original'] + [spec['display_name'] for spec in model_specs]
        for index, title in enumerate(titles):
            title_cell = header(Image.new('RGB', (thumb_width, 1), 'white'), title, height=30)
            grid.paste(title_cell, (index * thumb_width, 0))
        y = 30
        for row in rows:
            grid.paste(row, (0, y))
            y += row.height
        grid.save(Path(out_dir) / filename)
