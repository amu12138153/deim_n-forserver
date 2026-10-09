"""One process per checkpoint: comparison inference, evaluation, then Grad-CAM."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from common import ROOT, read_coco, sha256
from checkpoint_visuals import cam_overlay, draw_detections, header, local_metrics, match_predictions
from evaluate_coco import evaluate

sys.path.insert(0, str(ROOT))


def effective_deim_config(spec, out):
    import yaml
    from engine.core.yaml_utils import load_config
    config = load_config(spec['config'], {})
    config.pop('__include__', None)
    import torch
    from checkpoint_config import validate_deim_config
    checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if checkpoint.get('ema') is not None else checkpoint['model']
    validate_deim_config(config, state, spec.get('name', 'DEIM'))
    for backbone in ('HGNetv2', 'PResNet'):
        if backbone in config:
            config[backbone]['pretrained'] = False
    filename = out / 'effective_model_config.yml'
    filename.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return str(filename)


def external_args(job):
    spec = job['model']
    return SimpleNamespace(repo_workdir=spec['repo_workdir'], config=spec['config'],
                           checkpoint=spec['checkpoint'], ann=job['ann_file'],
                           image_dir=job['image_dir'], device=job['device'], imgsz=job['image_size'])


def infer(job, out):
    spec = job['model']
    target = out / 'predictions.json'
    if spec['family'] == 'deim':
        from deim_adapter import predict
        return predict(spec['config'], spec['checkpoint'], job['ann_file'], job['image_dir'],
                       target, job['device'], job.get('sensor_csv'))
    if spec['family'] == 'yolo':
        from ultralytics_adapter import predict
        return predict(spec['checkpoint'], job['ann_file'], job['image_dir'], target,
                       job['device'], job['image_size'], config=spec.get('config'))
    from external_worker import load_model, predict
    args = external_args(job)
    torch, model, post = load_model(args)
    predictions = predict(args, torch, model, post)
    target.write_text(json.dumps(predictions), encoding='utf-8')
    return predictions


def load_cam_model(job):
    import torch
    spec = job['model']
    if spec['family'] == 'deim':
        from deim_adapter import load
        _, model, post, loader, *_ = load(
            spec['config'], spec['checkpoint'], job['ann_file'], job['image_dir'],
            job['device'], job.get('sensor_csv'))
        return model, post, loader.dataset
    if spec['family'] == 'yolo':
        from ultralytics_adapter import load_model
        model = load_model(spec['checkpoint'], config=spec.get('config')).model.float().to(torch.device(job['device'])).eval()
        return model, None, None
    from external_worker import load_model
    _, model, post = load_model(external_args(job))
    return model.to(torch.device(job['device'])).eval(), post, None


def cam_pixels(job, image, image_id, dataset, model):
    import torch
    from torchvision import transforms as T
    family = job['model']['family']
    if family == 'deim':
        index = dataset.ids.index(image_id)
        pixels, target = dataset[index]
        water = target.get('water_quality')
        if water is not None:
            water = water.unsqueeze(0).to(job['device'])
        return pixels.unsqueeze(0).to(job['device']), water, None
    if family == 'yolo':
        from ultralytics.data.augment import LetterBox
        stride = int(model.stride.max())
        array = LetterBox(new_shape=(job['image_size'], job['image_size']), auto=True,
                          stride=stride)(image=np.asarray(image))
        ratio = min(job['image_size'] / image.height, job['image_size'] / image.width)
        resized_width, resized_height = round(image.width * ratio), round(image.height * ratio)
        left = round((array.shape[1] - resized_width) / 2 - 0.1)
        top = round((array.shape[0] - resized_height) / 2 - 0.1)
        crop = (left, top, resized_width, resized_height)
        pixels = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float() / 255
        return pixels.unsqueeze(0).to(job['device']), None, crop
    transform = T.Compose([T.Resize((job['image_size'], job['image_size'])), T.ToTensor()])
    return transform(image).unsqueeze(0).to(job['device']), None, None


def check_reference(job, metrics):
    source = job['model'].get('reference_summary')
    if not source:
        return {'status': 'not_provided'}
    reference = json.loads(Path(source).read_text(encoding='utf-8'))
    comparisons = {}
    for key in ('Precision', 'Recall', 'F1', 'mAP50', 'mAP50_95', 'Best_Confidence'):
        expected = reference.get(key)
        if expected is not None:
            delta = float(metrics[key]) - float(expected)
            comparisons[key] = {'reference': expected, 'actual': metrics[key], 'delta': delta,
                                'matches': abs(delta) <= job['metric_tolerance']}
    if not comparisons:
        raise ValueError('Reference summary has none of the comparison metrics')
    return {'status': 'matched' if all(item['matches'] for item in comparisons.values()) else 'different',
            'source': source, 'values': comparisons}


def run(job):
    import torch
    from gradcam_core import gradcam, resolve_layer, score_target, yolo_head_target
    torch.set_num_threads(int(job.get('cpu_threads', 4)))
    out = Path(job['out'])
    out.mkdir(parents=True, exist_ok=True)
    spec = job['model']
    if spec['family'] != 'yolo':
        from checkpoint_config import resolve_model_config
        spec['config'] = resolve_model_config(spec, out)
    else:
        spec['config_source'] = 'explicit' if spec.get('config') else 'checkpoint:model'
    if spec['family'] == 'deim':
        spec['source_config'] = spec['config']
        spec['config'] = effective_deim_config(spec, out)
    annotation, mapping = read_coco(job['ann_file'])
    image_infos = {int(row['id']): row for row in annotation['images']}
    names = {int(row['id']): row['name'] for row in annotation['categories']}

    if job.get('list_layers'):
        model, _, _ = load_cam_model(job)
        layers = [{'name': name, 'type': type(module).__name__}
                  for name, module in model.named_modules() if isinstance(module, torch.nn.Conv2d)]
        (out / 'cam_layers.json').write_text(json.dumps(layers, indent=2), encoding='utf-8')
        print(f'Available convolution layers saved to {out / "cam_layers.json"}')
        return

    predictions = infer(job, out)
    metrics = evaluate(job['ann_file'], out / 'predictions.json', out)
    metrics.update(checkpoint=str(Path(spec['checkpoint']).resolve()),
                   checkpoint_sha256=sha256(spec['checkpoint']),
                   predictions_sha256=sha256(out / 'predictions.json'))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    reference = check_reference(job, metrics)
    (out / 'reference_check.json').write_text(json.dumps(reference, indent=2), encoding='utf-8')
    print(f'{spec["name"]}: reference={reference["status"]}; '
          f'P={metrics["Precision"]:.6f} R={metrics["Recall"]:.6f} F1={metrics["F1"]:.6f}', flush=True)

    threshold = metrics['Best_Confidence']
    matches = match_predictions(annotation, predictions, threshold)
    for kind in ('TP', 'FP', 'FN'):
        if sum(len(row[kind]) for row in matches.values()) != metrics[kind]:
            raise AssertionError(f'Visualization {kind} count differs from the shared evaluator')

    model, post, dataset = load_cam_model(job)
    layer_name, layer = resolve_layer(model, spec.get('cam_layer', 'auto'), spec['family'])
    cam_class = spec.get('cam_class')
    class_index = None
    if cam_class is not None:
        lower_names = {name.lower(): category for category, name in names.items()}
        if str(cam_class).lower() not in lower_names:
            raise ValueError(f'Unknown cam_class: {cam_class}')
        wanted_id = lower_names[str(cam_class).lower()]
        class_index = next(index for index, category in mapping.items() if category == wanted_id)

    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    reports = []
    for image_id in job['image_ids']:
        info = image_infos[image_id]
        image = Image.open(Path(job['image_dir']) / info['file_name']).convert('RGB')
        pixels, water, crop = cam_pixels(job, image, image_id, dataset, model)
        if spec['family'] == 'yolo':
            def target_forward(x):
                return yolo_head_target(model, x, threshold, job['cam_topk'], class_index)
        else:
            def target_forward(x):
                outputs = model(x, water=water) if spec['family'] == 'deim' else model(x)
                logits = outputs['pred_logits']
                probabilities = logits.sigmoid() if getattr(post, 'use_focal_loss', True) else logits.softmax(-1)[..., :-1]
                return score_target(probabilities, threshold, job['cam_topk'], class_index)
        cam_tensor, details = gradcam(model, layer, pixels, target_forward)
        cam = cam_tensor.numpy()
        if crop:
            left, top, width, height = crop
            cam = cam[top:top+height, left:left+width]
        cam = np.asarray(Image.fromarray(cam).resize(image.size, Image.Resampling.BILINEAR))
        maximum = float(cam.max())
        if maximum > 0:
            cam = cam / maximum
        np.save(figures / f'{image_id}_gradcam.npy', cam)
        matched = matches.get(image_id, {'TP': [], 'FP': [], 'FN': []})
        per_image = local_metrics(matched)
        detections = sorted(matched['TP'] + matched['FP'], key=lambda row: -row['score'])
        header(draw_detections(image, detections, names, show_labels=not job.get('hide_labels', False)),
               spec['display_name'], height=30).save(figures / f'{image_id}_detections.png')
        header(cam_overlay(image, cam), spec['display_name'], height=30).save(figures / f'{image_id}_gradcam.png')
        reports.append({'image_id': image_id, 'file_name': info['file_name'], 'image_metrics': per_image,
                        'cam': details})
    manifest = {
        'model': spec, 'checkpoint_sha256': metrics['checkpoint_sha256'],
        'annotation_sha256': metrics['test_annotation_sha256'],
        'method': 'Grad-CAM: ReLU of feature channels weighted by spatially averaged score gradients',
        'cam_layer': layer_name, 'cam_target': f'top {job["cam_topk"]} confidence-passing class scores before NMS/query postprocessing',
        'cam_normalization': 'independent 0–1 normalization per image; relative activation, not comparable absolute confidence',
        'detection_colors': 'class-colored predictions; no TP/FP/FN or metric text drawn',
        'operating_threshold': threshold, 'reference_check': reference, 'images': reports,
    }
    (out / 'visualization_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.job.read_text(encoding='utf-8')))
