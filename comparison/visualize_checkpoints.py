"""Evaluate explicitly supplied model weights and create detections plus Grad-CAM."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

from common import ROOT, read_coco, sha256
from checkpoint_visuals import assemble_grids
from checkpoint_config import infer_family, read_config_tree


FAMILIES = {'deimv2_n': 'deim', 'ours': 'deim', 'dfine_n': 'dfine', 'dfine_s': 'dfine',
            'rtdetr_r18': 'rtdetr', 'rtdetrv2_s': 'rtdetrv2',
            'yolov8n': 'yolo', 'yolo11n': 'yolo', 'yolo26n': 'yolo'}


def run_worker(command, cwd, log_file, model_name):
    """Stream the actual worker error and keep a complete per-model log."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    with Path(log_file).open('w', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace') as worker:
            for line in worker.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = worker.wait()
    if returncode:
        raise RuntimeError(f'{model_name}: worker exited with status {returncode}; '
                           f'the original error is above and saved to {Path(log_file).resolve()}')


def resolve_path(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def prepare_spec(name, source, base, annotation_hash):
    spec = dict(source, name=name)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError(f'Invalid model key: {name}')
    spec.setdefault('display_name', name)
    for key in ('checkpoint', 'config', 'repo_workdir', 'python', 'reference_summary'):
        if spec.get(key):
            spec[key] = resolve_path(spec[key], base)
            if not Path(spec[key]).exists():
                raise FileNotFoundError(f'{name}.{key}: {spec[key]}')
    requested_family = spec.get('family') or FAMILIES.get(name)
    actual_family = infer_family(read_config_tree(spec['config'])) if spec.get('config') else None
    if actual_family:
        spec['family'] = actual_family
        spec['family_source'] = 'config'
    elif spec.get('checkpoint') and Path(spec['checkpoint']).suffix.lower() == '.pt':
        spec['family'] = 'yolo'
        spec['family_source'] = 'checkpoint_extension'
    else:
        spec['family'] = requested_family
        spec['family_source'] = 'family_hint'
    if spec['family'] not in {'deim', 'yolo', 'dfine', 'rtdetr', 'rtdetrv2'}:
        raise ValueError(f'{name}: specify a supported family')
    required = ['checkpoint']
    if spec['family'] not in {'deim', 'yolo'}:
        required.append('repo_workdir')
    for key in required:
        if not spec.get(key):
            raise ValueError(f'{name}: {key} must be explicitly supplied')
    if spec.get('reference_summary'):
        reference = json.loads(Path(spec['reference_summary']).read_text(encoding='utf-8'))
        if reference.get('checkpoint_sha256') != sha256(spec['checkpoint']):
            raise ValueError(f'{name}: specified weight does not match reference checkpoint SHA256')
        if reference.get('test_annotation_sha256', reference.get('Test_Annotation_SHA256')) != annotation_hash:
            raise ValueError(f'{name}: test annotation does not match reference result')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-file', type=Path, help='YAML with models: mapping; relative paths use its directory')
    parser.add_argument('--models', nargs='+', help='Run selected keys from --models-file')
    parser.add_argument('--model-key', help='Single-model mode; all checkpoint paths are explicit')
    parser.add_argument('--family', choices=['deim', 'yolo', 'dfine', 'rtdetr', 'rtdetrv2'])
    parser.add_argument('--checkpoint')
    parser.add_argument('--model-config', help='Optional explicit architecture configuration; used when supplied')
    parser.add_argument('--repo-workdir')
    parser.add_argument('--reference-summary', help='Original metrics.json or comparison experiment_summary.json')
    parser.add_argument('--cam-layer', default='auto')
    parser.add_argument('--cam-class', choices=['normal', 'hypoxia'])
    parser.add_argument('--list-layers', action='store_true')
    parser.add_argument('--hide-labels', action='store_true', help='Draw only boxes, without class/confidence text')
    parser.add_argument('--ann-file', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--sensor-csv', type=Path, help='Matching test sensors for water-fusion Ours')
    parser.add_argument('--image-ids', nargs='+', type=int)
    parser.add_argument('--images', nargs='+', help='Exact annotation file_name values to visualize')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--cam-topk', type=int, default=10)
    parser.add_argument('--metric-tolerance', type=float, default=1e-5)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true', help='Replace generated files in this output directory')
    args = parser.parse_args()
    if bool(args.models_file) == bool(args.model_key):
        parser.error('Supply either --models-file or --model-key')
    if args.image_size != 640:
        parser.error('Use 640 to match the existing comparison inference protocol')
    if args.cam_topk < 1 or args.cpu_threads < 1 or args.metric_tolerance < 0:
        parser.error('CAM topk/CPU threads must be positive; tolerance must be nonnegative')
    if args.image_ids and args.images:
        parser.error('Use --image-ids or --images')
    if args.models_file and any((args.checkpoint, args.model_config, args.repo_workdir, args.reference_summary, args.family)):
        parser.error('Use per-model entries in --models-file for model paths and family')
    annotation, _ = read_coco(args.ann_file)
    annotation_hash = sha256(args.ann_file)
    infos = {int(row['id']): row for row in annotation['images']}
    image_ids = args.image_ids or sorted(infos)[:3]
    if args.images:
        by_name = {row['file_name']: int(row['id']) for row in annotation['images']}
        if any(name not in by_name for name in args.images):
            parser.error('One of --images file names is absent from the annotation')
        image_ids = [by_name[name] for name in args.images]
    if not image_ids or len(set(image_ids)) != len(image_ids) or any(i not in infos for i in image_ids):
        parser.error('Select unique image IDs that exist in the annotation')
    # Every image is evaluated; only the selected IDs are drawn.
    for info in annotation['images']:
        if not (args.image_dir / info['file_name']).is_file():
            raise FileNotFoundError(args.image_dir / info['file_name'])
    if args.sensor_csv and not args.sensor_csv.is_file():
        raise FileNotFoundError(args.sensor_csv)
    if args.models_file:
        base = args.models_file.resolve().parent
        sources = yaml.safe_load(args.models_file.read_text(encoding='utf-8'))['models']
        selected = args.models or list(sources)
        if len(set(selected)) != len(selected) or any(name not in sources for name in selected):
            parser.error('--models must contain unique keys from --models-file')
        specs = [prepare_spec(name, sources[name], base, annotation_hash) for name in selected]
    else:
        if args.models:
            parser.error('--models requires --models-file')
        source = {'checkpoint': args.checkpoint, 'config': args.model_config, 'repo_workdir': args.repo_workdir,
                  'reference_summary': args.reference_summary, 'cam_layer': args.cam_layer, 'cam_class': args.cam_class}
        if args.family:
            source['family'] = args.family
        specs = [prepare_spec(args.model_key, source, Path.cwd(), annotation_hash)]
    if not specs:
        parser.error('No models selected')
    for spec in specs:
        if args.cam_layer != 'auto':
            spec['cam_layer'] = args.cam_layer
        if args.cam_class:
            spec['cam_class'] = args.cam_class
    for spec in specs:
        model_out = args.out.resolve() / spec['name']
        if model_out.exists() and any(model_out.iterdir()) and not args.overwrite:
            raise FileExistsError(f'{model_out} is not empty; choose a new --out or use --overwrite')
    for spec in specs:
        model_out = args.out.resolve() / spec['name']
        model_out.mkdir(parents=True, exist_ok=True)
        job = {'model': spec, 'ann_file': str(args.ann_file.resolve()), 'image_dir': str(args.image_dir.resolve()),
               'sensor_csv': str(args.sensor_csv.resolve()) if args.sensor_csv else None,
               'image_size': args.image_size, 'image_ids': image_ids, 'device': args.device,
               'cam_topk': args.cam_topk, 'metric_tolerance': args.metric_tolerance,
               'out': str(model_out), 'list_layers': args.list_layers, 'cpu_threads': args.cpu_threads,
               'hide_labels': args.hide_labels}
        job_file = model_out / 'job.json'
        job_file.write_text(json.dumps(job, indent=2), encoding='utf-8')
        command = [spec.get('python') or sys.executable, str(Path(__file__).with_name('checkpoint_visualization_worker.py')),
                   '--job', str(job_file)]
        print(f'===== {spec["name"]}: loader={spec["family"]}, checkpoint={spec["checkpoint"]} =====', flush=True)
        run_worker(command, ROOT, model_out / 'worker.log', spec['name'])
    if args.list_layers:
        return
    fields = ['Model', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50_95', 'Best_Confidence',
              'checkpoint', 'checkpoint_sha256', 'test_annotation_sha256', 'Reference_Status']
    with (args.out / 'visualization_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for spec in specs:
            folder = args.out / spec['name']
            metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
            reference = json.loads((folder / 'reference_check.json').read_text(encoding='utf-8'))
            row = {key: metrics.get(key) for key in fields}
            row.update(Model=spec['display_name'], Reference_Status=reference['status'])
            writer.writerow(row)
    assemble_grids(annotation, args.image_dir, specs, args.out, image_ids)
    print(f'Finished: {args.out.resolve()}')


if __name__ == '__main__':
    main()
