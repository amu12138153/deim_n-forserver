"""Sequential configuration + subprocess wrapper; no training implementation."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
GROUPS = {
    'variables': ['V0_visual', 'V1_T_DO_pH_Tur', 'V2_T_DO_pH', 'V3_DO_pH', 'V4_T_pH', 'V5_T_DO', 'V6_DO'],
    'modules': ['M0_visual', 'M1_QCA', 'M2_SA_QCA', 'M3_QCA_Gate', 'M4_full'],
}


def summarize(paths, output):
    fields = ['experiment', 'water_features', 'water_self_attn', 'query_water_cross_attn',
              'learnable_gate', 'Precision', 'Recall', 'F1', 'mAP50', 'mAP50_95', 'AP75', 'Params_M', 'GFLOPs']
    aliases = {'experiment': 'experiment_name', 'mAP50': 'mAP@0.5', 'mAP50_95': 'mAP@0.5:0.95',
               'AP75': 'AP@0.75', 'Params_M': 'Parameters_M'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in paths:
            summary = json.loads(Path(path).read_text(encoding='utf-8'))
            row = {key: summary[aliases.get(key, key)] for key in fields}
            row['water_features'] = '+'.join(row['water_features'])
            writer.writerow(row)
    print(f'Saved: {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=[*GROUPS, 'all'], default='variables')
    parser.add_argument('--root', default=None, help='Batch output directory')
    parser.add_argument('--config-dir', default='configs/deimv2/ablation_n',
                        help='Ablation YAML directory; use configs/deimv2/ablation for the historical Pico suite')
    parser.add_argument('--summary-only', action='store_true', help='Collect existing summaries under --root')
    parser.add_argument('--include-v6', action='store_true', help='Deprecated: V6 is now included by default')
    parser.add_argument('--only', nargs='+', choices=sum(GROUPS.values(), []),
                        help='Run only these experiments from the selected group, in group order')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--test-ann')
    parser.add_argument('--test-images')
    parser.add_argument('--test-sensor-csv')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--batch-size', type=int, help='Override the same training batch size for every ablation')
    parser.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    if args.summary_only and not args.root:
        parser.error('--summary-only requires --root')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    root = Path(args.root or HERE / 'outputs' / f'{datetime.datetime.now():%Y%m%d_%H%M%S}_ablation_n_query_{args.group}').resolve()
    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = HERE / config_dir
    config_dir = config_dir.resolve()
    if not config_dir.is_dir():
        raise FileNotFoundError(config_dir)
    if args.summary_only:
        paths = sorted(root.rglob('experiment_summary.json'))
        if not paths:
            raise FileNotFoundError(f'No experiment_summary.json under {root}')
        summarize(paths, root / 'ablation_summary.csv')
        return
    group_names = sum(GROUPS.values(), []) if args.group == 'all' else list(GROUPS[args.group])
    names = list(group_names)
    if args.only:
        invalid = set(args.only) - set(names)
        if invalid:
            parser.error(f'--only contains experiments outside --group {args.group}: {sorted(invalid)}')
        names = [name for name in names if name in set(args.only)]
    # Preserve completed rows when continuing only part of an existing batch.
    # Never overwrite ablation_summary.csv with just the newly selected names.
    completed = {}
    for name in group_names:
        summary_path = root / name / 'experiment_summary.json'
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            if summary.get('experiment_name') != name:
                raise ValueError(f'Summary experiment mismatch: {summary_path}')
            completed[name] = summary_path
    for name in names:
        if name in completed:
            print(f'Skipping completed experiment: {name}', flush=True)
            continue
        cmd = [sys.executable, str(HERE / 'run_experiment.py'),
               '-c', str(config_dir / f'{name}.yml'),
               '--name', name, '--dir', str(root / name), '--seed', str(args.seed), '--nproc', str(args.nproc)]
        if args.batch_size is not None:
            cmd += ['--batch-size', str(args.batch_size)]
        for flag, value in [('--test-ann', args.test_ann), ('--test-images', args.test_images),
                            ('--test-sensor-csv', args.test_sensor_csv), ('--device', args.device)]:
            if value:
                cmd += [flag, str(Path(value).resolve()) if flag != '--device' else value]
        if args.extra:
            cmd += ['--extra', *args.extra]
        print(subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=HERE, check=True)
            summary_path = root / name / 'experiment_summary.json'
            if not summary_path.is_file():
                raise FileNotFoundError(f'Experiment produced no summary: {summary_path}')
            completed[name] = summary_path
            summarize([completed[item] for item in group_names if item in completed],
                      root / 'ablation_summary.csv')


if __name__ == '__main__':
    main()
