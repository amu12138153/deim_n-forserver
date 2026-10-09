"""Create explicitly labeled illustrative AP curves from measured training logs.

This does not produce replacement training logs or measured evaluation results.
Only the two AP series are transformed; the original values are retained in CSV.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


TARGETS = {
    'baseline': {'map50': 0.8862546986334597, 'map50_95': 0.6709050382214434},
    'dfine_n': {'map50': 0.9139435953119819, 'map50_95': 0.7012819905186731},
}


def read_series(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    rows.sort(key=lambda row: row['epoch'])
    if [row['epoch'] for row in rows] != list(range(200)):
        raise ValueError(f'{path}: expected one record for each epoch 0–199')
    result = {'map50': [], 'map50_95': []}
    for row in rows:
        stats = row.get('test_coco_eval_bbox', [])
        values = (row.get('test_map_50', stats[1] if len(stats) > 1 else None),
                  row.get('test_map_50_95', stats[0] if stats else None))
        for key, value in zip(result, values):
            if value is None or not math.isfinite(float(value)) or not 0 <= value <= 1:
                raise ValueError(f'{path}: invalid {key} at epoch {row["epoch"]}')
            result[key].append(float(value))
    return result


def sample(values, position):
    left = min(int(position), len(values) - 1)
    right = min(left + 1, len(values) - 1)
    fraction = position - left
    return values[left] * (1 - fraction) + values[right] * fraction


def moving_average(values, index, radius=5):
    section = values[max(0, index-radius):min(len(values), index+radius+1)]
    return sum(section) / len(section)


def illustrate(values, target, peak_epoch=176, final_gap=0.0015):
    """Time-warp early rise; retain residual fluctuations around a late trend."""
    early = []
    for epoch in range(1, 101):
        if epoch <= 60:
            original_position = 30 * ((epoch - 1) / 59) ** 2
        else:
            original_position = 30 + 70 * (epoch - 60) / 40
        early.append(sample(values, original_position))

    # Keep every early point below the requested later maximum.
    headroom = 0.004 if target > 0.8 else 0.006
    ceiling = target - headroom
    initial = early[0]
    early_max = max(early)
    scale = min(1.0, (ceiling - initial) / (early_max - initial))
    early = [initial + scale * (value - initial) for value in early]

    residual = [value - moving_average(values, index) for index, value in enumerate(values)]
    adjusted = early[:]
    start = early[-1]
    for epoch in range(101, peak_epoch + 1):
        x = (epoch - 100) / (peak_epoch - 100)
        smoothstep = x * x * (3 - 2 * x)
        trend = start + (target - start) * smoothstep
        # Remove endpoint offsets so the target is attained exactly at epoch 176.
        offset = residual[epoch - 1] - ((1 - x) * residual[99] + x * residual[peak_epoch - 1])
        value = trend + 0.65 * offset
        if epoch != peak_epoch:
            value = min(value, target - 0.00001)
        adjusted.append(value)
    for epoch in range(peak_epoch + 1, 201):
        x = (epoch - peak_epoch) / (200 - peak_epoch)
        trend = target - final_gap * x
        offset = residual[epoch - 1] - ((1 - x) * residual[peak_epoch - 1] + x * residual[199])
        value = min(trend + 0.35 * offset, target - 0.00001)
        adjusted.append(value)
    if len(adjusted) != 200 or max(adjusted) != target or adjusted[-1] >= target:
        raise AssertionError('Illustrative curve did not meet its declared anchors')
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--second', type=Path, required=True,
                        help='Second source log (here dfine_n.txt)')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'kind': 'illustrative_synthetic_ap_curves_not_measured',
                'peak_epoch': 176, 'inputs': {}, 'method':
                'Early 0–30 source epochs time-warped to plotted epochs 1–60; '
                'source local residuals retained; post-100 broad trend reshaped to '
                'peak at 176 and end 0.0015 below peak. Other log fields are not rewritten.'}
    for name, source in [('baseline', args.baseline), ('dfine_n', args.second)]:
        raw = read_series(source)
        adjusted = {key: illustrate(raw[key], TARGETS[name][key]) for key in raw}
        output = args.out_dir / f'{name}_illustrative.csv'
        with output.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch', 'original_map50', 'original_map50_95',
                             'illustrative_map50', 'illustrative_map50_95', 'status'])
            for index in range(200):
                writer.writerow([index + 1, raw['map50'][index], raw['map50_95'][index],
                                 adjusted['map50'][index], adjusted['map50_95'][index],
                                 'synthetic_illustration_not_measured'])
        manifest['inputs'][name] = {
            'source': str(source.resolve()),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'output': str(output.resolve()),
            'targets': TARGETS[name],
            'original_peaks': {key: {'value': max(values), 'epoch': values.index(max(values)) + 1}
                               for key, values in raw.items()},
        }
        print(output)
    (args.out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
