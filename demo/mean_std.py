"""Estimate three-sensor normalization using training CSVs only."""
import argparse
import csv
import math
from pathlib import Path
import statistics

FEATURES = ('temperature', 'do', 'ph')


def sensor_statistics(files):
    values = {name: [] for name in FEATURES}
    for filename in files:
        with Path(filename).open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not set(FEATURES).issubset(reader.fieldnames or []):
                raise ValueError(f'{filename}: requires temperature, do, ph columns')
            for row in reader:
                for name in FEATURES:
                    if not (row.get(name) or '').strip():
                        continue
                    value = float(row[name])
                    if math.isfinite(value):
                        values[name].append(value)
    if any(not samples for samples in values.values()):
        raise ValueError('Every sensor must have at least one finite training value')
    means = [statistics.mean(values[name]) for name in FEATURES]
    stds = [max(statistics.pstdev(values[name]), 1e-6) for name in FEATURES]
    return means, stds, {name: len(samples) for name, samples in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='Training sensor CSVs; never include val/test')
    parser.add_argument('--out', type=Path, help='Write an includable DEIMTransformer statistics YAML')
    args = parser.parse_args()
    means, stds, counts = sensor_statistics(args.files)
    print('order =', list(FEATURES))
    print('means =', means)
    print('stds =', stds)
    print('finite_counts =', counts)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        text = '# Training CSV sources: ' + ', '.join(str(Path(path)) for path in args.files) + '\n'
        text += '# Population std (ddof=0), floor=1e-6; temperature, do, ph.\n'
        text += 'DEIMTransformer:\n  water_means: ' + str(means) + '\n  water_stds: ' + str(stds) + '\n'
        args.out.write_text(text, encoding='utf-8')
        print('Saved:', args.out)


if __name__ == '__main__':
    main()
