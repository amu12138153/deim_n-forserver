import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs='+', help='Training sensor CSV files; do not include val/test')
files = parser.parse_args().files
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

cols = ['temperature', 'do', 'ph', 'turbidity']  # 顺序必须和模型代码一致

stats = df[cols].agg(['mean', 'std', 'min', 'max']).round(3)
print(stats)

print('\nmeans =', df[cols].mean().round(3).tolist())
print('stds  =', df[cols].std().round(3).tolist())

print('\n与溶解氧DO的相关系数')
print(df[cols].corr()['do'].round(3))
