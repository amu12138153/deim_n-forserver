import json
import argparse

def remap_categories(ann_path, out_path):
    with open(ann_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    # 按原id排序，映射为 0, 1, ...
    old_ids = sorted(c['id'] for c in d['categories'])
    mapping = {old: new for new, old in enumerate(old_ids)}
    for c in d['categories']:
        c['id'] = mapping[c['id']]
    for a in d['annotations']:
        a['category_id'] = mapping[a['category_id']]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False)
    print(f'{out_path}  映射关系: {mapping}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('annotation', help='Source COCO annotation JSON')
    parser.add_argument('output', help='Destination annotation JSON')
    args = parser.parse_args()
    remap_categories(args.annotation, args.output)
