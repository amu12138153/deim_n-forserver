import json
from pathlib import Path

d = json.loads((Path(__file__).resolve().parents[1] / 'fish_dataset622/train/coco_detection_train0.json').read_text(encoding='utf-8'))
print('categories:', d['categories'])
print('标注里的category_id集合:', sorted(set(a['category_id'] for a in d['annotations'])))
