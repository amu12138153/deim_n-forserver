"""Check that every COCO image referenced by an annotation is readable by path."""

import json
from pathlib import Path


def validate_coco_images(ann_file, image_dir, *, val_image_dir=None):
    ann_file = Path(ann_file)
    image_dir = Path(image_dir)
    data = json.loads(ann_file.read_text(encoding='utf-8-sig'))
    images = data.get('images')
    if not isinstance(images, list):
        raise ValueError(f'COCO annotation has no images list: {ann_file}')

    missing = []
    in_validation = []
    for image in images:
        name = image.get('file_name')
        if not isinstance(name, str) or not name or not (image_dir / name).is_file():
            missing.append(name)
            if name and val_image_dir and (Path(val_image_dir) / name).is_file():
                in_validation.append(name)

    if missing:
        examples = ', '.join(repr(name) for name in missing[:5])
        message = (f'{len(missing)} of {len(images)} COCO images are missing from {image_dir}. '
                   f'Annotation: {ann_file}. First missing file_name values: {examples}. '
                   'Check that ann_file and img_folder refer to the same dataset split.')
        if in_validation:
            message += (f' {len(in_validation)} missing image(s) exist in the validation image '
                        f'directory {val_image_dir}; check whether the test annotation contains validation entries.')
        raise FileNotFoundError(message)
    return len(images)
