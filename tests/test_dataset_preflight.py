"""Checks for COCO image-path validation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_preflight import validate_coco_images


class DatasetPreflightTests(unittest.TestCase):
    def test_reports_validation_images_in_test_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / 'test'
            val_dir = root / 'val'
            test_dir.mkdir()
            val_dir.mkdir()
            (val_dir / 'val.jpg').touch()
            (test_dir / 'test.jpg').touch()
            annotation = test_dir / 'test.json'
            annotation.write_text(json.dumps({'images': [
                {'file_name': 'test.jpg'}, {'file_name': 'val.jpg'},
                {'file_name': 'absent.jpg'}]}), encoding='utf-8')
            with self.assertRaisesRegex(FileNotFoundError, '2 of 3 COCO images') as caught:
                validate_coco_images(annotation, test_dir, val_image_dir=val_dir)
            self.assertIn('validation image directory', str(caught.exception))
            annotation.write_text(json.dumps({'images': [{'file_name': 'test.jpg'}]}), encoding='utf-8')
            self.assertEqual(validate_coco_images(annotation, test_dir), 1)


if __name__ == '__main__':
    unittest.main()
