"""Small end-to-end check for saved-prediction visualizations."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison import visualize_results


class ComparisonVisualizationsTests(unittest.TestCase):
    def test_same_annotation_produces_detection_and_score_heatmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'
            results = root / 'results' / 'yolov8n'
            images.mkdir()
            results.mkdir(parents=True)
            Image.new('RGB', (64, 64), 'white').save(images / 'sample.jpg')
            annotation = root / 'test.json'
            annotation.write_text(json.dumps({
                'images': [{'id': 3, 'file_name': 'sample.jpg', 'width': 64, 'height': 64}],
                'categories': [{'id': 0, 'name': 'normal'}],
                'annotations': [],
            }), encoding='utf-8')
            digest = hashlib.sha256(annotation.read_bytes()).hexdigest()
            (results / 'metrics.json').write_text(
                json.dumps({'test_annotation_sha256': digest}), encoding='utf-8')
            (results / 'predictions.json').write_text(json.dumps([
                {'image_id': 3, 'category_id': 0, 'bbox': [16, 16, 20, 20], 'score': 0.9},
            ]), encoding='utf-8')
            out = root / 'out'
            argv = ['visualize_results.py', '--ann-file', str(annotation),
                    '--image-dir', str(images), '--results-root', str(root / 'results'),
                    '--models', 'yolov8n', '--out', str(out)]
            with patch.object(sys, 'argv', argv):
                visualize_results.main()
            model_dir = out / 'yolov8n'
            self.assertTrue((model_dir / 'sample_detections.png').is_file())
            self.assertTrue((model_dir / 'sample_score_heatmap.png').is_file())
            self.assertTrue((model_dir / 'sample_pair.png').is_file())
            self.assertTrue((out / 'sample_comparison.png').is_file())
            self.assertNotEqual(
                Image.open(model_dir / 'sample_score_heatmap.png').getpixel((26, 26)),
                Image.open(images / 'sample.jpg').getpixel((26, 26)),
            )
            self.assertEqual(json.loads((out / 'manifest.json').read_text())['annotation_sha256'], digest)

            (results / 'metrics.json').write_text(
                json.dumps({'test_annotation_sha256': 'wrong'}), encoding='utf-8')
            with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'different annotation'):
                visualize_results.main()


if __name__ == '__main__':
    unittest.main()
