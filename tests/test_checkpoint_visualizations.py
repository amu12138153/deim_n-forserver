"""Metric correspondence, explicit checkpoint identity, and Grad-CAM gradients."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'comparison'))
from checkpoint_visuals import local_metrics, match_predictions
from evaluate_coco import best_f1
from gradcam_core import gradcam, score_target, yolo_head_target
from visualize_checkpoints import prepare_spec, run_worker
from common import sha256
from checkpoint_config import resolve_model_config, validate_deim_config
from ultralytics_adapter import load_model as load_yolo_model


class CheckpointVisualizationTests(unittest.TestCase):
    def test_worker_log_preserves_the_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'worker.log'
            with patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'worker.log'):
                    run_worker([sys.executable, '-c', "raise ValueError('config and weights mismatch')"],
                               directory, log, 'test_model')
            self.assertIn('ValueError: config and weights mismatch', log.read_text(encoding='utf-8'))

    def test_wrong_class_count_and_backbone_water_config_are_explained(self):
        state = {'decoder.enc_score_head.weight': torch.zeros(2, 128)}
        with self.assertRaisesRegex(ValueError, 'num_classes: config=80, checkpoint=2') as error:
            validate_deim_config({'num_classes': 80, 'HGNetv2': {}}, state, 'baseline')
        self.assertIn('HGNetv2.use_water_quality: config=True, checkpoint=False', str(error.exception))
        validate_deim_config({'num_classes': 2, 'HGNetv2': {'use_water_quality': False}}, state, 'baseline')

    def test_config_can_be_omitted_for_every_family(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'weights.pth'
            checkpoint.write_bytes(b'parameters')
            for family in ('deim', 'dfine', 'rtdetr', 'rtdetrv2', 'yolo'):
                source = {'family': family, 'checkpoint': str(checkpoint), 'repo_workdir': str(root)}
                for config in (None, ''):
                    source['config'] = config
                    spec = prepare_spec('custom_label', source, root, 'annotation')
                    self.assertEqual(spec['family'], family)
                    self.assertFalse(spec.get('config'))

    def test_supplied_architecture_controls_loader_independently_of_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'chosen.pth'
            checkpoint.write_bytes(b'parameters')
            (root / 'base.yml').write_text('model: DEIM\nDEIMTransformer: {}\n')
            (root / 'explicit.yml').write_text('__include__: [base.yml]\n')
            spec = prepare_spec('yolo11n', {'family': 'yolo', 'checkpoint': str(checkpoint),
                                          'config': 'explicit.yml'}, root, 'test')
            self.assertEqual(spec['family'], 'deim')
            self.assertEqual(spec['family_source'], 'config')
            self.assertEqual(spec['display_name'], 'yolo11n')
            checkpoint = root / 'chosen.pt'
            checkpoint.write_bytes(b'saved YOLO')
            spec = prepare_spec('ours', {'family': 'deim', 'checkpoint': str(checkpoint)}, root, 'test')
            self.assertEqual(spec['family'], 'yolo')

    def test_config_is_recovered_from_weights_and_explicit_input_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'chosen.pth'
            torch.save({'model_config': {'model': 'DEIM', 'DEIMTransformer': {}}, 'model': {}}, checkpoint)
            spec = {'name': 'custom', 'checkpoint': str(checkpoint)}
            resolved = resolve_model_config(spec, root)
            self.assertIn('model: DEIM', Path(resolved).read_text())
            self.assertEqual(spec['config_source'], 'checkpoint:model_config')
            explicit = root / 'explicit.yml'
            explicit.write_text('model: DFINE\n')
            spec['config'] = str(explicit)
            with patch('torch.load', side_effect=AssertionError('Must not override explicit input')):
                self.assertEqual(resolve_model_config(spec, root), str(explicit))
            torch.save({'model': {}}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'no usable model configuration'):
                resolve_model_config({'name': 'custom', 'checkpoint': str(checkpoint)}, root)

    def test_yolo_supplied_config_receives_trained_weights_and_names(self):
        created = []
        def fake_yolo(path):
            channels = 3 if path == 'wrong.yaml' else 2
            model = torch.nn.Conv2d(3, channels, 1)
            model.names = {0: 'normal', 1: 'hypoxia'} if path == 'best.pt' else {0: 'untrained'}
            model.args = {'imgsz': 640}
            with torch.no_grad():
                model.weight.fill_(.25 if path == 'best.pt' else 0)
            instance = SimpleNamespace(model=model, overrides={'model': path})
            created.append(path)
            return instance
        with patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=fake_yolo)}):
            loaded = load_yolo_model('best.pt', 'structure.yaml')
            self.assertEqual(created, ['best.pt', 'structure.yaml'])
            self.assertTrue(torch.all(loaded.model.weight == .25))
            self.assertEqual(loaded.model.names, {0: 'normal', 1: 'hypoxia'})
            with self.assertRaisesRegex(ValueError, 'does not match checkpoint'):
                load_yolo_model('best.pt', 'wrong.yaml')

    def test_visualized_matches_reproduce_best_f1_counts(self):
        annotation = {'annotations': [
            {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10]},
            {'image_id': 1, 'category_id': 1, 'bbox': [20, 20, 10, 10]},
            {'image_id': 2, 'category_id': 0, 'bbox': [0, 0, 10, 10]},
        ]}
        predictions = [
            {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10], 'score': .9},
            {'image_id': 1, 'category_id': 0, 'bbox': [0, 0, 10, 10], 'score': .8},
            {'image_id': 1, 'category_id': 1, 'bbox': [20, 20, 10, 10], 'score': .7},
            {'image_id': 2, 'category_id': 1, 'bbox': [0, 0, 10, 10], 'score': .6},
        ]
        best = best_f1(annotation, predictions)
        matched = match_predictions(annotation, predictions, best['Best_Confidence'])
        for kind in ('TP', 'FP', 'FN'):
            self.assertEqual(sum(len(row[kind]) for row in matched.values()), best[kind])
        one = local_metrics(matched[1])
        self.assertEqual(one['TP'], 2)
        self.assertEqual(one['FP'], 1)
        self.assertAlmostEqual(one['F1'], .8)

    def test_explicit_checkpoint_must_match_reference_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'chosen.pt'
            checkpoint.write_bytes(b'weight A')
            reference = root / 'metrics.json'
            reference.write_text(json.dumps({'checkpoint_sha256': sha256(checkpoint),
                                             'test_annotation_sha256': 'test'}), encoding='utf-8')
            source = {'family': 'yolo', 'checkpoint': str(checkpoint), 'reference_summary': str(reference)}
            self.assertEqual(prepare_spec('yolov8n', source, root, 'test')['checkpoint'], str(checkpoint))
            checkpoint.write_bytes(b'weight B')
            with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
                prepare_spec('yolov8n', source, root, 'test')

    def test_gradcam_uses_actual_feature_gradients(self):
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 3, padding=1, bias=False), torch.nn.ReLU())
        with torch.no_grad():
            model[0].weight.fill_(.01)
        def target(pixels):
            score = model(pixels).mean()
            return score, [float(score.detach())]
        cam, report = gradcam(model, model[0], torch.ones(1, 3, 16, 16), target)
        self.assertEqual(tuple(cam.shape), (16, 16))
        self.assertGreater(float(cam.max()), 0)
        self.assertEqual(report['status'], 'ok')
        self.assertEqual(len(model[0]._forward_hooks), 0)

    def test_no_confident_cam_target_is_reported(self):
        target, scores = score_target(torch.tensor([[[.1, .2]]]), .5, 10)
        self.assertIsNone(target)
        self.assertEqual(scores, [])

    def test_yolo_one2one_class_replay_recovers_feature_gradients(self):
        class Head(torch.nn.Module):
            end2end = True
            def __init__(self):
                super().__init__()
                self.one2one_cv3 = torch.nn.ModuleList([torch.nn.Conv2d(2, 2, 1)])
                with torch.no_grad():
                    self.one2one_cv3[0].weight.fill_(.1)
                    self.one2one_cv3[0].bias.fill_(1)
            def forward(self, features):
                return self.one2one_cv3[0](features[0].detach())
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.ModuleList([torch.nn.Conv2d(3, 2, 1), Head()])
                with torch.no_grad():
                    self.model[0].weight.fill_(.1)
                    self.model[0].bias.fill_(0)
            def forward(self, pixels):
                return self.model[-1]([self.model[0](pixels)])
        model = Detector().eval()
        cam, report = gradcam(model, model.model[0], torch.ones(1, 3, 8, 8),
                              lambda x: yolo_head_target(model, x, .5, 10))
        self.assertGreater(float(cam.max()), 0)
        self.assertEqual(report['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
