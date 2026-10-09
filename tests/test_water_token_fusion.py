"""Shape, gradient, sparsity, mixed precision and shared-decoder contracts."""
import csv
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.data.dataset.coco_dataset import CocoDetection
from engine.deim.deim_decoder import DEIMTransformer
from engine.mynewblock.WaterQueryCrossAttention import WaterTokenEncoder, WaterAwareQueryFusion
from engine.mynewblock.water_token_fusion import RankAwareWaterAttention, GlobalSparseQueryWaterCrossAttention, MultiSparseQueryWaterCrossAttention, ComplementaryQueryWaterGate
from engine.solver._solver import BaseSolver
from engine.water_schema import WATER_FEATURES


def decoder(flags, batch_queries=10):
    return DEIMTransformer(num_classes=2, hidden_dim=128, feat_channels=[128,128],
        feat_strides=[16,32], num_levels=2, num_queries=batch_queries, num_layers=3,
        num_points=[2,2], dim_feedforward=256, num_denoising=0,
        use_water_quality=True, water_features=list(WATER_FEATURES),
        water_legacy_init_scale=0.01, use_rwa=flags[0], use_msqwca=flags[1], use_cqwg=flags[2])


class WaterTokenFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_requested_random_chain_forward_and_backward(self):
        for batch in (1,2):
            with self.subTest(batch=batch):
                encoder = WaterTokenEncoder(use_rwa=True)
                fusion = WaterAwareQueryFusion(use_msqwca=True, use_cqwg=True)
                water = torch.randn(batch,3, requires_grad=True)
                query = torch.randn(batch,300,128, requires_grad=True)
                output = fusion(query, encoder(water))
                self.assertEqual(tuple(output.shape), (batch,300,128))
                self.assertTrue(torch.isfinite(output).all())
                output.mean().backward()
                for module in (encoder, fusion):
                    for name, parameter in module.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertTrue(torch.isfinite(water.grad).all())
                self.assertTrue(torch.isfinite(query.grad).all())

    def test_global_sparse_branches_and_requested_mean_loss_gradients(self):
        torch.manual_seed(19)
        for sparse_k in (1, 2):
            with self.subTest(sparse_k=sparse_k):
                module = GlobalSparseQueryWaterCrossAttention(sparse_k=sparse_k)
                module.save_attention = True
                query = torch.randn(2,300,128, requires_grad=True)
                water = torch.randn(2,3,64, requires_grad=True)
                result = module(query, water)
                self.assertEqual(result.shape,query.shape)
                self.assertTrue(torch.isfinite(result).all())
                global_attn, sparse_attn = module.last_branch_attention
                for branch, count in ((global_attn, 3), (sparse_attn, sparse_k)):
                    self.assertEqual(tuple(branch.shape), (2,8,300,3))
                    self.assertTrue(((branch > 0).sum(-1) == count).all())
                    torch.testing.assert_close(branch.sum(-1), torch.ones(2,8,300))
                self.assertEqual(module.last_global_attention.shape,(2,300,3))
                self.assertEqual(module.last_sparse_attention.shape,(2,300,3))
                self.assertEqual(module.last_branch_weights.shape,(2,300,2))
                self.assertEqual(module.last_sparse_mask.shape,(2,8,300,3))
                self.assertTrue((module.last_sparse_mask.sum(-1)==sparse_k).all())
                torch.testing.assert_close(module.last_branch_weights,torch.full((2,300,2),0.5),rtol=0,atol=0)
                torch.testing.assert_close(module.last_attention,0.5*(module.last_global_attention+module.last_sparse_attention))
                for saved in (module.last_global_attention,module.last_sparse_attention,module.last_branch_weights,module.last_sparse_mask):
                    self.assertEqual(saved.device.type,'cpu')
                    self.assertFalse(saved.requires_grad)
                result.mean().backward()
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad,name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(),name)
                for projection in (module.water_proj,module.q_proj,module.k_proj,module.v_proj,module.out_proj,module.branch_router[-1]):
                    self.assertGreater(float(projection.weight.grad.abs().sum()),0)
                self.assertTrue((water.grad.abs().sum(-1)>0).all())
                # The zero final layer intentionally blocks first-router-layer gradients at initialization.
                self.assertEqual(float(module.branch_router[0].weight.grad.abs().sum()),0)
                module.save_attention = False
                module(query,water)
                for name in ('last_attention','last_branch_attention','last_global_attention','last_sparse_attention','last_branch_weights','last_sparse_mask'):
                    self.assertIsNone(getattr(module,name))
        for bad in (0,3,True,1.5):
            with self.assertRaises(ValueError):
                GlobalSparseQueryWaterCrossAttention(sparse_k=bad)

    def test_query_router_changes_by_query_and_receives_hidden_gradients_after_update(self):
        torch.manual_seed(23)
        module = GlobalSparseQueryWaterCrossAttention()
        query,water = torch.randn(2,20,128),torch.randn(2,3,64)
        optimizer = torch.optim.SGD(module.parameters(),lr=0.2)
        result = module(query,water)
        (result*torch.randn_like(result)).mean().backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        module.save_attention = True
        result = module(query,water)
        weights = module.last_branch_weights
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,20))
        self.assertGreater(float(weights[...,0].std()),0)
        result.square().mean().backward()
        self.assertGreater(float(module.branch_router[0].weight.grad.abs().sum()),0)

    def test_legacy_name_and_config_select_global_sparse_semantics(self):
        alias = MultiSparseQueryWaterCrossAttention(topk=[1,2])
        self.assertIsInstance(alias,GlobalSparseQueryWaterCrossAttention)
        self.assertEqual(alias.sparse_k,2)
        self.assertNotIn('branch_logits',dict(alias.named_parameters()))
        fusion = WaterAwareQueryFusion(use_msqwca=True,query_water_fusion={
            'type':'MultiSparseQueryWaterCrossAttention','topk':[1,2]})
        self.assertIsInstance(fusion.cross_attn,GlobalSparseQueryWaterCrossAttention)
        with self.assertRaises(ValueError):
            MultiSparseQueryWaterCrossAttention(topk=[0,2])

    def test_rank_statistics_and_degenerate_tokens_are_finite(self):
        module = RankAwareWaterAttention()
        module.save_attention = True
        for data in (torch.zeros(2,3,64),torch.ones(2,3,64),torch.randn(2,3,64)):
            data.requires_grad_(True)
            output = module(data)
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(tuple(module.last_rank_stats.shape),(2,4,2))
            self.assertTrue(torch.isfinite(module.last_effective_rank).all())
            self.assertTrue((module.last_effective_rank <= 2.001).all())
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(data.grad).all())
            module.zero_grad(set_to_none=True)

    def test_simplified_rwa_mean_backward_and_detached_router_statistics(self):
        torch.manual_seed(31)
        module=RankAwareWaterAttention(dim=64,num_heads=4,init_scale=0.01)
        module.save_attention=True
        data=torch.randn(2,3,64,requires_grad=True)
        statistics=[]
        router_inputs=[]
        original_statistics=module._rank_statistics
        def capture_statistics(y):
            stats,centered=original_statistics(y)
            stats.retain_grad()
            statistics.append(stats)
            return stats,centered
        module._rank_statistics=capture_statistics
        handle=module.router.register_forward_pre_hook(lambda m,a: router_inputs.append(a[0]))
        try:
            output=module(data)
            self.assertEqual(output.shape,(2,3,64))
            self.assertTrue(torch.isfinite(output).all())
            self.assertFalse(hasattr(module,'norm'))
            self.assertEqual(module.router[-1].out_features,1)
            self.assertAlmostEqual(float(module.alpha.detach()),0.01,places=7)
            self.assertEqual(router_inputs[0].shape,(2,4,2))
            self.assertFalse(router_inputs[0].requires_grad)
            self.assertEqual(module.last_relation_strength.shape,(2,4,1,1))
            self.assertEqual(module.last_relation_strength.device.type,'cpu')
            self.assertFalse(module.last_relation_strength.requires_grad)
            self.assertTrue(((module.last_relation_strength>0)&(module.last_relation_strength<1)).all())
            output.mean().backward()
            self.assertIsNone(statistics[0].grad)
            for projection in (module.q_proj,module.k_proj,module.v_proj,module.relation[0],module.router[0],module.out_proj):
                self.assertIsNotNone(projection.weight.grad)
                self.assertTrue(torch.isfinite(projection.weight.grad).all())
                self.assertGreater(float(projection.weight.grad.abs().sum()),0)
            self.assertIsNotNone(module.alpha.grad)
            self.assertTrue(torch.isfinite(module.alpha.grad))
            self.assertTrue(torch.isfinite(data.grad).all())
            module.save_attention=False
            module(data)
            for name in ('last_attention','last_rank_stats','last_effective_rank','last_relation_strength'):
                self.assertIsNone(getattr(module,name))
        finally:
            handle.remove()

    def test_old_rwa_checkpoint_shape_mismatches_are_limited_to_final_router(self):
        module=RankAwareWaterAttention()
        old_state={name:value.clone() for name,value in module.state_dict().items()}
        old_state['router.2.weight']=torch.randn(2,16)
        old_state['router.2.bias']=torch.randn(2)
        old_state['norm.weight']=torch.ones(64)
        old_state['norm.bias']=torch.zeros(64)
        with self.assertRaisesRegex(RuntimeError,'size mismatch for router.2.weight'):
            module.load_state_dict(old_state,strict=False)
        matched,info=BaseSolver._matched_state(module.state_dict(),old_state)
        self.assertEqual(set(info['unmatched']),{'router.2.weight','router.2.bias'})
        self.assertEqual(set(old_state)-set(module.state_dict()),{'norm.weight','norm.bias'})
        missing=module.load_state_dict(matched,strict=False)
        self.assertEqual(set(missing.missing_keys),{'router.2.weight','router.2.bias'})
        self.assertFalse(missing.unexpected_keys)

    def test_small_scale_zero_is_exact_identity_and_gate_is_vector(self):
        water, query = torch.randn(2,3,64), torch.randn(2,11,128)
        rwa = RankAwareWaterAttention(init_scale=0)
        gate = ComplementaryQueryWaterGate(init_scale=0)
        gate.save_attention = True
        torch.testing.assert_close(rwa(water), water, rtol=0, atol=0)
        torch.testing.assert_close(gate(query, torch.randn_like(query)), query, rtol=0, atol=0)
        self.assertEqual(tuple(gate.last_gate.shape), (2,11,128))
        self.assertTrue(((gate.last_gate > 0) & (gate.last_gate < 1)).all())

    def test_shared_encoder_runs_once_and_layer_parameters_are_independent(self):
        model = decoder((True,True,True)).eval()
        encoded, used = [], []
        handles = [model.water_encoder.register_forward_hook(lambda m,a,out: encoded.append(out))]
        for layer in model.decoder.layers:
            layer.water_fusion.save_attention = True
            handles.append(layer.water_fusion.register_forward_pre_hook(lambda m,a: used.append(a[1])))
        try:
            out = model([torch.randn(2,128,4,4),torch.randn(2,128,2,2)],water=torch.randn(2,3))
            self.assertEqual(len(encoded),1)
            self.assertEqual(len(used),3)
            self.assertTrue(all(item is encoded[0] for item in used))
            self.assertEqual(tuple(out['pred_logits'].shape),(2,10,2))
            self.assertEqual(len({id(layer.water_fusion.gate.lambda_gate) for layer in model.decoder.layers}),3)
            self.assertEqual(len({id(layer.water_fusion.cross_attn.branch_router[-1].weight) for layer in model.decoder.layers}),3)
            for row in model.water_fusion_diagnostics()['layers']:
                self.assertIsNotNone(row['mean_gate'])
                self.assertEqual(row['branch_weights_shape'],[2,10,2])
                self.assertEqual(row['sparse_k'],2)
                torch.testing.assert_close(torch.tensor(row['mean_branch_weights']),torch.tensor([0.5,0.5]))
        finally:
            for handle in handles: handle.remove()

    def test_all_eight_switch_combinations_support_detector_backward(self):
        for flags in itertools.product((False,True),repeat=3):
            with self.subTest(flags=flags):
                model = decoder(flags).train()
                features = [torch.randn(1,128,4,4,requires_grad=True),torch.randn(1,128,2,2,requires_grad=True)]
                result = model(features,water=torch.tensor([[27.4,4.8,8.0]]))
                loss = result['pred_logits'].square().mean()+result['pred_boxes'].square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_backbone_encoder_decoder_names_and_visual_behavior_are_preserved(self):
        full = decoder((True,True,True)).eval()
        baseline = DEIMTransformer(num_classes=2, hidden_dim=128, feat_channels=[128,128],
            feat_strides=[16,32], num_levels=2,num_queries=10,num_layers=3,num_points=[2,2],
            dim_feedforward=256,num_denoising=0).eval()
        matched, info = BaseSolver._matched_state(full.state_dict(),baseline.state_dict())
        self.assertFalse(info['unmatched'])
        self.assertTrue(all('water' in name for name in info['missed']))
        full.load_state_dict(matched,strict=False)
        features = [torch.randn(1,128,4,4),torch.randn(1,128,2,2)]
        with torch.no_grad():
            actual, expected = full(features,water=None), baseline(features)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)

    def test_three_column_sensor_csv_and_missing_image_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation={'images':[{'id':1,'file_name':'a.jpg','width':8,'height':8},{'id':2,'file_name':'b.jpg','width':8,'height':8}],
                        'annotations':[],'categories':[{'id':0,'name':'normal'},{'id':1,'name':'hypoxia'}]}
            (root/'ann.json').write_text(json.dumps(annotation))
            (root/'sensor.csv').write_text('file_name,temperature,do,ph\na.jpg,27.0,4.0,8.0\n')
            Image.new('RGB',(8,8)).save(root/'a.jpg')
            dataset=CocoDetection(str(root),str(root/'ann.json'),transforms=None,sensor_csv=str(root/'sensor.csv'),use_water_quality=True)
            self.assertEqual(tuple(dataset.get_water_quality(0).shape),(3,))
            torch.testing.assert_close(dataset.get_water_quality(0),torch.tensor([27.,4.,8.]))
            self.assertTrue(torch.isnan(dataset.get_water_quality(1)).all())
            self.assertEqual(tuple(dataset[0][1]['water_quality'].shape),(3,))
        with self.assertRaises(ValueError):
            WaterTokenEncoder()(torch.randn(2,4))
        with self.assertRaises(ValueError):
            WaterTokenEncoder(means=[1,2,3,4],stds=[1,1,1,1])

    def test_mixed_precision(self):
        devices=[('cpu',torch.bfloat16)]
        if torch.cuda.is_available(): devices.append(('cuda',torch.float16))
        for device,dtype in devices:
            with self.subTest(device=device):
                encoder=WaterTokenEncoder(use_rwa=True).to(device)
                fusion=WaterAwareQueryFusion(use_msqwca=True,use_cqwg=True).to(device)
                query=torch.randn(2,30,128,device=device,requires_grad=True)
                water=torch.tensor([[27.4,4.8,8.0],[27.1,4.1,8.1]],device=device)
                with torch.autocast(device_type=device,dtype=dtype):
                    output=fusion(query,encoder(water))
                    loss=output.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(output).all())
                for module in (encoder,fusion):
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None))

    def test_innovation_yamls_only_change_water_switches(self):
        names=('A0_baseline','A1_rwa','A2_rwa_msqwca','A3_full','A4_msqwca_only','A5_cqwg_only','A6_visual')
        base=load_config(str(ROOT/'configs/deimv2/water_innovations/A0_baseline.yml'),{})
        for name in names:
            config=load_config(str(ROOT/f'configs/deimv2/water_innovations/{name}.yml'),{})
            self.assertEqual(config['HGNetv2'],base['HGNetv2'])
            self.assertEqual(config['HybridEncoder'],base['HybridEncoder'])
            for split in ('train','val','test'):
                self.assertEqual(config[f'{split}_dataloader'],base[f'{split}_dataloader'])
            self.assertEqual(len(config['DEIMTransformer']['water_means']),3)
            self.assertEqual(len(config['DEIMTransformer']['water_stds']),3)
            self.assertEqual(config['DEIMTransformer']['water_num_params'],3)
            self.assertEqual(config['DEIMTransformer']['query_water_fusion']['type'],'GlobalSparseQueryWaterCrossAttention')
            self.assertEqual(config['DEIMTransformer']['query_water_fusion']['sparse_k'],2)


if __name__ == '__main__':
    unittest.main()
