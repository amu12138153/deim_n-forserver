"""CLI attention exports using synthetic weights and an image; no AP estimate."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig


class GSQWCAExportTests(unittest.TestCase):
    def test_cli_exports_three_layers_and_exact_sparse_selections(self):
        torch.set_num_threads(2)
        cfg=YAMLConfig(str(ROOT/'configs/deimv2/water_innovations/A3_full.yml'))
        cfg.yaml_cfg['HGNetv2']['pretrained']=False
        cfg.yaml_cfg['eval_spatial_size']=[128,128]
        cfg.yaml_cfg['DEIMTransformer'].update(num_queries=20,num_denoising=0)
        model=cfg.model.eval()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            configuration=dict(cfg.yaml_cfg)
            configuration.pop('__include__',None)
            (root/'config.yml').write_text(yaml.safe_dump(configuration),encoding='utf-8')
            torch.save({'model':model.state_dict()},root/'checkpoint.pth')
            Image.new('RGB',(128,128),(50,100,150)).save(root/'image.jpg')
            completed=subprocess.run([
                sys.executable,str(ROOT/'visualize_water_attention.py'),
                '-c',str(root/'config.yml'),'-r',str(root/'checkpoint.pth'),
                '--image',str(root/'image.jpg'),'--water','27.2','4.6','7.95',
                '--layer','all','--topk','3','--device','cpu','--outdir',str(root/'vis'),
            ],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
            self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
            for index in range(3):
                out=root/'vis'/f'layer_{index}'
                for filename in ('query_water_global_attention.png','query_water_sparse_attention.png','query_water_branch_weights.png'):
                    self.assertTrue((out/filename).is_file())
                data=json.loads((out/'attention_data.json').read_text(encoding='utf-8'))
                self.assertEqual(data['query_water_fusion_type'],'GlobalSparseQueryWaterCrossAttention')
                self.assertEqual(data['sparse_k'],2)
                self.assertEqual(data['branch_order'],['global','sparse'])
                for key,width in (('global_attention',3),('sparse_attention',3),('branch_weights',2)):
                    values=torch.tensor(data[key])
                    self.assertEqual(values.shape,(20,width))
                    torch.testing.assert_close(values.sum(-1),torch.ones(20))
                mask=torch.tensor(data['sparse_selection_mask'])
                self.assertEqual(mask.shape,(8,20,3))
                self.assertTrue((mask.sum(-1)==2).all())
                torch.testing.assert_close(torch.tensor(data['sparse_selection_fraction_per_query']),mask.float().mean(0))
                self.assertEqual(len(data['query_predicted_labels']),20)
                self.assertEqual(len(data['query_confidences']),20)


if __name__=='__main__':
    unittest.main()
