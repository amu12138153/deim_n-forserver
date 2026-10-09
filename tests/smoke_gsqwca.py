"""GSQWCA random chain and full detector/loss/optimizer smoke; no accuracy evaluation."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig
from engine.mynewblock.WaterQueryCrossAttention import WaterTokenEncoder, WaterAwareQueryFusion
from get_info_param_and_flops import profile_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--amp',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/gsqwca_smoke.json')
    args=parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(2)
    reports=[]
    for batch in (1,2):
        encoder=WaterTokenEncoder(use_rwa=True).to(args.device)
        fusion=WaterAwareQueryFusion(use_msqwca=True,use_cqwg=True).to(args.device)
        water=torch.randn(batch,3,device=args.device,requires_grad=True)
        query=torch.randn(batch,300,128,device=args.device,requires_grad=True)
        output=fusion(query,encoder(water))
        output.mean().backward()
        assert output.shape==(batch,300,128) and torch.isfinite(output).all()
        assert torch.isfinite(query.grad).all() and torch.isfinite(water.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fusion.cross_attn.parameters())
        reports.append({'check':'random_chain','batch':batch,'output_shape':list(output.shape),'passed':True})
        print(f'PASS requested chain B={batch}: forward + mean loss backward',flush=True)
    for batch in (1,2):
        cfg=YAMLConfig(str(ROOT/'configs/deimv2/water_innovations/A3_full.yml'))
        cfg.yaml_cfg['HGNetv2']['pretrained']=False
        cfg.yaml_cfg['eval_spatial_size']=[128,128]
        cfg.yaml_cfg['DEIMTransformer'].update(num_queries=20,num_denoising=10)
        model=cfg.model.to(args.device).train()
        criterion=cfg.criterion.to(args.device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        images=torch.randn(batch,3,128,128,device=args.device)
        water=torch.tensor([[27.2,4.6,7.95]],device=args.device).expand(batch,-1)
        targets=[{'labels':torch.tensor([0,1],device=args.device),
                  'boxes':torch.tensor([[.3,.3,.2,.2],[.7,.7,.2,.2]],device=args.device),
                  'water_quality':water[index]} for index in range(batch)]
        for layer in model.decoder.decoder.layers: layer.water_fusion.save_attention=True
        mode=(torch.autocast(device_type=torch.device(args.device).type,
                            dtype=torch.float16 if torch.device(args.device).type=='cuda' else torch.bfloat16)
              if args.amp else contextlib.nullcontext())
        with mode:
            losses=criterion(model(images,targets=targets),targets,epoch=0)
            loss=sum(losses.values())
        assert torch.isfinite(loss),losses
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report={'check':'detector_loss_optimizer','batch':batch,'image_size':128,'object_queries':20,
                'loss':float(loss.detach()),'diagnostics':model.decoder.water_fusion_diagnostics(),'passed':True}
        for layer in model.decoder.decoder.layers: layer.water_fusion.save_attention=False
        if args.profile and batch==1:
            report['profile']=profile_model(model.eval(),image_size=128,device=args.device)
            assert report['profile']['water_shape']==[1,3]
        reports.append(report)
        print(f'PASS detector B={batch}: full loss + backward + optimizer, loss={float(loss):.6f}',flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'fusion':'GSQWCA','device':args.device,'amp':args.amp,'checks':reports},indent=2),encoding='utf-8')
    print(f'Saved: {args.out}',flush=True)


if __name__=='__main__':
    main()
