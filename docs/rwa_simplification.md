# RWA 简化记录

## 修改范围

模型代码仅修改 `engine/mynewblock/water_token_fusion.py` 中的 `RankAwareWaterAttention`。
GSQWCA、CQWG、WaterTokenEncoder 输入编码和 DEIM Decoder 保持修改前内容不变。
协方差统计函数 `_rank_statistics` 保持不变，包括 redundancy、concentration 和低能量数值保护。

## 当前结构

1. 删除 `residual_gate` 的提取及其对分头 delta 的乘法，并删除相应的重新合并 heads 操作。
2. Router 改为 `Linear(2, router_hidden) → GELU → Linear(router_hidden, 1)`，默认 `router_hidden=16`。
3. 输入 Router 的统计量为 `stats.detach()`；Router 参数仍正常接收梯度。
4. `relation_strength = sigmoid(router(stats.detach()))[..., None]`，形状为 `[B,H,1,1]`。
5. Relation MLP 仍为 `Linear(Dw,Dw) → SiLU → Linear(Dw,Dw)`，没有卷积或空间 reshape。
6. 删除 RWA 的 `self.norm`，delta 路径仅保留输出 Linear 投影。
7. 保留可学习的 `alpha`，默认初始化为 `0.01`。

```text
Fg = LinearAttention(Q, K, V)                 # [B,H,3,Dh]
stats = [redundancy(Fg), concentration(Fg)]   # [B,H,2]
r = sigmoid(Router(stats.detach()))[...,None] # [B,H,1,1]
Fr = split_heads(RelationMLP(W))             # [B,H,3,Dh]
F = merge_heads(Fg + r * Fr)                 # [B,3,Dw]
W_out = W + alpha * OutProj(F)               # [B,3,Dw]
```

保留 `last_attention`、`last_rank_stats`、`last_effective_rank`，新增 `last_relation_strength`。
仅在 `save_attention=True` 时保存诊断张量，均 detached 并存于 CPU；每次 forward 先清空缓存。

## 验证结果

- `python -m unittest discover -s tests -p test_water_token_fusion.py`：14 项全部通过。
- 独立 RWA 输入 `[2,3,64]`、输出 `[2,3,64]`；`output.mean().backward()` 通过。
- q/k/v、relation、router、out_proj 及 alpha 均获得有限梯度；输出没有 NaN/Inf。
- Router 输入不要求梯度，统计量不接收 Router 路径的梯度。
- CPU bfloat16 和 CUDA float16 混合精度检查通过。
- `python tests/smoke_gsqwca.py --device cuda --amp --out outputs/rwa_simplification_smoke.json`：B=1/2 模块链与整模型 loss、backward、optimizer step 全部通过。整模型检查使用 128 像素图像、20 个普通 Query。
- 修改前源代码快照与修改后内容核对通过：其他融合类、WaterTokenEncoder 文件和 Decoder 文件不变；rank statistics 函数不变。

完整测试记录在 `outputs/rwa_simplification_checks.json` 和 `outputs/rwa_simplification_smoke.json`。
RWA 的 PNG/SVG/PDF 示意图已同步，目录为 `figures/water_modules`。

## 旧 RWA checkpoint

默认参数下，旧权重有两项形状不兼容：

| RWA 局部键名 | 旧形状 | 新形状 |
|---|---|---|
| router.2.weight | [2,16] | [1,16] |
| router.2.bias | [2] | [1] |

旧权重中的 `norm.weight`、`norm.bias` 已无对应成员，直接加载时属于 unexpected keys。
完整模型键名会带 RWA 的所属模块前缀。

`load_state_dict(old_state, strict=False)` 仍会因为 router 形状不同报错；strict=False 不会忽略同名参数的尺寸差异。
项目现有 `-t/--tuning` 会通过 `_matched_state` 筛选同名且同形状的参数，再加载。
筛选后 `router.2.weight`、`router.2.bias` 属于 missing keys，保留新模块的随机初始化；旧 norm 参数被排除。
这种方式用于初始化新结构并继续训练。新结构的推理应使用按新结构训练得到的 checkpoint。

```powershell
python train.py -c configs/deimv2/water_innovations/A3_full.yml -t "你的旧权重路径.pth" --use-amp
```

本次验证覆盖结构和数值运行；检测 AP 的变化需要重新训练和评估后获得。
