# GSQWCA：全局与稀疏查询水质交叉注意力

## 本次改动与完整代码

本次将第二个模块从 Top-1 + Top-2 的 MSQWCA 替换为 Global Dense + Sparse Top-K 的 GSQWCA。RWA 和 CQWG 的类源码已逐字比对，保持不变。Backbone、Encoder、原始 Decoder Self-Attention、MS-Deformable Attention、原 FFN 与检测头的实现没有修改。

| 文件 | 本次改动 |
|---|---|
| `engine/mynewblock/water_token_fusion.py` | 新增正式类 `GlobalSparseQueryWaterCrossAttention`；旧类名作为构造入口兼容；取消固定分支参数，使用 Query-Adaptive Router |
| `engine/mynewblock/WaterQueryCrossAttention.py` | 新跨注意力类型接入；接受新类型名称和旧类型名称 |
| `engine/deim/deim_decoder.py` | 更新诊断接口，报告动态分支均值、Sparse K、选择频率；融合调用位置不变 |
| `configs/deimv2/water_innovations/base.yml` | 正式类型改为 GSQWCA，默认 `sparse_k: 2` |
| `configs/deimv2/water_innovations/A3_full_sparse1.yml` | 额外 Full + Sparse Top-1 消融配置 |
| `visualize_water_attention.py` | 导出 Global/Sparse 注意力、每个 Query 的路由权重和实际选择掩码；新增三类热图 |
| `eval_curves.py`、`run_ablation.py` | 评估与汇总记录实际跨注意力类型和 Sparse K，避免混淆新旧实验 |
| `tests/test_water_token_fusion.py` | Global/Sparse、梯度、router、初始化、AMP、共享编码与消融测试 |
| `tests/test_gsqwca_export.py` | 使用临时随机 checkpoint/图片执行实际 CLI，验证三个层的图与 JSON |
| `tests/smoke_gsqwca.py` | 随机完整融合链与完整检测 loss/backward/optimizer 的可复用验证脚本 |

完整新类代码位于 [water_token_fusion.py](../engine/mynewblock/water_token_fusion.py) 的 `GlobalSparseQueryWaterCrossAttention`。该类只返回水质 context，最终 Query 更新仍由原 CQWG 执行。

## 实际公式与形状

输入 `query [B,N,Dq]`、`water_tokens [B,3,Dw]`；水质顺序为 temperature、DO、pH。当前 Full 为 Dw=64、Dq=128、8 heads。

```text
water_proj: [B,3,64] -> [B,3,128]
Q: [B,H,N,Dh]; K,V: [B,H,3,Dh]
scores = Q K^T / sqrt(Dh)             [B,H,N,3]
A_global = softmax(scores)             全部三个 Token
A_sparse = softmax(mask_TopK(scores))   默认每个 head/query 保留两个 Token
F_global = merge(A_global V)            [B,N,Dq]
F_sparse = merge(A_sparse V)            [B,N,Dq]
w = softmax(MLP(cat(query,F_global,F_sparse)))  [B,N,2]
context = w_global*F_global + w_sparse*F_sparse
output = Dropout(out_proj(context))     [B,N,Dq]
```

Router 是 `Linear(3Dq,Dq) -> SiLU -> Linear(Dq,2)`，每层独立。最后一层 weight/bias 为零，因此初始比例严格为 0.5/0.5；训练后不同 Query 可以学习不同的比例。末层零初始化使 router 第一层梯度在首次反传时为零，这是预期行为；末层获得梯度后，第二次反传已验证第一层具有非零梯度。

两条分支仍可能同时关注 DO；新增 Global 分支提供完整环境上下文和连续梯度，并不保证两条分支的变量集合互斥，也不保证模型一定学会降低 DO 权重。

跨注意力保持原融合封装中的输入 LayerNorm；router 使用传入跨注意力模块的 Query，而非 Q 投影结果。CQWG 继续接收原融合封装中的原 Query 与水质 context。

## YAML 与训练

正式配置：

```yaml
DEIMTransformer:
  use_rwa: True
  use_msqwca: True  # 保留旧开关名作为兼容入口；实际启用 GSQWCA
  use_cqwg: True
  query_water_fusion:
    type: GlobalSparseQueryWaterCrossAttention
    num_heads: 8
    sparse_k: 2
    dropout: 0.0
```

正式类、结构和诊断统一使用 GSQWCA 名称；已有 `use_msqwca` 开关和 A2/A4 文件名保留，以兼容既有调用。旧类名和 `topk: [1,2]` 配置入口也会执行 **Global + Sparse Top-2**，不会重现旧双稀疏结构。显式改用新的类型与 `sparse_k`，可避免歧义。

在项目根目录执行，创建新的 GSQWCA 训练输出：

```powershell
python run_experiment.py -c configs/deimv2/water_innovations/A3_full.yml --name A3_full_gsqwca --device cuda:0
```

指定旧 DEIM checkpoint 作微调初始化：

```powershell
python run_experiment.py -c configs/deimv2/water_innovations/A3_full.yml --name A3_full_gsqwca --device cuda:0 --extra -t D:/weights/old_deim_best.pth
```

主模块消融仍使用原配置入口，但第二个模块已更新为 GSQWCA：

```powershell
python run_ablation.py --group innovations --only A0_baseline A1_rwa A2_rwa_msqwca A3_full --device cuda:0
```

单独 GSQWCA：`A4_msqwca_only.yml`；Full 中 Sparse Top-1 对照：

```powershell
python run_experiment.py -c configs/deimv2/water_innovations/A3_full_sparse1.yml --name A3_full_gsqwca_sparse1 --device cuda:0
```

保持正式 Full 的 Sparse Top-2 默认值。消融脚本会跳过指定输出目录中的已完成实验，GSQWCA 实验应使用新的批次目录。

## 旧权重兼容

实际检查旧纯视觉 DEIM-N checkpoint：609 个原状态项全部匹配，无原始项尺寸不匹配或 unexpected，新增 101 个水质状态项由微调初始化。原检测参数的名称与维度保持。

旧 MSQWCA checkpoint 的 `cross_attn.branch_logits` 是 unexpected；新 `cross_attn.branch_router.*` 是 missing。water_proj、Q/K/V 和 out_proj 等名称/维度相同的项可通过 `-t` 迁移。新 router 没有旧固定权重的等价参数映射，不能将旧双稀疏 checkpoint 严格恢复为新的 GSQWCA。`-r` 断点续训和严格评估使用相同新结构的 checkpoint。

## Attention 与后续 Normal/Hypoxia 统计

```powershell
python visualize_water_attention.py -c D:/run/config_used.yml -r D:/run/best_stg1.pth --image D:/dataset/test/image.jpg --water 27.2 4.6 7.95 --layer all --outdir gs_attention_vis
```

每层新增：

- `query_water_global_attention.png`
- `query_water_sparse_attention.png`
- `query_water_branch_weights.png`

`attention_data.json` 保留所有 Query 的 `global_attention [N,3]`、`sparse_attention [N,3]`、`branch_weights [N,2]` 和 `sparse_selection_mask [H,N,3]`。类内捕获属性分别为 `last_global_attention [B,N,3]`、`last_sparse_attention [B,N,3]`、`last_branch_weights [B,N,2]`、`last_sparse_mask [B,H,N,3]`。全部仅在 save_attention=True 时 detach/cpu，下一次未捕获前向会清除缓存。

按 head 平均后的 Sparse 注意力可能三个列都非零，因为不同 head 可以选择不同变量。统计“被选择频率”应使用真实 bool mask，不能用平均注意力是否大于零替代。JSON 同时保存预测类别 ID 与置信度；可按预测 Normal/Hypoxia 分组，或后续用 GT/IoU 匹配后统计真实类别。预测类别不是 GT；中间层以 Query 索引关联最终预测。

## 测试与计算量

```powershell
python -m unittest discover -s tests -p test_water_token_fusion.py
python -m unittest discover -s tests -p test_gsqwca_export.py
python tests/smoke_gsqwca.py --device cuda --amp --profile --out outputs/gsqwca_smoke.json
```

本机已通过 12 项模块测试、6 项消融工具回归和 1 项真实 CLI 导出测试。直接执行 batch=2、N=300、Dq=128 的 `out.mean().backward()`，water_proj、q_proj、k_proj、v_proj、router 与 out_proj 均有有限梯度；Global 分支也使 Sparse 未选中的水质 Token 得到梯度。完整模型 CUDA AMP 下 batch=1/2 的检测 loss、反传和优化器更新通过。集成 smoke 使用 128×128 和 20 个 Query，验证计算链，不代表正式训练效果。

640×640 的算子估算如下：

| 配置 | 参数量 | GFLOPs |
|---|---:|---:|
| A6_visual | 5,147,619 | 6.9317 |
| A3_full GSQWCA | 5,794,271 | 7.2293 |

原始报告：`outputs/gsqwca_profiles.json`；AMP smoke：`outputs/gsqwca_smoke.json`。特殊算子可能未计入 FLOPs，实际延迟需实测；AP 收益需正式训练验证。

## Full Model 完整数据流

```text
temperature / DO / pH [B,3]
 -> WaterTokenEncoder [B,3,64]
 -> 原 RWA（每次检测器前向一次）
 -> Enhanced Water Tokens [B,3,64]，各层共享

每个 Decoder Layer：
原 Query Self-Attention
 -> GSQWCA（Global + Sparse Top-2 + Query-Adaptive Router）
 -> Water Context [B,N,128]
 -> 原 CQWG
 -> Updated Query [B,N,128]
 -> 原 MS-Deformable Attention -> 原 FFN -> 原检测头
```

## GSQWCA 完整类代码

```python
import math
import torch
from torch import nn

class GlobalSparseQueryWaterCrossAttention(nn.Module):
    """Dense + sparse water contexts, fused by a router for each object query."""
    def __init__(self, query_dim=128, water_dim=64, num_heads=8, sparse_k=2, dropout=0.0):
        super().__init__()
        if num_heads < 1 or query_dim % num_heads:
            raise ValueError('GSQWCA query_dim must be divisible by positive num_heads')
        if isinstance(sparse_k, bool) or not isinstance(sparse_k, int) or sparse_k not in (1, 2):
            raise ValueError('Three water tokens require sparse_k=1 or 2; default is 2')
        self.num_heads, self.head_dim = num_heads, query_dim // num_heads
        self.sparse_k = sparse_k
        self.water_proj = nn.Linear(water_dim, query_dim)
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj, self.v_proj = nn.Linear(query_dim, query_dim), nn.Linear(query_dim, query_dim)
        self.branch_router = nn.Sequential(nn.Linear(3 * query_dim, query_dim), nn.SiLU(), nn.Linear(query_dim, 2))
        nn.init.zeros_(self.branch_router[-1].weight)
        nn.init.zeros_(self.branch_router[-1].bias)
        self.out_proj, self.dropout = nn.Linear(query_dim, query_dim), nn.Dropout(dropout)
        self.save_attention = False
        self.last_attention = self.last_branch_attention = None
        self.last_global_attention = self.last_sparse_attention = self.last_branch_weights = None
        self.last_sparse_mask = None

    def _heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, query, water_tokens):
        if query.ndim != 3 or water_tokens.ndim != 3 or water_tokens.shape[1] != 3:
            raise ValueError('GSQWCA expects query [B,N,Dq] and water [B,3,Dw]')
        if query.shape[0] != water_tokens.shape[0]:
            raise ValueError('Query and water batch sizes must match')
        self.last_attention = self.last_branch_attention = None
        self.last_global_attention = self.last_sparse_attention = self.last_branch_weights = None
        self.last_sparse_mask = None
        water = self.water_proj(water_tokens)
        q, k, v = self._heads(self.q_proj(query)), self._heads(self.k_proj(water)), self._heads(self.v_proj(water))
        with torch.autocast(device_type=query.device.type, enabled=False):
            scores = q.float() @ k.float().transpose(-2, -1) / math.sqrt(self.head_dim)
            global_attn = scores.softmax(-1)
            indices = scores.topk(self.sparse_k, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)
            sparse_attn = scores.masked_fill(~mask, float('-inf')).softmax(-1)
            global_context = global_attn @ v.float()
            sparse_context = sparse_attn @ v.float()
        global_context = global_context.transpose(1, 2).reshape_as(query).to(query.dtype)
        sparse_context = sparse_context.transpose(1, 2).reshape_as(query).to(query.dtype)
        router_input = torch.cat((query, global_context, sparse_context), dim=-1)
        logits = self.branch_router(router_input)
        with torch.autocast(device_type=query.device.type, enabled=False):
            weights = logits.float().softmax(-1)
            context = weights[..., 0:1] * global_context.float() + weights[..., 1:2] * sparse_context.float()
        context = context.to(query.dtype)
        if self.save_attention:
            attention = weights[:, None, :, 0:1] * global_attn + weights[:, None, :, 1:2] * sparse_attn
            self.last_attention = attention.mean(1).detach().cpu()
            self.last_global_attention = global_attn.mean(1).detach().cpu()
            self.last_sparse_attention = sparse_attn.mean(1).detach().cpu()
            self.last_branch_weights = weights.detach().cpu()
            self.last_sparse_mask = mask.detach().cpu()
            self.last_branch_attention = (global_attn.detach().cpu(), sparse_attn.detach().cpu())
        return self.dropout(self.out_proj(context))
```
