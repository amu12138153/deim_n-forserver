# DEIMv2 Pico 模型架构核查（`visdrone_pico.yml`）

## 1. 核查范围与最终有效配置

分析对象是当前工作区的 `configs/deimv2/visdrone_pico.yml`，工作区提交为 `6c5b60b`。该文件在工作区有未提交的 batch size 修改；模型结构相关配置没有出现在该差异中。本地 Git remote 为 `amu12138153/newdeim.git`，与请求中给出的 `DEIM-improved.git` URL 不同。因此，本文和图严格对应**当前工作区源码及配置**，不声称已核实另一个远端仓库的内容。给定 URL 的在线页面本次无法获取。

`engine/core/yaml_utils.py:24-56` 中的 `load_config()` 按顺序递归读取 `__include__`，对字典进行递归合并，最后用目标文件覆盖同名配置。实际 include 链没有更深层 include：

1. `configs/dataset/visdrone.yml`
2. `configs/runtime.yml`
3. `configs/base/dataloader.yml`
4. `configs/base/deimv2.yml`
5. `configs/base/optimizer.yml`
6. `configs/deimv2/visdrone_pico.yml`（最终覆盖）

与模型实例化和输出直接相关的**合并后有效值**如下（未列出优化器、增强等与模型前向无关的键）：

| 配置项 | 最终值 | 依据 |
|---|---|---|
| `model` | `DEIM` | `configs/base/deimv2.yml` |
| `DEIM` | `backbone=HGNetv2`, `encoder=LiteEncoder`, `decoder=DEIMTransformer` | 目标配置覆盖基配置的 `HybridEncoder` |
| `num_classes` | `2` | `configs/dataset/visdrone.yml` |
| `eval_spatial_size` | `[640,640]` | `configs/base/deimv2.yml` |
| `HGNetv2` | `name=Pico`, `return_idx=[2]`, `use_lab=True`, `agg=se`, `use_water_quality=True`, `water_fuse_idx=2`, `pretrained=True`, `freeze_at=-1`, `freeze_norm=False` | 目标配置；其余构造默认值或基配置 |
| `LiteEncoder` | `in_channels=[512]`, `feat_strides=[16]`, `hidden_dim=112`, `expansion=0.34`, `depth_mult=0.5`, `act=silu`, `csp_type=csp2`, `use_water_quality=False` | 目标配置及 `LiteEncoder.__init__` 默认值 |
| `DEIMTransformer` | `feat_channels=[112,112]`, `feat_strides=[16,32]`, `num_levels=2`, `hidden_dim=112`, `num_points=[4,2]`, `num_layers=3`, `num_queries=200`, `eval_idx=-1`, `dim_feedforward=320`, `nhead=8`, `num_denoising=100`, `reg_max=32`, `reg_scale=4`, `layer_scale=1`, `share_bbox_head=True`, `use_gateway=False`, `use_water_quality=False` | 目标配置、基配置及构造默认值 |
| 解码器其余有效选项 | `query_select_method=default`, `cross_attn_method=default`, `learn_query_content=False`, `aux_loss=True`, `share_score_head=False`, `activation=silu`, `mlp_act=silu` | 目标配置、基配置及构造默认值 |
| 水质字段 | `temperature`, `do`, `ph`, `turbidity`；训练/验证/测试数据集的 `use_water_quality=True` 和 `sensor_csv` | 数据集配置与 `coco_dataset.py:44-71` |
| `PostProcessor` | `num_top_queries=300`, `use_focal_loss=True` | 基配置及 `PostProcessor.__init__` |

`HybridEncoder` 的基配置字典虽然仍存在于合并结果，但 `DEIM.encoder=LiteEncoder`，故不会实例化或进入此模型的 forward。解码器配置中 `water_features`、均值/标准差和水质注意力开关虽有值，`use_water_quality=False` 使相关水质模块不实例化、不执行。

配置文件名含 `visdrone`，但当前合并的数据集配置 `configs/dataset/visdrone.yml` 的图像、标注和传感器 CSV 路径指向 `./fish_dataset622/...`。本图按配置文件名标识模型，不把实际数据集来源写成已验证的 VisDrone 数据。

`engine/core/yaml_config.py:36-39` 使用 `create(model, global_cfg)`；`engine/core/workspace.py:93-173` 按注册表解析 `DEIM.__inject__=['backbone','encoder','decoder']`。对应已注册类分别为 `engine/deim/deim.py::DEIM`、`engine/backbone/hgnetv2.py::HGNetv2`、`engine/deim/lite_encoder.py::LiteEncoder`、`engine/deim/deim_decoder.py::DEIMTransformer`。`engine/backbone/__init__.py` 导入的是 `hgnetv2.py`，并非同目录的 `hgnetv2_org.py` 或 `hgnetv2_AAFM.py`。

## 2. 主模型与真实执行顺序

`DEIM.forward()`（`engine/deim/deim.py:26-42`）：

```text
Input Image [B,3,640,640] ───────────────────────┐
                                                v
Water [B,4] -> HGNetv2-Pico Stage 3 -> WaterCrossAttention
                                                |
                                    Stage 3 feature [B,512,40,40]
                                                v
                                    LiteEncoder (two-scale fusion)
                                                |
                         [B,112,40,40] + [B,112,20,20]
                                                v
                            DEIMTransformer / TransformerDecoder ×3
                                                v
                        pred_logits [B,200,2] + pred_boxes [B,200,4]
```

`DEIM.forward()` 把同一 `water` 参数传给三个子模块；实际只有 Backbone 的开关打开。水质数据可来自调用者显式传入，也可在 `water=None` 且训练 `targets` 含 `water_quality` 时由 `DEIM.forward()` 堆叠；`engine/solver/det_engine.py:135-173` 另有从 targets 提取并传入的路径。若调用时 `water=None`，Backbone 条件分支跳过，成为纯视觉前向。

## 3. HGNetv2-Pico 输出

`hgnetv2.py:476-483` 定义 Pico 仅有 3 个 stage；`hgnetv2.py:735-746` 在每个 stage 后先检查水质融合，再依据 `return_idx` 收集特征。`return_idx=[2]` 只输出第三个 stage（零基索引 2）。`_out_strides=[4,8,16,32]` 是类中的固定元数据，不能据此虚构 Pico 的第四个 stage。

| Feature | Source | Channels | Stride | Resolution（输入 640×640） | 是否传入 Encoder |
|---|---|---:|---:|---:|---|
| Stem | `StemBlock` | 16 | 4 | 160×160 | 否 |
| Stage 1 | `HG_Stage` / `HG_Block ×1` | 64 | 4 | 160×160 | 否 |
| Stage 2 | `HG_Stage` / `HG_Block ×1` | 256 | 8 | 80×80 | 否 |
| Stage 3（融合前） | `HG_Stage` / `HG_Block ×2` | 512 | 16 | 40×40 | 否 |
| Stage 3（返回特征） | `WaterCrossAttention`，仅当 `water` 非空 | 512 | 16 | 40×40 | 是，唯一返回层 |

Stem 的总 stride 4 来自两次 stride 2 卷积；Stage 1 不下采样，Stage 2/3 各下采样 2 倍。Stage 3 的两个 `HG_Block` 均由三个 `LightConvBNAct`、通道拼接及 `agg='se'` 聚合组成；第二个块带 residual。Stage 1/2 各一个 `HG_Block`，内部使用三个 `ConvBNAct`。以上尺寸还通过无预训练权重的 640×640 前向 hook 验证：Stage 1 `[1,64,160,160]`、Stage 2 `[1,256,80,80]`、Stage 3 和融合输出 `[1,512,40,40]`。

## 4. 水质融合（位置为 HGNetv2 Stage 3）

当前配置**启用了 water 模块，但只在 HGNetv2 位置启用**。`HGNetv2.__init__()` 在 `use_water_quality=True` 时构造 `WaterCrossAttention(in_channels=512, embed_dim=64, num_heads=4, ..., init_scale=0.01)`；`HGNetv2.forward()` 在 `idx==2` 且 `water is not None` 时将 Stage 3 特征和水质向量送入它（`hgnetv2.py:593-596,735-746`）。

实际路径（`engine/mynewblock/WaterVisionCrossAttention.py:9-47,54-180`）：

```text
Water Quality Parameters [temperature, DO, pH, turbidity]
  -> WaterTokenEncoder（标准化、数值编码 + 参数嵌入、token self-attention）
  -> water tokens [B,4,64] ───────────────────────────────┐
                                                        v
HGNetv2 Stage 3 [B,512,40,40] -> visual_proj -> visual-token queries
  -> WaterCrossAttention（visual query；water key/value）
  -> FFN -> visual_recover -> 带可学习系数的残差加回 Stage 3 特征
  -> [B,512,40,40] -> LiteEncoder
```

`LiteEncoder.use_water_quality=False`，因此其 `WaterCrossAttentionforencoder` 分支未构造。`DEIMTransformer.use_water_quality=False`，因此 `WaterAwareQueryCrossAttention` 不在任何 `TransformerDecoderLayer` 中构造或运行。**图中没有 Water → Object Query 的连线。**

## 5. LiteEncoder 两尺度数据流

`engine/deim/lite_encoder.py:44-126` 的实际路径是单输入尺度生成第二尺度，再执行一次上行融合和一次下行融合；它不是三尺度 `HybridEncoder`，也没有可画的 lateral-conv 分支。

```text
Stage 3 [512,40,40]
  -> input_proj（1×1 Conv + BN）-> P16 [112,40,40]
                                  |
                                  +-> down_sample1（AvgPool stride 2 + 1×1 Conv + BN + SiLU）
                                      -> GAP_Fusion（全局均值 + add + ConvNormLayer_fuse）
                                      -> P32 [112,20,20]
P16 + nearest upsample(P32, ×2) -> fuse_feat（逐元素 add）
  -> fpn_block：RepNCSPELAN4 -> FPN output [112,40,40]
P32 + down_sample2(FPN output, stride 2) -> fuse_feat（逐元素 add）
  -> pan_block：RepNCSPELAN4 -> PAN output [112,20,20]
return [FPN output, PAN output]
```

没有 `concat` 发生在两个尺度的融合点；跨尺度融合均为 `+`。`RepNCSPELAN4` 模块内部有 `split`、两段顺序 CSP 处理、`torch.cat`，这是模块内部路径。参数 `n=round(3×0.5)=2`、`c4=round(0.34×112//2)=19`；FPN/PAN 各一个独立拷贝。实际前向验证返回 `[1,112,40,40]` 与 `[1,112,20,20]`。

## 6. DEIMTransformer 解码与预测

`engine/deim/deim_decoder.py:410-561` 中，`_get_encoder_input()` 对 Encoder 两个输出分别使用 `input_proj`。两者通道已等于 `hidden_dim=112`，因此两个 projection 均是 `nn.Identity()`；随后展平并拼接成 `memory [B,2000,112]`，`spatial_shapes=[[40,40],[20,20]]`。这里的 `input_proj` 属于 Decoder，与 LiteEncoder 的真实 1×1 `input_proj` 是不同模块。

`_generate_anchors()` 为两个网格产生归一化 anchor，训练时按运行时 `spatial_shapes` 生成，评估时使用 `eval_spatial_size=[640,640]` 预生成的 40×40、20×20 anchor。`_get_decoder_input()` 将无效位置 mask 到 memory，`enc_score_head` 在 2 类上打分，按每位置最大类分数选前 200 个；`enc_bbox_head` 为这些位置预测 box 增量并加到 anchor。`learn_query_content=False`，所以 **object-query content 是选中的 memory 向量的 detached 拷贝**，而非独立可学习 query embedding；初始参考框也是预测框的 detached 拷贝。训练时还可能在前面拼接 100 个 denoising query（实际组大小受目标数影响），评估不拼接。

`TransformerDecoder` 包含 `TransformerDecoderLayer ×3`、`LQE ×3`。每层真实顺序（`deim_decoder.py:81-119,190-231`）：

1. 用当前参考框经 `query_pos_head` 形成位置编码（循环开始前生成，当前配置 `layer_scale=1`）。
2. `MultiheadAttention` 自注意力；残差和 `RMSNorm`。
3. `MSDeformableAttention` 对两尺度 memory 做 cross-attention（每层两个尺度分别使用 `[4,2]` sampling points）；残差和 `RMSNorm`。`use_gateway=False`。
4. `SwiGLUFFN`；残差和 `RMSNorm`。
5. 第 1 层后额外计算 `pre_bbox_head`、`pre_scores`，`pre_bboxes` 成为后续分布式回归的基准框。
6. 每层 `dec_bbox_head[i]` 预测 `4×(reg_max+1)=132` 个 corner 分布值；`Integral` + `distance2bbox` 得到该层预测框。`share_bbox_head=True`，三层索引指向同一个 bbox-head 模块。
7. 每层 `dec_score_head[i]` 作分类预测，`LQE[i]` 调整分数；框作为下一层参考框时执行 `detach()`。评估在 `eval_idx=-1` 对应的第 3 层停止；训练保留各层输出用于辅助监督。

`DEIMTransformer.forward()` 返回评估态 `pred_logits [B,200,2]` 与归一化 `pred_boxes [B,200,4]`；训练态另带 `aux_outputs`、`enc_aux_outputs`、`pre_outputs` 和 denoising 相关输出（`deim_decoder.py:551-614`）。`PostProcessor` 是模型返回之后在评估流程中单独调用的模块，将框转换为像素 `xyxy` 并从类别分数选取 top 300；它不在 `DEIM.forward()` 的子模块链内（`engine/deim/postprocessor.py:50-86`）。

## 7. 重复模块与图的抽象层级

| 模块 | 有效次数 | 主图表达 | 单模块细节框 |
|---|---:|---|---|
| `HG_Stage` | 3 | Stem → Stage 1/2/3 | Stage 的 downsample → HG_Block |
| `HG_Block` | 4（1+1+2） | 标在 Stage 内，不展开卷积 | layer ×3 → concat → SE 聚合；第二个 Stage 3 块有残差 |
| `LightConvBNAct` | 6（仅 Stage 3 两块） | 不逐一画 | 1×1 ConvBNAct → depthwise ConvBNAct |
| `RepNCSPELAN4` | 2（FPN/PAN 各 1） | 分别标在两尺度融合后 | split → 两段 CSPLayer2 → concat |
| `CSPLayer2` | 4（每个 RepNCSPELAN4 有 2 个） | 不逐一画 | 各含 VGGBlock ×2 |
| `TransformerDecoderLayer` | 3 | Decoder Layer ×3 | Self-Attn → MSDeformableAttention → SwiGLUFFN |
| `LQE` | 3 | 分类支路标注 | 每层类别分数质量调整 |
| `dec_bbox_head` | 同一 MLP 被 3 层复用 | BBox head（shared） | 132 维分布 → Integral → distance2bbox |

## 8. 验证与不确定性

使用 `deimv2` Python 环境从目标 YAML 实际构建 `DEIM`，只在内存中把 `HGNetv2.pretrained` 设为 `False` 以避免下载权重；没有改变任何影响网络连线的参数。以 `[1,3,640,640]` 图像和 `[1,4]` 水质向量做 eval 前向，hook 观察到的 Stage、water fusion、Encoder 尺寸以及最终预测尺寸与上表一致。

图表达的是**结构和给定 640×640 输入下的张量尺寸**。数值输出未使用训练权重；图不表达训练精度。运行时不传入 `water` 时，水质分支条件跳过。用户提供的另一个 GitHub URL 与本地 remote 不一致且页面不可访问，因此跨仓库一致性为 **Unknown / determined at runtime of that repository**；若需对应另一个仓库，需取得其源码快照重新核查。
