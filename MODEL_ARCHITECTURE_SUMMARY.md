# 模型结构说明：`visdrone_pico.yml`

## 1. Input

- 图像：`3 × 640 × 640`（本图的尺寸假设）。
- 水质向量：`[temperature, do, ph, turbidity]`，由配置中的数据集从 `sensor_csv` 读取；运行时可不传入，此时跳过水质融合。

## 2. HGNetv2-Pico Backbone

- `StemBlock`：`16 × 160 × 160`，stride 4。
- Stage 1：`HG_Block ×1`，`64 × 160 × 160`，stride 4。
- Stage 2：`HG_Block ×1`，`256 × 80 × 80`，stride 8。
- Stage 3：`HG_Block ×2`，`512 × 40 × 40`，stride 16。
- `return_idx=[2]`：只向 Encoder 返回 Stage 3 特征。

## 3. Water Fusion（HGNetv2 位置）

- 当前配置启用 `HGNetv2.use_water_quality=True`，`water_fuse_idx=2`。
- 水质向量经过 `WaterTokenEncoder` 形成 4 个 64 维 token。`WaterCrossAttention` 用 Stage 3 视觉 token 作 query、水质 token 作 key/value，经 FFN 和残差将结果融合回 `512 × 40 × 40` 特征。
- `LiteEncoder.use_water_quality=False`，`DEIMTransformer.use_water_quality=False`。**没有 Water → Object Query 的连接。**

## 4. LiteEncoder

- 单尺度输入先经 `input_proj` 从 512 通道压到 112 通道。
- `down_sample1` 和 `GAP_Fusion` 从 40×40 生成 20×20 特征。
- 上行：20×20 最近邻上采样，与 40×40 投影特征逐元素相加，再经 `fpn_block: RepNCSPELAN4`。
- 下行：FPN 输出经 `down_sample2`，与 20×20 特征逐元素相加，再经 `pan_block: RepNCSPELAN4`。
- 最终输出两尺度：`P16 = 112 × 40 × 40`、`P32 = 112 × 20 × 20`。跨尺度融合使用 `add`；`concat` 只存在于 `RepNCSPELAN4` 内部。

## 5. DEIMTransformer

- 两尺度特征在 Decoder 内展平拼接为 `2000 × 112` memory。
- 对两个网格生成 anchors，用 `enc_score_head` 选 top 200。默认 query content 是选中 memory 向量的 detached 拷贝；初始参考框由 `enc_bbox_head` 的预测叠加 anchor 后形成。
- `TransformerDecoderLayer ×3`：每层执行 Self-Attention → `MSDeformableAttention` 跨尺度注意力 → `SwiGLUFFN`，每步带残差和 `RMSNorm`。
- 第 1 层额外产生 pre-prediction；三层各自进行分类及边框细化。bbox MLP 在三层共享；参考框逐层更新。训练时还有 encoder、intermediate 和 denoising 辅助输出。

## 6. Detection Head / Output

- 模型评估态返回 `pred_logits: 200 × 2` 和 `pred_boxes: 200 × 4`。
- 独立的 `PostProcessor` 在模型返回后转换框格式、映射到原图尺寸，并按分数取 top 300 类别候选；它不属于 `DEIM.forward()` 内部的数据流。

## 最终网络主数据流

```text
Image → HGNetv2-Pico (Stem → Stage 1 → Stage 2 → Stage 3)
      → WaterCrossAttention at Stage 3
      → LiteEncoder (input_proj → FPN/PAN two-scale fusion)
      → DEIMTransformer (memory → top-200 queries → Decoder Layer ×3)
      → Classification + BBox Regression → Detection Output

Water Parameters → WaterTokenEncoder → WaterCrossAttention at HGNetv2 Stage 3
```

完整的配置合并、源码定位、张量尺寸和条件分支依据见 [`architecture_analysis.md`](architecture_analysis.md)。
