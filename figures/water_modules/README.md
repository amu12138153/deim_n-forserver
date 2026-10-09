# 三个多模态模块示意图

依据 `engine/mynewblock/water_token_fusion.py` 的实际 forward 绘制：

| 图 | 结构 |
|---|---|
| 01_RWA | 正值线性注意力、协方差代理统计、单值路由、关系 MLP、输出投影和 alpha 残差 |
| 02_GSQWCA | Global 全三个水质 Token + Sparse 默认 Top-2；按 Query 学习分支比例；仅输出 context |
| 03_CQWG | 四种联合特征构建门控，互补差值，差值 LayerNorm，lambda 残差与原 Query 跳连 |

每张均提供 PNG（3000 px 宽）、SVG（可编辑矢量文字与图形）和 PDF（矢量）。`water_modules_overview.png` 是三图并排预览。完整文件包为 `water_module_schematics.zip`。

## 符号与源码对应

- 水质 Token 顺序：temperature、DO、pH；Dw=64。
- RWA：Hw=4，dhw=16；relation_strength 为每个 head 的一个控制量，由 `Router(stats.detach())` 经 sigmoid 得到。其统计量来自 Y，真正的谱有效秩仅用于诊断。RWA 已删除 residual_gate 和差值路径的 LayerNorm，最终输出为 `W + alpha * OutProj(F)`。
- GSQWCA：Dq=128，H=8，dh=16；Q_in 为类方法传入的 query。外层 WaterAwareQueryFusion 在调用 GSQWCA 前执行 query_norm/water_norm，并将原 Query 留给 CQWG。
- Global/Sparse 共享同一组 scores、K 和 V。Global 保留三变量，Sparse 每个 head/query 选择两个变量；router 不会替代 CQWG。
- CQWG 的 LayerNorm 位于残差差值内部，严格对应 `query + lambda_gate * norm(gate * (water_context-query))`。
- 实线表示特征流，RWA 中的虚线表示控制系数。跨线处的弧形跳线表示两条路径不连接。

## 复现

在项目根目录、安装 matplotlib 和 Pillow 的环境中执行：

```powershell
python figures/water_modules/draw_water_module_schematics.py
```

`source_manifest.json` 记录源代码文件及 SHA256。RWA 图已同步本次简化后的实现。图中的张量维度为现有模块的默认配置；N 表示 Query 数，训练时可以包含去噪 Query。

## 检查

已逐图核对张量尺寸、箭头方向、分支和归一化位置；导出脚本会测量标签边界并检查文字不超出节点。SVG 中保留文本节点，不嵌入 PNG 图像。
