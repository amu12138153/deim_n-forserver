# 鱼类缺氧检测模型对比：HGNetv2-N 基线

在工程根目录运行。统一入口由 [comparison_models.yml](configs/comparison_models.yml) 控制，所有模型使用 `fish_dataset622`、640×640 输入、batch 8 和完整 200 轮训练。训练只读取 `train`，最优权重由 `val` 选择，指标只在独立 `test` 上计算。COCO 类别统一为 `0=normal, 1=hypoxia`。测试集原始 1/2 标注保存在 `fish_dataset622/test/coco_detection_test.json`，评估读取映射后的 `coco_detection_test0.json`。

激活装有工程依赖、Ultralytics 和官方对比模型依赖的 Python 环境，并确认 CUDA GPU 可用后，一行运行全部对比。未指定 `--ultralytics-python` 时，YOLO 与其他模型都使用当前 `python`：

```powershell
python comparison/run_comparison.py --all --train --eval --profile
```

可先运行 `python -c "import torch, yaml, ultralytics, faster_coco_eval, calflops; assert torch.cuda.is_available()"` 检查当前环境。若 YOLO 依赖只能放在另一套 Python 环境，才需要额外指定 `--ultralytics-python`；上述通用命令要求它们装在同一环境。

运行完成后读取 `comparison_outputs/fish_n_query_200e_common_aug/comparison_results.csv`。九个对比项依次为 YOLOv8n、YOLO11n、YOLO26n、DEIMv2-HGNetv2-N、Ours、D-FINE-N、D-FINE-S、RT-DETRv2-S 和 RT-DETR-R18。`deimv2_n` 使用视觉基线 `M0_visual.yml`；`ours` 使用 `M4_full.yml`，仅在 Query/Decoder 融合温度、溶氧和 pH，**不含浊度**。两者同为 HGNetv2-B0、2 个特征层、3 层 decoder，并使用同一 B0 backbone 预训练文件。YOLO 使用其官方 `.pt` 预训练；D-FINE/RT-DETR 使用各自配置指定的 backbone 预训练。

全部模型的训练增强统一为原始 640×640 图像和概率 0.5 的水平翻转；Mosaic、MixUp、CopyBlend、颜色扰动、随机裁剪、仿射缩放和多尺度训练均关闭。各模型使用 4 epoch 暖身和随后的余弦学习率衰减，保留各自的优化器与基准学习率规则。DEIM/D-FINE/RT-DETR 保留源码中的初始学习率；YOLO 保留 Ultralytics 的 `optimizer=auto`，它可能自动选择优化器并覆盖默认 `lr0`，实际选择记录在每次运行的 `training_metadata.json`。DEIM/D-FINE/RT-DETR 的 PyTorch 调度器是线性暖身加 CosineAnnealingLR；YOLO 使用其自身的暖身和 `cos_lr=True`，阶段一致但离散步进公式并非逐步完全相等。完整设置见[实验说明](../docs/fish_n_query_experiments.md)。

所需官方源码目录是工程下的 `D-FINE/` 与 `RT-DETR/`。YOLO 权重若本地尚无，Ultralytics 会按自身机制获取；网络不可用时应事先准备对应 `.pt` 文件。`--all` 遇到单个模型失败会继续执行其余模型，最后报告失败项并返回非零状态，成功模型的结果仍会汇总。已有非空训练目录会被保护，不覆盖旧结果；重跑正式实验时用新的 `--output-root`，或仅对失败项用 `--model <模型键>` 指定新目录。

单独运行示例：

```powershell
python comparison/run_comparison.py --model deimv2_n --train --eval --profile
python comparison/run_comparison.py --model ours --train --eval --profile
```

各模型先生成 COCO 格式的 `predictions.json`，由同一 `evaluate_coco.py` 得到 `metrics.json`，再生成参数量和 FLOPs 的 `profile.json`。只汇总测试标注哈希、checkpoint 哈希及训练元数据一致的完整结果。`--speed` 可额外生成统一口径的 batch=1 模型前向 FPS；默认命令不测 FPS。

快速合同测试：

```powershell
python -m unittest discover -s comparison -p test_*.py
```
