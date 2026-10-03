# HGNetv2-N 鱼类缺氧检测：Query/Decoder 水质融合实验

在另一台电脑上先激活已安装工程依赖、Ultralytics 和官方对比模型依赖的 Python 环境，再从工程根目录运行以下两行命令。脚本及其子进程默认使用当前 `python`；整套对比训练需要可用的 CUDA GPU。两个批量任务耗时较长，输出目录不能已有同名非空实验。

```powershell
python run_ablation.py --group all
python comparison/run_comparison.py --all --train --eval --profile
```

消融输出在 `outputs/<时间>_ablation_n_query_all/`，自动生成每项的 `experiment_summary.json` 和总表 `ablation_summary.csv`。对比输出在 `comparison_outputs/fish_n_query_200e_common_aug/`，总表是 `comparison_results.csv`。若需重跑对比，指定新的 `--output-root`；单项失败可用 `--model <模型键>` 在新目录补跑。

## 基线与融合位置

所有 DEIM 消融配置继承 [原始 N 配置](../configs/deimv2/deimv2_hgnetv2_n_coco.yml) 和 [鱼数据集配置](../configs/dataset/fish_01.yml)，保留 HGNetv2-B0、`return_idx=[2,3]`、HybridEncoder 的 2 个特征层和 128 维隐藏层、3 层 DEIM decoder。训练 200 轮，batch=8，输入 640×640，使用本地 `weight/hgnetv2/PPHGNetV2_B0_stage1.pth` backbone 预训练。

水质张量随数据载入，但 HGNetv2 与 HybridEncoder 不执行水质融合。水质只在 Query/Decoder 的水质自注意力、查询水质交叉注意力和可学习门控处使用。视觉基线 `M0_visual` 关闭这些模块；主模型 `M4_full` 开启全部三个模块，并且**只用温度、溶氧、pH，不含浊度**。

## 数据增强及水质对应

全部 12 个消融实验和 9 个对比模型采用同一训练增强策略：原图尺寸已经是 640×640，固定缩放到 640×640，并以 `p=0.5` 水平翻转，边框同步变换。Mosaic、MixUp、CopyBlend、YOLO 的 CutMix/CopyPaste、颜色扰动、随机裁剪、仿射缩放和多尺度训练均关闭。验证和独立测试只做确定性的 640×640 输入处理。这样每张训练图只对应自己的水质读数；全部 200 轮使用同一个增强策略，不再按轮数更换变换。

训练预算统一为 200 epoch、batch 8、随机种子 0、AMP 和 EMA。学习率统一分成前 4 epoch 暖身及后续余弦下降至 0；448 张训练图每轮产生 56 个 batch，因此 PyTorch 模型暖身 224 次更新。DEIM/D-FINE/RT-DETR 保留源码的初始学习率，并使用 `LinearWarmup` 和 `CosineAnnealingLR`。YOLO 保留 Ultralytics 的 `optimizer=auto` 与 `cos_lr=True`：自动优化器可能覆盖其默认 `lr0`，具体选择写入每次训练的 `training_metadata.json`。由于实现不同，暖身与余弦曲线的逐步数值不能声称完全相等；优化器分组和损失函数也仍随模型而异。

| 实验组 | 配置 | 改动 |
| --- | --- | --- |
| 变量消融 | V0–V6 | V0 无水质；V1 四变量含浊度；V2 是三变量主模型；V3–V6 为三变量的子集。水质模型均启用完整 Query/Decoder 融合。 |
| 组件消融 | M0–M4 | M0 视觉基线；M1 仅查询水质交叉注意力；M2 加水质自注意力；M3 加门控；M4 全部开启。**M1–M4 均固定温度、溶氧、pH。** |

`V2_T_DO_pH` 与 `M4_full` 是同一种结构的独立实验；`V0_visual` 与 `M0_visual` 也是同一种结构。保留两组结果便于各表独立阅读。对比实验的 `deimv2_n` 指向 M0 配置，`ours` 指向 M4 配置，各自按同样数据和训练预算重新训练。

## 数据与类别

训练、验证、测试分别读取 `fish_dataset622/{train,val,test}`；标注分别是 `coco_detection_train0.json`、`coco_detection_val0.json`、`coco_detection_test0.json`。类别统一为 `0=normal, 1=hypoxia`。原始测试标注是 1/2 编号，已由 [映射脚本](../tools/remap_fish_test_categories.py)生成 0/1 版本，原文件保留。测试集共 151 张图、1475 个框；与训练、验证集无同名图像，也与验证集无逐文件字节重复。模型选择只使用验证集，最终指标读取独立测试集。

## 运行前校验

```powershell
python -c "import torch, yaml, ultralytics, faster_coco_eval, calflops; assert torch.cuda.is_available()"
python tools/remap_fish_test_categories.py
python -m unittest discover -s tests -p test_n_query_suite.py
python run_ablation.py --group all --dry-run
```

YOLO 官方权重和 D-FINE、RT-DETR 的 backbone 预训练文件需由对应环境/仓库可访问。比较模型的批量见 [对比实验说明](../comparison/README.md)。
