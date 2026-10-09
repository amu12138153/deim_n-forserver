# 用指定权重生成检测图与 Grad-CAM

入口：`comparison/visualize_checkpoints.py`。运行环境需要 PyTorch、torchvision、Pillow、NumPy、Matplotlib、PyYAML、faster-coco-eval；YOLO 模型另外需要训练这些权重时兼容的 Ultralytics，D-FINE/RT-DETR 需要对应官方仓库及其依赖。Grad-CAM 在此直接用 PyTorch 实现，不额外依赖 grad-cam 包。

## 指标与 comparison_results.csv 的对应关系

当前 `comparison_results.csv` 由 `comparison/summarize_results.py` 汇总各模型的 `experiment_summary.json`，汇总时核对实际权重、metrics、profile 中的权重 SHA256 和测试标注 SHA256。

`comparison/run_comparison.py` 首先选择用于评估的权重。DEIM 默认 `best_stg1.pth`；D-FINE 优先存在的 `best_stg2.pth`，再用 `best_stg1.pth`；RT-DETR 按其配置的 best 文件；YOLO 用 `weights/best.pt`。显式 `--checkpoint` 会覆盖这些默认选择。因此，应使用原始结果记录的那一份权重，不能仅凭文件名包含 best 判断它与表格一致。

各模型生成 COCO `predictions.json` 后，`comparison/evaluate_coco.py` 统一计算 COCO mAP，并在 IoU=0.5 下按置信度排序、同类别一对一匹配，扫描预测置信度，报告最高 F1 对应的 P、R、F1 和 `Best_Confidence`。新可视化入口复用这些推理适配器和评估器，对完整测试集重新计算指标，再用每个模型自己的 `Best_Confidence` 绘图。

检测图只绘制阈值以上的预测框，按类别使用一致的颜色，并标注类别和置信度。图上只标模型名，不显示 TP/FP/FN 或 P/R/F1。完整测试集指标保存在 CSV/JSON；逐图匹配计数保存在清单中。几张选图上的指标不会等于整个测试集汇总值。小尺寸排版可添加 `--hide-labels`，只显示预测框。

Grad-CAM 展示分类得分对所选空间特征层的响应。目标是阈值以上的前 K 个类别得分之和，默认 K=10；YOLO 使用对应的推理分类分支，DETR 使用 query 类别概率。各图独立归一化至 0–1，色彩展示相对响应。热力图本身不计算或证明 P/R/F1；漏检、误检计数由标注匹配得到，保存到 JSON 中。

## 单模型：从另一台电脑上的最新权重推理

在存放最新权重的电脑上使用本项目代码，或把最新权重与对应的 `config_used.yml` 复制过来。路径均由参数指定，新脚本不会查找本地历史 outputs 中的权重。

所有模型的 `config` 都是可选字段，不按模型名称限制填写：提供时用这份配置构建模型，再加载指定权重；不提供（省略、`null` 或空字符串）时尝试使用权重保存的结构或配置。配置优先于 `family` 和模型键来识别 DEIM、D-FINE、RT-DETR、YOLO；未提供配置的 `.pt` 使用 YOLO 加载器。显示名称保持你填写的内容，实际加载类型与配置来源记录在清单中。

如果某个权重只有参数，没有可恢复的模型结构或配置，加载时会明确提示缺少结构信息；不会自动套用本地旧配置。YOLO 提供的配置应是模型结构 YAML，包含匹配的类别数和网络层；结构与权重不匹配会报错，不做部分参数加载。

鱼类 2 类权重不能直接搭配原始 80 类 COCO 配置。DEIM 还需要确认 `HGNetv2.use_water_quality` 与权重中是否存在 `backbone.water_fusion.*` 相符。出现分类头尺寸或水质模块缺失错误时，应换成对应训练配置；不要用 `strict=False` 跳过未训练模块。入口会预先检查这两项并报告配置值与权重值。

```powershell
python comparison/visualize_checkpoints.py `
  --model-key ours `
  --checkpoint "D:\latest_results\ours\best_stg1.pth" `
  --model-config "D:\latest_results\ours\config_used.yml" `
  --reference-summary "D:\latest_results\ours\metrics.json" `
  --ann-file "D:\fish_dataset\test\coco_detection_test0.json" `
  --image-dir "D:\fish_dataset\test" `
  --sensor-csv "D:\fish_dataset\test\test.csv" `
  --image-ids 0 1 2 `
  --device cuda:0 `
  --out "D:\visualizations\ours_latest"
```

`--reference-summary` 可用原比较实验的 `metrics.json` 或 `experiment_summary.json`。脚本在推理前核对权重与测试标注哈希，在推理后检查复算的 P/R/F1/mAP/阈值（默认绝对误差 1e-5）。复制文件到另一台电脑不会改变哈希。若没有这个参数，脚本仍会推理和生成真实指标，但不会声明与某份历史对比表已核对一致。

`--images image_a.jpg image_b.jpg` 可替代 `--image-ids`，文件名必须与 COCO 标注中的 `file_name` 一致。省略选图参数默认绘制前 3 张，**评估始终覆盖整个标注测试集**，所以初次运行比只处理选图慢。

YOLO：使用 `--model-key yolov8n --checkpoint "...\weights\best.pt"`，也可添加 `--model-config "...\yolov8n.yaml"` 指定匹配的模型结构。YOLO11n、YOLO26n 同理。

D-FINE：使用 `--model-key dfine_n --checkpoint "...\best_stg2.pth" --model-config "...\config_used.yml" --repo-workdir "D:\DEIMv2-main\D-FINE"`。D-FINE-S 用 `dfine_s`。

RT-DETR：使用 `--model-key rtdetr_r18` 或 `rtdetrv2_s`，填写权重，可填写匹配配置，并指定 `--repo-workdir "D:\DEIMv2-main\RT-DETR\rtdetrv2_pytorch"`。如果最新模型使用另一份官方源码，传它的工作目录（其中应含 `src/core`）。

## 九模型：每个模型填写自己的权重

复制并编辑 `comparison/checkpoint_visualization.example.yml`，把每个模型的 `checkpoint` 改成最新文件路径；每个条目的 `config` 都可自由填写或留空。可填写 `reference_summary`、`cam_layer`、`cam_class`，也可通过 `python` 为不同模型指定不同 Python 环境。相对路径相对于 YAML 文件所在目录。

```powershell
python comparison/visualize_checkpoints.py `
  --models-file comparison/checkpoint_visualization.yml `
  --ann-file "D:\fish_dataset\test\coco_detection_test0.json" `
  --image-dir "D:\fish_dataset\test" `
  --sensor-csv "D:\fish_dataset\test\test.csv" `
  --image-ids 0 1 2 `
  --device cuda:0 `
  --out "D:\visualizations\nine_models_latest"
```

填好部分模型后，可用 `--models ours yolov8n dfine_n` 只运行这些键。各模型在独立进程中加载，不会把 D-FINE 与 RT-DETR 两份 `src` 混到一起。

## CAM 层与输出

`cam_layer: auto` 为 DETR 类模型选择 encoder 最后的空间卷积，为 YOLO 选择检测头最后一个输入特征的来源模块。不同架构的层并不完全等价；实际层名写入清单，可按需要显式指定各模型的 CAM 层。保持选择规则一致，不按图片逐张挑层。

添加 `--list-layers` 可只加载模型，把可选卷积层写到每个模型目录的 `cam_layers.json`，不做完整测试评估。单模型通过 `--cam-layer encoder.xxx` 指定；多模型在 YAML 中写 `cam_layer`。`--cam-class hypoxia` 或 YAML 中 `cam_class: hypoxia` 可只解释缺氧类别。

输出结构：

```text
out/
  detections_comparison.png       # 原图 + 各模型检测图，跨模型总览
  gradcam_comparison.png          # 原图 + 各模型 Grad-CAM，跨模型总览
  visualization_metrics.csv      # 完整测试集指标与权重/标注哈希
  ours/
    job.json                     # 所有实际输入路径
    worker.log                   # 子进程完整输出与原始错误
    predictions.json             # 从指定权重重新推理的预测
    metrics.json                 # 统一评估器的实际指标
    reference_check.json         # 与原对比结果的逐指标检查
    visualization_manifest.json # CAM 层、目标、阈值、逐图计数
    figures/
      0_detections.png
      0_gradcam.png
      0_gradcam.npy               # 归一化响应矩阵
```

已存在的模型输出目录默认受到保护；重新生成可用新的 `--out` 或显式 `--overwrite`。这只替换生成文件，不更改输入权重或原始训练结果。
