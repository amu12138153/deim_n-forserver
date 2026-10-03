# 参考图检查记录

## 交付状态
- reference_figure.png：内置图像工具生成的单张主参考图，1536×1024，3:2，PNG 位图。
- 最初目标为3072×2048，工具实际返回1536×1024；未插值放大，未冒充可编辑矢量文件。
- 科学结构来自用户确认的 visdrone_pico.yml 与当前源码；本次无需 PDF。
- 图像借用例图的浅色模块、细线、虚线容器和矩形语言，没有沿用其 YOLO 科学模块。
- 完整模块层级、条件分支、参数和逐条连接在 diagram_spec.md。
- drawio_handoff.md 为自包含交接提示词；重绘时科学结构优先于 PNG。
- 未启动、未调用 Draw.io Scientific Illustrator；未创建 .drawio；未修改模型代码或配置。
- 未做模型前向运行验证；尺寸由源码静态推导。

## 最终 PNG 仍需在 Draw.io 重绘时修正的两处连线
1. 主图 Query initialization 有来自 Memory 的输入，但缺少通向 Decoder layer 1 的输出。重绘必须添加 Query initialization → Decoder layer 1，标签 Q0, r0。现有 Memory → Decoder layer 1 水平箭头不能作为 query 初始化捷径；删掉该箭头或明确合并到 visual-memory V 端口。三个 memory V 端口均与同一个视觉 memory 对应。
2. 水质详图的最终 T / 4×64 → Query-Water Cross-Attention 连线，在 T 端仍残留一个指向 T 的反向箭头。删去这个反向箭头，只保留 T → attention K,V 的单向流。不能解释成双向交互或 attention 反馈到 water encoder。

## 概括表示，不是额外科学关系
- 黄色 Per-layer heads and refinement 是容器级概括，不表示三个层各自只使用某一个预测头。每层都调用共享 bbox head/FDR；训练时各层独立 class/LQE，推理用末层。Pre-box head 仅作用于首层。
- query initialization 在主图合并表示 Top-200、参考框初始化；query_pos 用文字说明。Draw.io 重绘可按规范展开 D03/D04/D05。
- 两个编码器输出和三个 memory V 用同名端口表示连接；虚线端口连接不是训练专用边。本图底部训练部分仅文本说明。
- 水质 token 残差用公式表示，不能略去 interaction_scale；attn_scale 与 ffn_scale 同理。
- HG_Block 与 RepNCSPELAN4 为折叠模块，完整子结构与所有内部连接在结构说明；后续可附页展开。

## 已核对的核心内容
- Backbone 三阶段，stage3 两个 HG_Block，仅 stride16 输出。
- LiteEncoder 投影512→112，两路输出112×40×40及112×20×20，两处逐元素 Add。
- G32→Upsample→第一处 Add；P16→第一处 Add；G32 与 downsample(F16)→第二处 Add。
- 3 层decoder，200 queries，2类，4×33分布边框，bbox权重共享。
- 水质只进入decoder，4项水质特征，各层独立 WaterTokenEncoder。
- decoder 顺序 SA→水质融合→视觉CA→SwiGLUFFN，视觉 Gateway关闭。
- 残差尺度初始化0；不能解释为推理始终没有水质作用，训练后参数可变化。
- PostProcessor top-300选的是query-class pairs，无NMS。
- 无性能数值、类别名称、PDF来源、训练结果等无证据内容。

图像经历结构修正，但由于生成式图像的连线保真限制，以上两项未在PNG内可靠消除，明确保留为待重绘修正项；不得将PNG视为已逐箭头通过审校的最终发表图。

