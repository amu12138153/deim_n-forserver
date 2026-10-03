# DEIMv2 Pico 水质感知模型：科研架构图结构说明

交付PNG为1536×1024。已知两处连线偏差：缺少 Query initialization → Decoder layer 1；最终水质 T 输出端多余反向箭头。重绘必须按本文纠正，详见 visual_qa.md。本说明的科学结构不受图像偏差影响。

## 1. Figure purpose
按用户确认的 `configs/deimv2/visdrone_pico.yml` 与当前工作区实际代码，表示图像—水质双输入目标检测模型。强调 HGNetv2(Pico) → LiteEncoder → 三层 DEIMTransformer，以及每层查询自注意力之后的水质交叉注意力。本文是标签、层级、连接的权威；生成 PNG 只是视觉参考，不是可编辑矢量文件。

## 2. Scientific evidence boundary
- explicit：以下模块、参数、连接均来自配置、继承配置或 forward 实现。尺寸由 640×640 输入代入代码推导，不是运行实测。
- strongly implied：主图将 flatten/concat 结果称为 Multi-scale memory，将函数操作框、跨层数据总线作为绘图抽象；不是额外神经网络层。
- unknown：训练后权重、精度、速度、消融增益、类别名称。不得补造。
- 用户已确认本次无需 PDF；未使用任何 PDF 科学内容。
- 唯一风格输入为 codex-clipboard-17691b87-1bae-400f-b08d-1b9327505949.png。其 YOLO 模块、损失头及图注均无科学证据权威。
- 原请求路径 `configs/deimv2/visdrone/_pico.yml` 不存在；已由用户明确确认采用 `configs/deimv2/visdrone_pico.yml`。
- 名称含 visdrone，但当前 dataset 配置指向鱼类图像及传感器 CSV，num_classes=2。不得画无人机、VisDrone 10 类或 COCO 80 类标签。
- 本次只新增交付材料，不修改模型或配置，不调用 Draw.io。

### 源码索引
路径均相对项目根目录。
| 文件 / 入口 | 依据 |
|---|---|
| configs/deimv2/visdrone_pico.yml | 本次配置，water 开关、宽度、层数、queries 等 |
| configs/base/deimv2.yml | 继承 reg_max=32、reg_scale=4、n_denoising=100、损失和后处理 |
| configs/dataset/visdrone.yml | 2 类、图像与四维水质输入 |
| engine/deim/deim.py:13 | DEIM 的 backbone → encoder → decoder，water 传递 |
| engine/backbone/hgnetv2.py:476 | Pico 三阶段配置 |
| engine/backbone/hgnetv2.py:126,205,294,380,735 | stem、HG_Block、se 实际实现、stage、forward |
| engine/deim/lite_encoder.py:27,41,95 | GAP_Fusion、双尺度编码器与连接 |
| engine/deim/hybrid_encoder.py:192,220 | RepNCSPELAN4、CSPLayer2 |
| engine/deim/deim_decoder.py:31,122,235,559 | decoder layer、FDR、初始化、总 forward |
| engine/mynewblock/WaterQueryCrossAttention.py:22,202 | 水质编码与查询水质融合 |
| engine/deim/deim_utils.py | RMSNorm、SwiGLUFFN、MLP |
| engine/deim/dfine_decoder.py:246,270 | Integral、LQE |
| engine/deim/postprocessor.py:50 | sigmoid 与 top-300 query-class pairs |
| engine/deim/deim_criterion.py:165,362 | local、aux 与训练损失 |
| engine/data/dataset/coco_dataset.py:33 | 水质顺序、按文件名配对 |

## 3. Canvas
- 横向 3:2；目标 3072×2048（由图像工具实际支持能力决定，目标不是已验证结果）。
- 白色背景。主图左→右；骨干、解码层内部上→下。
- 主图区约上方 60%；下方并列 Decoder layer 与 Water fusion 两个核心详图。HG/Rep 保留宏观框，完整内部结构见本文，后续可在附页展开。
- 图内保留英文技术名称；中文解释留在本文件。
- 主标题：DEIMv2 Pico with Water-Aware Queries。
- 常规实线箭头：激活张量流；紫色实线箭头：原始水质输入；训练专用虚线箭头；无箭头细虚线仅用于详图关联。

## 4. Regions or panels
| ID | 标题 | 位置 | 用途 |
|---|---|---|---|
| R01 | HGNetv2 (Pico) | 主图左 | Stem + 三阶段 |
| R02 | LiteEncoder | 主图中左 | 单输入生成 stride 16/32 双尺度 |
| R03 | DEIMTransformer | 主图中右 | memory、Top-200、3 层及预测 |
| R04 | Outputs | 主图最右 | logits / boxes 及后处理 |
| R05 | Water input | 主图解码器上侧 | 四个传感器值广播到每个 decoder layer |
| R06 | HG_Block / RepNCSPELAN4 | 文字规范 / 后续附页 | 主图折叠，内部结构完整保留在本文 |
| R07 | TransformerDecoderLayer | 下方中部 | 每层真实操作顺序 |
| R08 | WaterAwareQueryCrossAttention | 下方右侧 | 水质 token 编码与两次标量门控残差 |
| R09 | Training only | 主图区底部窄条 | denoising 和损失关系概括 |

## 5. Modules
除圆形加法/拼接节点外均为小圆角矩形；region 用淡色虚线边框。所有 ID 均稳定，不取决于位置。
| ID | Exact label | Parent | 角色/参数 | 优先级 |
|---|---|---|---|---|
| I01 | RGB image | root | B×3×640×640 | high |
| B01 | StemBlock | R01 | 3→16；stride 4 | high |
| B02 | Stage 1 | R01 | HG_Block ×1；16→64；stride 4 | high |
| B03 | Stage 2 | R01 | DWConv s2 + HG_Block ×1；64→256；stride 8 | high |
| B04 | Stage 3 | R01 | DWConv s2 + HG_Block ×2；256→512；stride 16 | high |
| E01 | Input projection | R02 | Conv1×1 512→112 + BN；P16 | high |
| E02 | Downsample 1 | R02 | AvgPool3×3 s2 p1 → Conv1×1 112→112 → BN → SiLU | high |
| E03 | GAP_Fusion | R02 | G32=Conv1×1+BN+SiLU(x+GAP(x)) | high |
| E04 | Upsample ×2 | R02 | nearest；G32→stride16 | high |
| E05 | + | R02 | P16 与上采样 G32 逐元素相加 | high |
| E06 | RepNCSPELAN4 (FPN) | R02 | fpn_block，112 通道 | high |
| E07 | Downsample 2 | R02 | 结构同 E02，参数独立 | high |
| E08 | + | R02 | G32 与下采样 F16 逐元素相加 | high |
| E09 | RepNCSPELAN4 (PAN) | R02 | pan_block，112 通道 | high |
| F16 | F16 | R02 | B×112×40×40，编码器第一个输出 | high |
| F32 | F32 | R02 | B×112×20×20，编码器第二个输出 | high |
| D01 | Flatten + Concat | R03 | 输入投影为两个 Identity；序列拼接 | high |
| D02 | Multi-scale memory | R03 | B×2000×112；1600+400 tokens | high |
| D03 | Top-200 query selection | R03 | enc_score_head→每个位置 max class logit→topk；gather memory/anchors | high |
| D04 | Initial queries + refs | R03 | Q0=B×200×112；refs=B×200×4 | high |
| D05 | Query position MLP | R03 | sigmoid(refs)→MLP(4,112,112,3)→clamp[-10,10] | medium |
| L1 | Decoder layer 1 | R03 | 下述层模板实例；参数独立 | high |
| L2 | Decoder layer 2 | R03 | 同一层模板；参数独立 | high |
| L3 | Decoder layer 3 | R03 | 同一层模板；参数独立；eval_idx=2 | high |
| H00 | Pre-box head | R03 | 仅第一层输出，MLP112→112→112→4，建立固定 FDR reference | high |
| H01 | Shared bbox head + FDR | R03 | 三层共享 bbox MLP，输出 4×33=132 分布 logits | high |
| H02 | Class heads + LQE | R03 | 每层独立 Linear112→2 和独立 LQE | high |
| O01 | pred_logits | R04 | 最后层 B×200×2 | high |
| O02 | pred_boxes | R04 | 最后层 B×200×4，归一化 cxcywh | high |
| O03 | PostProcessor | R04 | sigmoid、top-300 query-class pairs、原尺寸 xyxy | medium |
| O04 | labels / scores / boxes | R04 | B 个检测结果列表；未执行置信度筛选时每图 300 项 | medium |
| W00 | Water input | R05 | temperature / do / ph / turbidity；B×4 | high |
| T01 | GT labels + boxes | R09 | 训练输入，不能画成推理输入 | medium |
| T02 | Contrastive denoising | R09 | num_denoising=100 为配置预算，并非固定精确增加100个查询 | medium |
| T03 | DEIMCriterion | R09 | mal / boxes / local，含 HungarianMatcher 和辅助监督 | medium |

### 5.1 Backbone 内部层级与尺寸
ConvBNAct = Conv2d → BatchNorm2d → ReLU → LearnableAffineBlock（use_lab=True、use_act=True 时）；关闭 use_act 时后两者为 Identity。
LightConvBNAct = Conv1×1+BN（无激活）→ depthwise Conv5×5+BN+ReLU+LAB。

StemBlock 子节点和连接：
- B01.s1：ConvBNAct k3 s2，3→16；640→320。
- B01.pad1：右侧/下侧 pad1，320→321。
- B01.pool：MaxPool k2 s1 ceil_mode=True，321→320。
- B01.s2a：ConvBNAct k2 s1，16→8，321→320。
- B01.pad2：右/下 pad1。
- B01.s2b：ConvBNAct k2 s1，8→16，321→320。
- B01.cat：Concat([pool, s2b])，32 通道。
- B01.s3：ConvBNAct k3 s2，32→16，320→160。
- B01.s4：ConvBNAct k1 s1，16→16。
- 连接：I01→s1→pad1；pad1→pool→cat；pad1→s2a→pad2→s2b→cat；cat→s3→s4。
- Stem 输出 B×16×160×160。

HG_Block 模板 HGB（在 B02/B03/B04 内实例化）：
- HGB.in→HGB.c1→HGB.c2→HGB.c3。
- HGB.in、c1、c2、c3 均→HGB.cat（沿 channel concat）。
- HGB.cat→HGB.a1（ConvBNAct k1，total_chs→out_chs/2）→HGB.a2（ConvBNAct k1，out_chs/2→out_chs）。
- residual=False：a2→out。residual=True：a2 与 in→HGB.add→out。drop_path 为 Identity。
- B02：1 个 HGB，mid=16，三次普通 k3 Conv；cat 16+3×16=64；聚合64→32→64；无残差。
- B03：DWConv k3 s2 groups64（Conv+BN，无激活）→1 个 HGB，mid=32，普通 k3 Conv；cat64+96=160；聚合160→128→256；无残差。
- B04：DWConv k3 s2 groups256→2 个 HGB；每个 HGB 含3个 LightConvBNAct k5，mid=64。第1个 cat256+192=448，聚合448→256→512，无残差；第2个 cat512+192=704，聚合704→256→512，有残差。
- 输出尺寸依次 B×64×160×160、B×256×80×80、B×512×40×40。
- 只返回 Stage 3（return_idx=[2]）；Stage 1/2 不直接连编码器。Pico 没有 Stage 4。
- `agg='se'` 实际是以上两个卷积，不能画为带 GAP/sigmoid/通道乘法的标准 SE/ESE。
- backbone.use_water_quality=False；不得连水质到骨干。

### 5.2 LiteEncoder 内部层级
- hidden_dim=112，expansion=0.34，depth_mult=0.5，act=silu，csp_type=csp2。
- RepNCSPELAN4(c1=112,c2=112,c3=224,c4=19,n=2)，其中 c4=round(0.34×112//2)=19，n=round(3×0.5)=2。
- E06 和 E09 结构相同、权重不同，不画为共享参数。
- RepNCSPELAN4 模板 REP：in→cv1(Conv1×1+BN+SiLU，112→224)→Split(112,112)，得到 a、b。
- b→cv2[CSPLayer2(112→19,n=2)→Conv3×3 19→19]→c。
- c→cv3[CSPLayer2(19→19,n=2)→Conv3×3 19→19]→d。
- [a,b,c,d]→Concat(262 channels)→cv4(Conv1×1+BN+SiLU，262→112)→out。
- 每个 CSPLayer2：Conv1×1 in→38→Split(19,19)；第一半旁路，第二半→VGGBlock×2；两半相加；conv3=Identity。
- VGGBlock：输入分成 Conv3×3+BN 和 Conv1×1+BN 两支→相加→SiLU；可在 deploy 时合并为单卷积。
- E03 单独包含 x→GAP(x)，x 与 GAP(x) 广播相加→Conv1×1+BN+SiLU。GAP 不是第三个输出尺度。
- 外部两处融合 E05/E08 是 Add，不能按风格例图画成 Concat。
- encoder.use_water_quality=False；不得添加水质注入。

### 5.3 Decoder 初始化、memory 与每层模板
- 两个 112 通道输入投影均为 Identity。F16/F32 展平后沿 sequence 维拼接；memory 同时提供查询选择输入及所有三层 MSDeformableAttention 的视觉 value。
- anchors 由两级规则网格生成；base grid_size=0.05，level 1 尺寸倍增。valid_mask 屏蔽不合法 anchors 对应的候选 memory。
- D03：enc_score_head Linear112→2；按位置的最大类别 logit 选 Top-200（不是单独选200×类别）。
- D04：Q0=selected_memory.detach()；未开启 learn_query_content，无可学习 query embedding。
- 初始 refs_unact=enc_bbox_head(selected_memory)+selected_anchor_logits，再 detach。enc_bbox_head=MLP112→112→112→4（SiLU）。
- D05 使用初始 refs 的 sigmoid 输出，生成 query position；当前代码只在层循环前计算一次，不按更新框逐层重算。
- 每层 Li 的子 ID 均使用 Li.SA、Li.W、Li.CA、Li.FFN 等前缀：
  1. Li.SA：MultiheadAttention 8 heads；Q=K=target+query_pos，V=target。
  2. Li.AN1：残差 Add → RMSNorm，dropout=0。
  3. Li.W：WaterAwareQueryCrossAttention；仅 water 非 None 时执行。
  4. Li.CA：MSDeformableAttention 8 heads、2 levels、num_points=[4,2]；Q=water-fused target+query_pos；视觉 value 来自 memory；采样参考框来自该层输入 refs。
  5. Li.AN2：残差 Add → RMSNorm。use_gateway=False，无视觉 Gateway。
  6. Li.FFN：SwiGLUFFN，配置 dim_feedforward=320，实际 w12 Linear112→320，split 两个160，SiLU(a)×b→Linear160→112。
  7. Li.AN3：残差 Add → clamp[-65504,65504] → RMSNorm。
- L1→L2→L3 传递 query content；更新 refs.detach() 送下一层采样。query_pos 始终是 D05 的初始位置编码。
- 三层 attention/水质编码/水质融合参数均独立；只有 bbox head 跨三层共享。

### 5.4 WaterAwareQueryCrossAttention 内部模板
为每个 Li.W 独立实例化以下 W 子节点；主图的同一个原始 W00 输入广播三层，不把 token encoder 画成跨层权重共享。
| 子ID | Exact label | 操作 |
|---|---|---|
| W01 | Normalize + Clamp | (water−mean)/std → nan_to_num(nan=0) → clamp[-5,5] |
| W02 | Value MLP | 对每个标量：Linear1→16 → SiLU → Linear16→64；值编码器在4个传感器间共享 |
| W03 | Parameter embedding | 4×64 个可学习参数；区分传感器类型 |
| W04 | + | Value tokens + parameter embedding；T0=B×4×64 |
| W05 | LN + Water self-attention | LayerNorm64 → MultiheadAttention64，4 heads，Q=K=V |
| W06 | Delta MLP | Linear64→64 → SiLU → Linear64→64 |
| W07 | Gated residual | T=T0+interaction_scale×Delta；标量初值0 |
| W08 | LayerNorm | water_norm64(T)；作为 query-water attention 的 K,V |
| W09 | LayerNorm | query_norm112(q)；作为 query-water attention 的 Q |
| W10 | Query-Water Cross-Attention | 8 heads，embed_dim112，kdim=vdim=64 |
| W11 | Gated residual | q1=q+attn_scale×attn_out；标量初值0 |
| W12 | Water fusion FFN | LN112 → Linear112→448 → GELU → Dropout0 → Linear448→112 → Dropout0 |
| W13 | Gated residual | q2=q1+ffn_scale×FFN(q1)；标量初值0 |
- water 特征顺序：temperature, do, ph, turbidity。
- mean=[27.878,4.547,8.069,0.125]；std=[0.673,1.939,0.229,0.914]。只写文档，不把长统计表塞进主图。
- 水质值按图像文件名从 CSV 配对；缺值置 NaN，标准化后置0。不能宣称传感器量纲或实测数值。
- 三个尺度是可学习标量，不是额外 sigmoid 门网络；不能把 `learnable_gate=True` 误认为视觉 `use_gateway=True`。
- 完整子图连接：W00→W01→W02→W04；W03→W04；W04→W05→W06→W07；W04→W07；W07→W08→W10(K,V)；q→W09→W10(Q)；W10→W11；q→W11；W11→W12→W13；W11→W13→Li.CA 的 query 输入。

### 5.5 Predictions 与跨层细化
令层索引 i=1,2,3；h_i 为 Li 输出；sg 表示 detach。变量命名仅用于说明现有代码。
- 第1层：b_pre=sigmoid(pre_bbox_head(h_1)+inverse_sigmoid(r0))，b_base=sg(b_pre)；所有三层 FDR 均相对同一个 b_base。
- h_prev=0（i=1）；i>1 时 h_prev=sg(h_(i−1))。
- d_prev=0（i=1）；i>1 时 d_prev=d_(i−1)，分布累加项没有 detach。
- d_i=shared_bbox_head(h_i+h_prev)+d_prev。
- bbox MLP：112→112→112→132；SiLU；132=4×(reg_max+1)=4×33。
- b_i=distance2bbox(b_base, Integral(d_i, project), reg_scale=4)；Integral 对每个33-bin分布做 softmax 与非均匀权重投影；up=0.5。
- b_i.detach() 是下一层视觉 deformable attention 的 reference points；不能画成重设 b_base。
- cls_i=Linear_i(h_i)，输出2类；score_i=LQE_i(cls_i,d_i)；LQE 取每个边分布 top4 概率及其均值，MLP20→64→1，质量 logit 广播加到类别 logits。
- 训练时每层执行 LQE/score 输出；普通 eval 仅第3层执行 score/LQE 输出（第1层仍计算 pre_scores）；前两层边框细化仍须执行。
- H01 是三层重复调用的共享权重子图，可在主图用总线容器概括；H02 是每层独立子图，不允许标 shared。
- 最终输出 O01=score_3，O02=b_3。
- PostProcessor 独立于 DEIM.forward：sigmoid(logits) 后在200×2=400个 query-class pairs 上选300，转 xyxy 并按原图大小缩放。300 不等于300个不同 object queries，也不保证300个不同物体。无 NMS 模块。

### 5.6 Training only
- T01→T02→D04：contrastive denoising 的 label embeddings / noisy boxes 追加到正常200个匹配 queries；同时提供 self-attention mask。
- num_denoising=100、label_noise_ratio=0.5、box_noise_scale=1.0；实际 denoising token 数由 batch 中 GT 和分组决定。
- T03 内含 HungarianMatcher；T01 与预测共同用于匹配及损失。
- losses=['mal','boxes','local']；weight_dict={loss_mal:1, loss_bbox:5, loss_giou:2, loss_fgl:0.15, loss_ddf:1.5}。
- boxes 包含 L1 + GIoU；local 包含 FGL，存在 teacher 字段等条件时计算 DDF。不能写成每个分支无条件具有全部五项损失。
- main、前两层 aux、encoder selected proposals、first-layer pre_outputs、denoising 各支按代码条件监督。末层分布/logits 作为 aux teacher 数据。
- 主图可将上述训练细节统一收进一个 Training only 容器，以虚线连接预测与 GT；不得把它串入推理路径。
- 优化器、学习率、数据增强不属于本次模型架构主图；不添加反向传播闭环。

## 6. Connections
以下主图有向边全部为 explicit；图中的概括框均展开于第5节。
| Source | Target | Arrow label |
|---|---|---|
| I01 | B01 | RGB |
| B01 | B02 | 16 / s4 |
| B02 | B03 | 64 / s4 |
| B03 | B04 | 256 / s8 |
| B04 | E01 | 512 / s16 |
| E01 | E02 | P16 |
| E02 | E03 | 112 / s32 |
| E03 | E04 | G32 |
| E04 | E05 | upsampled G32 |
| E01 | E05 | P16 |
| E05 | E06 | elementwise sum |
| E06 | F16 | 112 / s16 |
| F16 | E07 | F16 |
| E07 | E08 | downsampled F16 |
| E03 | E08 | G32 |
| E08 | E09 | elementwise sum |
| E09 | F32 | 112 / s32 |
| F16 | D01 | level 0 |
| F32 | D01 | level 1 |
| D01 | D02 | concat along sequence |
| D02 | D03 | candidate memory |
| D03 | D04 | selected content + regressed anchor refs |
| D04 | D05 | r0 |
| D04 | L1 | Q0, r0 |
| D05 | L1, L2, L3 | fixed query_pos |
| D02 | L1.CA, L2.CA, L3.CA | visual value + spatial shapes |
| W00 | L1.W, L2.W, L3.W | raw water; independent encoders |
| L1 | L2 | content; detached refined refs |
| L2 | L3 | content; detached refined refs |
| L1 | H00 | h1 |
| D04 | H00 | r0 |
| H00 | H01 | fixed b_base |
| L1, L2, L3 | H01 | per-layer h_i; previous detached h |
| H01 | L2, L3 | preceding refined boxes, detached |
| L1, L2, L3 | H02 | class-head inputs; eval only final + pre |
| H01 | H02 | per-layer distribution for LQE |
| H01 | O02 | b3 |
| H02 | O01 | score3 |
| O01, O02 | O03 | logits, boxes |
| O03 | O04 | labels, scores, boxes |
| T01 | T02 | training only |
| T02 | D04 | DN content + refs; self-attention mask |
| T01 | T03 | training only |
| R03 predictions | T03 | main / aux / encoder / pre / DN (training only) |

逗号目标表示分别连到各个目标，不是额外拼接节点。H01→L2/L3 为逐层参考框更新，不表示重复推理至收敛。实例展开边以5.3–5.5节为准。

## 7. Visual style
- 从例图提取：白底、浅色模块、细灰黑边框、淡色虚线分组、简洁正交箭头、小圆形操作节点、主干与模块详图的层次。
- 蓝 #B9D5EA：卷积/投影/视觉 attention；桃 #FBE5D5：HG/Rep 聚合；绿 #E4F2D7：池化/上采样；紫 #D9C6EA：水质融合；黄 #FFF2CC：预测/细化；粉 #F3CBCB：Add；浅灰 #F1F3F5：张量和输出。
- 边线 #58636D，正文 #25313B；框线约1.5–2px（2560尺度），箭头1.5px实线，小实心三角；容器淡色底、不透明、虚线边框。
- 圆角约4–8px，禁止胶囊形泛滥；Add用“+”，Concat用“C”或完整“Concat”，必须区分。
- 标题38–46px，分区28–32px，模块22–26px（3840画布按比例放大），Arial/Helvetica；足够内边距，不能把长类名截断。
- 無照片、3D堆叠特征图、纹理、阴影、渐变、装饰性鱼图标；RGB image 用简单矩形即可。

## 8. Label authority
- 原名 HGNetv2、LiteEncoder、DEIMTransformer、RepNCSPELAN4、WaterTokenEncoder、WaterAwareQueryCrossAttention 必须保留。
- 主图可用“Decoder layer 1/2/3”和“Water fusion”，并在详图给出完整类名。
- 框内长名可换行或用 ID，交接重绘时从本文件取完整标签，不从 PNG OCR 猜测。
- + 为逐元素相加，C 为拼接；σ 仅用于已证实的 sigmoid 操作，不给水质尺度参数添加 σ。
- 不显示任何来自风格图的 C2f、C2f-OP、P-DConv、EMBC、SPPF、YOLO 或 Bbox_Loss/Cls_Loss。

## 9. Simplification rules
- 必须保留三个骨干 stage、stage3 ×2、两级编码器输出、两处 Add 融合、memory 分支、Top-200初始化、三层decoder、水质逐层注入、共享bbox/FDR与独立class/LQE。
- 允许主图折叠 Conv/BN/activation、HG/Rep 和预测细节，但在详图或本文层级模板完整保留。
- Rep 及 HG 内部 Concat 与编码器外部 Add 不可相互替换。
- 详图用同一个模板代表重复实例，不表示权重共享；仅 bbox head 为明确共享。
- 损失放底部窄条；图内不塞水质统计数值或训练超参数。
- 不增加三尺度P3/P4/P5骨干输出、AIFI/HybridEncoder、P2检测头、SPPF、NMS、视觉Gate或水质→backbone/encoder边。

## 10. Known uncertainties / validation limits
- 已完成配置继承与 forward 的静态交叉核对；未加载权重、未运行训练或实测前向。
- 项目 .venv 指向不存在的旧 Python 路径；系统 Python 缺 torch/yaml。本次未安装依赖、未更改运行环境。
- 640×640 是示例/评估尺寸；训练时可能多尺度，不能把1600+400 tokens 宣称为所有输入下恒定。
- water=None 时解码器水质融合被跳过；图展示水质存在的已配置路径。
- 图像生成可能出现文字或箭头偏差；重绘必须以本说明为准，不能把未核对 PNG 作为唯一结构依据。
