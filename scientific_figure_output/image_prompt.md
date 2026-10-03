Use case: scientific-educational.
Create exactly ONE coherent, publication-oriented, flat scientific architecture reference image for later reconstruction as editable shapes in draw.io. Do not create a draw.io file.
Target: landscape 3:2, ideally 3072x2048 or the highest supported matching resolution, opaque white background. Clean readable technical labels, no logos.

INPUT IMAGE ROLE
The supplied image is STYLE REFERENCE ONLY. Borrow the pastel palette, thin dark outlines, dashed pale containers, simple rectangles, circular operators, orthogonal arrow style, and main architecture plus module-detail composition. Do NOT copy any scientific content, labels, model names, modules, losses, photos, or caption from that image. In particular, this is NOT YOLO; no C2f, P-DConv, EMBC, SPPF, or YOLO heads.

TITLE
"DEIMv2 Pico with Water-Aware Queries"
This is the actual customized model from the user's local configuration, not a generic DEIM diagram.

COMPOSITION
Upper 54%: clear left-to-right system diagram. Backbone is a compact vertical column at left; LiteEncoder is a two-resolution fusion graph in the middle-left; decoder occupies middle-right; outputs at far right.
Below: four clearly separated module-detail inset panels, HG_Block (small), RepNCSPELAN4 (small), TransformerDecoderLayer (medium), WaterAwareQueryCrossAttention (largest). Insets are explanatory expansions, not additional pipeline stages. Use small letter labels (a)-(e) if useful, but no unrelated panels.
A narrow bottom footer contains a clearly marked dashed "Training only" strip, and a small + / C legend. Do not crowd text. Keep all arrows from crossing text.

MAIN DIAGRAM: exact modules and topology
RGB image [3 x 640 x 640]
 -> container "HGNetv2 (Pico)":
   "StemBlock" [16 / s4]
   -> "Stage 1" [HG_Block x1; 64 / s4]
   -> "Stage 2" [DWConv s2; HG_Block x1; 256 / s8]
   -> "Stage 3" [DWConv s2; HG_Block x2; 512 / s16].
Only Stage 3 connects to LiteEncoder. No Stage 4, no lateral outputs from Stage 1 or Stage 2.

Container "LiteEncoder", width 112:
P16 = "Input projection" [1x1 Conv + BN; 512 ->112; s16], receives Stage 3.
P16 -> "Downsample 1" [AvgPool s2 + 1x1 Conv / BN / SiLU]
 -> "GAP_Fusion" [x + GAP(x), then 1x1 Conv / BN / SiLU]; call its output G32.
G32 -> "Upsample x2" [nearest] -> pink circle "+".
P16 ALSO feeds that same "+" via a skip path.
"+" -> "RepNCSPELAN4" [FPN] -> "F16" [112 x 40 x 40].
F16 -> "Downsample 2" [AvgPool s2 + 1x1 Conv / BN / SiLU] -> second pink "+".
G32 ALSO feeds the second "+" via a skip path.
second "+" -> "RepNCSPELAN4" [PAN] -> "F32" [112 x 20 x 20].
The FPN and PAN are distinct weights. These two external fusion nodes are ADD, not concatenation. The GAP branch does not create a third output scale.

Container "DEIMTransformer":
F16 and F32 -> "Flatten + Concat" -> "Multi-scale memory" [2000 x 112].
Memory forks into:
1) "Top-200 query selection" [class scores + anchors] -> "Initial queries + refs" [200 x 112; refs 200 x 4]
2) a visual-memory bus to the MSDeformableAttention of ALL THREE decoder layers.
"Initial queries + refs" -> "Decoder layer 1" -> "Decoder layer 2" -> "Decoder layer 3".
Draw the three layers explicitly as three boxes, not a single ambiguous x3 box.
Initial refs -> small "Query position MLP" -> position bus to all three layers.
Query positions are calculated ONCE from initial refs and reused. Annotate "fixed query_pos" if there is room.
A purple input rectangle above decoder says "Water input" and lists "temperature | do | ph | turbidity", [4 values].
Its purple bus feeds a water-fusion input of EACH of the three decoder boxes. It must NOT connect to the backbone or LiteEncoder.
Label decoder width "d = 112" and "3 layers". It is OK to show a compact note "independent water encoders per layer".

PREDICTION AND REFINEMENT:
Below the three decoder boxes draw a pale yellow prediction/refinement strip representing per-layer heads:
"Shared bbox head + FDR" [4 x 33 bins] receives all three layer outputs in sequence.
A small "Pre-box head" receives layer 1 and initial refs; establishes a fixed base box for FDR.
FDR updates sampling refs for layer 2 and layer 3; route arrows as short lower bypasses labeled "refined refs", not as a loop back to the entire network.
"Class heads + LQE" [2 classes] receives decoder hidden states and box distributions. Class heads and LQE are NOT shared. Mark "bbox weights shared" only on bbox head.
Keep these as explicit containers, and use a neat arrow bus if individual head invocations would make the main graph unreadable.
Final layer FDR -> "pred_boxes" [200 x 4].
Final layer Class heads + LQE -> "pred_logits" [200 x 2].
Both outputs -> "PostProcessor" [sigmoid; top-300 pairs] -> "labels / scores / boxes".
300 means query-class pairs from 200 queries x 2 classes; NOT 300 unique queries. No NMS.
Only final-layer output is used for evaluation; earlier-layer score outputs are training auxiliary outputs.

DETAIL INSET 1: "HG_Block"
input x -> Conv 1 -> Conv 2 -> Conv 3 in series.
x and all three conv outputs -> circle "C" -> "1x1 Conv" -> "1x1 Conv" -> out.
Label "agg = 'se': two convolutions".
Show optional residual as a labeled bypass "Stage 3, block 2 only". Its addition must be AFTER the two aggregation convolutions.
Compact note: "Stage 1/2: ConvBNAct k3; Stage 3: LightConvBNAct k5".
ConvBNAct is Conv-BN-ReLU-LAB; LightConv is 1x1 Conv-BN then depthwise 5x5 Conv-BN-ReLU-LAB.
Do not draw squeeze-excitation pooling/sigmoid gates here: local code's 'se' name is just the two 1x1 conv aggregator.

DETAIL INSET 2: "RepNCSPELAN4"
input -> "1x1 Conv" [112 -> 224] -> "Split" [112 + 112].
Two split outputs a,b. b -> "CSPLayer2 + Conv3x3" -> c [19 channels]
c -> "CSPLayer2 + Conv3x3" -> d [19 channels].
a,b,c,d -> "C" -> "1x1 Conv" [262 ->112] -> out.
Note "n = 2; c4 = 19". Each CSPLayer2 contains VGGBlock x2; mention with a small label, no unsupported bottleneck names.
This is a template for both FPN/PAN with separate weights.

DETAIL INSET 3: "TransformerDecoderLayer"
draw vertical exact sequence:
"Query self-attention" [8 heads] -> "Add + RMSNorm"
 -> purple "Water fusion" (linked by annotation to next inset)
 -> blue "MSDeformableAttention" [2 levels; points 4, 2; 8 heads]
 -> "Add + RMSNorm"
 -> "SwiGLUFFN" [112 -> (160,160) ->112]
 -> "Add + RMSNorm".
Add residual bypasses around self-attention, visual cross-attention and SwiGLUFFN.
The water fusion is AFTER query self-attention and BEFORE visual cross-attention.
Memory -> visual cross-attention V; refs -> sampling; fixed query_pos -> self-attention Q/K and visual cross-attention Q.
Footer small "use_gateway = False". This does NOT disable the water branch's learned residual scales.

DETAIL INSET 4: "WaterAwareQueryCrossAttention"
Two input paths, raw Water [4 values] and Query [200 x112].
Within a light lavender dashed sub-container "WaterTokenEncoder":
"Normalize + Clamp" -> "Value MLP" [1 ->16 ->64]
 -> "+" receiving separate "Parameter embedding" [4 x64]
 -> tokens T0 [4 x64].
T0 -> "LN + Water self-attention" [4 heads] -> "Delta MLP" [64 ->64 ->64]
 -> scaled residual "+" also receiving T0 bypass; label scale "interaction_scale".
Output T [4 x64] -> "LN" -> K,V of "Query-Water Cross-Attention" [8 heads].
Query q -> "LN" -> Q of that same attention.
Attention output -> scaled residual "+" also receiving original q bypass; label "attn_scale".
Result q1 -> "Water fusion FFN" [LN;112 ->448 ->112; GELU]
 -> second scaled residual "+" also receiving q1 bypass; label "ffn_scale".
Result q2 is fused query -> layer's visual cross-attention.
All three scales are trainable SCALARS initialized to 0; no sigmoid-gating network.
Annotate "per decoder layer; independent weights" at container bottom.
If space tight use short labels "Water self-attn", "Query-Water Cross-Attn" inside boxes, with full class name as the panel title.

TRAINING FOOTER
Clearly separated thin dashed strip:
"GT labels + boxes" -> "Contrastive denoising" -> annotation "append DN queries (training only)" directed toward query initialization.
"Predictions + GT" -> "DEIMCriterion" [mal / boxes / local; auxiliary supervision].
Keep training arrows dashed. No training losses inserted along inference arrows.
No performance claims, no accuracy metrics, no guessed category names.

VISUAL STYLE
Flat vector-like scientific infographic, not photorealism and not an actual editable vector file.
Pastel blue #B9D5EA for visual transforms, peach #FBE5D5 for HG/Rep, pale green #E4F2D7 for resampling, lavender #D9C6EA for water, pale yellow #FFF2CC for heads, pink #F3CBCB for Add circles, gray #F1F3F5 for tensors.
Dark gray #58636D thin outlines, #25313B text. White background; dashed group boundaries; consistent small corner radii.
Straight and orthogonal arrows with small filled triangle heads. Logical junctions explicit; no ambiguous accidental junctions.
Use legible Arial/Helvetica-like type, generous whitespace, aligned modules. Labels 1-5 words where possible, line-break long class names instead of truncating.
No heavy shadows, gradients, 3D feature stacks, photo inserts, microscopic text, decorative scenes, logos or watermarks.
Required structure takes precedence over copying the style image's arrangement. Make a new composition appropriate to this actual architecture.
Every named main stage must appear. Never invent modules. Infer no numeric parameters beyond those supplied.
This image is a reference design; accompanying diagram_spec.md remains authoritative for all precise internals and connections.

---
## Structural correction pass

Edit this architecture reference diagram to correct scientifically incorrect arrows. Produce one corrected primary figure, preserve title, all module names, pastel flat-vector style, white background, dashed containers and landscape format. Increase size/spacing if needed. The input image is an imperfect generated draft: the rules below are authoritative, do NOT preserve its wrong connections. Arrow topology has priority over matching exact placement.

CRITICAL CORRECTIONS:

1. RGB IMAGE MUST ENTER STEM. Remove the horizontal RGB -> Stage 2 connection. Route RGB image arrow upwards into the LEFT side of StemBlock. Backbone chain is only StemBlock -> Stage 1 -> Stage 2 -> Stage 3 -> Input projection. No other RGB input connection.

2. REWIRE LiteEncoder from scratch, maintaining the existing boxes. Top label on Input projection output is P16. GAP_Fusion output is G32.
P16 -> Downsample 1 -> GAP_Fusion -> G32.
G32 -> Upsample x2 -> FIRST (+).
P16 -> FIRST (+) by its own bypass.
FIRST (+) -> RepNCSPELAN4 (FPN) -> F16.
F16 -> Downsample 2 -> SECOND (+).
G32 -> SECOND (+).
SECOND (+) -> RepNCSPELAN4 (PAN) -> F32.
Remove P16 -> Upsample connection.
Remove direct G32 -> FIRST (+) connection: G32 must pass through Upsample before FIRST (+).
Each (+) has EXACTLY TWO incoming edges and one outgoing edge. The only encoder outputs are F16 and F32.
Use a clean orthogonal route from G32 to Upsample that does not intersect FIRST (+).
Both F16 and F32 must have clear outbound arrows into Flatten+Concat; current F32 shows a disconnected destination stub. Use paired labeled F16/F32 ports if long arrows clutter the graph.

3. REWIRE THE MAIN DECODER:
- Flatten+Concat -> Multi-scale memory.
- Memory -> Top-200 query selection -> Initial queries+refs -> Decoder layer 1. The content initialization MUST enter layer 1, not layer 2.
- Initial refs -> Query position MLP. Its output "fixed query_pos" feeds every layer; it does NOT feed Initial queries backwards.
- A separate memory bus feeds the visual attention input port of layers 1,2,3. Label those inputs V.
- Raw Water input bus feeds water port of all layers (already correct).
- Layer 1 -> Layer 2 -> Layer 3 is the query content chain.
- Pre-box head receives Layer1 output and initial refs. Its output is "fixed base box" into FDR.
- Three decoder outputs feed Shared bbox head+FDR; bbox distribution feeds Class heads+LQE. Class heads also receive hidden states. Final pred_boxes exits FDR, final pred_logits exits Class heads+LQE.
- Refined refs from FDR feed the sampling-reference input of layer2 and layer3 only, no arrow from class heads to reference updates.
- The two head rectangles are repeated per-layer computations, not an extra sequential layer after the decoder. Show small caption "per-layer heads; final outputs from layer 3".
- Keep shared label ONLY on bbox head, independent label on class heads.
- If complete buses would tangle, use clearly named paired connector ports "memory V", "fixed query_pos", "refined refs"; matching port names signify identical tensors. Do NOT draw misleading substitute lines.

4. FIX (a) HG_Block:
Place Conv1, Conv2, Conv3 in a HORIZONTAL SEQUENTIAL CHAIN.
x -> Conv1 -> Conv2 -> Conv3.
Separate tap arrows from x, Conv1, Conv2, Conv3 enter C.
NO x bus feeding all three Conv inputs in parallel.
C -> 1x1 Conv -> 1x1 Conv -> (+) -> out.
Dashed x bypass -> final (+), with "Stage 3, block 2 only".
All HG conv chains are serial with concatenated intermediate outputs.

5. KEEP (b) RepNCSPELAN4:
x -> 1x1 -> Split -> a,b.
b -> CSPLayer2+Conv3x3 -> c -> CSPLayer2+Conv3x3 -> d.
a,b,c,d -> C -> final1x1 -> out. Ensure branches entering C have arrowheads pointing TO C.

6. FIX (c) TransformerDecoderLayer RESIDUALS:
queries -> Query self-attention -> Add+RMSNorm -> Water fusion -> MSDeformableAttention -> Add+RMSNorm -> SwiGLUFFN -> Add+RMSNorm -> output queries.
There are exactly THREE local visual residual skips:
 original queries -> first Add+RMSNorm,
 water fusion output -> second Add+RMSNorm,
 second Add+RMSNorm output -> last Add+RMSNorm.
Remove the misleading long raw-queries -> output-queries bypass.
Memory V enters only MSDeformableAttention (NOT query self-attention).
Refs enter only MSDeformableAttention.
Fixed query_pos enters Query self-attention Q/K AND MSDeformableAttention Q.
Water raw enters only Water fusion.

7. FIX (d) WATER FUSION RESIDUALS:
Raw water -> Normalize+Clamp -> ValueMLP -> token (+); parameter embedding ALSO -> token (+).
Call output T0; route T0 -> LN+Water self-attn -> DeltaMLP -> SCALE interaction_scale -> token residual (+).
T0 bypass -> SAME token residual (+). Output T -> LN -> K,V ports of Query-Water Cross-Attention.
Query q -> LN -> Q port of Query-Water Cross-Attention.
Original query q BEFORE LN bypasses attention to first QUERY residual (+).
Cross-attention output -> SCALE attn_scale -> first QUERY residual (+); output q1.
q1 -> Water fusion FFN -> SCALE ffn_scale -> second QUERY residual (+).
q1 bypass -> SAME second QUERY residual (+).
second QUERY residual (+) -> fused query.
The current draft omits q1 bypass around Water fusion FFN: it MUST be added.
Do not use a bypass from a normalized query in place of original q.
Scalar scale labels can sit on incoming transformed branch arrows, with a tiny multiplication symbol. No sigmoid gates.

8. TRAINING FOOTER:
DN queries training-only dashed arrow must feed Initial queries+refs, not the query position MLP. GT+prediction supervision stays separate. Do not route GT into inference outputs.

Preserve all numeric labels already supplied. No new modules, no missing nodes. No replication of RGB photo. Do not copy the original style-reference architecture. This is a technical arrow-correction edit, not a variant. Return one corrected diagram with large readable labels.

---
## Final compact hierarchical composition

前两版存在复杂连线生成偏差，最终改为宏观层级图 + 两个核心详图；HG/Rep内部详解完整保留在结构说明，水质残差用准确公式表达。

Create ONE NEW clean scientific model architecture reference diagram, using the supplied image ONLY for pastel colors, simple module rectangles, thin arrows and dashed grouping style. Ignore its scientific labels and topology. It is a style reference, not an edit target.

Landscape 3:2, opaque WHITE background, target 3072x2048, flat vector-like rendering for subsequent draw.io reconstruction. Title "DEIMv2 Pico with Water-Aware Queries". Prioritize accurate legible topology and ample spacing over density. This is a hierarchical overview with two detail panels; do NOT draw HG_Block or RepNCSPELAN4 internals.

LAYOUT: top 60% main architecture, left-to-right across three pale dashed containers. Bottom 40% two wide detail panels. Use simple rectangles, orthogonal arrows, small pink "+" circles. Blue visual operators, peach HG/Rep, green resampling, purple water, yellow heads. Black/gray text and thin strokes. Short English labels, no photographs, no 3D, no shadows, no strong gradients, no logos.

MAIN LEFT CONTAINER "HGNetv2 (Pico)":
Vertically arranged: "RGB image / 3 x 640 x 640" -> "StemBlock / 16, s4" -> "Stage 1 / HG_Block x1 / 64, s4" -> "Stage 2 / DWConv s2 + HG_Block x1 / 256, s8" -> "Stage 3 / DWConv s2 + HG_Block x2 / 512, s16".
Input enters StemBlock ONLY. Stage3 ONLY feeds LiteEncoder.

MAIN CENTER CONTAINER "LiteEncoder / width 112":
Use TWO VERTICAL COLUMNS within this container, to avoid crossings.
Left column from top to bottom: "Projection / Conv1x1 + BN" -> "Downsample 1 / AvgPool s2 + Conv1x1" -> "GAP_Fusion / x + GAP(x)" -> "G32 / 112 x20 x20".
Right column from TOP to BOTTOM:
"Upsample x2 / nearest"
-> circle "+" (call it first plus)
-> "RepNCSPELAN4 / FPN"
-> "F16 / 112 x40 x40"
-> "Downsample 2 / AvgPool s2 + Conv1x1"
-> circle "+" (call it second plus)
-> "RepNCSPELAN4 / PAN"
-> "F32 / 112 x20 x20".
EXACT CROSS-COLUMN EDGES:
Projection output has a fork labeled P16 to FIRST PLUS.
G32 goes up through a dedicated edge corridor to the INPUT of Upsample x2.
G32 also goes directly to SECOND PLUS.
There is NO edge Projection -> Upsample.
There is NO edge G32 -> first plus except through Upsample.
Each plus has exactly TWO incoming arrows.
If a long G32 -> Upsample edge would cross any other edge, replace it with a pair of explicitly labeled off-page ports BOTH named "G32": one originating at G32, one entering Upsample. These ports are a wire abstraction, not new modules.
F16 and F32 exit via matching labeled ports to the decoder.
At bottom, small note: "Rep blocks: separate weights, n=2; c4=19".
GAP_Fusion includes 1x1 Conv+BN+SiLU after the sum; put this in a tiny secondary label if room.
Do not introduce any extra plus input.

MAIN RIGHT CONTAINER "DEIMTransformer / d=112":
F16 and F32 paired ports -> "Flatten + Concat" -> "Memory / 2000 x112".
Memory -> "Query initialization / Top-200 + anchor refinement" -> "Decoder layer 1" -> "Decoder layer 2" -> "Decoder layer 3".
The three decoder layer boxes are shown explicitly horizontally. Decoder layers receive query content in sequence.
A purple "Water input / temperature, do, ph, turbidity" above them feeds a purple input arrow into EACH layer.
Each layer has a small paired input port labeled "memory V", denoting visual memory from the Memory box. Use three identical labeled ports to avoid wiring the memory bus through query initialization.
A small note underneath: "fixed query_pos from initial refs; refined sampling refs between layers".
A pale yellow container below the three layers titled "Per-layer heads and refinement". Feed it from each layer by three downward arrows.
Inside this yellow container put THREE separate subrectangles with NO speculative internal arrows:
"Pre-box head / layer 1 only";
"Shared bbox head + FDR / 4 x33 bins";
"Class heads + LQE / 2 classes; independent".
A small exact label within the yellow container: "fixed base box; iterative distribution refinement".
Give this yellow container two precisely labeled output ports: "final boxes" coming from Shared bbox head+FDR and "final logits" coming from Class heads+LQE.
Those lead respectively to "pred_boxes / 200 x4" and "pred_logits / 200 x2".
Both -> "PostProcessor / sigmoid; top-300 query-class pairs" -> "labels / scores / boxes".
Do NOT draw reference-feedback arrows between main head rectangles, since the ports and detail text already describe the refinements. No NMS, no extra detector heads.
Water is injected in decoder ONLY, no water edges into backbone/encoder.

BOTTOM LEFT DETAIL PANEL "TransformerDecoderLayer":
One vertical, unbranched chain of SEVEN rectangles with clear downward arrows:
1 "Query self-attention / 8 heads"
2 "Add + RMSNorm"
3 purple "WaterAwareQueryCrossAttention"
4 blue "MSDeformableAttention / 2 levels; points [4,2]"
5 "Add + RMSNorm"
6 "SwiGLUFFN / 112 -> (160,160) ->112"
7 "Add + RMSNorm".
NO raw-query-to-output bypass arrow. NO memory arrow to self-attention.
Use three concise text annotations beside the chain, not crossing arrows:
"Self-attention: Q,K = query + pos; V = query"
"Visual attention: Q = fused query + pos; V = memory"
"Local residuals around SA, visual CA and FFN".
Footer "use_gateway=False". This is the exact operation order.

BOTTOM RIGHT DETAIL PANEL "WaterAwareQueryCrossAttention":
Use a clean two-column hierarchical description with short operation chains and equations rather than a tangled bypass graph.
Left column dashed subcontainer title "WaterTokenEncoder (per layer)".
Vertical chain: "4 raw values" -> "Normalize + Clamp" -> "Value MLP / 1 ->16 ->64" -> "Add parameter embedding" -> "T0 / 4 x64" -> "LN + Water self-attention / 4 heads" -> "Delta MLP / 64 ->64 ->64".
Below chain display readable equation "T = T0 + interaction_scale * Delta".
Do not draw an unsupported arrow that changes this formula.
Right column:
"Query q / 200 x112" -> "Query-Water Cross-Attention / Q=LN(q); K,V=LN(T) / 8 heads".
T from the left column feeds the K,V port of that attention.
Then vertical steps:
"q1 = q + attn_scale * Attention"
-> "Water fusion FFN / LN;112 ->448 ->112; GELU"
-> "q2 = q1 + ffn_scale * FFN(q1)".
Bottom note "All scales learnable, initialized to 0; independent encoder in each decoder layer".
The residual equations replace all bypass arrows in this panel. Do NOT add extra bypass arrows or sigmoid gates.

FOOTNOTE under main/bottom:
"Training only: contrastive denoising at query initialization; DEIMCriterion (mal / boxes / local) with auxiliary supervision."
This is plain text in a narrow gray dashed box, not a new inference stage; no arrows needed.
Tiny legend " +  elementwise addition     Concat  sequence/channel concatenation as labeled".

Do not copy any YOLO scientific content from the provided style image. Do not draw the old example's insets. Do not invent model modules. This deliberately compact hierarchy preserves named modules while full nested connections live in accompanying diagram_spec.md. Ensure all text clear and all stated main edges exact.

---
## Final arrow corrections

Make ONLY THREE localized arrow corrections to this image. Keep every box, label, title, color, size, layout and all other connections unchanged. This is a surgical technical edit; do not redesign or reflow the figure.

CORRECTION 1 — Backbone output:
Currently Stage 3 at lower left points RIGHT into G32. That is wrong.
ERASE the entire arrow from Stage 3 to G32.
Draw a new orthogonal arrow from Stage 3's RIGHT edge into Projection's LEFT edge (the Projection box is at the TOP of the LiteEncoder panel).
Route it upward along the empty gutter immediately inside the LEFT edge of the LiteEncoder container, then turn right into Projection.
Label this wire "512 / s16" in the gutter.
There MUST be NO wire from Stage 3 to G32.
G32 must receive ONLY the downward edge from GAP_Fusion.
Projection MUST receive the Stage3 arrow, and retain its two existing outgoing branches.

CORRECTION 2 — Three water and memory ports:
Currently each purple Water input line visually enters a small BLUE box "memory V", causing it to look as though water becomes visual memory. That is wrong.
For EACH of Decoder layer 1,2,3:
Keep its small blue "memory V" port, but move that small blue port horizontally slightly to the LEFT of the purple vertical water arrow.
Connect that blue "memory V" port with a BLACK short arrow to the TOP-LEFT input of the associated decoder-layer rectangle.
Connect the purple water line DIRECTLY from Water input to the TOP-RIGHT input of that decoder-layer rectangle.
Purple water lines must never touch or pass through the blue "memory V" ports.
Use separate spatially distinct arrowheads for purple WATER and black MEMORY.
No water arrow may enter a memory port.

CORRECTION 3 — Water token output T:
In the bottom-center WaterTokenEncoder subpanel, a line currently exits the RIGHT edge of the peach "T0 / 4x64" box and is labeled T before entering Query-Water Cross-Attention. That is wrong.
ERASE the full T0-to-attention side arrow.
KEEP T0's downward arrow to LN + Water self-attention, and keep its upstream arrow from Add parameter embedding.
At the BOTTOM of the WaterTokenEncoder subpanel, the equation "T = T0 + interaction_scale * Delta" defines the FINAL token output.
Add a small output terminal labeled "T / 4x64" beside or directly below that EQUATION, inside the panel.
Route a new line FROM THIS FINAL T terminal upwards through the gutter between the two subpanels, then RIGHT into the LEFT edge (K,V input) of the existing Query-Water Cross-Attention box.
Label the arrow "T". It must originate at the FINAL T equation/terminal, NOT at T0, not at raw values, not at the Delta MLP box by itself.
The attention box must keep "Q=LN(q); K,V=LN(T)" exactly as it currently says.

All other content stays untouched. Produce one image. Flat crisp legible scientific diagram. No new scientific modules.

---
## Restore separate initialization and first decoder layer

Surgical edit of supplied diagram: preserve EVERYTHING except the following two corrections. No redesign, no changes to other boxes.

1. Restore missing "Decoder layer 1" as a DISTINCT box after query initialization.
The blue box currently labeled "Query initialization / Top-200 + anchor refinement" sits immediately left of Decoder layer 2 and receives purple water and black memory V. THAT PARTICULAR BOX MUST BE RELABELED EXACTLY "Decoder layer 1" and remain in place with its current water and memory V inputs.
Then ADD a separate small blue rectangle ABOVE the Memory box, within the currently empty upper-left space of the DEIMTransformer container (left of the Water input box).
Label the new rectangle "Query initialization" on line1 and "Top-200 + anchor refinement" on line2.
Keep "Memory / 2000x112" in place.
ERASE only the old horizontal Memory -> newly relabeled Decoder layer1 arrow.
Draw Memory -> NEW Query initialization -> Decoder layer1 with orthogonal arrows using that upper-left empty space.
Preserve Decoder layer1 -> Decoder layer2 -> Decoder layer3 arrows.
Do NOT remove or rename Decoder layer2 or layer3.
Result: THREE clearly named layer boxes, exactly Decoder layer1, Decoder layer2, Decoder layer3, PLUS a FOURTH separate Query initialization box.
Water input goes into the THREE LAYERS ONLY, not Query initialization.
Black memory V paired ports enter the THREE LAYERS ONLY, not Query initialization.

2. At the bottom WaterTokenEncoder, final token box "T / 4x64" is the SOURCE of the line going right and up into Query-Water Cross-Attention.
Remove the arrowhead pointing LEFT into this T box.
Keep only the arrowhead pointing RIGHT INTO the Query-Water Cross-Attention K,V input.
This line must be single-direction T -> Cross-Attention. Keep both endpoint boxes and the connecting line in place.

Everything else pixel-layout consistent. One corrected image, not alternatives. Do not remove any existing unrelated box or label.
