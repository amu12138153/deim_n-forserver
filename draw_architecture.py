"""Draw the source-verified visdrone_pico.yml architecture.

Run with a Python environment containing matplotlib:
    python draw_architecture.py

Outputs next to this script: DEIMv2_Pico_architecture.png and .pdf.
The structural evidence is documented in architecture_analysis.md.
"""

from pathlib import Path
import subprocess
import sys

try:
    import matplotlib as mpl
except ModuleNotFoundError as exc:
    if exc.name != "matplotlib":
        raise
    # The repository's conda environment supplies matplotlib when base does not.
    env_python = Path(sys.prefix) / "envs" / "deimv2" / ("python.exe" if sys.platform == "win32" else "bin/python")
    if not env_python.is_file() or env_python.resolve() == Path(sys.executable).resolve():
        raise
    raise SystemExit(subprocess.call([str(env_python), __file__, *sys.argv[1:]]))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
INK = "#1C2530"
MUTED = "#52616F"
BLUE = "#E7F0F8"
GREEN = "#E8F3EA"
ORANGE = "#FAEDDE"
PURPLE = "#F1EAF6"
PINK = "#F8EAF0"
NEUTRAL = "#F5F7F8"
WHITE = "#FFFFFF"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 9,
        "savefig.facecolor": "white",
    }
)


def panel(ax, x, y, w, h, color, title, subtitle=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.25,rounding_size=1.3",
            linewidth=1.05, edgecolor=INK, facecolor=color, zorder=1,
        )
    )
    ax.text(x + 2, y + h - 3.4, title, color=INK, fontsize=11, weight="bold",
            ha="left", va="top", zorder=3)
    if subtitle:
        ax.text(x + 2, y + h - 7.0, subtitle, color=MUTED, fontsize=7.7,
                ha="left", va="top", zorder=3)


def node(ax, x, y, w, h, text, fill=WHITE, size=8.7, weight="normal", edge=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.13,rounding_size=0.8",
            linewidth=0.85, edgecolor=edge, facecolor=fill, zorder=3,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, color=INK, fontsize=size,
            weight=weight, ha="center", va="center", linespacing=1.12, zorder=4)


def route(ax, points, label=None, label_xy=None, lw=1.0, color=INK, head=7):
    xs, ys = zip(*points)
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=2,
            solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=head, linewidth=lw,
                                 color=color, zorder=5))
    if label and label_xy:
        ax.text(*label_xy, label, color=MUTED, fontsize=7.1,
                ha="center", va="center", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))


def detail_frame(ax, x, y, w, h, title, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.18,rounding_size=1.1",
            linewidth=0.9, linestyle=(0, (4, 3)),
            edgecolor=MUTED, facecolor=color, zorder=1,
        )
    )
    ax.text(x + 2, y + h - 2.9, title, fontsize=9.4, color=INK,
            weight="bold", va="top", zorder=3)


fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=300)
fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975)
ax.set_xlim(0, 192)
ax.set_ylim(0, 108)
ax.set_aspect("auto")
ax.axis("off")

ax.text(3, 105, "DEIMv2-Pico  |  visdrone_pico.yml", color=INK,
        fontsize=17, weight="bold", ha="left", va="top")
ax.text(3, 100.8, "Source-verified data flow  ·  640 × 640 image  ·  water fusion at HGNetv2 Stage 3",
        color=MUTED, fontsize=9.5, ha="left", va="top")

# Main image path and HGNetv2 stages.
node(ax, 2, 67.3, 12.5, 9.0, "Input image\n3 × 640 × 640", NEUTRAL, 9.2, "bold")
panel(ax, 18, 45.5, 51.5, 44, BLUE, "HGNetv2-Pico Backbone")
node(ax, 21, 77, 21, 6.7, "StemBlock  ·  s4", WHITE, 8.6)
node(ax, 21, 69, 21, 6.7, "Stage 1  ·  HG_Block ×1", WHITE, 8.4)
node(ax, 21, 61, 21, 6.7, "Stage 2  ·  HG_Block ×1", WHITE, 8.4)
node(ax, 21, 53, 21, 6.7, "Stage 3  ·  HG_Block ×2", WHITE, 8.4)
node(ax, 47, 53, 19, 6.7, "WaterCrossAttention", PURPLE, 8.1, "bold")
route(ax, [(14.5, 71.8), (16.2, 71.8), (16.2, 80.35), (21, 80.35)])
route(ax, [(31.5, 77), (31.5, 75.7)])
route(ax, [(31.5, 69), (31.5, 67.7)])
route(ax, [(31.5, 61), (31.5, 59.7)])
route(ax, [(42, 56.35), (47, 56.35)])
ax.text(45, 50.5, "C3: 512 × 40 × 40  ·  stride 16", fontsize=7.7,
        color=MUTED, ha="center")

# Water enters the stage-3 feature, not the object query.
node(ax, 45, 92.1, 16.4, 6.5, "Water parameters\nT · DO · pH", PURPLE, 7.7, "bold")
node(ax, 64, 92.1, 14.0, 6.5, "WaterTokenEncoder\n3 × 64 tokens", PURPLE, 7.7, "bold")
route(ax, [(61.4, 95.35), (64, 95.35)])
route(ax, [(71, 92.1), (71, 86.0), (57, 86.0), (57, 59.7)],
      label="water K/V", label_xy=(64, 86.9), lw=0.95)

# Two-scale LiteEncoder. All cross-scale fusion points use addition.
panel(ax, 74.5, 45.5, 50.5, 44, GREEN, "LiteEncoder", "One input scale → two output scales")
node(ax, 77.5, 72, 12.2, 7.2, "input_proj\n512 → 112", WHITE, 8.2, "bold")
node(ax, 77.5, 53.5, 12.2, 8, "down_sample1\n+ GAP_Fusion", WHITE, 7.8)
node(ax, 94.5, 72, 12.2, 7.2, "add + fpn_block\nRepNCSPELAN4", WHITE, 7.7, "bold")
node(ax, 94.5, 53.5, 12.2, 8, "add + pan_block\nRepNCSPELAN4", WHITE, 7.7, "bold")
node(ax, 111.0, 72, 10.5, 7.2, "P16\n112 × 40 × 40", WHITE, 7.7, "bold")
node(ax, 111.0, 53.5, 10.5, 8, "P32\n112 × 20 × 20", WHITE, 7.7, "bold")
route(ax, [(66, 56.35), (72.0, 56.35), (72.0, 75.6), (77.5, 75.6)],
      label="C3", label_xy=(72, 65.7))
route(ax, [(89.7, 75.6), (94.5, 75.6)])
route(ax, [(83.6, 72), (83.6, 61.5)])
route(ax, [(89.7, 57.5), (91.8, 57.5), (91.8, 73.4), (94.5, 73.4)],
      label="upsample ×2", label_xy=(91.4, 65.5))
route(ax, [(106.7, 75.6), (111, 75.6)])
route(ax, [(100.6, 72), (100.6, 61.5)],
      label="down_sample2", label_xy=(108.7, 66.4))
route(ax, [(89.7, 57.5), (94.5, 57.5)])
route(ax, [(106.7, 57.5), (111, 57.5)])

# Both feature scales feed memory. Selected memory yields query content.
panel(ax, 129, 45.5, 44.0, 44, ORANGE, "DEIMTransformer", "Two-level memory · 200 queries · 3 layers")
node(ax, 132, 70.5, 12.0, 8.3, "Flatten + concat\nMemory: 2000 × 112", WHITE, 7.9, "bold")
node(ax, 148, 70.5, 10.5, 8.3, "Anchors + top 200\nQuery init", WHITE, 7.8, "bold")
node(ax, 162, 70.5, 8.5, 8.3, "Decoder\nLayer ×3", WHITE, 8.1, "bold")
route(ax, [(121.5, 75.6), (126.2, 75.6), (126.2, 76.1), (132, 76.1)])
route(ax, [(121.5, 57.5), (126.2, 57.5), (126.2, 73.3), (132, 73.3)])
route(ax, [(144, 74.65), (148, 74.65)])
route(ax, [(158.5, 74.65), (162, 74.65)])
route(ax, [(138, 70.5), (138, 62.1), (166.25, 62.1), (166.25, 70.5)],
      label="memory / value", label_xy=(151.4, 62.1))
ax.text(151.2, 54.4, "Initial references: top-200 anchors + box offsets",
        fontsize=7.5, color=MUTED, ha="center")

# Parallel prediction branches and final output.
ax.text(181.5, 87, "Detection output", fontsize=11, weight="bold",
        color=INK, ha="center")
node(ax, 177.4, 72, 12.8, 7.2, "Classification\n200 × 2 logits", PINK, 8.3, "bold")
node(ax, 177.4, 58.5, 12.8, 7.2, "BBox regression\n200 × 4 boxes", PINK, 8.3, "bold")
route(ax, [(170.5, 74.65), (174.6, 74.65), (174.6, 75.6), (177.4, 75.6)])
route(ax, [(170.5, 74.65), (175.0, 74.65), (175.0, 62.1), (177.4, 62.1)])
ax.text(182, 52.5, "Training: intermediate, encoder and\ndenoising auxiliary predictions",
        color=MUTED, fontsize=7.8, ha="center", va="center")

# Separated repeated-module details.
detail_frame(ax, 3, 6, 58, 33.5, "A  |  HG_Block  ×4 total", BLUE)
node(ax, 6, 19, 16, 8, "ConvBNAct or\nLightConvBNAct ×3", WHITE, 8.0)
node(ax, 26, 19, 13, 8, "concat\nintermediates", WHITE, 8.0)
node(ax, 43, 19, 14.5, 8, "SE aggregation\n1×1 projections", WHITE, 8.0)
route(ax, [(22, 23), (26, 23)])
route(ax, [(39, 23), (43, 23)])
ax.text(32, 11.7, "Stage 3 uses LightConvBNAct; its second block has a residual add.",
        fontsize=7.5, color=MUTED, ha="center")

detail_frame(ax, 65, 6, 58, 33.5, "B  |  RepNCSPELAN4  ×2", GREEN)
node(ax, 68, 19, 12.5, 8, "cv1 + split", WHITE, 8.1)
node(ax, 83.5, 19, 14, 8, "CSPLayer2\nVGGBlock ×2", WHITE, 8.0)
node(ax, 100.5, 19, 18.7, 8, "CSPLayer2 → concat\n+ cv4", WHITE, 7.9)
route(ax, [(80.5, 23), (83.5, 23)])
route(ax, [(97.5, 23), (100.5, 23)])
ax.text(94, 11.7, "FPN and PAN use separate copies; scale fusion is add.",
        fontsize=7.7, color=MUTED, ha="center")

detail_frame(ax, 127, 6, 63, 33.5, "C  |  TransformerDecoderLayer  ×3", ORANGE)
node(ax, 130, 19, 12.2, 8, "Self-Attn\n+ RMSNorm", WHITE, 7.8)
node(ax, 145, 19, 14.5, 8, "MSDeformable\nCross-Attn", WHITE, 7.8)
node(ax, 162, 19, 12.5, 8, "SwiGLUFFN\n+ RMSNorm", WHITE, 7.8)
node(ax, 177, 19, 10.5, 8, "Score/LQE\nBBox refine", WHITE, 7.6)
route(ax, [(142.2, 23), (145, 23)])
route(ax, [(159.5, 23), (162, 23)])
route(ax, [(174.5, 23), (177, 23)])
ax.text(158.5, 11.7, "Reference boxes update each layer; bbox MLP is shared.",
        fontsize=7.5, color=MUTED, ha="center")

ax.text(96, 2.5,
        "Solid arrows: executed forward connections  |  Water branch runs when water is supplied  |  Decoder water-query fusion is disabled",
        color=MUTED, fontsize=7.8, ha="center")

png = OUT / "DEIMv2_Pico_architecture.png"
pdf = OUT / "DEIMv2_Pico_architecture.pdf"
fig.savefig(png, dpi=300, facecolor="white", edgecolor="none")
fig.savefig(pdf, facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved {png} and {pdf}")
