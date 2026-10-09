import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import to_hex
from adjustText import adjust_text

import xml.etree.ElementTree as ET
from datetime import datetime


# ============================================================
# 1. 只需要修改这里
# ============================================================

CSV_PATH = r"data.csv"

OUTPUT_NAME = "model_compare_map5095"

# 高亮模型
HIGHLIGHT_MODEL = "Ours"


# ============================================================
# 2. Matplotlib 全局设置
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"

# PDF保留字体
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# SVG中保留文字，方便后期编辑
plt.rcParams["svg.fonttype"] = "none"

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 3. 图像基本设置
# ============================================================

FIG_W = 12
FIG_H = 6.5

DPI = 600


# 主图气泡面积范围
MIN_BUBBLE_SIZE = 250
MAX_BUBBLE_SIZE = 1500


# Matplotlib图例统一小圆大小
LEGEND_MARKER_SIZE = 110


# ============================================================
# 4. 读取 CSV
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"找不到 CSV 文件：{CSV_PATH}"
    )


df = pd.read_csv(CSV_PATH)


required_columns = [
    "Model",
    "FLOPs",
    "Parameters",
    "mAP50_95"
]


for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"CSV 缺少必要列：{col}"
        )


# ============================================================
# 5. 数值转换
# ============================================================

df["FLOPs"] = pd.to_numeric(
    df["FLOPs"],
    errors="coerce"
)

df["Parameters"] = pd.to_numeric(
    df["Parameters"],
    errors="coerce"
)

df["mAP50_95"] = pd.to_numeric(
    df["mAP50_95"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "Model",
        "FLOPs",
        "Parameters",
        "mAP50_95"
    ]
).reset_index(drop=True)


# ============================================================
# 6. mAP 自动转百分数
#
# 0.706418 -> 70.6418
# ============================================================

if df["mAP50_95"].max() <= 1.0:

    df["mAP_plot"] = (
        df["mAP50_95"] * 100.0
    )

else:

    df["mAP_plot"] = (
        df["mAP50_95"]
    )


# ============================================================
# 7. 参数量 -> 气泡大小
#
# 使用 sqrt 压缩参数量差距
# 防止20M模型的气泡特别夸张
# ============================================================

pmin = df["Parameters"].min()
pmax = df["Parameters"].max()


def parameter_normalized(value):

    if np.isclose(
        pmin,
        pmax
    ):

        return 0.5


    return (
        np.sqrt(value)
        -
        np.sqrt(pmin)
    ) / (
        np.sqrt(pmax)
        -
        np.sqrt(pmin)
    )


def parameter_to_size(value):

    """
    Matplotlib scatter 的 s 表示面积
    """

    norm = parameter_normalized(
        value
    )

    return (
        MIN_BUBBLE_SIZE
        +
        norm
        *
        (
            MAX_BUBBLE_SIZE
            -
            MIN_BUBBLE_SIZE
        )
    )


def parameter_to_drawio_diameter(value):

    """
    draw.io 主图气泡直径
    """

    norm = parameter_normalized(
        value
    )

    MIN_D = 32
    MAX_D = 85

    return (
        MIN_D
        +
        norm
        *
        (
            MAX_D
            -
            MIN_D
        )
    )


df["bubble_size"] = (
    df["Parameters"].apply(
        parameter_to_size
    )
)


# ============================================================
# 8. 颜色
# ============================================================

cmap = plt.get_cmap("Set2")


colors = [

    to_hex(
        cmap(
            i % cmap.N
        )
    )

    for i in range(
        len(df)
    )

]


# ============================================================
# 9. 创建 Matplotlib 图
# ============================================================

fig, ax = plt.subplots(
    figsize=(
        FIG_W,
        FIG_H
    )
)


# 右边专门给 Parameters 图例留空间
fig.subplots_adjust(
    left=0.10,
    right=0.72,
    bottom=0.14,
    top=0.95
)


# ============================================================
# 10. 坐标范围
# ============================================================

x_min_data = (
    df["FLOPs"].min()
)

x_max_data = (
    df["FLOPs"].max()
)

y_min_data = (
    df["mAP_plot"].min()
)

y_max_data = (
    df["mAP_plot"].max()
)


x_margin = max(
    (
        x_max_data
        -
        x_min_data
    ) * 0.10,

    1.0
)


y_margin = max(
    (
        y_max_data
        -
        y_min_data
    ) * 0.20,

    1.0
)


X_MIN = max(
    0,
    x_min_data - x_margin
)

X_MAX = (
    x_max_data
    +
    x_margin
)

Y_MIN = (
    y_min_data
    -
    y_margin
)

Y_MAX = (
    y_max_data
    +
    y_margin
)


ax.set_xlim(
    X_MIN,
    X_MAX
)

ax.set_ylim(
    Y_MIN,
    Y_MAX
)


# ============================================================
# 11. 绘制普通模型
# ============================================================

for i, row in df.iterrows():

    model = str(
        row["Model"]
    )

    if model == HIGHLIGHT_MODEL:
        continue


    ax.scatter(

        row["FLOPs"],

        row["mAP_plot"],

        s=row["bubble_size"],

        color=colors[i],

        alpha=0.72,

        edgecolors="white",

        linewidths=1.2,

        zorder=3

    )


# ============================================================
# 12. 绘制 Ours
# ============================================================

for i, row in df.iterrows():

    model = str(
        row["Model"]
    )


    if model != HIGHLIGHT_MODEL:
        continue


    ax.scatter(

        row["FLOPs"],

        row["mAP_plot"],

        s=row["bubble_size"],

        color="#E64B35",

        alpha=0.90,

        edgecolors="black",

        linewidths=2.0,

        zorder=5

    )


# ============================================================
# 13. 添加模型名称
# ============================================================

texts = []


for i, row in df.iterrows():

    model = str(
        row["Model"]
    )

    x = float(
        row["FLOPs"]
    )

    y = float(
        row["mAP_plot"]
    )


    if model == HIGHLIGHT_MODEL:

        txt = ax.text(

            x,
            y,

            model,

            fontsize=11.5,

            fontweight="bold",

            color="#B22222",

            ha="center",
            va="center",

            zorder=10

        )

    else:

        txt = ax.text(

            x,
            y,

            model,

            fontsize=10.5,

            color="black",

            ha="center",
            va="center",

            zorder=10

        )


    texts.append(
        txt
    )


# ============================================================
# 14. adjustText 自动避免模型名称重叠
# ============================================================

adjust_text(

    texts,

    x=df["FLOPs"].values,

    y=df["mAP_plot"].values,

    ax=ax,


    # 标签之间留空间
    expand=(
        1.35,
        1.60
    ),


    # 标签之间排斥
    force_text=(
        0.9,
        1.3
    ),


    # 标签避开数据点
    force_static=(
        0.7,
        1.0
    ),


    # 防止文字离对应点太远
    force_pull=(
        0.02,
        0.03
    ),


    # 每次允许移动距离
    max_move=(
        30,
        30
    ),


    # 保证标签不会跑出坐标轴
    ensure_inside_axes=True,


    # 尽量避免连线交叉
    prevent_crossings=True,


    # 自动指示线
    arrowprops=dict(

        arrowstyle="-",

        color="#777777",

        lw=0.7,

        alpha=0.75

    )

)


# ============================================================
# 15. 保存 adjustText 调整后的标签坐标
#
# 后面 draw.io 使用同样的位置
# ============================================================

label_positions = {}


for txt, (_, row) in zip(
    texts,
    df.iterrows()
):

    model = str(
        row["Model"]
    )

    label_positions[
        model
    ] = (
        txt.get_position()
    )


# ============================================================
# 16. 坐标轴
# ============================================================

ax.set_xlabel(

    "FLOPs (G)",

    fontsize=15,

    fontweight="bold"

)


ax.set_ylabel(

    r"mAP$_{50:95}$ (%)",

    fontsize=15,

    fontweight="bold"

)


# ============================================================
# 17. 坐标刻度一位小数
# ============================================================

ax.xaxis.set_major_formatter(
    FormatStrFormatter(
        "%.1f"
    )
)


ax.yaxis.set_major_formatter(
    FormatStrFormatter(
        "%.1f"
    )
)


ax.tick_params(

    axis="both",

    which="major",

    labelsize=12,

    direction="in",

    length=5,

    width=1

)


# ============================================================
# 18. 坐标轴边框
# ============================================================

ax.spines[
    "top"
].set_visible(
    False
)

ax.spines[
    "right"
].set_visible(
    False
)

ax.spines[
    "left"
].set_linewidth(
    1.2
)

ax.spines[
    "bottom"
].set_linewidth(
    1.2
)


# ============================================================
# 19. 网格
# ============================================================

ax.grid(

    True,

    linestyle="--",

    linewidth=0.6,

    alpha=0.20,

    zorder=0

)


# ============================================================
# 20. 参数量图例
#
# 新方案：
#
# 所有气泡大小完全相同
# 保留每个模型颜色
# 写：
#
# 模型名称 + 参数量
# ============================================================

legend_handles = []
legend_labels = []


for i, row in df.iterrows():

    model = str(
        row["Model"]
    )

    param = float(
        row["Parameters"]
    )


    # --------------------------------------------------------
    # Ours
    # --------------------------------------------------------

    if model == HIGHLIGHT_MODEL:

        handle = ax.scatter(

            [],
            [],

            s=LEGEND_MARKER_SIZE,

            color="#E64B35",

            alpha=0.90,

            edgecolors="black",

            linewidths=1.2

        )


    # --------------------------------------------------------
    # 其他模型
    # --------------------------------------------------------

    else:

        handle = ax.scatter(

            [],
            [],

            s=LEGEND_MARKER_SIZE,

            color=colors[i],

            alpha=0.72,

            edgecolors="white",

            linewidths=1.0

        )


    legend_handles.append(
        handle
    )


    # 参数量只显示一位小数
    legend_labels.append(

        f"{model}    {param:.1f} M"

    )


legend = ax.legend(

    legend_handles,

    legend_labels,

    title="Parameters",

    loc="upper left",

    bbox_to_anchor=(
        1.02,
        1.00
    ),

    fontsize=9.5,

    title_fontsize=11,

    frameon=True,

    borderpad=0.9,

    labelspacing=0.75,

    handletextpad=0.8,

    scatterpoints=1

)


legend.get_frame().set_alpha(
    0.95
)

legend.get_frame().set_linewidth(
    0.8
)


# ============================================================
# 21. 保存 PNG / PDF / SVG
# ============================================================

PNG_FILE = (
    OUTPUT_NAME
    +
    ".png"
)

PDF_FILE = (
    OUTPUT_NAME
    +
    ".pdf"
)

SVG_FILE = (
    OUTPUT_NAME
    +
    ".svg"
)


plt.savefig(

    PNG_FILE,

    dpi=DPI,

    bbox_inches="tight"

)


plt.savefig(

    PDF_FILE,

    bbox_inches="tight"

)


plt.savefig(

    SVG_FILE,

    bbox_inches="tight"

)


# ============================================================
# ============================================================
#
#                   DRAW.IO 部分
#
# ============================================================
# ============================================================


DRAWIO_FILE = (
    OUTPUT_NAME
    +
    ".drawio"
)


# ============================================================
# 22. draw.io画布
# ============================================================

CANVAS_W = 1500
CANVAS_H = 760


LEFT = 110
RIGHT = 1030

TOP = 70
BOTTOM = 630


PLOT_W = (
    RIGHT
    -
    LEFT
)

PLOT_H = (
    BOTTOM
    -
    TOP
)


# ============================================================
# 23. 数据坐标 -> draw.io坐标
# ============================================================

def data_to_drawio(
        x,
        y
):

    px = (

        LEFT

        +

        (
            x
            -
            X_MIN
        )

        /

        (
            X_MAX
            -
            X_MIN
        )

        *

        PLOT_W

    )


    # draw.io坐标Y向下增大
    py = (

        BOTTOM

        -

        (
            y
            -
            Y_MIN
        )

        /

        (
            Y_MAX
            -
            Y_MIN
        )

        *

        PLOT_H

    )


    return (
        px,
        py
    )


# ============================================================
# 24. XML初始化
# ============================================================

mxfile = ET.Element(

    "mxfile",

    {

        "host":
            "app.diagrams.net",

        "modified":
            datetime.now().isoformat(),

        "agent":
            "Python Bubble Chart Generator",

        "version":
            "24.7.17"

    }

)


diagram = ET.SubElement(

    mxfile,

    "diagram",

    {

        "id":
            "model-comparison",

        "name":
            "Model Comparison"

    }

)


graph_model = ET.SubElement(

    diagram,

    "mxGraphModel",

    {

        "dx":
            str(CANVAS_W),

        "dy":
            str(CANVAS_H),

        "grid":
            "1",

        "gridSize":
            "10",

        "guides":
            "1",

        "tooltips":
            "1",

        "connect":
            "1",

        "arrows":
            "1",

        "fold":
            "1",

        "page":
            "1",

        "pageScale":
            "1",

        "pageWidth":
            str(CANVAS_W),

        "pageHeight":
            str(CANVAS_H),

        "math":
            "0",

        "shadow":
            "0"

    }

)


root = ET.SubElement(
    graph_model,
    "root"
)


ET.SubElement(

    root,

    "mxCell",

    {
        "id": "0"
    }

)


ET.SubElement(

    root,

    "mxCell",

    {
        "id": "1",
        "parent": "0"
    }

)


cell_id = 2


# ============================================================
# 25. 新建ID
# ============================================================

def new_id():

    global cell_id

    result = str(
        cell_id
    )

    cell_id += 1

    return result


# ============================================================
# 26. 添加文字
# ============================================================

def add_text(

        text,

        center_x,
        center_y,

        width,
        height,

        font_size=14,

        bold=False,

        color="#000000",

        rotation=0

):

    cid = new_id()


    style = (

        "text;"

        "html=1;"

        "strokeColor=none;"

        "fillColor=none;"

        "align=center;"

        "verticalAlign=middle;"

        "whiteSpace=wrap;"

        "overflow=visible;"

        "resizable=1;"

        "fontFamily=Times New Roman;"

        f"fontSize={font_size};"

        f"fontColor={color};"

    )


    if bold:

        style += (
            "fontStyle=1;"
        )


    if rotation != 0:

        style += (
            f"rotation={rotation};"
        )


    cell = ET.SubElement(

        root,

        "mxCell",

        {

            "id":
                cid,

            "value":
                str(text),

            "style":
                style,

            "vertex":
                "1",

            "parent":
                "1"

        }

    )


    ET.SubElement(

        cell,

        "mxGeometry",

        {

            "x":
                str(
                    center_x
                    -
                    width / 2
                ),

            "y":
                str(
                    center_y
                    -
                    height / 2
                ),

            "width":
                str(width),

            "height":
                str(height),

            "as":
                "geometry"

        }

    )


    return cid


# ============================================================
# 27. 添加气泡
# ============================================================

def add_bubble(

        center_x,
        center_y,

        diameter,

        fill_color,

        stroke_color,

        stroke_width=1.5

):

    cid = new_id()


    style = (

        "ellipse;"

        "whiteSpace=wrap;"

        "html=1;"

        "aspect=fixed;"

        f"fillColor={fill_color};"

        f"strokeColor={stroke_color};"

        f"strokeWidth={stroke_width};"

        "opacity=75;"

    )


    cell = ET.SubElement(

        root,

        "mxCell",

        {

            "id":
                cid,

            "value":
                "",

            "style":
                style,

            "vertex":
                "1",

            "parent":
                "1"

        }

    )


    ET.SubElement(

        cell,

        "mxGeometry",

        {

            "x":
                str(
                    center_x
                    -
                    diameter / 2
                ),

            "y":
                str(
                    center_y
                    -
                    diameter / 2
                ),

            "width":
                str(diameter),

            "height":
                str(diameter),

            "as":
                "geometry"

        }

    )


    return cid


# ============================================================
# 28. 添加直线
# ============================================================

def add_line(

        x1,
        y1,

        x2,
        y2,

        color="#000000",

        width=1,

        dashed=False

):

    cid = new_id()


    style = (

        "edgeStyle=none;"

        "html=1;"

        "rounded=0;"

        "endArrow=none;"

        "startArrow=none;"

        f"strokeColor={color};"

        f"strokeWidth={width};"

    )


    if dashed:

        style += (
            "dashed=1;"
        )


    cell = ET.SubElement(

        root,

        "mxCell",

        {

            "id":
                cid,

            "style":
                style,

            "edge":
                "1",

            "parent":
                "1"

        }

    )


    geometry = ET.SubElement(

        cell,

        "mxGeometry",

        {

            "relative":
                "1",

            "as":
                "geometry"

        }

    )


    ET.SubElement(

        geometry,

        "mxPoint",

        {

            "x":
                str(x1),

            "y":
                str(y1),

            "as":
                "sourcePoint"

        }

    )


    ET.SubElement(

        geometry,

        "mxPoint",

        {

            "x":
                str(x2),

            "y":
                str(y2),

            "as":
                "targetPoint"

        }

    )


    return cid


# ============================================================
# 29. 添加标签与气泡之间连接线
# ============================================================

def add_connector(
        source_id,
        target_id
):

    cid = new_id()


    style = (

        "edgeStyle=orthogonalEdgeStyle;"

        "rounded=0;"

        "orthogonalLoop=1;"

        "jettySize=auto;"

        "html=1;"

        "endArrow=none;"

        "startArrow=none;"

        "strokeColor=#888888;"

        "strokeWidth=1;"

    )


    cell = ET.SubElement(

        root,

        "mxCell",

        {

            "id":
                cid,

            "style":
                style,

            "edge":
                "1",

            "parent":
                "1",

            "source":
                source_id,

            "target":
                target_id

        }

    )


    ET.SubElement(

        cell,

        "mxGeometry",

        {

            "relative":
                "1",

            "as":
                "geometry"

        }

    )


# ============================================================
# 30. draw.io 坐标轴
# ============================================================

add_line(

    LEFT,
    BOTTOM,

    RIGHT,
    BOTTOM,

    width=2

)


add_line(

    LEFT,
    TOP,

    LEFT,
    BOTTOM,

    width=2

)


# ============================================================
# 31. 获取 Matplotlib 实际刻度
# ============================================================

fig.canvas.draw()


x_ticks = (
    ax.get_xticks()
)

y_ticks = (
    ax.get_yticks()
)


# ============================================================
# 32. draw.io X轴
# ============================================================

for value in x_ticks:

    if not (
        X_MIN
        <=
        value
        <=
        X_MAX
    ):

        continue


    x, _ = data_to_drawio(

        value,

        Y_MIN

    )


    # 网格线
    add_line(

        x,
        TOP,

        x,
        BOTTOM,

        color="#DDDDDD",

        width=0.7,

        dashed=True

    )


    # 刻度
    add_line(

        x,
        BOTTOM,

        x,
        BOTTOM + 7,

        width=1.2

    )


    # 数值
    add_text(

        f"{value:.1f}",

        x,
        BOTTOM + 24,

        60,
        25,

        font_size=12

    )


# ============================================================
# 33. draw.io Y轴
# ============================================================

for value in y_ticks:

    if not (
        Y_MIN
        <=
        value
        <=
        Y_MAX
    ):

        continue


    _, y = data_to_drawio(

        X_MIN,

        value

    )


    # 网格
    add_line(

        LEFT,
        y,

        RIGHT,
        y,

        color="#DDDDDD",

        width=0.7,

        dashed=True

    )


    # 刻度
    add_line(

        LEFT - 7,
        y,

        LEFT,
        y,

        width=1.2

    )


    # 数值
    add_text(

        f"{value:.1f}",

        LEFT - 42,
        y,

        60,
        24,

        font_size=12

    )


# ============================================================
# 34. draw.io坐标轴标题
# ============================================================

add_text(

    "FLOPs (G)",

    LEFT
    +
    PLOT_W / 2,

    BOTTOM + 70,

    180,
    35,

    font_size=18,

    bold=True

)


add_text(

    "mAP@0.5:0.95 (%)",

    35,

    TOP
    +
    PLOT_H / 2,

    210,
    35,

    font_size=18,

    bold=True,

    rotation=-90

)


# ============================================================
# 35. draw.io 主图模型
# ============================================================

bubble_ids = {}
label_ids = {}


for i, row in df.iterrows():

    model = str(
        row["Model"]
    )


    # --------------------------------------------------------
    # 数据点位置
    # --------------------------------------------------------

    bubble_x, bubble_y = data_to_drawio(

        row["FLOPs"],

        row["mAP_plot"]

    )


    diameter = (
        parameter_to_drawio_diameter(
            row["Parameters"]
        )
    )


    # --------------------------------------------------------
    # 颜色
    # --------------------------------------------------------

    if model == HIGHLIGHT_MODEL:

        fill_color = (
            "#E64B35"
        )

        stroke_color = (
            "#000000"
        )

        stroke_width = 2.5

    else:

        fill_color = (
            colors[i]
        )

        stroke_color = (
            "#FFFFFF"
        )

        stroke_width = 1.5


    # --------------------------------------------------------
    # 气泡
    # --------------------------------------------------------

    bubble_id = add_bubble(

        bubble_x,
        bubble_y,

        diameter,

        fill_color,

        stroke_color,

        stroke_width

    )


    bubble_ids[
        model
    ] = bubble_id


    # --------------------------------------------------------
    # adjustText后的标签位置
    # --------------------------------------------------------

    (
        label_x_data,
        label_y_data

    ) = label_positions[
        model
    ]


    label_x, label_y = data_to_drawio(

        label_x_data,

        label_y_data

    )


    # 标签宽度自动适应名称长度
    label_width = max(

        75,

        min(

            220,

            len(model) * 9

        )

    )


    label_height = 28


    # --------------------------------------------------------
    # 高亮Ours
    # --------------------------------------------------------

    if model == HIGHLIGHT_MODEL:

        label_color = (
            "#B22222"
        )

        label_bold = True

        font_size = 15

    else:

        label_color = (
            "#000000"
        )

        label_bold = False

        font_size = 13


    # --------------------------------------------------------
    # 创建标签
    # --------------------------------------------------------

    label_id = add_text(

        model,

        label_x,
        label_y,

        label_width,
        label_height,

        font_size=font_size,

        bold=label_bold,

        color=label_color

    )


    label_ids[
        model
    ] = label_id


    # --------------------------------------------------------
    # 标签离气泡比较远才加引线
    # --------------------------------------------------------

    distance = np.sqrt(

        (
            label_x
            -
            bubble_x
        ) ** 2

        +

        (
            label_y
            -
            bubble_y
        ) ** 2

    )


    if distance > (
        diameter / 2
        +
        14
    ):

        add_connector(

            bubble_id,

            label_id

        )


# ============================================================
# 36. draw.io右侧参数量图例
#
# 每个模型：
#
# 同样大小的小圆
# +
# 模型名
# +
# 参数量
# ============================================================

LEGEND_X = 1120

LEGEND_TOP = 115

LEGEND_DIAMETER = 22

LEGEND_ROW_GAP = 48


# 标题
add_text(

    "Parameters",

    LEGEND_X + 115,

    70,

    260,
    35,

    font_size=16,

    bold=True

)


for i, row in df.iterrows():

    model = str(
        row["Model"]
    )

    param = float(
        row["Parameters"]
    )


    yy = (

        LEGEND_TOP

        +

        i
        *
        LEGEND_ROW_GAP

    )


    # --------------------------------------------------------
    # 颜色与主图对应
    # --------------------------------------------------------

    if model == HIGHLIGHT_MODEL:

        fill_color = (
            "#E64B35"
        )

        stroke_color = (
            "#000000"
        )

        stroke_width = 1.5

    else:

        fill_color = (
            colors[i]
        )

        stroke_color = (
            "#FFFFFF"
        )

        stroke_width = 1


    # --------------------------------------------------------
    # 统一大小圆点
    # --------------------------------------------------------

    add_bubble(

        LEGEND_X,

        yy,

        LEGEND_DIAMETER,

        fill_color,

        stroke_color,

        stroke_width

    )


    # --------------------------------------------------------
    # 模型名称
    # --------------------------------------------------------

    add_text(

        model,

        LEGEND_X + 120,

        yy,

        190,
        28,

        font_size=12,

        bold=(
            model
            ==
            HIGHLIGHT_MODEL
        ),

        color=(

            "#B22222"

            if model
            ==
            HIGHLIGHT_MODEL

            else

            "#000000"

        )

    )


    # --------------------------------------------------------
    # 参数量
    # --------------------------------------------------------

    add_text(

        f"{param:.1f} M",

        LEGEND_X + 255,

        yy,

        75,
        28,

        font_size=12,

        bold=(
            model
            ==
            HIGHLIGHT_MODEL
        ),

        color=(

            "#B22222"

            if model
            ==
            HIGHLIGHT_MODEL

            else

            "#000000"

        )

    )


# ============================================================
# 37. 保存 draw.io
# ============================================================

tree = ET.ElementTree(
    mxfile
)


try:

    ET.indent(
        tree,
        space="  "
    )

except AttributeError:

    pass


tree.write(

    DRAWIO_FILE,

    encoding="utf-8",

    xml_declaration=True

)


# ============================================================
# 38. 输出信息
# ============================================================

print()
print(
    "============================================"
)

print(
    "全部生成完成"
)

print(
    "============================================"
)

print(
    f"PNG    : {PNG_FILE}"
)

print(
    f"PDF    : {PDF_FILE}"
)

print(
    f"SVG    : {SVG_FILE}"
)

print(
    f"DRAWIO : {DRAWIO_FILE}"
)

print()
print(
    "draw.io 中模型名称、气泡、参数量图例都可以单独编辑。"
)

print()


plt.show()