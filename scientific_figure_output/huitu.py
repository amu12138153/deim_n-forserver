"""
脚本名称: draw_fish_monitoring_setup.py
说明: 绘制鱼类缺氧监测系统数据采集装置示意图 (2.5D 轴测透视、低饱和度科研风格)
输出: 
  - fish_hypoxia_setup.png (300 DPI 超清位图)
  - fish_hypoxia_setup.pdf (期刊投稿无损矢量图)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyBboxPatch, Circle, Wedge
import numpy as np

# 1. 字体配置 (优先使用黑体/Arial确保中文与英文标注清晰)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_canvas():
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    return fig, ax

# 论文学术低饱和度调色盘 (Muted Academic Palette)
PALETTE = {
    "edge": "#2C3E50",         # 边框深灰蓝
    "glass_back": "#F4F8FA",   # 玻璃后壁衬底
    "water_side": "#81D4FA",   # 侧面半透明水体
    "water_front": "#B3E5FC",  # 正面半透明水体
    "water_top": "#4FC3F7",    # 水面
    "rpi": "#A5D6A7",          # 树莓派浅草绿
    "rpi_chip": "#37474F",     # 芯片深灰
    "cam": "#90A4AE",          # 摄像头本体灰蓝
    "cam_lens": "#455A64",     # 镜头深蓝灰
    "pump": "#FFCC80",         # 循环水泵柔和橙
    "sensor": "#B39DDB",       # 传感器柔和紫
    "fish": "#FFAB91",         # 鱼群浅珊瑚红
    "bulb": "#FFF59D",         # 灯泡发光黄
    "power": "#CFD8DC",        # 电源模块浅灰
    "cable": "#78909C",        # 连接线灰
    "label_line": "#37474F"    # 标注指示线
}

def draw_device_schematic():
    fig, ax = create_canvas()

    # ==========================================
    # 1. 绘制 2.5D 方形玻璃鱼缸及水体
    # ==========================================
    # 视角设计：左前方看向右侧鱼缸 (主视+右侧微斜视)
    # 前表面原点 (64, 18)，宽度 72，高度 46
    # 侧立面投射向量: dx=16, dy=11
    dx, dy = 16, 11
    x0, y0 = 64, 18
    w_tank, h_tank = 72, 46
    h_water = 38  # 水位高度

    # 1.1 鱼缸后玻璃内壁
    tank_back = Polygon([
        [x0 + dx, y0 + dy],
        [x0 + w_tank + dx, y0 + dy],
        [x0 + w_tank + dx, y0 + h_tank + dy],
        [x0 + dx, y0 + h_tank + dy]
    ], closed=True, facecolor=PALETTE["glass_back"], edgecolor="#CFD8DC", linewidth=1.2, zorder=1)
    ax.add_patch(tank_back)

    # 1.2 半透明水体分层 (侧立面、水表面、正立面)
    water_side = Polygon([
        [x0 + w_tank, y0],
        [x0 + w_tank + dx, y0 + dy],
        [x0 + w_tank + dx, y0 + h_water + dy],
        [x0 + w_tank, y0 + h_water]
    ], closed=True, facecolor=PALETTE["water_side"], edgecolor="none", alpha=0.45, zorder=2)
    ax.add_patch(water_side)

    water_top = Polygon([
        [x0, y0 + h_water],
        [x0 + w_tank, y0 + h_water],
        [x0 + w_tank + dx, y0 + h_water + dy],
        [x0 + dx, y0 + h_water + dy]
    ], closed=True, facecolor=PALETTE["water_top"], edgecolor="#4FC3F7", alpha=0.55, linewidth=1.0, zorder=3)
    ax.add_patch(water_top)

    water_front = Polygon([
        [x0, y0],
        [x0 + w_tank, y0],
        [x0 + w_tank, y0 + h_water],
        [x0, y0 + h_water]
    ], closed=True, facecolor=PALETTE["water_front"], edgecolor="none", alpha=0.35, zorder=4)
    ax.add_patch(water_front)

    # ==========================================
    # 2. 内部装置前后错位布置 (避免遮挡鱼群视觉轴线)
    # ==========================================
    # 2.1 水泵 (紧靠右侧壁底部，沉入水底后方)
    pump = FancyBboxPatch((138, 22), 8.5, 7.5, boxstyle="round,pad=0.2",
                          facecolor=PALETTE["pump"], edgecolor=PALETTE["edge"], linewidth=1.2, zorder=5)
    ax.add_patch(pump)
    ax.text(142.25, 25.75, "Pump\n水泵", fontsize=7, color="#E65100", weight="bold", ha="center", va="center", zorder=6)

    # 2.2 三组水质传感器 (不同高度与进深错开，探头深入水中)
    # 传感器 1: 挂于后壁偏左 (浅中层)
    ax.plot([78, 78], [66, 45], color=PALETTE["cable"], lw=1.3, zorder=2)
    s1 = FancyBboxPatch((76.5, 41), 3.0, 7.0, boxstyle="round,pad=0.1",
                        facecolor=PALETTE["sensor"], edgecolor=PALETTE["edge"], linewidth=1.0, zorder=3)
    ax.add_patch(s1)

    # 传感器 2: 挂于右后侧壁 (深水层)
    ax.plot([143, 143], [67, 34], color=PALETTE["cable"], lw=1.3, zorder=2)
    s2 = FancyBboxPatch((141.5, 30), 3.0, 7.0, boxstyle="round,pad=0.1",
                        facecolor=PALETTE["sensor"], edgecolor=PALETTE["edge"], linewidth=1.0, zorder=3)
    ax.add_patch(s2)

    # 传感器 3: 挂于右前侧壁 (中层近景)
    ax.plot([134, 134], [58, 46], color=PALETTE["cable"], lw=1.3, zorder=7)
    s3 = FancyBboxPatch((132.5, 42), 3.0, 7.0, boxstyle="round,pad=0.1",
                        facecolor=PALETTE["sensor"], edgecolor=PALETTE["edge"], linewidth=1.0, zorder=8)
    ax.add_patch(s3)

    # 2.3 鱼群 (均匀散布在水体核心透视区，姿态自然)
    fish_positions = [(86, 29), (98, 41), (92, 49), (114, 32), (120, 46)]
    for fx, fy in fish_positions:
        # 鱼身
        body = patches.Ellipse((fx, fy), 5.6, 2.4, angle=-4,
                               facecolor=PALETTE["fish"], edgecolor=PALETTE["edge"], lw=1.0, zorder=6)
        # 鱼尾
        tail = Polygon([[fx - 2.5, fy], [fx - 4.6, fy + 1.4], [fx - 4.6, fy - 1.4]],
                       closed=True, facecolor=PALETTE["fish"], edgecolor=PALETTE["edge"], lw=0.9, zorder=6)
        ax.add_patch(body)
        ax.add_patch(tail)

    # ==========================================
    # 3. 鱼缸整体外轮廓线框 (加在水体上方体现玻璃质感)
    # ==========================================
    tank_front = Polygon([[x0, y0], [x0 + w_tank, y0], [x0 + w_tank, y0 + h_tank], [x0, y0 + h_tank]],
                         closed=True, facecolor='none', edgecolor=PALETTE["edge"], linewidth=1.6, zorder=9)
    tank_right = Polygon([[x0 + w_tank, y0], [x0 + w_tank + dx, y0 + dy], [x0 + w_tank + dx, y0 + h_tank + dy], [x0 + w_tank, y0 + h_tank]],
                         closed=True, facecolor='none', edgecolor=PALETTE["edge"], linewidth=1.6, zorder=9)
    tank_top = Polygon([[x0, y0 + h_tank], [x0 + w_tank, y0 + h_tank], [x0 + w_tank + dx, y0 + h_tank + dy], [x0 + dx, y0 + h_tank + dy]],
                       closed=True, facecolor='none', edgecolor=PALETTE["edge"], linewidth=1.6, zorder=9)
    ax.add_patch(tank_front)
    ax.add_patch(tank_right)
    ax.add_patch(tank_top)

    # ==========================================
    # 4. 左侧控制单元 (树莓派 + 摄像头)
    # ==========================================
    # 4.1 树莓派主控板 (PCB + 接口插座)
    rpi = FancyBboxPatch((14, 32), 19, 15, boxstyle="round,pad=0.3",
                         facecolor=PALETTE["rpi"], edgecolor=PALETTE["edge"], linewidth=1.4, zorder=7)
    ax.add_patch(rpi)
    # 板载细节 (网口、USB口、SoC)
    ax.add_patch(patches.Rectangle((15, 34), 4.2, 5.5, facecolor="#ECEFF1", edgecolor=PALETTE["edge"], lw=0.8, zorder=8))
    ax.add_patch(patches.Rectangle((27, 33), 4.8, 4.0, facecolor="#ECEFF1", edgecolor=PALETTE["edge"], lw=0.8, zorder=8))
    ax.add_patch(patches.Rectangle((20, 37), 5.0, 5.0, facecolor=PALETTE["rpi_chip"], edgecolor=PALETTE["edge"], lw=0.8, zorder=8))
    ax.text(22.5, 41, "Raspberry Pi", fontsize=8.2, weight="bold", color="#1B5E20", ha="center", zorder=9)

    # 4.2 工业摄像头组件 (正对鱼缸拍摄)
    cam_body = FancyBboxPatch((39, 36.5), 6.5, 8.0, boxstyle="round,pad=0.1",
                              facecolor=PALETTE["cam"], edgecolor=PALETTE["edge"], linewidth=1.2, zorder=8)
    cam_lens = Polygon([[45.5, 38], [49.5, 36], [49.5, 45], [45.5, 43]],
                       closed=True, facecolor=PALETTE["cam_lens"], edgecolor=PALETTE["edge"], linewidth=1.2, zorder=8)
    ax.add_patch(cam_body)
    ax.add_patch(cam_lens)

    # 树莓派与摄像头排线连接
    ax.plot([33, 39], [40.5, 40.5], color="#37474F", lw=3.0, zorder=7)

    # 摄像头视野角度虚线 (FOV: 照亮鱼缸正立面视野，穿过玻璃)
    ax.plot([49.5, 64], [44.5, 52], color="#90A4AE", linestyle=":", lw=1.2, zorder=3)
    ax.plot([49.5, 64], [36.5, 23], color="#90A4AE", linestyle=":", lw=1.2, zorder=3)

    # 4.3 供电电源模块 (左下方方形模块)
    power = FancyBboxPatch((15, 14), 14, 9, boxstyle="round,pad=0.2",
                           facecolor=PALETTE["power"], edgecolor=PALETTE["edge"], linewidth=1.2, zorder=7)
    ax.add_patch(power)
    ax.text(22, 18.5, "Power Supply\n电源模块", fontsize=7.5, weight="bold", ha="center", va="center", zorder=8)

    # ==========================================
    # 5. 上方光源系统 (灯泡结构与光线辐射)
    # ==========================================
    bulb_x, bulb_y = 104, 76
    # 吊线与灯头
    ax.plot([bulb_x, bulb_x], [88, bulb_y + 3], color=PALETTE["edge"], lw=1.5, zorder=8)
    bulb_cap = patches.Rectangle((bulb_x - 1.5, bulb_y + 3), 3, 2,
                                 facecolor="#78909C", edgecolor=PALETTE["edge"], lw=1.0, zorder=9)
    # 灯泡本体
    bulb = Circle((bulb_x, bulb_y), 3.4, facecolor=PALETTE["bulb"], edgecolor=PALETTE["edge"], lw=1.2, zorder=9)
    ax.add_patch(bulb_cap)
    ax.add_patch(bulb)

    # 辐射光线 (虚线放射状)
    for angle in [-45, -25, 0, 25, 45]:
        rad = np.radians(angle)
        dx_l = 6.5 * np.sin(rad)
        dy_l = -6.5 * np.cos(rad)
        ax.plot([bulb_x + 0.65 * dx_l, bulb_x + dx_l],
                [bulb_y + 0.65 * dy_l, bulb_y + dy_l],
                color="#FBC02D", lw=1.1, linestyle="--", zorder=8)

    # ==========================================
    # 6. 系统供电与数据连接线缆
    # ==========================================
    # 电源至树莓派 (实线)
    ax.plot([22, 22], [23, 32], color=PALETTE["cable"], lw=1.4, linestyle="-", zorder=6)
    # 传感器总线引出至树莓派 GPIO (虚线)
    ax.plot([78, 55, 33], [66, 66, 44], color=PALETTE["cable"], lw=1.2, linestyle="--", zorder=5)

    # ==========================================
    # 7. 中英双语学术引线标注 (Professional Leader Lines)
    # ==========================================
    def annotate_item(target_pt, text_pos, text, align="left"):
        ax.annotate(
            text,
            xy=target_pt,
            xytext=text_pos,
            fontsize=8.5,
            fontweight="bold",
            color="#1A252C",
            ha=align,
            va="center",
            arrowprops=dict(
                arrowstyle="-|>",
                color=PALETTE["label_line"],
                lw=0.9,
                shrinkA=3,
                shrinkB=3,
                connectionstyle="angle,angleA=0,angleB=90,rad=2"
            ),
            bbox=dict(boxstyle="square,pad=0.2", fc="#FFFFFF", ec="none", alpha=0.88),
            zorder=20
        )

    # 各核心组件引线指引
    annotate_item((22, 47), (15, 61), "树莓派主控单元\nRaspberry Pi", align="center")
    annotate_item((45, 44), (43, 56), "工业摄像头\nCamera", align="center")
    annotate_item((bulb_x, bulb_y), (128, 82), "光源系统\nLight source", align="left")
    annotate_item((x0 + w_tank, 58), (146, 68), "方形透明水缸\nRectangular tank", align="left")
    annotate_item((141.5, 33), (146, 45), "水质传感器 (3组)\nWater quality sensors", align="left")
    annotate_item((142, 22), (146, 19), "水泵\nWater pump", align="left")
    annotate_item((98, 41), (92, 10), "鱼群受试对象\nExperimental fish", align="center")
    annotate_item((70, 26), (50, 10), "水体水环境\nWater body", align="center")
    annotate_item((22, 14), (22, 6), "电源模块\nPower supply", align="center")

    # ==========================================
    # 8. 保存与导出
    # ==========================================
    png_path = "fish_hypoxia_setup.png"
    pdf_path = "fish_hypoxia_setup.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="#FFFFFF")
    plt.close()

    print(f"示意图生成成功！")
    print(f" -> 论文高清位图: {png_path} (300 DPI)")
    print(f" -> 矢量文件:     {pdf_path} (直接插入 LaTeX)")

if __name__ == "__main__":
    draw_device_schematic()