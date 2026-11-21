# -*- coding: utf-8 -*-
r"""
论文插图生成（全部数据来自 key_data 原始文件，可复现）
=====================================================
输出目录：论文\figures\（300 dpi PNG）

  fig2_1_quant_accum.png   图 2-1  定点卷积逐层量化误差累积
  fig3_1_pipeline.png      图 3-1  困难样本筛选与校准集构建流程
  fig4_1_false_positive.png 图 4-1 量化引入的边缘伪响应（低置信度复核）
  fig4_2_latency.png       图 4-2  端到端延迟构成
  fig4_3_strata.png        图 4-3  按困难类型分层的量化误差
  fig4_4_cdf.png           图 4-4  中心点误差累积分布（CDF）
  fig4_5_scatter.png       图 4-5  目标面积与归一化误差的相关性

数据来源：
  · 分层/逐框误差：key_data\analysis\matched_{B_random,C_hard}.csv、quant_error_summary.json
  · 延迟分解：key_data\results_lat\summary.txt（42 帧实测）
  · 定点卷积：论文\code\quant_demo.py 运行输出（表 2-1 同源，4 层，累加器截断 10 位）
  · 虚警：key_data\results_0.05\board_results.csv、analysis_lowconf\golden_results.csv
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

KEY = r"D:\maixcam_exp\model_data\key_data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
IMG_TEST = r"D:\maixcam_exp\yolo_data_test\images\test"

C_B = "#4C72B0"     # 随机校准
C_C = "#C44E52"     # 困难校准
C_G = "#55A868"     # FP32 / 真值
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved:", p)
    return p


def read_matched(name):
    """读逐框对比明细 → list[dict]"""
    p = os.path.join(KEY, "analysis", name)
    rows = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "img": r["image"], "type": r["hard_type"],
                "ae": [float(r["ae_x1"]), float(r["ae_y1"]),
                       float(r["ae_x2"]), float(r["ae_y2"])],
                "ae_conf": float(r["ae_conf"]),
                "center": float(r["center_err"]),
                "norm": float(r["norm_err_pct"]),
                "area": float(r["gt_area"]),
            })
    return rows


def read_summary_json():
    with open(os.path.join(KEY, "analysis", "quant_error_summary.json"),
              encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 图 2-1
def fig_quant_accum():
    # 与 表 2-1 同源：code\quant_demo.py 输出（4 层定点卷积演示）
    layers = [1, 2, 3, 4]
    rmse_r = [0.098539, 0.158682, 0.417819, 1.042874]
    rmse_t = [0.086841, 0.173866, 0.431204, 1.071259]
    maxae_r = [0.270133, 0.397148, 1.686222, 3.436698]
    maxae_t = [0.226320, 0.442046, 1.327811, 3.436698]

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ax[0].plot(layers, rmse_r, "o-", color=C_B, label="舍入（round）")
    ax[0].plot(layers, rmse_t, "s--", color=C_C, label="截断（truncate）")
    for x, y in zip(layers, rmse_t):
        ax[0].annotate("%.3f" % y, (x, y), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=8)
    ax[0].set_xlabel("卷积层序号")
    ax[0].set_ylabel("RMSE")
    ax[0].set_title("(a) RMSE 随层数单调累积")
    ax[0].set_xticks(layers)
    ax[0].grid(alpha=0.3, ls=":")
    ax[0].legend(fontsize=9)

    ax[1].plot(layers, maxae_r, "o-", color=C_B, label="舍入（round）")
    ax[1].plot(layers, maxae_t, "s--", color=C_C, label="截断（truncate）")
    ax[1].set_xlabel("卷积层序号")
    ax[1].set_ylabel("MaxAE")
    ax[1].set_title("(b) 最大绝对误差")
    ax[1].set_xticks(layers)
    ax[1].grid(alpha=0.3, ls=":")
    ax[1].legend(fontsize=9)
    fig.suptitle("定点卷积逐层量化误差累积（4 层，累加器截断 10 位）", fontsize=11)
    return save(fig, "fig2_1_quant_accum.png")


# ---------------------------------------------------------------- 图 3-1
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10.5, 3.3))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.3)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF2FB", ec="#4C72B0", fs=8.6):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.1))

    box(0.1, 2.15, 1.7, 0.85,
        "训练集 181 张\n（YOLO 标注）")
    box(2.05, 2.15, 2.0, 0.85,
        "三维度指标统计\n目标面积 / 平均亮度\nLaplacian 方差", fc="#FFF4E5", ec="#D68910")
    box(4.3, 2.15, 2.2, 0.85,
        "优先级分桶\n小目标 > 低光照 > 模糊\n（阈值：0.02 / 0.30 / 48）", fc="#FFF4E5", ec="#D68910")
    box(6.75, 2.15, 1.9, 0.85,
        "等量抽取\n各 20 张\n（配额 N=60）")
    box(8.9, 2.15, 1.5, 0.85, "困难混合\n校准集 60 张",
        fc="#FDE9EA", ec=C_C)

    arrow(1.8, 2.57, 2.05, 2.57)
    arrow(4.05, 2.57, 4.3, 2.57)
    arrow(6.5, 2.57, 6.75, 2.57)
    arrow(8.65, 2.57, 8.9, 2.57)

    box(0.1, 0.75, 2.4, 0.75, "随机采样 60 张\n（对照组，同规模）",
        fc="#EEF3EE", ec=C_G)
    box(3.2, 0.75, 2.6, 0.75, "tpu-mlir 量化编译\n（INT8，cv181x）",
        fc="#F2EDF7", ec="#7D3C98")
    box(6.4, 0.75, 2.3, 0.75, "板端部署\nMaixCAM-Pro")
    box(9.1, 0.75, 1.3, 0.75, "黄金参考\n对比分析",
        fc="#FDE9EA", ec=C_C)

    arrow(9.65, 2.15, 9.65, 1.5)
    arrow(2.5, 2.15, 1.3, 1.5)
    arrow(2.5, 1.12, 3.2, 1.12)
    arrow(5.8, 1.12, 6.4, 1.12)
    arrow(8.7, 1.12, 9.1, 1.12)
    ax.text(5.2, 0.32, "校准集构造策略是唯一变量：两组规模一致（各 60 张），仅分布不同",
            fontsize=8.6, ha="center", color="#555555")
    ax.set_title("困难样本校准集构建与量化部署流程", fontsize=11)
    return save(fig, "fig3_1_pipeline.png")


# ---------------------------------------------------------------- 图 4-1
def fig_false_positive():
    rec = []
    with open(os.path.join(KEY, "results_0.05", "board_results.csv"),
              encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rec.append(r)
    with open(os.path.join(KEY, "results_random_0.05", "board_results.csv"),
              encoding="utf-8-sig", newline="") as f:
        rnd = [r for r in csv.DictReader(f)]
    with open(os.path.join(KEY, "analysis_lowconf", "golden_results.csv"),
              encoding="utf-8-sig", newline="") as f:
        gold = [r for r in csv.DictReader(f)]

    target = "1007085849_phone_14.jpg"
    img = mpimg.imread(os.path.join(IMG_TEST, target))

    def boxes(target_ax, rows, label, color, ls="-", lw=1.6):
        got = False
        for r in rows:
            if r["image"] != target:
                continue
            x, y = float(r["x1"]), float(r["y1"])
            w, h = float(r["x2"]) - x, float(r["y2"]) - y
            target_ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                                          ls=ls, lw=lw,
                                          label=label if not got else None))
            got = True
        return got

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.4),
                           gridspec_kw={"width_ratios": [1.35, 1]})
    ax[0].imshow(img)
    boxes(ax[0], gold, "FP32 黄金参考（仅 1 框）", C_G, "-", 1.8)
    boxes(ax[0], rec, "INT8 困难校准 → 边缘虚警", C_C, "-", 1.6)
    boxes(ax[0], rnd, "INT8 随机校准 → 边缘虚警", "#E8A33D", "--", 1.6)
    ax[0].set_title("(a) 测试图像与检测框", fontsize=10)
    ax[0].legend(fontsize=7.5, loc="lower left", framealpha=0.85)

    # 上边缘放大（0~30 px）
    ax[1].imshow(img)
    ax[1].set_xlim(40, 170)
    ax[1].set_ylim(34, 0)
    ax[1].set_title("(b) 上边缘放大（y = 0~30 px）", fontsize=10)
    ax[1].set_xticks([]); ax[1].set_yticks([])
    for r in rec:
        if r["image"] != target:
            continue
        x, y = float(r["x1"]), float(r["y1"])
        w, h = float(r["x2"]) - x, float(r["y2"]) - y
        c = C_C if float(r["conf"]) < 0.2 else C_G
        ax[1].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=c, lw=1.8))
        ax[1].annotate("conf=%.4f" % float(r["conf"]),
                       (x, y + h + 1.2), fontsize=8,
                       color=c, ha="left")
    for r in rnd:
        if r["image"] != target or float(r["conf"]) >= 0.2:
            continue
        x, y = float(r["x1"]), float(r["y1"])
        w, h = float(r["x2"]) - x, float(r["y2"]) - y
        ax[1].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#E8A33D",
                                  ls="--", lw=1.8))
    ax[1].text(45, 29, "红色实线=困难校准 0.0947\n橙色虚线=随机校准 0.0786\n绿色=同图真检 0.8359",
               fontsize=7.8, color="#333333", va="top")
    ax[1].annotate("", xy=(125, 4), xytext=(125, 26),
                   arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))
    fig.suptitle("量化引入的 letterbox 填边边界伪响应（conf = 0.05 复核）", fontsize=11)
    return save(fig, "fig4_1_false_positive.png")


# ---------------------------------------------------------------- 图 4-2
def fig_latency():
    total, lb, post, rest = 12.230, 3.451, 0.500, 8.279
    labels = ["letterbox 预处理\n(缩放+填边)",
              "后处理\n(YOLO 解码 + NMS)",
              "非 resize 预处理 + NPU 推理"]
    vals = [lb, post, rest]
    cols = ["#E8A33D", C_G, C_B]

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8),
                           gridspec_kw={"width_ratios": [1.7, 1]})
    left = 0
    for v, c, lb_ in zip(vals, cols, labels):
        ax[0].barh(0, v, left=left, color=c, edgecolor="white", height=0.5,
                   label="%s %.3f ms (%.1f%%)" % (lb_.replace("\n", " "), v, 100 * v / total))
        ax[0].text(left + v / 2, 0, "%.3f" % v, ha="center", va="center",
                   color="white", fontsize=9, fontweight="bold")
        left += v
    ax[0].set_xlim(0, total)
    ax[0].set_ylim(-0.75, 0.75)
    ax[0].set_yticks([])
    ax[0].set_xlabel("单帧延迟 (ms)")
    ax[0].set_title("(a) 端到端 %.3f ms 的构成（42 帧实测）" % total, fontsize=10)
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1,
                 frameon=False)
    ax[0].annotate("≥ 32.3% 与 NPU 算力无关", xy=(lb + post / 2, 0.42),
                   xytext=(lb + post / 2, 0.68), ha="center", fontsize=8.8, color=C_C,
                   arrowprops=dict(arrowstyle="->", color=C_C, lw=1.0))

    names = ["FP32 参考\n(训练机)", "INT8 随机\n校准", "INT8 困难\n校准"]
    lat = [16.5, 12.36, 12.24]
    fps = [None, 81.04, 81.80]
    bars = ax[1].bar(names, lat, color=["#BBBBBB", C_B, C_C], width=0.6)
    for b, v, f in zip(bars, lat, fps):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.25,
                   "%.2f ms" % v + ("\n%.2f FPS" % f if f else "\n(CPU 推理)"),
                   ha="center", fontsize=8.4)
    ax[1].set_ylim(0, 20)
    ax[1].set_ylabel("单帧延迟 (ms)")
    ax[1].set_title("(b) 三组端到端延迟对比", fontsize=10)
    ax[1].grid(axis="y", alpha=0.3, ls=":")
    return save(fig, "fig4_2_latency.png")


# ---------------------------------------------------------------- 图 4-3
def fig_strata():
    js = read_summary_json()
    arms = {a["arm"]: a for a in js["arms"]}
    order = ["小目标", "低光照", "模糊", "普通"]
    b, c = arms["B_random"]["by_type"], arms["C_hard"]["by_type"]
    keys = [k for k in order if k in b]
    mae_b = [b[k]["mae"] for k in keys]
    mae_c = [c[k]["mae"] for k in keys]
    nrm_b = [b[k]["norm_mae_pct"] for k in keys]
    nrm_c = [c[k]["norm_mae_pct"] for k in keys]
    n_img = [b[k]["img"] for k in keys]

    x = range(len(keys))
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.8))
    w = 0.38
    r1 = ax[0].bar([i - w / 2 for i in x], mae_b, w, color=C_B, label="随机校准")
    r2 = ax[0].bar([i + w / 2 for i in x], mae_c, w, color=C_C, label="困难校准")
    for bars in (r1, r2):
        for bb in bars:
            ax[0].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.03,
                       "%.3f" % bb.get_height(), ha="center", fontsize=7.8)
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(["%s\n(%d 张)" % (k, n) for k, n in zip(keys, n_img)])
    ax[0].set_ylabel("坐标 MAE (px)")
    ax[0].set_title("(a) 各困难分层的坐标绝对误差", fontsize=10)
    ax[0].legend(fontsize=9)
    ax[0].grid(axis="y", alpha=0.3, ls=":")

    r3 = ax[1].bar([i - w / 2 for i in x], nrm_b, w, color=C_B, label="随机校准")
    r4 = ax[1].bar([i + w / 2 for i in x], nrm_c, w, color=C_C, label="困难校准")
    for bars in (r3, r4):
        for bb in bars:
            ax[1].text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.03,
                       "%.2f%%" % bb.get_height(), ha="center", fontsize=7.8)
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(["%s\n(%d 张)" % (k, n) for k, n in zip(keys, n_img)])
    ax[1].set_ylabel("归一化误差 (% 目标对角线)")
    ax[1].set_title("(b) 归一化误差：小目标的相对误差最大", fontsize=10)
    ax[1].legend(fontsize=9)
    ax[1].grid(axis="y", alpha=0.3, ls=":")
    return save(fig, "fig4_3_strata.png")


# ---------------------------------------------------------------- 图 4-4
def fig_cdf():
    rows = {"随机校准": read_matched("matched_B_random.csv"),
            "困难校准": read_matched("matched_C_hard.csv")}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for (name, rs), col in zip(rows.items(), (C_B, C_C)):
        v = sorted(r["center"] for r in rs)
        n = len(v)
        y = [(i + 1) / n for i in range(n)]
        ax.plot(v, y, "-", color=col, lw=1.8, label="%s（n=%d）" % (name, n))
        p50 = v[int(0.5 * n)]
        p90 = v[min(n - 1, int(0.9 * n))]
        ax.annotate("P50=%.2f" % p50, (p50, 0.5), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color=col)
        ax.annotate("P90=%.2f" % p90, (p90, 0.9), textcoords="offset points",
                    xytext=(6, -12), fontsize=8, color=col)
    ax.axhline(0.9, color="#999999", ls=":", lw=1)
    ax.set_xlabel("中心点定位误差 (px)")
    ax.set_ylabel("累积概率")
    ax.set_title("中心点误差累积分布（板端 INT8 vs FP32 黄金参考，45 框全匹配）",
                 fontsize=10)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=9, loc="lower right")
    return save(fig, "fig4_4_cdf.png")


# ---------------------------------------------------------------- 图 4-5
def fig_scatter():
    rows = {"随机校准": read_matched("matched_B_random.csv"),
            "困难校准": read_matched("matched_C_hard.csv")}
    js = read_summary_json()
    rs = {a["arm"]: a for a in js["arms"]}
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for (name, rr), col, arm in zip(rows.items(), (C_B, C_C),
                                    ("B_random", "C_hard")):
        xs = [r["area"] for r in rr]
        ys = [r["norm"] for r in rr]
        ax.scatter(xs, ys, s=26, color=col, alpha=0.75, label=name, zorder=3)
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        import math
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        k = sxy / sxx
        b0 = my - k * mx
        xr = [min(xs), max(xs)]
        ax.plot(xr, [k * v + b0 for v in xr], "--", color=col, lw=1.4, zorder=2)
        ax.annotate("r = %.3f" % rs[arm]["r_area_norm"],
                    (0.62, 0.86 if arm == "B_random" else 0.78),
                    xycoords="axes fraction", fontsize=9.5, color=col)
    ax.axhline(2.0, color="#999999", ls=":", lw=1)
    ax.text(0.30, 2.15, "归一化误差 2% 参考线", fontsize=8, color="#777777")
    ax.set_xlabel("目标面积（占图像比例）")
    ax.set_ylabel("归一化中心点误差 (% 目标对角线)")
    ax.set_title("目标尺度与相对定位误差的相关性（目标越小，相对误差越大）", fontsize=10)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=9)
    return save(fig, "fig4_5_scatter.png")


def main():
    print("[插图] 输出目录:", OUT)
    fig_quant_accum()
    fig_pipeline()
    fig_false_positive()
    fig_latency()
    fig_strata()
    fig_cdf()
    fig_scatter()
    print("[插图] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
