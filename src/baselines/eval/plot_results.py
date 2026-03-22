"""
读取 run_eval.py 输出的 results.json，绘制对比图。

生成四张图：
  1. 主指标分组柱状图（best_score / success_rate / spectrum_mae）
  2. best_score 箱线图（展示多 case 下的稳定性）
  3. 推理时间 vs 质量散点图
  4. 多样性对比柱状图

用法：
  cd /data/GraduationProject
  python src/baselines/eval/plot_results.py \
      --results samples/eval_compare/results.json \
      --save_dir samples/eval_compare/figures
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 配色：每个方法一个颜色 ────────────────────────────────────────────
METHOD_COLORS = {
    "random"    : "#78909C",   # 灰蓝
    "cvae"      : "#42A5F5",   # 蓝
    "cgan"      : "#AB47BC",   # 紫
    "diffusion" : "#EF5350",   # 红（你的方法，突出显示）
}
METHOD_LABELS = {
    "random"    : "Random+TopoOpt",
    "cvae"      : "CVAE",
    "cgan"      : "cGAN",
    "diffusion" : "Ours (Diffusion)",
}
BG = "#FAFAFA"


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_valid(results: dict, key: str) -> dict[str, list]:
    out = {}
    for method, cases in results.items():
        vals = [c[key] for c in cases if key in c and "error" not in c]
        if vals:
            out[method] = vals
    return out


def _method_order(results: dict) -> list[str]:
    preferred = ["random", "cvae", "cgan", "diffusion"]
    present   = list(results.keys())
    return [m for m in preferred if m in present] + \
           [m for m in present if m not in preferred]


# ── 图1：主指标分组柱状图 ──────────────────────────────────────────────

def plot_main_metrics(results: dict, save_dir: str):
    metrics = [
        ("best_score",       "Best Score ↑",        True),
        ("top1_success_rate","Success Rate ↑",       True),
        ("spectrum_mae",     "Spectrum MAE ↓",       False),
    ]
    methods = _method_order(results)
    n_m  = len(methods)
    n_g  = len(metrics)
    x    = np.arange(n_g)
    w    = 0.7 / n_m

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for i, method in enumerate(methods):
        means, stds = [], []
        for key, _, _ in metrics:
            vals = [c[key] for c in results[method]
                    if key in c and "error" not in c]
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals)  if vals else 0.0)

        offset = (i - n_m / 2 + 0.5) * w
        color  = METHOD_COLORS.get(method, "#999999")
        label  = METHOD_LABELS.get(method, method)
        bars   = ax.bar(x + offset, means, w * 0.9,
                        yerr=stds, capsize=3,
                        color=color, alpha=0.88,
                        label=label,
                        error_kw={"elinewidth": 1.2, "ecolor": "#555"})

        # 数值标注
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([lb for _, lb, _ in metrics], fontsize=11)
    ax.set_ylabel("Score / Rate / MAE", fontsize=11)
    ax.set_title("Comparison of Inverse Design Methods", fontsize=13,
                 fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, framealpha=0.9, edgecolor="#ccc")
    ax.grid(axis="y", ls="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")
    ax.set_ylim(0, None)

    plt.tight_layout()
    out = os.path.join(save_dir, "fig1_main_metrics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


# ── 图2：best_score 箱线图 ────────────────────────────────────────────

def plot_score_boxplot(results: dict, save_dir: str):
    methods   = _method_order(results)
    data      = [_get_valid(results, "best_score").get(m, []) for m in methods]
    labels    = [METHOD_LABELS.get(m, m) for m in methods]
    colors    = [METHOD_COLORS.get(m, "#999") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best Score", fontsize=11)
    ax.set_title("Score Distribution Across Test Cases", fontsize=13,
                 fontweight="bold", pad=10)
    ax.axhline(0.5, color="#E57373", ls="--", lw=1.2, alpha=0.7,
               label="success threshold=0.5")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.grid(axis="y", ls="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")

    plt.tight_layout()
    out = os.path.join(save_dir, "fig2_score_boxplot.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


# ── 图3：推理时间 vs 质量 散点图 ──────────────────────────────────────

def plot_time_vs_quality(results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for method in _method_order(results):
        valid = [c for c in results[method] if "error" not in c
                 and "best_score" in c and "inference_time_s" in c]
        if not valid:
            continue
        t  = np.mean([c["inference_time_s"] for c in valid])
        sc = np.mean([c["best_score"]       for c in valid])
        sc_std = np.std([c["best_score"]    for c in valid])
        color  = METHOD_COLORS.get(method, "#999")
        label  = METHOD_LABELS.get(method, method)

        ax.scatter(t, sc, s=120, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.errorbar(t, sc, yerr=sc_std, fmt="none",
                    ecolor=color, capsize=4, elinewidth=1.5)
        ax.annotate(label, xy=(t, sc),
                    xytext=(t * 1.08, sc + 0.01),
                    fontsize=9, color=color)

    ax.set_xlabel("Average Inference Time (s)", fontsize=11)
    ax.set_ylabel("Average Best Score", fontsize=11)
    ax.set_title("Quality vs. Inference Time", fontsize=13,
                 fontweight="bold", pad=10)
    ax.grid(ls="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")

    plt.tight_layout()
    out = os.path.join(save_dir, "fig3_time_vs_quality.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


# ── 图4：多样性对比 ───────────────────────────────────────────────────

def plot_diversity(results: dict, save_dir: str):
    methods = _method_order(results)
    means, stds, labels, colors = [], [], [], []

    for method in methods:
        vals = [c["diversity"] for c in results[method]
                if "diversity" in c and "error" not in c]
        if not vals:
            continue
        means.append(np.mean(vals))
        stds.append(np.std(vals))
        labels.append(METHOD_LABELS.get(method, method))
        colors.append(METHOD_COLORS.get(method, "#999"))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    bars = ax.bar(labels, means, yerr=stds, capsize=4,
                  color=colors, alpha=0.85, width=0.5,
                  error_kw={"elinewidth": 1.5, "ecolor": "#555"})

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9.5)

    ax.set_ylabel("Average Hamming Diversity", fontsize=11)
    ax.set_title("Structural Diversity of Generated Candidates",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.set_ylim(0, None)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")

    plt.tight_layout()
    out = os.path.join(save_dir, "fig4_diversity.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"saved → {out}")


# ── 主程序 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="samples/eval_compare/results.json")
    parser.add_argument("--save_dir", default="samples/eval_compare/figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results = load_results(args.results)

    print(f"[plot] methods: {list(results.keys())}")
    plot_main_metrics(results, args.save_dir)
    plot_score_boxplot(results, args.save_dir)
    plot_time_vs_quality(results, args.save_dir)
    plot_diversity(results, args.save_dir)
    print("[plot] all done.")


if __name__ == "__main__":
    main()
