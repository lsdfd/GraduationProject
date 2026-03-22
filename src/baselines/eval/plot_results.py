"""
读取 run_eval.py 输出的 results.json，绘制对比图。

生成五张图：
  1. 质量三连柱状图（best_score / mean_score / top3_score）
  2. 成功率 & 多样性 & 二值化程度柱状图
  3. all_scores 小提琴图（展示候选分数分布形状）
  4. 推理时间 vs 质量散点图（误差棒 = best→mean 差距）
  5. spectrum_mae 柱状图

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

# ── 配色：5 个方法，diffusion+guide 红色高亮（= Ours）────────────────
METHOD_COLORS = {
    "random"          : "#78909C",   # 灰
    "cvae"            : "#42A5F5",   # 蓝
    "cgan"            : "#AB47BC",   # 紫
    "diffusion"       : "#FFA726",   # 橙（扩散无引导）
    "diffusion+guide" : "#EF5350",   # 红（Ours）
}
METHOD_LABELS = {
    "random"          : "Random",
    "cvae"            : "CVAE",
    "cgan"            : "cGAN",
    "diffusion"       : "Diffusion (CFG only)",
    "diffusion+guide" : "Ours (CFG + DPS)",
}
BG = "#FAFAFA"
PREFERRED_ORDER = ["random", "cvae", "cgan", "diffusion", "diffusion+guide"]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _method_order(results: dict) -> list[str]:
    present = list(results.keys())
    return [m for m in PREFERRED_ORDER if m in present] + \
           [m for m in present if m not in PREFERRED_ORDER]


def _val(results: dict, method: str, key: str, default: float = 0.0) -> float:
    r = results[method]
    if isinstance(r, list):   # 旧格式兼容
        vals = [c[key] for c in r if key in c and "error" not in c]
        return float(np.mean(vals)) if vals else default
    return float(r.get(key, default)) if "error" not in r else default


def _bar_group(ax, methods, metrics_keys, results):
    """通用分组柱状图绘制。metrics_keys: list of (key, label)"""
    n_m = len(methods)
    n_g = len(metrics_keys)
    x   = np.arange(n_g)
    w   = 0.7 / n_m
    for i, method in enumerate(methods):
        vals   = [_val(results, method, k) for k, _ in metrics_keys]
        offset = (i - n_m / 2 + 0.5) * w
        color  = METHOD_COLORS.get(method, "#999")
        bars   = ax.bar(x + offset, vals, w * 0.9,
                        color=color, alpha=0.88,
                        label=METHOD_LABELS.get(method, method),
                        error_kw={"elinewidth": 1.2, "ecolor": "#555"})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([lb for _, lb in metrics_keys], fontsize=10)
    ax.set_ylim(0, None)
    ax.grid(axis="y", ls="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")


# ── 图1：质量三连 ────────────────────────────────────────────────────

def plot_quality(results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    _bar_group(ax, _method_order(results), [
        ("best_score",  "Best Score ↑"),
        ("mean_score",  "Mean Score ↑"),
        ("top3_score",  "Top-3 Score ↑"),
    ], results)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Generation Quality: Best / Mean / Top-3 Score",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#ccc")
    plt.tight_layout()
    out = os.path.join(save_dir, "fig1_quality.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 图2：成功率 / 多样性 / 二值化 ────────────────────────────────────

def plot_aux_metrics(results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    _bar_group(ax, _method_order(results), [
        ("top1_success_rate", "Success Rate ↑\n(score > 0.5)"),
        ("diversity",         "Diversity ↑"),
        ("binary_rate",       "Binary Rate ↑"),
    ], results)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("Success Rate / Structural Diversity / Binary Quality",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#ccc")
    plt.tight_layout()
    out = os.path.join(save_dir, "fig2_aux_metrics.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 图3：all_scores 小提琴图 ─────────────────────────────────────────

def plot_score_violin(results: dict, save_dir: str):
    methods = _method_order(results)
    data, labels, colors = [], [], []
    for m in methods:
        r = results[m]
        if isinstance(r, list):
            sc = [s for c in r if "error" not in c
                  for s in c.get("all_scores", [])]
        else:
            sc = r.get("all_scores", []) if "error" not in r else []
        if not sc:
            continue
        data.append(sc)
        labels.append(METHOD_LABELS.get(m, m))
        colors.append(METHOD_COLORS.get(m, "#999"))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    vp = ax.violinplot(data, positions=range(len(data)),
                       showmedians=True, showextrema=True)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color); body.set_alpha(0.75)
    vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
    for part in ("cmins", "cmaxes", "cbars"):
        vp[part].set_color("#555")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Second-Order Score", fontsize=11)
    ax.set_title("Score Distribution Across All Generated Candidates",
                 fontsize=13, fontweight="bold", pad=10)
    ax.axhline(0.5, color="#E57373", ls="--", lw=1.2, alpha=0.7,
               label="success threshold = 0.5")
    ax.legend(fontsize=9.5, framealpha=0.9)
    ax.grid(axis="y", ls="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")

    plt.tight_layout()
    out = os.path.join(save_dir, "fig3_score_violin.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 图4：推理时间 vs 质量 散点图 ─────────────────────────────────────

def plot_time_vs_quality(results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

    for method in _method_order(results):
        r = results[method]
        if not isinstance(r, list) and "error" in r:
            continue
        t   = _val(results, method, "inference_time_s")
        sc  = _val(results, method, "best_score")
        mn  = _val(results, method, "mean_score")
        color = METHOD_COLORS.get(method, "#999")
        label = METHOD_LABELS.get(method, method)
        ax.scatter(t, sc, s=130, color=color, zorder=5,
                   edgecolors="white", linewidths=1.2)
        # 误差棒：best 到 mean 的差距（越小说明一致性越好）
        ax.errorbar(t, sc, yerr=[[sc - mn], [0.0]], fmt="none",
                    ecolor=color, capsize=4, elinewidth=1.5)
        ax.annotate(label, xy=(t, sc),
                    xytext=(t * 1.06, sc + 0.01),
                    fontsize=8.5, color=color)

    ax.set_xlabel("Inference Time (s)", fontsize=11)
    ax.set_ylabel("Best Score", fontsize=11)
    ax.set_title("Quality vs. Inference Time\n"
                 "(error bar: best → mean, smaller = more consistent)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.grid(ls="--", alpha=0.35)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")
    plt.tight_layout()
    out = os.path.join(save_dir, "fig4_time_vs_quality.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 图5：spectrum_mae ──────────────────────────────────────────────────

def plot_spectrum_mae(results: dict, save_dir: str):
    methods = _method_order(results)
    means, labels, colors = [], [], []
    for m in methods:
        v = _val(results, m, "spectrum_mae", default=float("nan"))
        if not np.isfinite(v):
            continue
        means.append(v)
        labels.append(METHOD_LABELS.get(m, m))
        colors.append(METHOD_COLORS.get(m, "#999"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.bar(labels, means, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_ylabel("Spectrum MAE ↓", fontsize=11)
    ax.set_title("Best-Candidate Spectrum Error vs. Target",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="y", ls="--", alpha=0.35)
    ax.set_ylim(0, None)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7); sp.set_color("#aaa")
    plt.tight_layout()
    out = os.path.join(save_dir, "fig5_spectrum_mae.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 主程序 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  default="samples/eval_compare/results.json")
    parser.add_argument("--save_dir", default="samples/eval_compare/figures")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    results = load_results(args.results)

    print(f"[plot] methods: {list(results.keys())}")
    plot_quality(results, args.save_dir)
    plot_aux_metrics(results, args.save_dir)
    plot_score_violin(results, args.save_dir)
    plot_time_vs_quality(results, args.save_dir)
    plot_spectrum_mae(results, args.save_dir)
    print("[plot] all done.")


if __name__ == "__main__":
    main()
