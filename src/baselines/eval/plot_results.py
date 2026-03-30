"""
读取 run_eval.py 输出的 results.json，绘制对比图。

生成四张图：
  1. 质量三连柱状图（best_score / mean_score / top3_score）
  2. all_scores 小提琴图（展示候选分数分布形状）
  3. spectrum_mae 柱状图
  4. 各方法 top-1 候选在目标波长下的 theta 响应曲线

用法：
  cd /data/GraduationProject
  python src/baselines/eval/plot_results.py \
      --results samples/eval_compare/results.json \
      --save_dir samples/eval_compare/figures
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ── 配色：当前 baseline 方法集 ───────────────────────────────────────
METHOD_COLORS = {
    "topo_opt"        : "#78909C",   # 灰（拓扑优化）
    "cvae"            : "#42A5F5",   # 蓝
    "cgan"            : "#AB47BC",   # 紫
    "diffusion"       : "#FFA726",   # 橙（扩散无引导）
}
METHOD_LABELS = {
    "topo_opt"        : "Topo-Opt",
    "cvae"            : "CVAE",
    "cgan"            : "cGAN",
    "diffusion"       : "Diffusion",
}
BG = "#FAFAFA"
PREFERRED_ORDER = ["topo_opt", "cvae", "cgan", "diffusion"]


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_default_results_path(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path

    root = Path("samples") / "eval_compare"
    candidates: list[Path] = []
    if root.exists():
        candidates.extend(p for p in root.rglob("results.json") if p.is_file())

    fallback = root / "results.json"
    if fallback.exists():
        candidates.append(fallback)
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)
    return str(fallback)


def resolve_default_save_dir(explicit_dir: str | None = None, results_path: str | None = None) -> str:
    if explicit_dir:
        return explicit_dir

    base = Path(results_path).resolve().parent if results_path else (Path("samples") / "eval_compare")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(base / f"figures_{stamp}")


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


def _sorted_methods(results: dict, keys: list[str], descending: bool = False) -> list[str]:
    methods = [m for m in _method_order(results) if not (isinstance(results[m], dict) and "error" in results[m])]
    return sorted(
        methods,
        key=lambda m: float(np.mean([_val(results, m, k, default=0.0) for k in keys])),
        reverse=descending,
    )


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
    methods = _sorted_methods(results, ["best_score", "mean_score", "top3_score"], descending=False)
    _bar_group(ax, methods, [
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

# ── 图2：all_scores 小提琴图 ─────────────────────────────────────────

def plot_score_violin(results: dict, save_dir: str):
    methods = _sorted_methods(results, ["mean_score"], descending=False)
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
    out = os.path.join(save_dir, "fig2_score_violin.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 图3：spectrum_mae ──────────────────────────────────────────────────

def plot_spectrum_mae(results: dict, save_dir: str):
    methods = _sorted_methods(results, ["spectrum_mae"], descending=False)
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
    out = os.path.join(save_dir, "fig3_spectrum_mae.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 图4：各方法 top-1 候选在目标波长下的 theta 响应曲线 ───────────────

def plot_top1_theta_curves(results: dict, save_dir: str):
    methods = _sorted_methods(results, ["best_score"], descending=False)
    usable = [m for m in methods if not (isinstance(results[m], list) or "error" in results[m])]
    if not usable:
        return

    ref = results[usable[0]]
    thetas = np.asarray(ref.get("thetas_deg", []), dtype=np.float32)
    target = np.asarray(ref.get("target_theta_curve", []), dtype=np.float32)
    if thetas.size == 0 or target.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(BG)
        ax.plot(thetas, target, "k--", lw=2.0, label="Target ~ |sin(theta)|^2")
        ax.grid(ls="--", alpha=0.35)
        ax.set_ylim(-0.05, 1.05)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
            sp.set_color("#aaa")

    for method in usable:
        r = results[method]
        color = METHOD_COLORS.get(method, "#999")
        label = METHOD_LABELS.get(method, method)

        tpp = np.asarray(r.get("best_tpp_theta_curve", []), dtype=np.float32)
        tss = np.asarray(r.get("best_tss_theta_curve", []), dtype=np.float32)

        if tpp.size == thetas.size:
            tpp = tpp / max(float(np.max(tpp)), 1e-8)
            axes[0].plot(thetas, tpp, lw=2.0, color=color, label=label)
        if tss.size == thetas.size:
            tss = tss / max(float(np.max(tss)), 1e-8)
            axes[1].plot(thetas, tss, lw=2.0, color=color, label=label)

    target_lambda = ref.get("target_lambda_nm", "target")
    axes[0].set_title(rf"Top-1 $|t_{{pp}}|$ vs theta @ {target_lambda} nm", fontsize=12, fontweight="bold")
    axes[1].set_title(rf"Top-1 $|t_{{ss}}|$ vs theta @ {target_lambda} nm", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("theta (deg)")
    axes[1].set_xlabel("theta (deg)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[1].set_ylabel("Normalized amplitude")
    axes[0].legend(fontsize=8.5, framealpha=0.9, edgecolor="#ccc")

    out = os.path.join(save_dir, "fig4_top1_theta_curves.png")
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


def plot_summary_table(results: dict, save_dir: str):
    methods = _sorted_methods(results, ["best_score"], descending=False)
    if not methods:
        return

    metric_defs = [
        ("best_score", "Best Score", True, ".4f"),
        ("mean_score", "Mean Score", True, ".4f"),
        ("top3_score", "Top-3 Score", True, ".4f"),
        ("spectrum_mae", "Spectrum MAE", False, ".4f"),
    ]

    values = {key: np.asarray([_val(results, m, key, default=float("nan")) for m in methods], dtype=np.float64)
              for key, _, _, _ in metric_defs}

    fig_w = 10.0
    fig_h = 1.4 + 0.72 * (len(methods) + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    n_rows = len(methods) + 1
    n_cols = len(metric_defs) + 1
    col_widths = [1.9] + [1.35] * len(metric_defs)
    x_edges = np.cumsum([0.0] + col_widths)
    table_width = x_edges[-1]
    row_h = 0.88
    table_height = n_rows * row_h

    ax.set_xlim(0, table_width)
    ax.set_ylim(0, table_height)

    header_fc = "#EEF2F6"
    method_fc = "#FBFBFC"
    cell_fc = "#FFFFFF"
    border_c = "#D8DEE6"
    text_c = "#1F2933"
    muted_c = "#52606D"
    diffusion_row_fc = "#F6F9FC"
    best_fc = "#E8F1FB"

    best_mask: dict[str, np.ndarray] = {}
    for key, _, higher_is_better, _ in metric_defs:
        col = values[key]
        if not np.isfinite(col).any():
            best_mask[key] = np.zeros_like(col, dtype=bool)
            continue
        best_val = np.nanmax(col) if higher_is_better else np.nanmin(col)
        best_mask[key] = np.isclose(col, best_val, equal_nan=False)

    y_top = table_height
    headers = ["Method"] + [label for _, label, _, _ in metric_defs]
    for c, header in enumerate(headers):
        x0 = x_edges[c]
        w = col_widths[c]
        rect = plt.Rectangle((x0, y_top - row_h), w, row_h, facecolor=header_fc, edgecolor=border_c, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y_top - row_h / 2, header, ha="center", va="center",
                fontsize=11, fontweight="bold", color=text_c)

    for r, method in enumerate(methods, start=1):
        y0 = y_top - (r + 1) * row_h
        row_bg = diffusion_row_fc if method == "diffusion" else cell_fc

        rect = plt.Rectangle((x_edges[0], y0), col_widths[0], row_h, facecolor=method_fc if method != "diffusion" else diffusion_row_fc,
                             edgecolor=border_c, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x_edges[0] + 0.12, y0 + row_h / 2, METHOD_LABELS.get(method, method),
                ha="left", va="center", fontsize=10.5,
                fontweight="bold" if method == "diffusion" else "normal", color=text_c)

        for c, (key, _, higher_is_better, fmt) in enumerate(metric_defs, start=1):
            val = values[key][r - 1]
            x0 = x_edges[c]
            w = col_widths[c]
            base_fc = best_fc if best_mask[key][r - 1] else row_bg
            rect = plt.Rectangle((x0, y0), w, row_h, facecolor=base_fc, edgecolor=border_c, linewidth=1.0)
            ax.add_patch(rect)
            text = format(val, fmt) if np.isfinite(val) else "NA"
            ax.text(
                x0 + w / 2,
                y0 + row_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold" if best_mask[key][r - 1] else "normal",
                color=text_c,
            )

    # outer border
    outer = plt.Rectangle((x_edges[0], y_top - n_rows * row_h), table_width, n_rows * row_h,
                          fill=False, edgecolor="#C7D0D9", linewidth=1.2)
    ax.add_patch(outer)

    ax.text(
        0,
        table_height + 0.2,
        "Baseline Comparison Summary",
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=text_c,
    )
    ax.text(
        0,
        table_height + 0.02,
        "Methods sorted by Best Score (ascending). Best values are lightly highlighted.",
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=muted_c,
    )

    plt.tight_layout()
    out = os.path.join(save_dir, "fig5_summary_table.png")
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor=BG)
    plt.close(); print(f"saved → {out}")


# ── 主程序 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default=None,
        help="results.json path; default is the most recently modified file under samples/eval_compare",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="output directory; default creates a new timestamped folder next to results.json",
    )
    args = parser.parse_args()

    results_path = resolve_default_results_path(args.results)
    save_dir = resolve_default_save_dir(args.save_dir, results_path)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[plot] results={results_path}")
    print(f"[plot] save_dir={save_dir}")
    results = load_results(results_path)

    print(f"[plot] methods: {list(results.keys())}")
    plot_quality(results, save_dir)
    plot_score_violin(results, save_dir)
    plot_spectrum_mae(results, save_dir)
    plot_top1_theta_curves(results, save_dir)
    plot_summary_table(results, save_dir)
    print("[plot] all done.")


if __name__ == "__main__":
    main()
