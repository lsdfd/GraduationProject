from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one sample's raw tpp grid and a value table without interpolation."
    )
    parser.add_argument("--npz", type=str, default="data/train_data.npz")
    parser.add_argument("--sample_idx", type=int, default=514)
    parser.add_argument("--out_png", type=str, default="data/sample_514_tpp_raw_grid.png")
    return parser.parse_args()


def build_table_text(spec: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray) -> tuple[list[str], list[list[str]]]:
    col_labels = ["lambda/theta"] + [f"{theta:.0f}" for theta in thetas]
    rows: list[list[str]] = []
    for lam, row in zip(lambdas, spec):
        rows.append([f"{lam:.0f}"] + [f"{float(v):.4f}" for v in row])
    return col_labels, rows


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    out_png = resolve_from_root(args.out_png)

    data = np.load(npz_path)
    tpp = np.asarray(data["tpp_mag"], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float64)
    thetas = np.asarray(data["thetas"], dtype=np.float64)

    if args.sample_idx < 0 or args.sample_idx >= len(tpp):
        raise IndexError(f"sample_idx out of range: {args.sample_idx}, dataset size={len(tpp)}")

    spec = tpp[args.sample_idx]
    col_labels, table_rows = build_table_text(spec, lambdas, thetas)

    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.4])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    im = ax_map.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="turbo",
        interpolation="nearest",
        extent=[float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])],
        vmin=max(0.0, float(np.nanmin(spec))),
        vmax=min(1.0, float(np.nanmax(spec))),
    )
    ax_map.set_title(f"Sample {args.sample_idx} raw tpp grid")
    ax_map.set_xlabel("theta (deg)")
    ax_map.set_ylabel("lambda (nm)")
    ax_map.set_xticks(thetas)
    ax_map.set_yticks(lambdas)
    for i, lam in enumerate(lambdas):
        for j, theta in enumerate(thetas):
            ax_map.text(float(theta), float(lam), f"{float(spec[i, j]):.3f}", ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02, label="tpp magnitude")

    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    ax_tbl.set_title("Raw values table", pad=10)

    fig.suptitle(
        f"Raw tpp values for sample {args.sample_idx} | dataset={npz_path.name} | shape={spec.shape[0]}x{spec.shape[1]}",
        fontsize=14,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved_png: {out_png}")


if __name__ == "__main__":
    main()
