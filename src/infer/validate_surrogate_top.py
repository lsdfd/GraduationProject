"""Validate whether the forward surrogate preserves second-order structure on top-ranked samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from infer.common import denormalize_with_stats, lambda_theta_grid, load_model, load_stats, resolve_default_forward_ckpt, resolve_default_stats_path, second_order_score_row, second_order_target  # noqa: E402
from model.models import ForwardSurrogate  # noqa: E402


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path

def load_top_sample_ids(csv_path: Path, target_lambda: float, topk: int) -> list[int]:
    rows: list[tuple[int, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["lambda_nm"]) == float(target_lambda):
                rows.append((int(row["rank"]), int(row["sample_idx"])))
    rows.sort(key=lambda x: x[0])
    return [sample_idx for _, sample_idx in rows[:topk]]


def load_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    structures = np.asarray(data["structures"], dtype=np.float32)
    tpp = np.asarray(data["tpp_mag"], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    return structures, tpp, lambdas, thetas


def plot_sample_compare(
    out_png: Path,
    structure: np.ndarray,
    real_map: np.ndarray,
    pred_map: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    sample_idx: int,
    real_score: dict,
    pred_score: dict,
    target_lambda: float,
) -> None:
    lam_idx = int(np.argmin(np.abs(lambdas - float(target_lambda))))
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    finite = np.concatenate([real_map[np.isfinite(real_map)], pred_map[np.isfinite(pred_map)]])
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    target = second_order_target(thetas)
    real_row = real_map[lam_idx].astype(np.float64)
    pred_row = pred_map[lam_idx].astype(np.float64)
    real_row_n = real_row / max(float(np.max(real_row)), 1e-8)
    pred_row_n = pred_row / max(float(np.max(pred_row)), 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), gridspec_kw={"width_ratios": [0.8, 1.05, 1.05, 1.15]}, constrained_layout=True)
    a0, a1, a2, a3 = axes
    a0.imshow(structure, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
    a0.set_title(f"id={sample_idx}")
    a0.axis("off")

    hm1 = a1.imshow(real_map, cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, interpolation="bicubic")
    a1.axhline(float(lambdas[lam_idx]), color="w", ls="--", lw=1.0)
    a1.set_title(f"real score={real_score['score']:.3f}")
    a1.set_xlabel("theta (deg)")
    a1.set_ylabel("lambda (nm)")

    hm2 = a2.imshow(pred_map, cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, interpolation="bicubic")
    a2.axhline(float(lambdas[lam_idx]), color="w", ls="--", lw=1.0)
    a2.set_title(f"pred score={pred_score['score']:.3f}")
    a2.set_xlabel("theta (deg)")
    a2.set_ylabel("lambda (nm)")

    a3.plot(thetas, target, "k--", lw=1.8, label="ideal ~ |sin(theta)|^2")
    a3.plot(thetas, real_row_n, lw=2.0, color="#1f77b4", label="real (normalized)")
    a3.plot(thetas, pred_row_n, lw=2.0, color="#d62728", label="surrogate (normalized)")
    a3.set_ylim(-0.05, 1.05)
    a3.set_xlabel("theta (deg)")
    a3.set_ylabel("normalized |tpp|")
    a3.grid(alpha=0.25)
    a3.legend(fontsize=8, loc="lower right")
    a3.set_title(
        f"{target_lambda:.0f}nm row\nreal center={real_score['center']:.3f} pred center={pred_score['center']:.3f}",
        fontsize=10,
    )

    fig.colorbar(hm2, ax=[a1, a2], shrink=0.92, pad=0.02, label="|tpp_mag|")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Validate surrogate predictions on top-ranked second-order samples at a given wavelength.")
    p.add_argument("--train_npz", default=str(ROOT / "data" / "train_data.npz"))
    p.add_argument("--topk_csv", default=str(ROOT / "data" / "second_order_scores" / "tpp_mag_top5_per_lambda.csv"))
    p.add_argument("--stats", default=str(resolve_default_stats_path(ROOT)))
    p.add_argument("--forward_ckpt", default=str(resolve_default_forward_ckpt(ROOT)))
    p.add_argument("--target_lambda", type=float, default=1000.0)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "surrogate_validate"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.train_npz = str(resolve_from_root(args.train_npz))
    args.topk_csv = str(resolve_from_root(args.topk_csv))
    args.stats = str(resolve_from_root(args.stats))
    args.forward_ckpt = str(resolve_from_root(args.forward_ckpt))
    args.save_dir = str(resolve_from_root(args.save_dir))

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    sample_ids = load_top_sample_ids(Path(args.topk_csv), args.target_lambda, args.topk)
    structures, tpp, lambdas, thetas = load_dataset(Path(args.train_npz))
    mean, std = load_stats(args.stats)
    cond_ch = int(mean.shape[1])
    model = load_model(args.forward_ckpt, ForwardSurrogate(cond_ch).to(args.device), "model", args.device)

    x = torch.from_numpy(structures[sample_ids, None]).to(args.device)
    pred = model(x)
    pred_raw = denormalize_with_stats(pred.cpu().numpy(), mean, std)
    pred_tpp = pred_raw[:, 0].astype(np.float32)

    summary: list[dict] = []
    lam_idx = int(np.argmin(np.abs(lambdas - args.target_lambda)))
    for local_idx, sample_idx in enumerate(sample_ids):
        real_map = tpp[sample_idx]
        pred_map = pred_tpp[local_idx]
        real_score = second_order_score_row(real_map[lam_idx], thetas)
        pred_score = second_order_score_row(pred_map[lam_idx], thetas)
        row_mae = float(np.mean(np.abs(pred_map[lam_idx] - real_map[lam_idx])))
        full_mae = float(np.mean(np.abs(pred_map - real_map)))
        item = {
            "sample_idx": int(sample_idx),
            "row_mae_target_lambda": row_mae,
            "full_mae": full_mae,
            "real_second_order_score": float(real_score["score"]),
            "pred_second_order_score": float(pred_score["score"]),
            "real_center_score": float(real_score["center"]),
            "pred_center_score": float(pred_score["center"]),
            "real_shape_score": float(real_score["shape"]),
            "pred_shape_score": float(pred_score["shape"]),
            "real_edge_score": float(real_score["edge"]),
            "pred_edge_score": float(pred_score["edge"]),
        }
        summary.append(item)
        plot_sample_compare(
            save_dir / f"sample_{sample_idx}.png",
            structures[sample_idx],
            real_map,
            pred_map,
            lambdas,
            thetas,
            sample_idx,
            real_score,
            pred_score,
            args.target_lambda,
        )

    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("saved_to:", save_dir)
    print("top_sample_ids:", sample_ids)
    print("summary:", summary)


if __name__ == "__main__":
    main()
