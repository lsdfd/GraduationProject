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

from infer.common import denormalize_with_stats, load_model, load_stats, second_order_score_row, second_order_target  # noqa: E402
from model.models import ForwardSurrogate  # noqa: E402
from model.train_utils import resolve_latest_run  # noqa: E402


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


def load_dataset(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    data = np.load(npz_path)
    structures = np.asarray(data["structures"], dtype=np.float32)
    tpp = np.asarray(data["tpp_mag"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    target_lambda = float(data["target_lambda"])
    if tpp.ndim != 2:
        raise ValueError(f"validate_surrogate_top 只支持 only-tpp 数据集 [N,17]，实际得到 {tpp.shape}")
    return structures, tpp, thetas, target_lambda


def plot_sample_compare(
    out_png: Path,
    structure: np.ndarray,
    real_row: np.ndarray,
    pred_row: np.ndarray,
    thetas: np.ndarray,
    sample_idx: int,
    real_score: dict,
    pred_score: dict,
    target_lambda: float,
) -> None:
    target = second_order_target(thetas)
    real_row = real_row.astype(np.float64)
    pred_row = pred_row.astype(np.float64)
    real_row_n = real_row / max(float(np.max(real_row)), 1e-8)
    pred_row_n = pred_row / max(float(np.max(pred_row)), 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [0.8, 1.2]}, constrained_layout=True)
    a0, a1 = axes
    a0.imshow(structure, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
    a0.set_title(f"id={sample_idx}")
    a0.axis("off")

    a1.plot(thetas, target, "k--", lw=1.8, label="ideal ~ |sin(theta)|^2")
    a1.plot(thetas, real_row_n, lw=2.0, color="#1f77b4", label="real (normalized)")
    a1.plot(thetas, pred_row_n, lw=2.0, color="#d62728", label="surrogate (normalized)")
    a1.set_ylim(-0.05, 1.05)
    a1.set_xlabel("theta (deg)")
    a1.set_ylabel("normalized |tpp|")
    a1.grid(alpha=0.25)
    a1.legend(fontsize=8, loc="lower right")
    a1.set_title(
        f"{target_lambda:.0f}nm row\nreal center={real_score['center']:.3f} pred center={pred_score['center']:.3f}",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    latest_forward = resolve_latest_run(ROOT / "checkpoints", "forward")
    p = argparse.ArgumentParser(description="Validate surrogate predictions on top-ranked second-order samples at a given wavelength.")
    p.add_argument("--train_npz", default=str(ROOT / "data" / "train_data.npz"))
    p.add_argument("--topk_csv", default=str(ROOT / "data" / "second_order_scores" / "tpp_mag_top5_per_lambda.csv"))
    p.add_argument("--stats", default=str((latest_forward / "cond_stats.npz") if latest_forward is not None else ""))
    p.add_argument("--forward_ckpt", default=str((latest_forward / "forward_best.pt") if latest_forward is not None else ""))
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "surrogate_validate"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if not args.stats or not args.forward_ckpt:
        raise FileNotFoundError("未找到 latest forward run；请先运行 train_forward.py 或显式传入 --stats 和 --forward_ckpt")

    args.train_npz = str(resolve_from_root(args.train_npz))
    args.topk_csv = str(resolve_from_root(args.topk_csv))
    args.stats = str(resolve_from_root(args.stats))
    args.forward_ckpt = str(resolve_from_root(args.forward_ckpt))
    args.save_dir = str(resolve_from_root(args.save_dir))

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    structures, tpp, thetas, target_lambda = load_dataset(Path(args.train_npz))
    sample_ids = load_top_sample_ids(Path(args.topk_csv), target_lambda, args.topk)
    mean, std = load_stats(args.stats)
    model = load_model(args.forward_ckpt, ForwardSurrogate(out_dim=17).to(args.device), "model", args.device)

    x = torch.from_numpy(structures[sample_ids, None]).to(args.device)
    pred = model(x)
    pred_raw = denormalize_with_stats(pred.cpu().numpy(), mean, std).astype(np.float32)

    summary: list[dict] = []
    for local_idx, sample_idx in enumerate(sample_ids):
        real_row = tpp[sample_idx]
        pred_row = pred_raw[local_idx]
        real_score = second_order_score_row(real_row, thetas)
        pred_score = second_order_score_row(pred_row, thetas)
        row_mae = float(np.mean(np.abs(pred_row - real_row)))
        item = {
            "sample_idx": int(sample_idx),
            "row_mae_target_lambda": row_mae,
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
            real_row,
            pred_row,
            thetas,
            sample_idx,
            real_score,
            pred_score,
            target_lambda,
        )

    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("saved_to:", save_dir)
    print("top_sample_ids:", sample_ids)
    print("summary:", summary)


if __name__ == "__main__":
    main()
