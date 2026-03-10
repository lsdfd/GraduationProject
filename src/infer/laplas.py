# -*- coding: utf-8 -*-
"""Construct 1250nm second-order targets and run diffusion inference."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model.diffusion import GaussianDiffusion  # noqa: E402
from model.models import ConditionalUNet, ForwardSurrogate  # noqa: E402
from infer.common import (  # noqa: E402
    build_weight,
    denormalize_with_stats,
    lambda_theta_grid,
    load_model,
    load_stats,
    normalize_with_stats,
    plot_map,
    plot_structure,
    rcwa_eval_1250,
    second_order_score_row,
    second_order_target,
)

TARGET_SWEEP = [
    {"name": "floor_0p01_off_0p90", "center_floor": 0.01, "off_target": 0.90},
    {"name": "floor_0p03_off_0p90", "center_floor": 0.03, "off_target": 0.90},
    {"name": "floor_0p05_off_0p90", "center_floor": 0.05, "off_target": 0.90},
]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def build_target(cond_ch: int, center_floor=0.02, off_target=0.90, edge_target=1.0):
    lambdas, thetas = lambda_theta_grid()
    kx2 = second_order_target(thetas)
    profile = (center_floor + (edge_target - center_floor) * kx2).astype(np.float32)
    tpp = np.full((len(lambdas), len(thetas)), off_target, dtype=np.float32)
    tpp[np.argmin(np.abs(lambdas - 1250.0))] = profile
    if cond_ch == 2:
        return np.stack([tpp, tpp.copy()], axis=0), lambdas, thetas
    return tpp[None], lambdas, thetas


def compute_second_order_metrics(pred_raw: np.ndarray, err: torch.Tensor, lambdas: np.ndarray, thetas: np.ndarray) -> tuple[list[dict], np.ndarray]:
    tpp_maps = pred_raw[:, 0].astype(np.float64)
    lam_idx = int(np.argmin(np.abs(lambdas - 1250.0)))
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    metrics = []
    for i in range(tpp_maps.shape[0]):
        s = second_order_score_row(tpp_maps[i, lam_idx], thetas)
        metrics.append(
            {
                "sample_idx": int(i),
                "mse_norm": float(err[i].item()),
                "second_order_score": float(s["score"]),
                "center_score": float(s["center"]),
                "shape_score": float(s["shape"]),
                "edge_score": float(s["edge"]),
                "r2": float(s["r2"]),
                "tpp_at_40": float(tpp_maps[i, lam_idx, t40_idx]),
            }
        )
    rank = np.array(sorted(range(len(metrics)), key=lambda i: metrics[i]["second_order_score"], reverse=True), dtype=np.int32)
    return metrics, rank


def plot_ranked_samples(out_png: Path, samples: np.ndarray, tpp_maps: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray, ranked_idx: np.ndarray, metrics: list[dict]) -> None:
    k = len(ranked_idx)
    if k == 0:
        return
    finite = tpp_maps[np.isfinite(tpp_maps)]
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    lam_idx = int(np.argmin(np.abs(lambdas - 1250.0)))
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    target = second_order_target(thetas)

    fig, axes = plt.subplots(k, 3, figsize=(13.5, max(2.7 * k, 5.0)), gridspec_kw={"width_ratios": [0.75, 1.1, 1.0]}, constrained_layout=True)
    axes = np.atleast_2d(axes)
    hm = None
    for r, i in enumerate(ranked_idx):
        a_struct, a_hm, a_curve = axes[r]
        m = metrics[int(i)]
        a_struct.imshow(samples[int(i), 0], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        a_struct.set_title(f"id={int(i)}", fontsize=10)
        a_struct.axis("off")

        hm = a_hm.imshow(tpp_maps[int(i)], cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, interpolation="bicubic")
        a_hm.axhline(float(lambdas[lam_idx]), color="w", ls="--", lw=1.0)
        a_hm.set_title(f"2nd={m['second_order_score']:.3f} | mse={m['mse_norm']:.4f}", fontsize=10)
        a_hm.set_xlabel("theta (deg)")
        a_hm.set_ylabel("lambda (nm)")

        y = tpp_maps[int(i), lam_idx].astype(np.float64)
        yn = y / max(float(np.max(y)), 1e-8)
        a_curve.plot(thetas, target, "k--", lw=1.7, label="target ~ |sin(theta)|^2")
        a_curve.plot(thetas, yn, lw=1.9, color="#1f77b4", label="candidate (normalized)")
        a_curve.set_ylim(-0.05, 1.05)
        a_curve.set_xlabel("theta (deg)")
        a_curve.set_ylabel("normalized |tpp|")
        a_curve.grid(alpha=0.25)
        if r == 0:
            a_curve.legend(fontsize=8, loc="lower right")
        tag = f"|tpp|@{float(thetas[t40_idx]):.1f}deg={m['tpp_at_40']:.3f}"
        a_hm.text(0.98, 0.03, tag, transform=a_hm.transAxes, ha="right", va="bottom", fontsize=8, color="white", bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"})
        a_curve.text(0.02, 0.03, tag, transform=a_curve.transAxes, ha="left", va="bottom", fontsize=8)

    fig.suptitle(f"Top-{k} by second-order score @1250nm", fontsize=12)
    if hm is not None:
        fig.colorbar(hm, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp_mag|")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_case(case: dict, args, mean: np.ndarray, std: np.ndarray, cond_ch: int, weight: torch.Tensor, surrogate: torch.nn.Module, diffusion: torch.nn.Module, root_save_dir: Path):
    save_dir = root_save_dir / case["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    target_raw, lambdas, thetas = build_target(cond_ch, center_floor=float(case["center_floor"]), off_target=float(case["off_target"]), edge_target=1.0)
    target = normalize_with_stats(target_raw, mean, std, args.device)
    cond_batch = target.repeat(args.num_samples, 1, 1, 1)
    samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)
    pred = surrogate(samples)
    err = ((pred - cond_batch).abs() * weight).sum(dim=(1, 2, 3)) / weight.sum()
    topk = torch.topk(err, k=min(5, args.num_samples), largest=False).indices
    best = samples[topk[0]:topk[0] + 1]

    pred_raw = denormalize_with_stats(pred.cpu().numpy(), mean, std)
    metrics, rank_second = compute_second_order_metrics(pred_raw, err, lambdas, thetas)
    topk_second = rank_second[: min(max(1, args.topk_second), len(rank_second))]
    topk_second_t = torch.from_numpy(topk_second).to(samples.device, dtype=torch.long)

    info = {
        "sweep_case": case["name"],
        "center_floor": float(case["center_floor"]),
        "off_target": float(case["off_target"]),
        "edge_target": 1.0,
        "best_surrogate_mae_norm": float(err[topk[0]].item()),
        "best_second_order_sample_idx": int(topk_second[0]),
        "best_second_order_score": float(metrics[int(topk_second[0])]["second_order_score"]),
        "best_second_order_tpp_at_40": float(metrics[int(topk_second[0])]["tpp_at_40"]),
    }

    rcwa_pred = None
    if args.rcwa_eval:
        out = rcwa_eval_1250(best, target_raw, cond_ch, args.device)
        if out is not None:
            info["best_rcwa_mae_raw"], rcwa_pred = out
            np.save(save_dir / "best_rcwa_pred.npy", rcwa_pred)

    np.save(save_dir / "target_cond_raw.npy", target_raw)
    np.save(save_dir / "target_cond_norm.npy", target.cpu().numpy())
    np.save(save_dir / "target_weight.npy", weight.cpu().numpy())
    np.save(save_dir / "lambdas.npy", lambdas)
    np.save(save_dir / "thetas.npy", thetas)
    np.save(save_dir / "all_samples.npy", samples.cpu().numpy())
    np.save(save_dir / "all_pred_cond.npy", pred.cpu().numpy())
    np.save(save_dir / "all_pred_cond_raw.npy", pred_raw)
    np.save(save_dir / "all_errors.npy", err.cpu().numpy())
    np.save(save_dir / "rank_second_order.npy", topk_second)
    np.save(save_dir / "topk_indices.npy", topk.cpu().numpy())
    np.save(save_dir / "topk_samples.npy", samples[topk].cpu().numpy())
    np.save(save_dir / "topk_pred_cond.npy", pred[topk].cpu().numpy())
    np.save(save_dir / "topk_pred_cond_raw.npy", pred_raw[topk.cpu().numpy()])
    np.save(save_dir / "topk_second_samples.npy", samples[topk_second_t].cpu().numpy())
    np.save(save_dir / "topk_second_pred_cond_raw.npy", pred_raw[topk_second])

    vmax_tpp = max(float(target_raw[0].max()), float(pred_raw[topk[0].item(), 0].max()), 1e-6)
    plot_map(save_dir / "target_tpp.png", target_raw, lambdas, thetas, "Target tpp_mag", vmax_tpp, channel_idx=0)
    plot_map(save_dir / "best_tpp.png", pred_raw[topk[0].item()], lambdas, thetas, "Best surrogate tpp_mag", vmax_tpp, channel_idx=0)
    if cond_ch == 2:
        vmax_tss = max(float(target_raw[1].max()), float(pred_raw[topk[0].item(), 1].max()), 1e-6)
        plot_map(save_dir / "target_tss.png", target_raw, lambdas, thetas, "Target tss_mag", vmax_tss, channel_idx=1)
        plot_map(save_dir / "best_tss.png", pred_raw[topk[0].item()], lambdas, thetas, "Best surrogate tss_mag", vmax_tss, channel_idx=1)
    plot_structure(save_dir / "best_structure.png", best.cpu().numpy(), "Best binary structure")

    if rcwa_pred is not None:
        row_tpp = np.zeros((1, len(thetas)), dtype=np.float32)
        row_tpp[0] = rcwa_pred[0]
        vmax_tpp = max(vmax_tpp, float(row_tpp.max()))
        plot_map(save_dir / "best_rcwa_tpp.png", row_tpp, np.array([1250.0], dtype=np.float32), thetas, "Best RCWA tpp_mag @1250nm", vmax_tpp)
        if cond_ch == 2:
            row_tss = np.zeros((1, len(thetas)), dtype=np.float32)
            row_tss[0] = rcwa_pred[1]
            vmax_tss = max(vmax_tss, float(row_tss.max()))
            plot_map(save_dir / "best_rcwa_tss.png", row_tss, np.array([1250.0], dtype=np.float32), thetas, "Best RCWA tss_mag @1250nm", vmax_tss)

    with (save_dir / "second_order_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_ranked_samples(save_dir / f"top{len(topk_second)}_second_order.png", samples.cpu().numpy(), pred_raw[:, 0].astype(np.float32), lambdas, thetas, topk_second, metrics)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("saved_to:", save_dir)
    print(info)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Run diffusion inference with swept second-order targets at 1250nm.")
    p.add_argument("--stats", default=str(ROOT / "checkpoints" / "cond_stats.npz"))
    p.add_argument("--forward_ckpt", default=str(ROOT / "checkpoints" / "forward_best.pt"))
    p.add_argument("--diffusion_ckpt", default=str(ROOT / "checkpoints" / "diffusion_best.pt"))
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "laplas"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rcwa_eval", action="store_true")
    p.add_argument("--topk_second", type=int, default=5)
    args = p.parse_args()

    args.stats = str(resolve_from_root(args.stats))
    args.forward_ckpt = str(resolve_from_root(args.forward_ckpt))
    args.diffusion_ckpt = str(resolve_from_root(args.diffusion_ckpt))
    args.save_dir = str(resolve_from_root(args.save_dir))

    root_save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root_save_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_stats(args.stats)
    cond_ch = int(mean.shape[1])
    weight = torch.from_numpy(build_weight(cond_ch)[None]).to(args.device)
    surrogate = load_model(args.forward_ckpt, ForwardSurrogate(cond_ch).to(args.device), "model", args.device)
    diffusion = load_model(
        args.diffusion_ckpt,
        GaussianDiffusion(ConditionalUNet(cond_ch).to(args.device), timesteps=1000, image_size=64).to(args.device),
        "diffusion",
        args.device,
    )

    for case in TARGET_SWEEP:
        run_case(case, args, mean, std, cond_ch, weight, surrogate, diffusion, root_save_dir)


if __name__ == "__main__":
    main()
