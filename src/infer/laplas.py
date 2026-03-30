# -*- coding: utf-8 -*-
"""Construct 1000nm second-order targets and run diffusion inference."""

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

from model.diffusion import GaussianDiffusion  # noqa: E402
from model.models import ConditionalUNet  # noqa: E402
from infer.common import (  # noqa: E402
    lambda_theta_grid,
    load_model,
    load_stats,
    normalize_with_stats,
    plot_map,
    plot_structure,
    rcwa_eval_full_map,
    rcwa_eval_target_lambda,
    resolve_default_diffusion_ckpt,
    resolve_default_stats_path,
    second_order_score_map,
    second_order_score_row,
    second_order_target,
)

TARGET_SWEEP = [
    {"name": "floor_0p00_off_0p90", "center_floor": 0.00, "off_target": 0.90},
]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def smooth_lambda_axis(spec_map: np.ndarray) -> np.ndarray:
    if spec_map.shape[0] < 3:
        return spec_map
    kernel = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    out = spec_map.copy()
    for i in range(1, spec_map.shape[0] - 1):
        out[i] = kernel[0] * spec_map[i - 1] + kernel[1] * spec_map[i] + kernel[2] * spec_map[i + 1]
    return out


def load_template_spectrum(
    cond_ch: int,
    train_npz_path: Path,
    topk_csv_path: Path,
    target_lambda: float = 1000.0,
    sample_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(train_npz_path)
    if "tpp_mag" not in data.files:
        raise ValueError(f"tpp_mag not found in {train_npz_path}")
    if cond_ch == 2 and "tss_mag" not in data.files:
        raise ValueError(f"tss_mag not found in {train_npz_path}")

    lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    tpp = np.asarray(data["tpp_mag"], dtype=np.float32)
    tss = np.asarray(data["tss_mag"], dtype=np.float32) if cond_ch == 2 else None
    if sample_idx is None:
        with topk_csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row["lambda_nm"]) == float(target_lambda) and int(row["rank"]) == 1:
                    sample_idx = int(row["sample_idx"])
                    break
    if sample_idx is None:
        raise ValueError(f"rank-1 sample at {target_lambda} nm not found in {topk_csv_path}")
    if sample_idx < 0 or sample_idx >= len(tpp):
        raise ValueError(f"sample_idx out of range: {sample_idx}")

    template_tpp = tpp[sample_idx].copy()
    if cond_ch == 2:
        template = np.stack([template_tpp, tss[sample_idx].copy()], axis=0)
    else:
        template = template_tpp[None]
    return template.astype(np.float32), lambdas, thetas, sample_idx


def build_target(
    cond_ch: int,
    center_floor=0.00,
    off_target=0.90,
    edge_target=1.0,
    band_sigma_nm=55.0,
    target_lambda: float = 1000.0,
    train_npz_path: Path | None = None,
    topk_csv_path: Path | None = None,
    sample_idx: int | None = None,
):
    lambdas, thetas = lambda_theta_grid()
    kx2 = second_order_target(thetas)
    template_idx = None
    if train_npz_path is not None and topk_csv_path is not None and train_npz_path.exists() and topk_csv_path.exists():
        target, lambdas, thetas, template_idx = load_template_spectrum(
            cond_ch,
            train_npz_path,
            topk_csv_path,
            target_lambda=target_lambda,
            sample_idx=sample_idx,
        )
        tpp = target[0].copy()
    else:
        tpp = np.empty((len(lambdas), len(thetas)), dtype=np.float32)

    # Build a smooth wavelength envelope so the target-lambda second-order row
    # gradually relaxes into the background/template spectrum instead of
    # switching abruptly to a flat constant map.
    lam_dist = lambdas.astype(np.float64) - float(target_lambda)
    lam_weight = np.exp(-0.5 * (lam_dist / max(float(band_sigma_nm), 1e-6)) ** 2)

    for i, w in enumerate(lam_weight):
        target_row = center_floor * (1.0 - kx2) + edge_target * kx2
        if template_idx is not None:
            base_row = tpp[i].astype(np.float64)
            row = (1.0 - w) * base_row + w * target_row
            row = np.clip(row, 0.0, max(edge_target, off_target))
            tpp[i] = row.astype(np.float32)
        else:
            row_floor = off_target * (1.0 - w)
            row_edge = off_target + (edge_target - off_target) * w
            tpp[i] = (row_floor + (row_edge - row_floor) * kx2).astype(np.float32)

    # Add one more wavelength-wise smoothing pass so neighboring lambda rows
    # transition naturally instead of changing too abruptly from row to row.
    tpp = smooth_lambda_axis(tpp)

    if cond_ch == 2:
        return np.stack([tpp, tpp.copy()], axis=0), lambdas, thetas, template_idx
    return tpp[None], lambdas, thetas, template_idx


def compute_second_order_metrics(
    pred_raw: np.ndarray,
    err: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    target_lambda: float,
) -> tuple[list[dict], np.ndarray]:
    tpp_maps = pred_raw[:, 0].astype(np.float64)
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    metrics = []
    for i in range(tpp_maps.shape[0]):
        s = second_order_score_map(tpp_maps[i], lambdas, thetas, target_lambda=target_lambda)
        metrics.append(
            {
                "sample_idx": int(i),
                "rcwa_mae_raw": float(err[i]),
                "second_order_score": float(s["score"]),
                "main_second_order_score": float(s["main_score"]),
                "bandwidth_second_order_score": float(s["bandwidth_score"]),
                "center_score": float(s["center"]),
                "shape_score": float(s["shape"]),
                "edge_score": float(s["edge"]),
                "r2": float(s["r2"]),
                "tpp_at_40": float(tpp_maps[i, np.argmin(np.abs(lambdas - float(target_lambda))), t40_idx]),
            }
        )
    rank = np.array(sorted(range(len(metrics)), key=lambda i: metrics[i]["second_order_score"], reverse=True), dtype=np.int32)
    return metrics, rank


def compute_second_order_metrics_rows(
    pred_rows: np.ndarray,
    err: np.ndarray,
    thetas: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    metrics = []
    for i in range(pred_rows.shape[0]):
        s = second_order_score_row(pred_rows[i, 0], thetas)
        metrics.append(
            {
                "sample_idx": int(i),
                "rcwa_mae_raw": float(err[i]),
                "second_order_score": float(s["score"]),
                "main_second_order_score": float(s["score"]),
                "bandwidth_second_order_score": float(s["score"]),
                "center_score": float(s["center"]),
                "shape_score": float(s["shape"]),
                "edge_score": float(s["edge"]),
                "r2": float(s["r2"]),
                "tpp_at_40": float(pred_rows[i, 0, t40_idx]),
            }
        )
    rank = np.array(sorted(range(len(metrics)), key=lambda i: metrics[i]["second_order_score"], reverse=True), dtype=np.int32)
    return metrics, rank


def plot_ranked_samples(
    out_png: Path,
    samples: np.ndarray,
    tpp_maps: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    ranked_idx: np.ndarray,
    metrics: list[dict],
    target_lambda: float,
) -> None:
    k = len(ranked_idx)
    if k == 0:
        return
    finite = tpp_maps[np.isfinite(tpp_maps)]
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    lam_idx = int(np.argmin(np.abs(lambdas - float(target_lambda))))
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
        a_hm.set_title(f"2nd={m['second_order_score']:.3f} | mae={m['rcwa_mae_raw']:.4f}", fontsize=10)
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

    fig.suptitle(f"Top-{k} by second-order score @{target_lambda:.0f}nm", fontsize=12)
    if hm is not None:
        fig.colorbar(hm, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp_mag|")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_target_curve(
    path: Path,
    target_raw: np.ndarray,
    thetas: np.ndarray,
    lambdas: np.ndarray,
    title: str,
    target_lambda: float,
) -> None:
    lam_idx = int(np.argmin(np.abs(lambdas - float(target_lambda))))
    target = second_order_target(thetas)
    y = target_raw[0, lam_idx].astype(np.float64)
    yn = y / max(float(np.max(y)), 1e-8)
    plt.figure(figsize=(5, 3.6))
    plt.plot(thetas, target, "k--", lw=1.7, label="ideal ~ |sin(theta)|^2")
    plt.plot(thetas, yn, lw=1.9, color="#d62728", label="laplas target (normalized)")
    plt.xlabel("theta (deg)")
    plt.ylabel("normalized |tpp|")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_case(case: dict, args, mean: np.ndarray, std: np.ndarray, cond_ch: int, weight: torch.Tensor, diffusion: torch.nn.Module, root_save_dir: Path):
    save_dir = root_save_dir / case["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    train_npz_path = resolve_from_root(args.train_npz)
    topk_csv_path = resolve_from_root(args.topk_csv)
    if args.raw_target_only:
        target_raw, lambdas, thetas, template_idx = load_template_spectrum(
            cond_ch,
            train_npz_path,
            topk_csv_path,
            target_lambda=args.target_lambda,
            sample_idx=args.target_sample_idx,
        )
    else:
        target_raw, lambdas, thetas, template_idx = build_target(
            cond_ch,
            center_floor=float(case["center_floor"]),
            off_target=float(case["off_target"]),
            edge_target=1.0,
            target_lambda=args.target_lambda,
            train_npz_path=train_npz_path,
            topk_csv_path=topk_csv_path,
            sample_idx=args.target_sample_idx,
        )
    target = normalize_with_stats(target_raw, mean, std, args.device)
    cond_batch = target.repeat(args.num_samples, 1, 1, 1)
    samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)
    rcwa_maps = []
    err = []
    target_row = target_raw[:, int(np.argmin(np.abs(lambdas - float(args.target_lambda))))]
    for idx in range(samples.shape[0]):
        print(f"[laplas-rcwa] sample {idx + 1}/{samples.shape[0]}", flush=True)
        if args.eval_mode == "target_only":
            out = rcwa_eval_target_lambda(
                samples[idx: idx + 1],
                target_raw,
                cond_ch,
                args.device,
                target_lambda=args.target_lambda,
                rcwa_orders=args.rcwa_orders,
            )
            if out is None:
                raise RuntimeError("RCWA backend unavailable during laplas evaluation.")
            mae, rcwa_row = out
            rcwa_maps.append(rcwa_row)
            err.append(float(mae))
        else:
            rcwa_map = rcwa_eval_full_map(
                samples[idx: idx + 1],
                cond_ch,
                args.device,
                rcwa_orders=args.rcwa_orders,
            )
            if rcwa_map is None:
                raise RuntimeError("RCWA backend unavailable during laplas evaluation.")
            rcwa_maps.append(rcwa_map)
            err.append(float(np.mean(np.abs(rcwa_map - target_raw))))

    pred_raw = np.stack(rcwa_maps, axis=0).astype(np.float32)
    err = np.asarray(err, dtype=np.float32)
    if args.eval_mode == "target_only":
        metrics, rank_second = compute_second_order_metrics_rows(pred_raw, err, thetas)
    else:
        metrics, rank_second = compute_second_order_metrics(pred_raw, err, lambdas, thetas, args.target_lambda)
    topk_second = rank_second[: min(max(1, args.topk_second), len(rank_second))]
    topk_second_t = torch.from_numpy(topk_second).to(samples.device, dtype=torch.long)
    topk = topk_second_t
    best_idx = int(topk_second[0])
    best = samples[best_idx:best_idx + 1]

    info = {
        "sweep_case": case["name"],
        "center_floor": float(case["center_floor"]),
        "off_target": float(case["off_target"]),
        "edge_target": 1.0,
        "target_mode": "raw_dataset_sample" if args.raw_target_only else "laplas_shaped",
        "target_lambda_nm": float(args.target_lambda),
        "template_sample_idx": None if template_idx is None else int(template_idx),
        "best_rcwa_mae_raw": float(err[best_idx]),
        "best_second_order_sample_idx": best_idx,
        "best_second_order_score": float(metrics[best_idx]["second_order_score"]),
        "best_second_order_tpp_at_40": float(metrics[best_idx]["tpp_at_40"]),
    }

    np.save(save_dir / "target_cond_raw.npy", target_raw)
    np.save(save_dir / "target_cond_norm.npy", target.cpu().numpy())
    np.save(save_dir / "target_weight.npy", weight.cpu().numpy())
    np.save(save_dir / "lambdas.npy", lambdas)
    np.save(save_dir / "thetas.npy", thetas)
    np.save(save_dir / "all_samples.npy", samples.cpu().numpy())
    np.save(save_dir / "all_pred_cond.npy", pred_raw)
    np.save(save_dir / "all_pred_cond_raw.npy", pred_raw)
    np.save(save_dir / "all_errors.npy", err)
    np.save(save_dir / "rank_second_order.npy", topk_second)
    np.save(save_dir / "topk_indices.npy", topk.cpu().numpy())
    np.save(save_dir / "topk_samples.npy", samples[topk].cpu().numpy())
    np.save(save_dir / "topk_pred_cond.npy", pred_raw[topk.cpu().numpy()])
    np.save(save_dir / "topk_pred_cond_raw.npy", pred_raw[topk.cpu().numpy()])
    np.save(save_dir / "topk_second_samples.npy", samples[topk_second_t].cpu().numpy())
    np.save(save_dir / "topk_second_pred_cond_raw.npy", pred_raw[topk_second])
    if args.eval_mode == "target_only":
        np.save(save_dir / "target_row_raw.npy", target_row)
        np.save(save_dir / "best_row_raw.npy", pred_raw[best_idx])
    else:
        vmax_tpp = max(float(target_raw[0].max()), float(pred_raw[best_idx, 0].max()), 1e-6)
        plot_map(save_dir / "target_tpp.png", target_raw, lambdas, thetas, "Target tpp_mag", vmax_tpp, channel_idx=0)
        plot_target_curve(
            save_dir / "target_second_order_curve.png",
            target_raw,
            thetas,
            lambdas,
            f"Target {args.target_lambda:.0f}nm second-order curve",
            args.target_lambda,
        )
        plot_map(save_dir / "best_tpp.png", pred_raw[best_idx], lambdas, thetas, "Best RCWA tpp_mag", vmax_tpp, channel_idx=0)
        if cond_ch == 2:
            vmax_tss = max(float(target_raw[1].max()), float(pred_raw[best_idx, 1].max()), 1e-6)
            plot_map(save_dir / "target_tss.png", target_raw, lambdas, thetas, "Target tss_mag", vmax_tss, channel_idx=1)
            plot_map(save_dir / "best_tss.png", pred_raw[best_idx], lambdas, thetas, "Best RCWA tss_mag", vmax_tss, channel_idx=1)
    plot_structure(save_dir / "best_structure.png", best.cpu().numpy(), "Best binary structure")

    with (save_dir / "second_order_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    if args.eval_mode != "target_only":
        plot_ranked_samples(
            save_dir / f"top{len(topk_second)}_second_order.png",
            samples.cpu().numpy(),
            pred_raw[:, 0].astype(np.float32),
            lambdas,
            thetas,
            topk_second,
            metrics,
            args.target_lambda,
        )
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("saved_to:", save_dir)
    print(info)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="Run diffusion inference with swept second-order targets at 1000nm.")
    p.add_argument("--stats", default=None)
    p.add_argument("--diffusion_ckpt", default=None)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "laplas"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train_npz", default=str(ROOT / "data" / "train_data.npz"))
    p.add_argument("--topk_csv", default=str(ROOT / "data" / "second_order_scores" / "tpp_mag_top5_per_lambda.csv"))
    p.add_argument("--target_lambda", type=float, default=1000.0)
    p.add_argument("--target_sample_idx", type=int, default=None, help="直接指定数据集 sample id；不传则按 topk_csv 取目标波长 rank-1")
    p.add_argument("--raw_target_only", action="store_true", help="直接使用数据集原始样本谱作为目标，不做 laplas 目标修正")
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--topk_second", type=int, default=5)
    p.add_argument("--eval_mode", choices=["full", "target_only"], default="full")
    args = p.parse_args()

    args.stats = str(resolve_from_root(args.stats) if args.stats else resolve_default_stats_path(ROOT))
    args.diffusion_ckpt = str(resolve_from_root(args.diffusion_ckpt) if args.diffusion_ckpt else resolve_default_diffusion_ckpt(ROOT))
    args.train_npz = str(resolve_from_root(args.train_npz))
    args.topk_csv = str(resolve_from_root(args.topk_csv))
    args.save_dir = str(resolve_from_root(args.save_dir))

    root_save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root_save_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_stats(args.stats)
    cond_ch = int(mean.shape[1])
    weight = torch.empty(0, device=args.device)
    diffusion = load_model(
        args.diffusion_ckpt,
        GaussianDiffusion(ConditionalUNet(cond_ch).to(args.device), timesteps=1000, image_size=64).to(args.device),
        "diffusion",
        args.device,
    )

    for case in TARGET_SWEEP:
        run_case(case, args, mean, std, cond_ch, weight, diffusion, root_save_dir)


if __name__ == "__main__":
    main()
