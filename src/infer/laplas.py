# -*- coding: utf-8 -*-
"""Construct second-order targets at configurable wavelength and run diffusion inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model.diffusion import GaussianDiffusion  # noqa: E402
from model.models import ConditionalUNet, ForwardSurrogate  # noqa: E402
from infer.common import (  # noqa: E402
    lambda_theta_grid,
    load_model,
    load_stats,
    normalize_with_stats,
    plot_map,
    plot_structure,
    rcwa_eval_full_map,
    second_order_score_map,
    second_order_target,
)

TARGET_SWEEP = [
    {"name": "floor_0p00_off_0p90", "center_floor": 0.00, "off_target": 0.90},
]


def parse_devices(devices_arg: str | None, device_arg: str | None) -> list[str]:
    if devices_arg:
        devices = [d.strip() for d in devices_arg.split(",") if d.strip()]
        if not devices:
            raise ValueError("--devices 为空，请传入类似 cuda:0,cuda:1")
        return devices
    if device_arg:
        return [device_arg]
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]


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
    sample_idx = None
    with topk_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["lambda_nm"]) == float(target_lambda) and int(row["rank"]) == 1:
                sample_idx = int(row["sample_idx"])
                break
    if sample_idx is None:
        raise ValueError(f"rank-1 sample at {target_lambda} nm not found in {topk_csv_path}")

    template_tpp = tpp[sample_idx].copy()
    if cond_ch == 2:
        template_tss = tss[sample_idx].copy()
        template = np.stack([template_tpp, template_tss], axis=0)
    else:
        template = template_tpp[None]
    return template.astype(np.float32), lambdas, thetas, sample_idx


def build_target(
    cond_ch: int,
    center_floor=0.00,
    off_target=0.90,
    edge_target=1.0,
    band_sigma_nm=55.0,
    max_target_mix=0.50,
    target_lambda: float = 1000.0,
    train_npz_path: Path | None = None,
    topk_csv_path: Path | None = None,
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
        )
        # 直接用数据集最好样本的原始谱作为目标，不做任何混合修改
        # 物理自洽：tpp 和 tss 来自同一真实结构
        return target, lambdas, thetas, template_idx
    else:
        tpp = np.empty((len(lambdas), len(thetas)), dtype=np.float32)
        tss = None

    # Build a smooth wavelength envelope so the target-lambda second-order row
    # gradually relaxes into the background/template spectrum instead of
    # switching abruptly to a flat constant map.
    lam_dist = lambdas.astype(np.float64) - float(target_lambda)
    lam_weight = max(float(max_target_mix), 0.0) * np.exp(-0.5 * (lam_dist / max(float(band_sigma_nm), 1e-6)) ** 2)

    for i, w in enumerate(lam_weight):
        if template_idx is not None:
            base_row = tpp[i].astype(np.float64)
            # 用模板最大角度处的边界均值缩放理想目标，保持边界透过率不变
            edge_val = (float(base_row[0]) + float(base_row[-1])) / 2.0
            scaled_ideal = kx2 * edge_val
            # theta 方向混合权重：kx2 在最大角度处=1，所以 (1-kx2) 在边界处=0，边界值完全保留
            theta_mask = 1.0 - kx2
            blend = w * theta_mask
            row = (1.0 - blend) * base_row + blend * scaled_ideal
            row = np.clip(row, 0.0, 1.0)
            tpp[i] = row.astype(np.float32)
        else:
            row_floor = off_target * (1.0 - w)
            row_edge = off_target + (edge_target - off_target) * w
            tpp[i] = (row_floor + (row_edge - row_floor) * kx2).astype(np.float32)

    # Add one more wavelength-wise smoothing pass so neighboring lambda rows
    # transition naturally instead of changing too abruptly from row to row.
    tpp = smooth_lambda_axis(tpp)

    if cond_ch == 2:
        if tss is None:
            return np.stack([tpp, tpp.copy()], axis=0), lambdas, thetas, template_idx
        return np.stack([tpp, tss], axis=0), lambdas, thetas, template_idx
    return tpp[None], lambdas, thetas, template_idx


def compute_second_order_metrics(
    pred_raw: np.ndarray,
    err: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    target_lambda: float,
) -> tuple[list[dict], np.ndarray]:
    tpp_maps = pred_raw[:, 0].astype(np.float64)
    edge_idx = int(np.argmax(np.abs(thetas)))
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
                "tpp_at_edge": float(tpp_maps[i, np.argmin(np.abs(lambdas - float(target_lambda))), edge_idx]),
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
    edge_idx = int(np.argmax(np.abs(thetas)))
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
        tag = f"|tpp|@{float(thetas[edge_idx]):.1f}deg={m['tpp_at_edge']:.3f}"
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


def _laplas_eval_worker(samples_np, target_raw, cond_ch, indices, device, rcwa_orders, queue):
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        maps = []
        errs = []
        for idx in indices:
            print(f"[laplas-rcwa {device}] sample {idx + 1}/{len(samples_np)}", flush=True)
            sample_t = torch.from_numpy(samples_np[idx: idx + 1]).to(device)
            rcwa_map = rcwa_eval_full_map(sample_t, cond_ch, device, rcwa_orders=rcwa_orders)
            if rcwa_map is None:
                raise RuntimeError("RCWA backend unavailable during laplas evaluation.")
            maps.append(rcwa_map.astype(np.float32))
            errs.append(float(np.mean(np.abs(rcwa_map - target_raw))))
        queue.put(
            {
                "ok": True,
                "device": device,
                "indices": np.asarray(indices, dtype=np.int64),
                "maps": np.stack(maps, axis=0) if maps else np.empty((0, cond_ch, *target_raw.shape[-2:]), dtype=np.float32),
                "errs": np.asarray(errs, dtype=np.float32),
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "device": device, "error": str(exc)})


def evaluate_rcwa_candidates(samples: torch.Tensor, target_raw: np.ndarray, cond_ch: int, devices: list[str], rcwa_orders: int) -> tuple[np.ndarray, np.ndarray]:
    samples_np = samples.cpu().numpy().astype(np.float32)
    num_samples = samples_np.shape[0]
    if len(devices) == 1:
        pred_raw = np.empty((num_samples, cond_ch, *target_raw.shape[-2:]), dtype=np.float32)
        err = np.empty((num_samples,), dtype=np.float32)
        for idx in range(num_samples):
            print(f"[laplas-rcwa {devices[0]}] sample {idx + 1}/{num_samples}", flush=True)
            rcwa_map = rcwa_eval_full_map(samples[idx: idx + 1], cond_ch, devices[0], rcwa_orders=rcwa_orders)
            if rcwa_map is None:
                raise RuntimeError("RCWA backend unavailable during laplas evaluation.")
            pred_raw[idx] = rcwa_map.astype(np.float32)
            err[idx] = float(np.mean(np.abs(rcwa_map - target_raw)))
        return pred_raw, err

    all_indices = np.arange(num_samples, dtype=np.int64)
    split_indices = [chunk.tolist() for chunk in np.array_split(all_indices, len(devices)) if len(chunk) > 0]
    active_devices = devices[: len(split_indices)]
    ctx = get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for dev, idxs in zip(active_devices, split_indices):
        proc = ctx.Process(
            target=_laplas_eval_worker,
            args=(samples_np, target_raw, cond_ch, idxs, dev, rcwa_orders, queue),
        )
        proc.start()
        procs.append(proc)

    pred_raw = np.empty((num_samples, cond_ch, *target_raw.shape[-2:]), dtype=np.float32)
    err = np.empty((num_samples,), dtype=np.float32)
    received = 0
    while received < len(procs):
        msg = queue.get()
        received += 1
        if not msg.get("ok", False):
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            raise RuntimeError(f"laplas worker {msg.get('device')} 失败: {msg.get('error')}")
        idxs = msg["indices"]
        pred_raw[idxs] = msg["maps"]
        err[idxs] = msg["errs"]

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"laplas worker 异常退出，exitcode={proc.exitcode}")
    return pred_raw, err


def run_case(case: dict, args, mean: np.ndarray, std: np.ndarray, cond_ch: int, weight: torch.Tensor, diffusion: torch.nn.Module, root_save_dir: Path, devices: list[str], surrogate: torch.nn.Module | None = None):
    save_dir = root_save_dir / case["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    target_raw, lambdas, thetas, template_idx = build_target(
        cond_ch,
        center_floor=float(case["center_floor"]),
        off_target=float(case["off_target"]),
        edge_target=1.0,
        target_lambda=args.target_lambda,
        train_npz_path=ROOT / "data" / "train_data.npz",
        topk_csv_path=ROOT / "data" / "second_order_scores" / "tpp_mag_top5_per_lambda.csv",
    )
    target = normalize_with_stats(target_raw, mean, std, args.device)
    cond_batch = target.repeat(args.num_samples, 1, 1, 1)
    if surrogate is not None:
        samples = diffusion.sample_guided(
            cond_batch, cfg_scale=args.cfg_scale,
            surrogate=surrogate, target_norm=target,
            guidance_scale=args.guidance_scale,
            guide_start_t=args.guide_start_t,
            guide_every=args.guide_every,
        )
    else:
        samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)
    pred_raw, err = evaluate_rcwa_candidates(samples, target_raw, cond_ch, devices, args.rcwa_orders)
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
        "target_lambda_nm": float(args.target_lambda),
        "template_sample_idx": None if template_idx is None else int(template_idx),
        "best_rcwa_mae_raw": float(err[best_idx]),
        "best_second_order_sample_idx": best_idx,
        "best_second_order_score": float(metrics[best_idx]["second_order_score"]),
        "best_second_order_tpp_at_edge": float(metrics[best_idx]["tpp_at_edge"]),
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
def _run_with_args(args) -> None:
    """Core execution logic; accepts a pre-parsed args namespace.

    Called both by main() (direct execution) and by band-specific wrapper
    scripts (e.g. band_900nm/laplas.py) that only override default values.
    """
    args.stats = str(resolve_from_root(args.stats))
    args.diffusion_ckpt = str(resolve_from_root(args.diffusion_ckpt))
    args.forward_ckpt = str(resolve_from_root(args.forward_ckpt))
    args.save_dir = str(resolve_from_root(args.save_dir))
    devices = parse_devices(args.devices, args.device)
    args.device = devices[0]

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

    surrogate = None
    if args.guidance_scale > 0:
        forward_path = Path(args.forward_ckpt)
        if forward_path.exists():
            surrogate = ForwardSurrogate(out_ch=cond_ch).to(args.device)
            ckpt = torch.load(str(forward_path), map_location=args.device)
            surrogate.load_state_dict(ckpt["model"])
            surrogate.eval()
            for param in surrogate.parameters():
                param.requires_grad_(False)
            print(f"[laplas] 物理引导已启用: guidance_scale={args.guidance_scale}, guide_start_t={args.guide_start_t}, guide_every={args.guide_every}")
        else:
            print(f"[laplas] 警告: forward_ckpt 不存在 ({forward_path})，禁用物理引导")

    for case in TARGET_SWEEP:
        run_case(case, args, mean, std, cond_ch, weight, diffusion, root_save_dir, devices, surrogate=surrogate)


def main():
    p = argparse.ArgumentParser(description="Run diffusion inference with swept second-order targets at 1000nm.")
    p.add_argument("--stats", default=str(ROOT / "checkpoints" / "cond_stats.npz"))
    p.add_argument("--diffusion_ckpt", default=str(ROOT / "checkpoints" / "diffusion_best.pt"))
    p.add_argument("--forward_ckpt", default=str(ROOT / "checkpoints" / "forward_best.pt"))
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "laplas"))
    p.add_argument("--device", default=None, help="主设备；默认自动使用全部可见 GPU，并以首张卡做扩散采样")
    p.add_argument("--devices", default=None, help="逗号分隔设备列表，如: cuda:0,cuda:1")
    p.add_argument("--target_lambda", type=float, default=1000.0)
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--topk_second", type=int, default=5)
    p.add_argument("--guidance_scale", type=float, default=0.1, help="物理引导强度，0 表示禁用")
    p.add_argument("--guide_start_t", type=int, default=300, help="开始物理引导的时间步阈值（t < 此值才引导）")
    p.add_argument("--guide_every", type=int, default=1, help="每隔几步做一次物理引导（1=每步，5=每5步）")
    args = p.parse_args()
    _run_with_args(args)


if __name__ == "__main__":
    main()
