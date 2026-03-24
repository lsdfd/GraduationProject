# -*- coding: utf-8 -*-
"""Construct one-lambda targets and run diffusion inference."""

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
from model.train_utils import resolve_latest_run  # noqa: E402
from infer.common import (  # noqa: E402
    load_model,
    load_stats,
    normalize_with_stats,
    plot_structure,
    rcwa_eval_target_lambda,
    second_order_score_row,
    second_order_target,
)

TARGET_SWEEP = [
    {"name": "template_top1"},
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


def resolve_infer_artifacts(
    stats_path: str | Path | None,
    diffusion_ckpt: str | Path | None,
    forward_ckpt: str | Path | None,
) -> tuple[Path, Path, Path | None]:
    ckpt_root = ROOT / "checkpoints"
    latest_forward = resolve_latest_run(ckpt_root, "forward")
    latest_diffusion = resolve_latest_run(ckpt_root, "diffusion")

    if stats_path is None:
        if latest_forward is None:
            raise FileNotFoundError("未找到 forward run 的 cond_stats.npz；请先运行 train_forward.py 或显式传入 --stats")
        stats = latest_forward / "cond_stats.npz"
    else:
        stats = resolve_from_root(stats_path)

    if diffusion_ckpt is None:
        if latest_diffusion is None:
            raise FileNotFoundError("未找到 diffusion_best.pt；请先运行 train_diffusion.py 或显式传入 --diffusion_ckpt")
        diffusion = latest_diffusion / "diffusion_best.pt"
    else:
        diffusion = resolve_from_root(diffusion_ckpt)

    if forward_ckpt is None:
        forward = latest_forward / "forward_best.pt" if latest_forward is not None else None
    else:
        forward = resolve_from_root(forward_ckpt)

    if not stats.exists():
        raise FileNotFoundError(f"cond stats 不存在: {stats}")
    if not diffusion.exists():
        raise FileNotFoundError(f"diffusion checkpoint 不存在: {diffusion}")
    if forward is not None and not forward.exists():
        raise FileNotFoundError(f"forward checkpoint 不存在: {forward}")
    return stats, diffusion, forward


def load_template_row(
    cond_ch: int,
    train_npz_path: Path,
    topk_csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(train_npz_path)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    target_lambda = float(data["target_lambda"]) if "target_lambda" in data.files else 1000.0
    lambdas = np.asarray([target_lambda], dtype=np.float32)

    sample_idx = None
    with topk_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["lambda_nm"]) == float(target_lambda) and int(row["rank"]) == 1:
                sample_idx = int(row["sample_idx"])
                break
    if sample_idx is None:
        raise ValueError(f"rank-1 sample at {target_lambda} nm not found in {topk_csv_path}")

    tpp_all = np.asarray(data["tpp_mag"], dtype=np.float32)
    tss_all = np.asarray(data["tss_mag"], dtype=np.float32)
    tpp_row = np.asarray(tpp_all[sample_idx], dtype=np.float32)
    if cond_ch == 2:
        tss_row = np.asarray(tss_all[sample_idx], dtype=np.float32)
        target = np.stack([tpp_row, tss_row], axis=0)
    else:
        target = tpp_row[None]
    return target.astype(np.float32), lambdas.astype(np.float32), thetas, sample_idx


def compute_second_order_metrics(
    pred_raw: np.ndarray,
    err: np.ndarray,
    thetas: np.ndarray,
) -> tuple[list[dict], np.ndarray]:
    metrics = []
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    for i in range(pred_raw.shape[0]):
        score = second_order_score_row(pred_raw[i, 0], thetas)
        metrics.append(
            {
                "sample_idx": int(i),
                "rcwa_mae_raw": float(err[i]),
                "second_order_score": float(score["score"]),
                "center_score": float(score["center"]),
                "shape_score": float(score["shape"]),
                "edge_score": float(score["edge"]),
                "outer_score": float(score["outer"]),
                "r2": float(score["r2"]),
                "tpp_at_40": float(pred_raw[i, 0, t40_idx]),
            }
        )
    rank = np.array(sorted(range(len(metrics)), key=lambda i: metrics[i]["second_order_score"], reverse=True), dtype=np.int32)
    return metrics, rank


def plot_curve(path: Path, target_row: np.ndarray, pred_row: np.ndarray | None, thetas: np.ndarray, title: str) -> None:
    ideal = second_order_target(thetas)
    target_norm = target_row / max(float(np.max(target_row)), 1e-8)
    plt.figure(figsize=(5, 3.6))
    plt.plot(thetas, ideal, "k--", lw=1.6, label="ideal ~ |sin(theta)|^2")
    plt.plot(thetas, target_norm, lw=1.9, color="#d62728", label="target")
    if pred_row is not None:
        pred_norm = pred_row / max(float(np.max(pred_row)), 1e-8)
        plt.plot(thetas, pred_norm, lw=1.9, color="#1f77b4", label="candidate")
    plt.xlabel("theta (deg)")
    plt.ylabel("normalized |tpp|")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_ranked_samples(
    out_png: Path,
    samples: np.ndarray,
    pred_rows: np.ndarray,
    thetas: np.ndarray,
    ranked_idx: np.ndarray,
    metrics: list[dict],
) -> None:
    k = len(ranked_idx)
    if k == 0:
        return
    ideal = second_order_target(thetas)
    fig, axes = plt.subplots(k, 2, figsize=(9.5, max(2.7 * k, 5.0)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, i in enumerate(ranked_idx):
        a_struct, a_curve = axes[r]
        m = metrics[int(i)]
        a_struct.imshow(samples[int(i), 0], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        a_struct.set_title(f"id={int(i)}", fontsize=10)
        a_struct.axis("off")

        y = pred_rows[int(i)].astype(np.float64)
        yn = y / max(float(np.max(y)), 1e-8)
        a_curve.plot(thetas, ideal, "k--", lw=1.7, label="ideal")
        a_curve.plot(thetas, yn, lw=1.9, color="#1f77b4", label="candidate")
        a_curve.set_ylim(-0.05, 1.05)
        a_curve.set_xlabel("theta (deg)")
        a_curve.set_ylabel("normalized |tpp|")
        a_curve.grid(alpha=0.25)
        a_curve.set_title(f"2nd={m['second_order_score']:.3f} | mae={m['rcwa_mae_raw']:.4f}", fontsize=10)
        if r == 0:
            a_curve.legend(fontsize=8, loc="lower right")

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _laplas_eval_worker(samples_np, target_raw, cond_ch, indices, device, target_lambda, rcwa_orders, queue):
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        rows = []
        errs = []
        for idx in indices:
            print(f"[laplas-rcwa {device}] sample {idx + 1}/{len(samples_np)}", flush=True)
            sample_t = torch.from_numpy(samples_np[idx: idx + 1]).to(device)
            result = rcwa_eval_target_lambda(
                sample_t,
                target_raw,
                cond_ch,
                device,
                target_lambda=target_lambda,
                rcwa_orders=rcwa_orders,
            )
            if result is None:
                raise RuntimeError("RCWA backend unavailable during laplas evaluation.")
            mae, pred = result
            rows.append(pred.astype(np.float32))
            errs.append(float(mae))
        queue.put(
            {
                "ok": True,
                "device": device,
                "indices": np.asarray(indices, dtype=np.int64),
                "rows": np.stack(rows, axis=0) if rows else np.empty((0, cond_ch, target_raw.shape[-1]), dtype=np.float32),
                "errs": np.asarray(errs, dtype=np.float32),
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "device": device, "error": str(exc)})


def evaluate_rcwa_candidates(
    samples: torch.Tensor,
    target_raw: np.ndarray,
    cond_ch: int,
    devices: list[str],
    target_lambda: float,
    rcwa_orders: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples_np = samples.cpu().numpy().astype(np.float32)
    num_samples = samples_np.shape[0]
    if len(devices) == 1:
        pred_raw = np.empty((num_samples, cond_ch, target_raw.shape[-1]), dtype=np.float32)
        err = np.empty((num_samples,), dtype=np.float32)
        for idx in range(num_samples):
            print(f"[laplas-rcwa {devices[0]}] sample {idx + 1}/{num_samples}", flush=True)
            result = rcwa_eval_target_lambda(
                samples[idx: idx + 1],
                target_raw,
                cond_ch,
                devices[0],
                target_lambda=target_lambda,
                rcwa_orders=rcwa_orders,
            )
            if result is None:
                raise RuntimeError("RCWA backend unavailable during laplas evaluation.")
            err[idx], pred_raw[idx] = result
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
            args=(samples_np, target_raw, cond_ch, idxs, dev, target_lambda, rcwa_orders, queue),
        )
        proc.start()
        procs.append(proc)

    pred_raw = np.empty((num_samples, cond_ch, target_raw.shape[-1]), dtype=np.float32)
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
        pred_raw[idxs] = msg["rows"]
        err[idxs] = msg["errs"]

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"laplas worker 异常退出，exitcode={proc.exitcode}")
    return pred_raw, err


def run_case(case: dict, args, mean: np.ndarray, std: np.ndarray, cond_ch: int, diffusion: torch.nn.Module, root_save_dir: Path, devices: list[str], surrogate: torch.nn.Module | None = None):
    save_dir = root_save_dir / case["name"]
    save_dir.mkdir(parents=True, exist_ok=True)

    target_raw, lambdas, thetas, template_idx = load_template_row(
        cond_ch,
        ROOT / "data" / "train_data.npz",
        ROOT / "data" / "second_order_scores" / "tpp_mag_top5_per_lambda.csv",
    )
    target = normalize_with_stats(target_raw, mean, std, args.device)
    cond_batch = target.repeat(args.num_samples, 1, 1)

    if surrogate is not None:
        samples = diffusion.sample_guided(
            cond_batch,
            cfg_scale=args.cfg_scale,
            surrogate=surrogate,
            target_norm=target,
            guidance_scale=args.guidance_scale,
            guide_start_t=args.guide_start_t,
            guide_every=args.guide_every,
        )
    else:
        samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)

    pred_raw, err = evaluate_rcwa_candidates(
        samples,
        target_raw,
        cond_ch,
        devices,
        target_lambda=1000.0,
        rcwa_orders=args.rcwa_orders,
    )
    metrics, rank_second = compute_second_order_metrics(pred_raw, err, thetas)
    topk_second = rank_second[: min(max(1, args.topk_second), len(rank_second))]
    topk_second_t = torch.from_numpy(topk_second).to(samples.device, dtype=torch.long)
    best_idx = int(topk_second[0])
    best = samples[best_idx:best_idx + 1]

    info = {
        "sweep_case": case["name"],
        "target_lambda_nm": 1000.0,
        "template_sample_idx": int(template_idx),
        "best_rcwa_mae_raw": float(err[best_idx]),
        "best_second_order_sample_idx": best_idx,
        "best_second_order_score": float(metrics[best_idx]["second_order_score"]),
        "best_second_order_tpp_at_40": float(metrics[best_idx]["tpp_at_40"]),
    }

    np.save(save_dir / "target_cond_raw.npy", target_raw)
    np.save(save_dir / "target_cond_norm.npy", target.cpu().numpy())
    np.save(save_dir / "lambdas.npy", lambdas)
    np.save(save_dir / "thetas.npy", thetas)
    np.save(save_dir / "all_samples.npy", samples.cpu().numpy())
    np.save(save_dir / "all_pred_cond_raw.npy", pred_raw)
    np.save(save_dir / "all_errors.npy", err)
    np.save(save_dir / "rank_second_order.npy", topk_second)
    np.save(save_dir / "topk_second_samples.npy", samples[topk_second_t].cpu().numpy())
    np.save(save_dir / "topk_second_pred_cond_raw.npy", pred_raw[topk_second])

    plot_curve(
        save_dir / "target_second_order_curve.png",
        target_raw[0],
        None,
        thetas,
        "Target 1000nm second-order curve",
    )
    plot_curve(
        save_dir / "best_second_order_curve.png",
        target_raw[0],
        pred_raw[best_idx, 0],
        thetas,
        "Best RCWA row @ 1000nm",
    )
    plot_structure(save_dir / "best_structure.png", best.cpu().numpy(), "Best binary structure")
    with (save_dir / "second_order_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    plot_ranked_samples(
        save_dir / f"top{len(topk_second)}_second_order.png",
        samples.cpu().numpy(),
        pred_raw[:, 0].astype(np.float32),
        thetas,
        topk_second,
        metrics,
    )
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("saved_to:", save_dir)
    print(info)


@torch.no_grad()
def _run_with_args(args) -> None:
    stats_path, diffusion_path, forward_path = resolve_infer_artifacts(
        args.stats,
        args.diffusion_ckpt,
        args.forward_ckpt,
    )
    args.stats = str(stats_path)
    args.diffusion_ckpt = str(diffusion_path)
    args.forward_ckpt = str(forward_path) if forward_path is not None else None
    args.save_dir = str(resolve_from_root(args.save_dir))
    devices = parse_devices(args.devices, args.device)
    args.device = devices[0]

    root_save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    root_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[laplas] stats={args.stats}")
    print(f"[laplas] diffusion_ckpt={args.diffusion_ckpt}")
    print(f"[laplas] forward_ckpt={args.forward_ckpt}")

    mean, std = load_stats(args.stats)
    cond_ch = int(mean.shape[1])
    diffusion = load_model(
        args.diffusion_ckpt,
        GaussianDiffusion(ConditionalUNet(cond_ch).to(args.device), timesteps=1000, image_size=64).to(args.device),
        "diffusion",
        args.device,
    )

    surrogate = None
    if args.guidance_scale > 0:
        if args.forward_ckpt is None:
            raise FileNotFoundError("启用物理引导时需要 forward checkpoint；请先训练 forward 或显式传入 --forward_ckpt")
        forward_ckpt = Path(args.forward_ckpt)
        if forward_ckpt.exists():
            surrogate = ForwardSurrogate(out_ch=cond_ch).to(args.device)
            ckpt = torch.load(str(forward_ckpt), map_location=args.device)
            surrogate.load_state_dict(ckpt["model"])
            surrogate.eval()
            for param in surrogate.parameters():
                param.requires_grad_(False)
            print(f"[laplas] 物理引导已启用: guidance_scale={args.guidance_scale}, guide_start_t={args.guide_start_t}, guide_every={args.guide_every}")
        else:
            print(f"[laplas] 警告: forward_ckpt 不存在 ({forward_ckpt})，禁用物理引导")

    for case in TARGET_SWEEP:
        run_case(case, args, mean, std, cond_ch, diffusion, root_save_dir, devices, surrogate=surrogate)


def main():
    p = argparse.ArgumentParser(description="Run diffusion inference with one-lambda second-order targets.")
    p.add_argument("--stats", default=None)
    p.add_argument("--diffusion_ckpt", default=None)
    p.add_argument("--forward_ckpt", default=None)
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "laplas"))
    p.add_argument("--device", default=None, help="主设备；默认自动使用全部可见 GPU，并以首张卡做扩散采样")
    p.add_argument("--devices", default=None, help="逗号分隔设备列表，如: cuda:0,cuda:1")
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--topk_second", type=int, default=5)
    p.add_argument("--guidance_scale", type=float, default=0.1, help="物理引导强度，0 表示禁用")
    p.add_argument("--guide_start_t", type=int, default=300, help="开始物理引导的时间步阈值（t < 此值才引导）")
    p.add_argument("--guide_every", type=int, default=1, help="每隔几步做一次物理引导（1=每步，5=每5步）")
    args = p.parse_args()
    _run_with_args(args)


if __name__ == "__main__":
    main()
