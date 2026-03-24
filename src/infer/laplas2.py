# -*- coding: utf-8 -*-
"""Run diffusion inference using the raw top-ranked dataset spectrum as target."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

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
    second_order_score_map,
    second_order_target,
)


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


def load_raw_top_condition(
    train_npz_path: Path,
    topk_csv_path: Path,
    target_lambda: float,
    target_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    data = np.load(train_npz_path)
    if "tpp_mag" not in data.files:
        raise ValueError(f"tpp_mag not found in {train_npz_path}")
    if "tss_mag" not in data.files:
        raise ValueError(f"tss_mag not found in {train_npz_path}")

    sample_idx = None
    with topk_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["lambda_nm"]) == float(target_lambda) and int(row["rank"]) == int(target_rank):
                sample_idx = int(row["sample_idx"])
                break
    if sample_idx is None:
        raise ValueError(f"rank-{target_rank} sample at {target_lambda} nm not found in {topk_csv_path}")

    target_raw = np.stack(
        [
            np.asarray(data["tpp_mag"][sample_idx], dtype=np.float32),
            np.asarray(data["tss_mag"][sample_idx], dtype=np.float32),
        ],
        axis=0,
    )
    lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    return target_raw, lambdas, thetas, sample_idx


def build_physics_target(
    target_raw: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    target_lambda: float = 1000.0,
    band_sigma_nm: float = 25.0,
) -> np.ndarray:
    """
    物理约束目标构建：
    - 保留模板在每个波长最大角度处的边界透过率值
    - 将角度分布修正为理想 sin²θ 形状（用边界值缩放）
    - 以高斯权重从 target_lambda 向两侧衰减，远处保持原始谱
    - tss 直接复制修正后的 tpp（C4 对称性 tpp≈tss）
    """
    ideal = second_order_target(thetas).astype(np.float64)
    lam_dist = lambdas.astype(np.float64) - float(target_lambda)
    gauss_w = np.exp(-0.5 * (lam_dist / max(float(band_sigma_nm), 1e-6)) ** 2)

    tpp = target_raw[0].copy().astype(np.float64)
    for li in range(len(lambdas)):
        w = float(gauss_w[li])
        if w < 1e-6:
            continue
        row = tpp[li]
        edge_val = (float(row[0]) + float(row[-1])) / 2.0
        ideal_row = ideal * edge_val
        tpp[li] = np.clip((1.0 - w) * row + w * ideal_row, 0.0, 1.0)

    tpp = tpp.astype(np.float32)
    tss = tpp.copy()  # C4 对称：tss = tpp
    return np.stack([tpp, tss], axis=0)


def compute_second_order_metrics(
    pred_raw: np.ndarray,
    err: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    target_lambda: float,
) -> tuple[list[dict], np.ndarray]:
    tpp_maps = pred_raw[:, 0].astype(np.float64)
    edge_idx = int(np.argmax(np.abs(thetas)))
    lam_idx = int(np.argmin(np.abs(lambdas - float(target_lambda))))
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
                "tpp_at_edge": float(tpp_maps[i, lam_idx, edge_idx]),
            }
        )
    rank = np.array(sorted(range(len(metrics)), key=lambda i: metrics[i]["rcwa_mae_raw"]), dtype=np.int32)
    return metrics, rank


def _eval_worker(samples_np, target_raw, cond_ch, indices, device, rcwa_orders, queue):
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        maps = []
        errs = []
        for idx in indices:
            print(f"[laplas2-rcwa {device}] sample {idx + 1}/{len(samples_np)}", flush=True)
            sample_t = torch.from_numpy(samples_np[idx: idx + 1]).to(device)
            rcwa_map = rcwa_eval_full_map(sample_t, cond_ch, device, rcwa_orders=rcwa_orders)
            if rcwa_map is None:
                raise RuntimeError("RCWA backend unavailable during laplas2 evaluation.")
            maps.append(rcwa_map.astype(np.float32))
            errs.append(float(np.mean(np.abs(rcwa_map - target_raw))))
        queue.put(
            {
                "ok": True,
                "indices": np.asarray(indices, dtype=np.int64),
                "maps": np.stack(maps, axis=0) if maps else np.empty((0, cond_ch, *target_raw.shape[-2:]), dtype=np.float32),
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
    rcwa_orders: int,
) -> tuple[np.ndarray, np.ndarray]:
    samples_np = samples.cpu().numpy().astype(np.float32)
    num_samples = samples_np.shape[0]
    if len(devices) == 1:
        pred_raw = np.empty((num_samples, cond_ch, *target_raw.shape[-2:]), dtype=np.float32)
        err = np.empty((num_samples,), dtype=np.float32)
        for idx in range(num_samples):
            print(f"[laplas2-rcwa {devices[0]}] sample {idx + 1}/{num_samples}", flush=True)
            rcwa_map = rcwa_eval_full_map(samples[idx: idx + 1], cond_ch, devices[0], rcwa_orders=rcwa_orders)
            if rcwa_map is None:
                raise RuntimeError("RCWA backend unavailable during laplas2 evaluation.")
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
        proc = ctx.Process(target=_eval_worker, args=(samples_np, target_raw, cond_ch, idxs, dev, rcwa_orders, queue))
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
            raise RuntimeError(f"laplas2 worker {msg.get('device')} 失败: {msg.get('error')}")
        idxs = msg["indices"]
        pred_raw[idxs] = msg["maps"]
        err[idxs] = msg["errs"]

    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"laplas2 worker 异常退出，exitcode={proc.exitcode}")
    return pred_raw, err


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description="Run diffusion inference using the raw top-ranked dataset spectrum as target.")
    p.add_argument("--train_npz", default=str(ROOT / "data" / "train_data.npz"))
    p.add_argument("--topk_csv", default=str(ROOT / "data" / "second_order_scores" / "tpp_mag_top5_per_lambda.csv"))
    p.add_argument("--stats", default=str(ROOT / "checkpoints" / "cond_stats.npz"))
    p.add_argument("--diffusion_ckpt", default=str(ROOT / "checkpoints" / "diffusion_best.pt"))
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "laplas2"))
    p.add_argument("--device", default=None)
    p.add_argument("--devices", default=None)
    p.add_argument("--target_lambda", type=float, default=1000.0)
    p.add_argument("--target_rank", type=int, default=1)
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--band_sigma_nm", type=float, default=25.0, help="高斯扩散宽度(nm)，控制理想形状向周围波长的扩散范围")
    args = p.parse_args()

    args.train_npz = str(resolve_from_root(args.train_npz))
    args.topk_csv = str(resolve_from_root(args.topk_csv))
    args.stats = str(resolve_from_root(args.stats))
    args.diffusion_ckpt = str(resolve_from_root(args.diffusion_ckpt))
    args.save_dir = str(resolve_from_root(args.save_dir))

    devices = parse_devices(args.devices, args.device)
    args.device = devices[0]
    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    target_raw, lambdas, thetas, template_idx = load_raw_top_condition(
        Path(args.train_npz),
        Path(args.topk_csv),
        args.target_lambda,
        args.target_rank,
    )
    target_raw = build_physics_target(
        target_raw, lambdas, thetas,
        target_lambda=args.target_lambda,
        band_sigma_nm=args.band_sigma_nm,
    )
    mean, std = load_stats(args.stats)
    cond_ch = int(mean.shape[1])
    if cond_ch != target_raw.shape[0]:
        raise ValueError(f"cond channel mismatch: stats={cond_ch}, target={target_raw.shape[0]}")

    target = normalize_with_stats(target_raw, mean, std, args.device)
    cond_batch = target.repeat(args.num_samples, 1, 1, 1)
    diffusion = load_model(
        args.diffusion_ckpt,
        GaussianDiffusion(ConditionalUNet(cond_ch).to(args.device), timesteps=1000, image_size=64).to(args.device),
        "diffusion",
        args.device,
    )

    samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)
    pred_raw, err = evaluate_rcwa_candidates(samples, target_raw, cond_ch, devices, args.rcwa_orders)
    metrics, rank = compute_second_order_metrics(pred_raw, err, lambdas, thetas, args.target_lambda)

    best_idx = int(rank[0])
    info = {
        "target_lambda_nm": float(args.target_lambda),
        "target_rank": int(args.target_rank),
        "template_sample_idx": int(template_idx),
        "num_samples": int(args.num_samples),
        "best_sample_idx": int(best_idx),
        "best_rcwa_mae_raw": float(err[best_idx]),
        "best_second_order_score": float(metrics[best_idx]["second_order_score"]),
        "best_second_order_tpp_at_edge": float(metrics[best_idx]["tpp_at_edge"]),
    }

    np.save(save_dir / "target_cond_raw.npy", target_raw)
    np.save(save_dir / "target_cond_norm.npy", target.cpu().numpy())
    np.save(save_dir / "lambdas.npy", lambdas)
    np.save(save_dir / "thetas.npy", thetas)
    np.save(save_dir / "all_samples.npy", samples.cpu().numpy())
    np.save(save_dir / "all_pred_cond_raw.npy", pred_raw)
    np.save(save_dir / "all_errors.npy", err)
    np.save(save_dir / "rank_by_rcwa_mae.npy", rank)
    np.save(save_dir / "best_structure.npy", samples[best_idx:best_idx + 1].cpu().numpy())
    np.save(save_dir / "best_pred_cond_raw.npy", pred_raw[best_idx:best_idx + 1])

    vmax_tpp = max(float(target_raw[0].max()), float(pred_raw[best_idx, 0].max()), 1e-6)
    plot_map(save_dir / "target_tpp.png", target_raw, lambdas, thetas, "Target raw tpp_mag", vmax_tpp, channel_idx=0)
    plot_map(save_dir / "best_tpp.png", pred_raw[best_idx], lambdas, thetas, "Best RCWA tpp_mag", vmax_tpp, channel_idx=0)
    if cond_ch == 2:
        vmax_tss = max(float(target_raw[1].max()), float(pred_raw[best_idx, 1].max()), 1e-6)
        plot_map(save_dir / "target_tss.png", target_raw, lambdas, thetas, "Target raw tss_mag", vmax_tss, channel_idx=1)
        plot_map(save_dir / "best_tss.png", pred_raw[best_idx], lambdas, thetas, "Best RCWA tss_mag", vmax_tss, channel_idx=1)
    plot_structure(save_dir / "best_structure.png", samples[best_idx:best_idx + 1].cpu().numpy(), "Best binary structure")

    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("saved_to:", save_dir)
    print(info)


if __name__ == "__main__":
    main()
