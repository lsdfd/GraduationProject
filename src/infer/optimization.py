# -*- coding: utf-8 -*-
"""Light topology optimization over diffusion-generated initial structures."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from model.models import ForwardSurrogate  # noqa: E402
from infer.common import (  # noqa: E402
    build_weight,
    denormalize_with_stats,
    lambda_theta_grid,
    load_stats,
    plot_structure,
    rcwa_eval_1250,
    second_order_band_score_torch,
    second_order_score_map,
    second_order_score_row,
    second_order_target,
)

BAND_OFFSET_NM = 20.0
BAND_SCORE_WEIGHT = 0.15
BAND_LOSS_WEIGHT = 0.12


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def latest_laplas_file(name: str) -> str:
    root = ROOT / "samples" / "laplas"
    files = sorted([p for p in root.glob(f"**/{name}") if p.is_file()], key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("未找到 samples/laplas 下的推理结果，请先运行 python src/infer/laplas.py")
    return str(files[-1])


def load_target(path: str, stats_path: str, device: str):
    x = np.load(path).astype(np.float32)
    x = x[0] if x.ndim == 4 else x
    mean, std = load_stats(stats_path)
    x = (x[None] - mean) / std
    return torch.from_numpy(x).to(device), mean, std


def load_inits(path: str, device: str, max_inits=5) -> torch.Tensor:
    x = np.load(path).astype(np.float32)
    if x.ndim == 4:
        arr = x
    elif x.ndim == 3:
        arr = x[:, None, :, :]
    elif x.ndim == 2:
        arr = x[None, None, :, :]
    else:
        raise ValueError(f"Unsupported init shape: {x.shape}")
    arr = arr[: max(1, int(max_inits))]
    return torch.from_numpy(arr).to(device)


def smooth(x, k=5):
    return F.avg_pool2d(x, k, stride=1, padding=k // 2)


def symmetrize(x):
    rots = [torch.rot90(x, k, (-2, -1)) for k in range(4)]
    flips = [t.flip(-2) for t in rots]
    return sum(rots + flips) / 8.0


def finalize_binary(x):
    x = symmetrize(smooth(smooth(x, 9), 7))
    return (x > 0.5).float()


def project(x, beta=10.0):
    x = symmetrize(smooth(smooth(x, 7), 5))
    return torch.sigmoid(beta * (x - 0.5))


def tv_loss(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def min_feature_loss(x):
    return (x - smooth(x, 9)).abs().mean()


def plot_tpp_map(path: Path, tpp_map: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray, title: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(
        tpp_map,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])],
        interpolation="bicubic",
    )
    plt.xlabel("theta (deg)")
    plt.ylabel("lambda (nm)")
    plt.title(title)
    plt.colorbar(label="|tpp|")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_second_order_curve(path: Path, row: np.ndarray, thetas: np.ndarray, title: str):
    target = second_order_target(thetas)
    y = row.astype(np.float64)
    yn = y / max(float(np.max(y)), 1e-8)
    plt.figure(figsize=(5, 3.6))
    plt.plot(thetas, target, "k--", lw=1.7, label="target ~ |sin(theta)|^2")
    plt.plot(thetas, yn, lw=1.9, color="#1f77b4", label="candidate (normalized)")
    plt.xlabel("theta (deg)")
    plt.ylabel("normalized |tpp|")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def optimize_one(init, target, weight, surrogate, steps, lr, target_mae, mean_t, std_t, lambdas, thetas):
    logits = torch.logit(init.clamp(1e-4, 1 - 1e-4), eps=1e-4).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=lr)
    hist, best = [], (1e9, None, None)
    for step in range(steps):
        x = project(torch.sigmoid(logits), beta=min(4.0 + step / 20.0, 20.0))
        pred = surrogate(x)
        pred_raw = pred * std_t + mean_t
        loss_fit = ((pred - target).abs() * weight).sum() / weight.sum()
        loss_bg = (((pred - target).abs() * (1.0 - weight / weight.max()))).mean()
        band_pack = second_order_band_score_torch(
            pred_raw[:, 0],
            lambdas,
            thetas,
            target_lambda=1250.0,
            band_offset_nm=BAND_OFFSET_NM,
            w_band=BAND_SCORE_WEIGHT,
        )
        loss_band = 1.0 - band_pack["bandwidth_score"].mean()
        loss_bin = (x * (1 - x)).mean()
        loss_sym = (x - symmetrize(x)).abs().mean()
        loss_tv = tv_loss(x)
        loss_feat = min_feature_loss(x)
        loss = (
            loss_fit
            + 0.20 * loss_bg
            + BAND_LOSS_WEIGHT * loss_band
            + 0.15 * loss_bin
            + 0.08 * loss_sym
            + 0.05 * loss_tv
            + 0.08 * loss_feat
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        item = {
            "step": step,
            "loss": float(loss.item()),
            "fit": float(loss_fit.item()),
            "bg": float(loss_bg.item()),
            "band": float(loss_band.item()),
            "bin": float(loss_bin.item()),
            "sym": float(loss_sym.item()),
            "tv": float(loss_tv.item()),
            "feat": float(loss_feat.item()),
        }
        hist.append(item)
        if item["loss"] < best[0]:
            best = (item["loss"], x.detach().clone(), pred.detach().clone())
        if step % 20 == 0 or step + 1 == steps:
            print(
                f"[opt] step={step:03d} loss={item['loss']:.4f} fit={item['fit']:.4f} "
                f"bg={item['bg']:.4f} band={item['band']:.4f} bin={item['bin']:.4f} sym={item['sym']:.4f}"
            )
        if item["fit"] <= target_mae:
            print(f"[opt] early stop at step={step:03d}, fit={item['fit']:.4f} <= target_mae={target_mae:.4f}")
            break
    return best[1], best[2], hist


def run_candidate(idx: int, init: torch.Tensor, target: torch.Tensor, target_raw: np.ndarray, weight: torch.Tensor, surrogate: torch.nn.Module, args, mean: np.ndarray, std: np.ndarray, save_dir: Path) -> dict:
    lambdas, thetas = lambda_theta_grid()
    lam_idx = int(np.argmin(np.abs(lambdas - 1250.0)))
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    cand_dir = save_dir / f"candidate_{idx:02d}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    mean_t = torch.from_numpy(mean).to(args.device)
    std_t = torch.from_numpy(std).to(args.device)
    best_x, best_pred, hist = optimize_one(
        init,
        target,
        weight,
        surrogate,
        args.steps,
        args.lr,
        args.target_mae,
        mean_t,
        std_t,
        lambdas,
        thetas,
    )
    best_bin = finalize_binary(best_x)
    pred_raw = denormalize_with_stats(best_pred.cpu().numpy(), mean, std)
    tpp_map = pred_raw[0, 0]
    second = second_order_score_map(
        tpp_map,
        lambdas,
        thetas,
        target_lambda=1250.0,
        band_offset_nm=BAND_OFFSET_NM,
        w_band=BAND_SCORE_WEIGHT,
    )
    out = {
        "surrogate_mae_norm": float((((best_pred - target).abs() * weight).sum() / weight.sum()).item()),
        "surrogate_mae_raw": float(np.mean(np.abs(pred_raw - target_raw))),
        "second_order_score": float(second["score"]),
        "main_second_order_score": float(second["main_score"]),
        "bandwidth_second_order_score": float(second["bandwidth_score"]),
        "band_left_score": float(second["band_left_score"]),
        "band_right_score": float(second["band_right_score"]),
        "center_score": float(second["center"]),
        "shape_score": float(second["shape"]),
        "edge_score": float(second["edge"]),
        "r2": float(second["r2"]),
        "tpp_at_40": float(tpp_map[lam_idx, t40_idx]),
    }
    if not args.skip_rcwa_eval:
        rcwa = rcwa_eval_1250(best_bin, target_raw[0], int(target.shape[1]), args.device)
        if rcwa is not None:
            rcwa_mae, rcwa_pred = rcwa
            out["rcwa_mae_raw"] = float(rcwa_mae)
            out["rcwa_tpp_at_40"] = float(rcwa_pred[0, t40_idx])
            np.save(cand_dir / "best_rcwa_pred.npy", rcwa_pred)

    np.save(cand_dir / "optimized_continuous.npy", best_x.cpu().numpy())
    np.save(cand_dir / "optimized_binary.npy", best_bin.cpu().numpy())
    np.save(cand_dir / "optimized_pred_cond.npy", best_pred.cpu().numpy())
    np.save(cand_dir / "optimized_pred_cond_raw.npy", pred_raw)
    np.save(cand_dir / "target_cond.npy", target.cpu().numpy())
    plot_structure(cand_dir / "optimized_continuous.png", best_x.cpu().numpy(), "Optimized continuous")
    plot_structure(cand_dir / "optimized_binary.png", best_bin.cpu().numpy(), "Optimized binary")
    plot_tpp_map(cand_dir / "optimized_tpp_map.png", tpp_map, lambdas, thetas, "Optimized tpp_mag")
    plot_second_order_curve(cand_dir / "optimized_second_order_curve.png", tpp_map[lam_idx], thetas, "1250nm second-order fit")
    with (cand_dir / "optimization_log.json").open("w", encoding="utf-8") as f:
        json.dump({"target": args.target, "init": args.init, "candidate_idx": idx, "metrics": out, "history_tail": hist[-20:]}, f, ensure_ascii=False, indent=2)
    return {"candidate_idx": idx, **out}


def main():
    p = argparse.ArgumentParser(description="Optimize one or multiple initial structures with surrogate guidance.")
    p.add_argument("--target")
    p.add_argument("--init")
    p.add_argument("--stats", default=str(ROOT / "checkpoints" / "cond_stats.npz"))
    p.add_argument("--forward_ckpt", default=str(ROOT / "checkpoints" / "forward_best.pt"))
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--target_mae", type=float, default=0.01)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "optimized"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_rcwa_eval", action="store_true")
    p.add_argument("--max_inits", type=int, default=5)
    args = p.parse_args()

    if args.target:
        args.target = str(resolve_from_root(args.target))
    if args.init:
        args.init = str(resolve_from_root(args.init))
    args.stats = str(resolve_from_root(args.stats))
    args.forward_ckpt = str(resolve_from_root(args.forward_ckpt))
    args.save_dir = str(resolve_from_root(args.save_dir))

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    args.target = args.target or latest_laplas_file("target_cond_raw.npy")
    try:
        default_init = latest_laplas_file("topk_second_samples.npy")
    except FileNotFoundError:
        default_init = latest_laplas_file("topk_samples.npy")
    args.init = args.init or default_init

    target, mean, std = load_target(args.target, args.stats, args.device)
    target_raw = np.load(args.target).astype(np.float32)
    target_raw = target_raw[None] if target_raw.ndim == 3 else target_raw
    inits = load_inits(args.init, args.device, max_inits=args.max_inits).clamp(0, 1)

    cond_ch = int(target.shape[1])
    weight = torch.from_numpy(build_weight(cond_ch)[None]).to(args.device)
    surrogate = ForwardSurrogate(cond_ch).to(args.device)
    surrogate.load_state_dict(torch.load(args.forward_ckpt, map_location=args.device)["model"])
    surrogate.eval()
    for p_ in surrogate.parameters():
        p_.requires_grad = False

    rows = []
    for i in range(inits.shape[0]):
        print(f"[opt] candidate {i + 1}/{inits.shape[0]}")
        rows.append(run_candidate(i, inits[i:i + 1], target, target_raw, weight, surrogate, args, mean, std, save_dir))

    rows = sorted(rows, key=lambda d: d["second_order_score"], reverse=True)
    with (save_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("saved_to:", save_dir)
    print("target_from:", args.target)
    print("init_from:", args.init)
    print("ranking_by_second_order:", [{"candidate_idx": r["candidate_idx"], "second_order_score": r["second_order_score"], "tpp_at_40": r["tpp_at_40"]} for r in rows])


if __name__ == "__main__":
    main()
