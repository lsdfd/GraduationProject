# -*- coding: utf-8 -*-
"""构造 1250nm 中心二阶微分目标，按 [tpp_mag, tss_mag] 条件做扩散采样。"""

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

from model.models import ConditionalUNet, ForwardSurrogate  # noqa: E402
from model.diffusion import GaussianDiffusion  # noqa: E402

try:
    from dataset.rcwa.rcwa import torcwa_simulation
except Exception:
    torcwa_simulation = None


def load_model(path, model, key, device):
    model.load_state_dict(torch.load(path, map_location=device)[key])
    return model.eval()


def build_target(cond_ch):
    lambdas = np.arange(1000.0, 1500.1, 50.0, dtype=np.float32)
    thetas = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)
    kx = np.sin(np.deg2rad(thetas)) / np.sin(np.deg2rad(40.0))
    amp_theta = (np.abs(kx) ** 2) * (1.0 - np.exp(-(np.abs(thetas) / 8.0) ** 2))
    edge = 1.0 - 0.10 * np.clip((np.abs(thetas) - 30.0) / 10.0, 0.0, 1.0)
    profile = (0.08 + 0.72 * amp_theta * edge).astype(np.float32)
    tpp = np.full((len(lambdas), len(thetas)), 0.08, dtype=np.float32)
    tpp[np.argmin(np.abs(lambdas - 1250.0))] = profile
    if cond_ch == 2:
        # channel 0: tpp_mag target, channel 1: tss_mag target (low baseline)
        tss = np.full_like(tpp, 0.02, dtype=np.float32)
        target = np.stack([tpp, tss], axis=0)
    else:
        target = tpp[None]
    return target.astype(np.float32), lambdas, thetas


def build_weight(cond_ch):
    lambdas = np.arange(1000.0, 1500.1, 50.0, dtype=np.float32)
    thetas = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)
    w = np.full((len(lambdas), len(thetas)), 0.05, dtype=np.float32)
    w[np.argmin(np.abs(lambdas - 1250.0))] = 1.0
    return np.stack([w, w], axis=0) if cond_ch == 2 else w[None]


def normalize(target, stats_path, device):
    stats = np.load(stats_path)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    return torch.from_numpy((target[None] - mean) / std).to(device)


def denormalize(x, stats_path):
    stats = np.load(stats_path)
    return x * stats["std"].astype(np.float32) + stats["mean"].astype(np.float32)


def rcwa_eval(structure, target_raw, cond_ch, device):
    if torcwa_simulation is None:
        return None
    layer = structure.squeeze().to(device)
    tpp = np.full((17,), np.nan, np.float32)
    tss = np.full_like(tpp, np.nan)
    lam = 1250.0
    for j, theta in enumerate(np.arange(-40.0, 40.1, 5.0)):
        out = torcwa_simulation({"periodicity": 500.0, "h": 500.0, "lam": lam, "tet": theta, "phi": 0.0, "angle_unit": "deg", "angle_layer": "input", "input_medium": "air", "output_medium": "SiO2", "structure": "Si", "n_input": 1.0, "n_output": 1.45, "n_structure": 3.4}, layer, rcwa_orders=7, project=False, device=device)
        tpp[j] = float(out["tpp_mag"].detach().cpu().item())
        tss[j] = float(out["tss_mag"].detach().cpu().item())
    i = int(np.argmin(np.abs(np.arange(1000.0, 1500.1, 50.0) - 1250.0)))
    pred = np.stack([tpp, tss], axis=0) if cond_ch == 2 else tpp[None]
    return float(np.mean(np.abs(pred - target_raw[:, i]))), pred


def plot_map(path, cond, lambdas, thetas, title, vmax, channel_idx=0):
    img = cond if cond.ndim == 2 else cond[channel_idx]
    plt.figure(figsize=(5, 4))
    plt.imshow(img, aspect="auto", origin="lower", cmap="turbo", extent=[thetas[0], thetas[-1], lambdas[0], lambdas[-1]], vmin=0.0, vmax=vmax)
    plt.xlabel("theta (deg)")
    plt.ylabel("lambda (nm)")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_structure(path, x, title):
    img = x.squeeze()
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description="按 1250nm 二阶微分目标做扩散采样")
    p.add_argument("--stats", default="checkpoints/cond_stats.npz")
    p.add_argument("--forward_ckpt", default="checkpoints/forward_best.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion_best.pt")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--save_dir", default="samples/laplas")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rcwa_eval", action="store_true")
    a = p.parse_args()

    save_dir = Path(a.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    cond_ch = np.load(a.stats)["mean"].shape[1]
    target_raw, lambdas, thetas = build_target(cond_ch)
    target = normalize(target_raw, a.stats, a.device)
    weight = torch.from_numpy(build_weight(cond_ch)[None]).to(a.device)
    surrogate = load_model(a.forward_ckpt, ForwardSurrogate(cond_ch).to(a.device), "model", a.device)
    diffusion = load_model(a.diffusion_ckpt, GaussianDiffusion(ConditionalUNet(cond_ch).to(a.device), timesteps=1000, image_size=64).to(a.device), "diffusion", a.device)

    cond_batch = target.repeat(a.num_samples, 1, 1, 1)
    samples = diffusion.sample(cond_batch, cfg_scale=a.cfg_scale)
    pred = surrogate(samples)
    err = ((pred - cond_batch).abs() * weight).sum(dim=(1, 2, 3)) / weight.sum()
    topk = torch.topk(err, k=min(5, a.num_samples), largest=False).indices
    best = samples[topk[0]:topk[0] + 1]
    pred_raw = denormalize(pred.cpu().numpy(), a.stats)
    info = {"best_surrogate_mae_norm": float(err[topk[0]].item())}

    if a.rcwa_eval:
        out = rcwa_eval(best, target_raw, cond_ch, a.device)
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
    np.save(save_dir / "topk_indices.npy", topk.cpu().numpy())
    np.save(save_dir / "topk_samples.npy", samples[topk].cpu().numpy())
    np.save(save_dir / "topk_pred_cond.npy", pred[topk].cpu().numpy())
    np.save(save_dir / "topk_pred_cond_raw.npy", pred_raw[topk.cpu().numpy()])
    vmax_tpp = max(float(target_raw[0].max()), float(pred_raw[topk[0].item(), 0].max()), 1e-6)
    plot_map(save_dir / "target_tpp.png", target_raw, lambdas, thetas, "Target tpp_mag", vmax_tpp, channel_idx=0)
    plot_map(save_dir / "best_tpp.png", pred_raw[topk[0].item()], lambdas, thetas, "Best surrogate tpp_mag", vmax_tpp, channel_idx=0)
    if cond_ch == 2:
        vmax_tss = max(float(target_raw[1].max()), float(pred_raw[topk[0].item(), 1].max()), 1e-6)
        plot_map(save_dir / "target_tss.png", target_raw, lambdas, thetas, "Target tss_mag", vmax_tss, channel_idx=1)
        plot_map(save_dir / "best_tss.png", pred_raw[topk[0].item()], lambdas, thetas, "Best surrogate tss_mag", vmax_tss, channel_idx=1)
    plot_structure(save_dir / "best_structure.png", best.cpu().numpy(), "Best binary structure")
    if a.rcwa_eval and "rcwa_pred" in locals():
        row_tpp = np.zeros((1, len(thetas)), dtype=np.float32)
        row_tpp[0] = rcwa_pred[0]
        vmax_tpp = max(vmax_tpp, float(row_tpp.max()))
        plot_map(save_dir / "best_rcwa_tpp.png", row_tpp, np.array([1250.0], dtype=np.float32), thetas, "Best RCWA tpp_mag @1250nm", vmax_tpp)
        if cond_ch == 2:
            row_tss = np.zeros((1, len(thetas)), dtype=np.float32)
            row_tss[0] = rcwa_pred[1]
            vmax_tss = max(vmax_tss, float(row_tss.max()))
            plot_map(save_dir / "best_rcwa_tss.png", row_tss, np.array([1250.0], dtype=np.float32), thetas, "Best RCWA tss_mag @1250nm", vmax_tss)
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("saved_to:", save_dir)
    print(info)


if __name__ == "__main__":
    main()
