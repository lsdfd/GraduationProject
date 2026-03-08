# -*- coding: utf-8 -*-
"""用前向代理模型对扩散生成结构做轻量拓扑优化。"""

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

try:
    from dataset.rcwa.rcwa import torcwa_simulation
except Exception:
    torcwa_simulation = None


def latest_laplas_file(name):
    root = ROOT / "samples" / "laplas"
    runs = sorted([p for p in root.glob("*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError("未找到 samples/laplas 下的推理结果，请先运行 python src/infer/laplas.py")
    path = runs[-1] / name
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    return str(path)


def load_target(path, stats_path, device):
    x = np.load(path).astype(np.float32)
    x = x[0] if x.ndim == 4 else x
    stats = np.load(stats_path)
    x = (x[None] - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)
    return torch.from_numpy(x).to(device), stats


def build_weight(cond_ch):
    lambdas = np.arange(1000.0, 1500.1, 50.0, dtype=np.float32)
    thetas = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)
    w = np.full((len(lambdas), len(thetas)), 0.05, dtype=np.float32)
    w[np.argmin(np.abs(lambdas - 1250.0))] = 1.0
    return torch.from_numpy((np.stack([w, w], axis=0) if cond_ch == 2 else w[None])[None])


def load_init(path, device):
    x = np.load(path).astype(np.float32)
    x = x[0] if x.ndim == 4 else x
    x = x[0] if x.ndim == 3 else x
    return torch.from_numpy(x[None, None]).to(device)


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


def plot_structure(path, x, title, binary=False):
    img = x.squeeze()
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def eval_rcwa(x, target_raw, device):
    if torcwa_simulation is None:
        return None
    layer = x.squeeze().to(device)
    real = np.full((17,), np.nan, np.float32)
    imag = np.full_like(real, np.nan)
    lam = 1250.0
    for j, theta in enumerate(np.arange(-40.0, 40.1, 5.0)):
        out = torcwa_simulation({"periodicity": 500.0, "h": 500.0, "lam": lam, "tet": theta, "phi": 0.0, "angle_unit": "deg", "angle_layer": "input", "input_medium": "air", "output_medium": "SiO2", "structure": "Si", "n_input": 1.0, "n_output": 1.45, "n_structure": 3.4}, layer, rcwa_orders=7, project=False, device=device)
        z = complex(out["tpp"].detach().cpu().item())
        real[j], imag[j] = z.real, z.imag
    i = int(np.argmin(np.abs(np.arange(1000.0, 1500.1, 50.0) - 1250.0)))
    pred = np.stack([real, imag], axis=0) if target_raw.shape[0] == 2 else np.sqrt(real ** 2 + imag ** 2)[None]
    return float(np.mean(np.abs(pred - target_raw[:, i])))


def main():
    p = argparse.ArgumentParser(description="用 surrogate 对初始结构做后处理优化")
    p.add_argument("--target")
    p.add_argument("--init")
    p.add_argument("--stats", default="checkpoints/cond_stats.npz")
    p.add_argument("--forward_ckpt", default="checkpoints/forward_best.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--target_mae", type=float, default=0.01)
    p.add_argument("--save_dir", default="samples/optimized")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_rcwa_eval", action="store_true")
    a = p.parse_args()

    save_dir = Path(a.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    a.target = a.target or latest_laplas_file("target_cond_raw.npy")
    a.init = a.init or latest_laplas_file("topk_samples.npy")
    target, stats = load_target(a.target, a.stats, a.device)
    target_raw = np.load(a.target).astype(np.float32)
    init = load_init(a.init, a.device).clamp(0, 1)
    cond_ch = target.shape[1]
    weight = build_weight(cond_ch).to(a.device)
    surrogate = ForwardSurrogate(cond_ch).to(a.device)
    surrogate.load_state_dict(torch.load(a.forward_ckpt, map_location=a.device)["model"])
    surrogate.eval()
    for p_ in surrogate.parameters():
        p_.requires_grad = False

    logits = torch.logit(init.clamp(1e-4, 1 - 1e-4), eps=1e-4).requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=a.lr)
    hist, best = [], (1e9, None, None)
    for step in range(a.steps):
        x = project(torch.sigmoid(logits), beta=min(4.0 + step / 20.0, 20.0))
        pred = surrogate(x)
        loss_fit = ((pred - target).abs() * weight).sum() / weight.sum()
        loss_bg = (((pred - target).abs() * (1.0 - weight / weight.max()))).mean()
        loss_bin = (x * (1 - x)).mean()
        loss_sym = (x - symmetrize(x)).abs().mean()
        loss_tv = tv_loss(x)
        loss_feat = min_feature_loss(x)
        loss = loss_fit + 0.20 * loss_bg + 0.15 * loss_bin + 0.08 * loss_sym + 0.05 * loss_tv + 0.08 * loss_feat
        opt.zero_grad()
        loss.backward()
        opt.step()
        item = {"step": step, "loss": float(loss.item()), "fit": float(loss_fit.item()), "bg": float(loss_bg.item()), "bin": float(loss_bin.item()), "sym": float(loss_sym.item()), "tv": float(loss_tv.item()), "feat": float(loss_feat.item())}
        hist.append(item)
        if item["loss"] < best[0]:
            best = (item["loss"], x.detach().clone(), pred.detach().clone())
        if step % 20 == 0 or step + 1 == a.steps:
            print(f"[opt] step={step:03d} loss={item['loss']:.4f} fit={item['fit']:.4f} bg={item['bg']:.4f} bin={item['bin']:.4f} sym={item['sym']:.4f}")
        if item["fit"] <= a.target_mae:
            print(f"[opt] early stop at step={step:03d}, fit={item['fit']:.4f} <= target_mae={a.target_mae:.4f}")
            break

    best_x, best_pred = best[1], best[2]
    best_bin = finalize_binary(best_x)
    mean, std = stats["mean"].astype(np.float32), stats["std"].astype(np.float32)
    pred_raw = best_pred.cpu().numpy() * std + mean
    target_raw = target_raw[None] if target_raw.ndim == 3 else target_raw
    out = {"surrogate_mae_norm": float((((best_pred - target).abs() * weight).sum() / weight.sum()).item()), "surrogate_mae_raw": float(np.mean(np.abs(pred_raw - target_raw)))}
    if not a.skip_rcwa_eval:
        out["rcwa_mae_raw"] = eval_rcwa(best_bin, target_raw[0], a.device)

    np.save(save_dir / "optimized_continuous.npy", best_x.cpu().numpy())
    np.save(save_dir / "optimized_binary.npy", best_bin.cpu().numpy())
    np.save(save_dir / "optimized_pred_cond.npy", best_pred.cpu().numpy())
    np.save(save_dir / "target_cond.npy", target.cpu().numpy())
    plot_structure(save_dir / "optimized_continuous.png", best_x.cpu().numpy(), "Optimized continuous")
    plot_structure(save_dir / "optimized_binary.png", best_bin.cpu().numpy(), "Optimized binary", binary=True)
    with open(save_dir / "optimization_log.json", "w", encoding="utf-8") as f:
        json.dump({"target": a.target, "init": a.init, "metrics": out, "history_tail": hist[-20:]}, f, ensure_ascii=False, indent=2)
    print("saved_to:", save_dir)
    print("target_from:", a.target)
    print("init_from:", a.init)
    print(out)


if __name__ == "__main__":
    main()
