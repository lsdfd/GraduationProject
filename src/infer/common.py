# -*- coding: utf-8 -*-
"""Shared infer utilities: grids, scoring, plotting, normalization, RCWA eval."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from dataset.rcwa.rcwa import torcwa_simulation
except Exception:
    torcwa_simulation = None


def lambda_theta_grid() -> tuple[np.ndarray, np.ndarray]:
    lambdas = np.asarray([1000.0], dtype=np.float32)
    thetas = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)
    return lambdas, thetas


def load_model(path: str, model: torch.nn.Module, key: str, device: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location=device)[key])
    return model.eval()


def load_stats(stats_path: str) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(stats_path)
    mean = stats["mean"].astype(np.float32)
    std = stats["std"].astype(np.float32)
    return mean, std


def normalize_with_stats(target: np.ndarray, mean: np.ndarray, std: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy((target[None] - mean) / std).to(device)


def denormalize_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def build_weight(cond_ch: int, target_lambda: float = 1000.0) -> np.ndarray:
    _, thetas = lambda_theta_grid()
    w = np.ones((len(thetas),), dtype=np.float32)
    return np.stack([w, w], axis=0) if cond_ch == 2 else w[None]


def second_order_target(thetas: np.ndarray) -> np.ndarray:
    sin_theta = np.sin(np.deg2rad(thetas.astype(np.float64)))
    smax = max(float(np.max(np.abs(sin_theta))), 1e-8)
    x = (np.abs(sin_theta) / smax) ** 2
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    return x.astype(np.float64)


def _outer_band_pairs(thetas: np.ndarray) -> list[tuple[int, int]]:
    thetas = np.asarray(thetas, dtype=np.float64)
    idx30 = int(np.argmin(np.abs(np.abs(thetas) - 30.0)))
    idx35 = int(np.argmin(np.abs(np.abs(thetas) - 35.0)))
    idx40 = int(np.argmin(np.abs(np.abs(thetas) - 40.0)))
    return [(idx30, idx35), (idx35, idx40)]


def second_order_score_row(
    y: np.ndarray,
    thetas: np.ndarray,
    w_center: float = 0.50,
    w_shape: float = 0.20,
    w_edge: float = 0.15,
    w_outer: float = 0.15,
) -> dict:
    y = y.astype(np.float64)
    x = second_order_target(thetas)
    if (not np.isfinite(y).all()) or float(np.max(y)) <= 0:
        return {"score": -1.0, "center": 0.0, "shape": 0.0, "edge": 0.0, "outer": 0.0, "r2": -1.0}

    y_norm = y / max(float(np.max(y)), 1e-8)
    denom = max(float(np.sum(x * x)), 1e-8)
    a = float(np.sum(x * y_norm) / denom)
    y_fit = a * x

    ss_res = float(np.sum((y_norm - y_fit) ** 2))
    ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    center_idx = int(np.argmin(np.abs(thetas)))
    edge_mask = np.abs(thetas) >= 0.85 * float(np.max(np.abs(thetas)))
    if not edge_mask.any():
        edge_mask[[0, -1]] = True

    edge_mean = float(np.mean(y[edge_mask]))
    center_val = float(y[center_idx])
    center_score = float(np.clip(1.0 - center_val / max(edge_mean, 1e-8), 0.0, 1.0))
    shape_score = float(np.clip(r2, 0.0, 1.0)) if a >= 0 else 0.0
    edge_score = float(np.clip(edge_mean, 0.0, 1.0))
    outer_vals = np.asarray([y_norm[i] for i, _ in _outer_band_pairs(thetas)] + [y_norm[_outer_band_pairs(thetas)[-1][1]]], dtype=np.float64)
    outer_rank_terms = []
    for lo, hi in _outer_band_pairs(thetas):
        outer_rank_terms.append(float(np.clip(y_norm[hi] - y_norm[lo], 0.0, 1.0)))
    outer_score = float(np.mean(outer_rank_terms)) if outer_rank_terms else 0.0
    return {
        "score": float(w_center * center_score + w_shape * shape_score + w_edge * edge_score + w_outer * outer_score),
        "center": float(center_score),
        "shape": float(shape_score),
        "edge": float(edge_score),
        "outer": float(outer_score),
        "r2": float(r2),
    }


def second_order_score_row_torch(
    y: torch.Tensor,
    thetas: np.ndarray,
    w_center: float = 0.50,
    w_shape: float = 0.20,
    w_edge: float = 0.15,
    w_outer: float = 0.15,
) -> dict[str, torch.Tensor]:
    if y.ndim == 1:
        y = y.unsqueeze(0)

    target = torch.as_tensor(second_order_target(thetas), device=y.device, dtype=y.dtype).unsqueeze(0)
    row_max = y.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    y_norm = y / row_max
    denom = (target * target).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    a = (y_norm * target).sum(dim=-1, keepdim=True) / denom
    y_fit = a * target

    ss_res = ((y_norm - y_fit) ** 2).sum(dim=-1)
    y_mean = y_norm.mean(dim=-1, keepdim=True)
    ss_tot = ((y_norm - y_mean) ** 2).sum(dim=-1).clamp_min(1e-8)
    r2 = 1.0 - ss_res / ss_tot

    thetas_arr = np.asarray(thetas, dtype=np.float64)
    center_idx = int(np.argmin(np.abs(thetas_arr)))
    edge_mask = np.abs(thetas_arr) >= 0.85 * float(np.max(np.abs(thetas_arr)))
    if not edge_mask.any():
        edge_mask[[0, -1]] = True
    edge_mask_t = torch.as_tensor(edge_mask, device=y.device, dtype=torch.bool)

    edge_mean = y[:, edge_mask_t].mean(dim=-1)
    center_val = y[:, center_idx]
    center_score = (1.0 - center_val / edge_mean.clamp_min(1e-8)).clamp(0.0, 1.0)
    shape_score = torch.where(a.squeeze(-1) >= 0.0, r2.clamp(0.0, 1.0), torch.zeros_like(r2))
    edge_score = edge_mean.clamp(0.0, 1.0)
    outer_pairs = _outer_band_pairs(thetas_arr)
    outer_terms = []
    for lo, hi in outer_pairs:
        outer_terms.append((y_norm[:, hi] - y_norm[:, lo]).clamp(0.0, 1.0))
    outer_score = torch.stack(outer_terms, dim=0).mean(dim=0) if outer_terms else torch.zeros_like(center_score)
    score = w_center * center_score + w_shape * shape_score + w_edge * edge_score + w_outer * outer_score
    return {
        "score": score,
        "center": center_score,
        "shape": shape_score,
        "edge": edge_score,
        "outer": outer_score,
        "r2": r2,
    }


def rcwa_eval_target_lambda(
    structure: torch.Tensor,
    target_raw: np.ndarray,
    cond_ch: int,
    device: str,
    target_lambda: float = 1000.0,
    rcwa_orders: int = 7,
):
    if torcwa_simulation is None:
        return None
    _, thetas = lambda_theta_grid()
    layer = structure.squeeze().to(device)
    tpp = np.full((len(thetas),), np.nan, np.float32)
    tss = np.full_like(tpp, np.nan)
    for j, theta in enumerate(thetas):
        out = torcwa_simulation(
            {
                "periodicity": 500.0,
                "h": 500.0,
                "lam": float(target_lambda),
                "tet": float(theta),
                "phi": 0.0,
                "angle_unit": "deg",
                "angle_layer": "input",
                "input_medium": "air",
                "output_medium": "SiO2",
                "structure": "Si",
                "n_input": 1.0,
                "n_output": 1.45,
                "n_structure": 3.4,
            },
            layer,
            rcwa_orders=rcwa_orders,
            project=False,
            device=device,
        )
        tpp[j] = float(out["tpp_mag"].detach().cpu().item())
        tss[j] = float(out["tss_mag"].detach().cpu().item())
    pred = np.stack([tpp, tss], axis=0) if cond_ch == 2 else tpp[None]
    if target_raw.ndim == 2:
        target_row = target_raw
    elif target_raw.ndim == 3 and target_raw.shape[0] == 1:
        target_row = target_raw[0]
    else:
        target_row = target_raw[:, int(np.argmin(np.abs(lambda_theta_grid()[0] - float(target_lambda)))), :]
    mae = float(np.mean(np.abs(pred - target_row)))
    return mae, pred


def rcwa_second_order_metrics_target_lambda(
    structure: torch.Tensor,
    target_raw: np.ndarray,
    cond_ch: int,
    device: str,
    target_lambda: float = 1000.0,
    rcwa_orders: int = 7,
) -> dict | None:
    out = rcwa_eval_target_lambda(
        structure,
        target_raw,
        cond_ch,
        device,
        target_lambda=target_lambda,
        rcwa_orders=rcwa_orders,
    )
    if out is None:
        return None
    mae, pred = out
    _, thetas = lambda_theta_grid()
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    score = second_order_score_row(pred[0], thetas)
    result = {
        "rcwa_mae_raw": float(mae),
        "rcwa_second_order_score": float(score["score"]),
        "rcwa_center_score": float(score["center"]),
        "rcwa_shape_score": float(score["shape"]),
        "rcwa_edge_score": float(score["edge"]),
        "rcwa_r2": float(score["r2"]),
        "rcwa_tpp_at_40": float(pred[0, t40_idx]),
        "rcwa_pred": pred,
    }
    if cond_ch == 2:
        result["rcwa_tss_at_40"] = float(pred[1, t40_idx])
    return result


def plot_map(path: Path, cond: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray, title: str, vmax: float, channel_idx: int = 0):
    img = cond if cond.ndim == 2 else cond[channel_idx]
    if len(lambdas) == 1:
        half_step = 5.0
        y0 = float(lambdas[0] - half_step)
        y1 = float(lambdas[0] + half_step)
    else:
        y0 = float(lambdas[0])
        y1 = float(lambdas[-1])
    plt.figure(figsize=(5, 4))
    plt.imshow(
        img,
        aspect="auto",
        origin="lower",
        cmap="turbo",
        extent=[float(thetas[0]), float(thetas[-1]), y0, y1],
        vmin=0.0,
        vmax=vmax,
    )
    plt.xlabel("theta (deg)")
    plt.ylabel("lambda (nm)")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_structure(path: Path, x: np.ndarray, title: str):
    img = x.squeeze()
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
