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

from dataset.rcwa.rcwa import torcwa_simulation  # noqa: E402
from infer.common import (  # noqa: E402
    load_model,
    load_stats,
    plot_map,
    plot_structure,
    resolve_default_diffusion_ckpt,
    resolve_default_stats_path,
)
from infer.task_library import (  # noqa: E402
    TaskCase,
    build_case_weight,
    case_output_dir,
    default_train_npz,
    list_task_keys,
    original_input_score,
    select_cases,
    task_score_details,
    task_score_details_at_lambda,
)
from model.diffusion import GaussianDiffusion  # noqa: E402
from model.models import ConditionalUNet  # noqa: E402


def load_target_from_dataset(train_npz: Path, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(train_npz)
    structure = np.asarray(data["structures"][sample_idx], dtype=np.float32)
    tpp = np.asarray(data["tpp_mag"][sample_idx], dtype=np.float32)
    tss = np.asarray(data["tss_mag"][sample_idx], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    return structure, np.stack([tpp, tss], axis=0), lambdas, thetas


def selected_pairs_from_weight(weight: np.ndarray) -> list[tuple[int, int]]:
    active = np.any(weight > 0, axis=0)
    return [(i, j) for i, j in zip(*np.where(active))]


def rcwa_physics_kwargs(target_lambda: float, theta: float) -> dict:
    return {
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
    }


def simulate_selected_map(
    structure: torch.Tensor,
    cond_ch: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    selected_pairs: list[tuple[int, int]],
    device: str,
    rcwa_orders: int,
) -> np.ndarray:
    layer = structure.squeeze(0).squeeze(0)
    pred = np.full((cond_ch, len(lambdas), len(thetas)), np.nan, dtype=np.float32)
    for lam_idx, theta_idx in selected_pairs:
        out = torcwa_simulation(
            rcwa_physics_kwargs(float(lambdas[lam_idx]), float(thetas[theta_idx])),
            layer,
            rcwa_orders=rcwa_orders,
            project=False,
            device=device,
        )
        pred[0, lam_idx, theta_idx] = float(out["tpp_mag"].detach().cpu().item())
        if cond_ch > 1:
            pred[1, lam_idx, theta_idx] = float(out["tss_mag"].detach().cpu().item())
    return pred


def simulate_full_map(
    structure: torch.Tensor,
    cond_ch: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    device: str,
    rcwa_orders: int,
) -> np.ndarray:
    all_pairs = [(i, j) for i in range(len(lambdas)) for j in range(len(thetas))]
    return simulate_selected_map(structure, cond_ch, lambdas, thetas, all_pairs, device, rcwa_orders)


def simulate_lambda_rows(
    structure: torch.Tensor,
    cond_ch: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lam_indices: list[int],
    device: str,
    rcwa_orders: int,
) -> np.ndarray:
    selected_pairs = [(int(lam_idx), theta_idx) for lam_idx in lam_indices for theta_idx in range(len(thetas))]
    return simulate_selected_map(structure, cond_ch, lambdas, thetas, selected_pairs, device, rcwa_orders)


def weighted_mae_numpy(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    valid = np.isfinite(pred) & np.isfinite(target)
    if float(np.sum(weight)) <= 1e-8:
        if not np.any(valid):
            return float("nan")
        return float(np.mean(np.abs(pred[valid] - target[valid])))
    valid &= weight > 0
    if not np.any(valid):
        return float("nan")
    w = weight[valid]
    return float(np.sum(np.abs(pred[valid] - target[valid]) * w) / np.sum(w))


def normalized_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    valid = np.isfinite(row)
    if not np.any(valid):
        return np.full_like(row, np.nan, dtype=np.float64)
    scale = max(float(np.max(row[valid])), 1e-8)
    out = np.full_like(row, np.nan, dtype=np.float64)
    out[valid] = row[valid] / scale
    return out


def format_metric(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{float(value):.3f}"


def second_like_curve(thetas: np.ndarray, order: int) -> np.ndarray:
    thetas = np.asarray(thetas, dtype=np.float64)
    tmax = float(np.max(np.abs(thetas)))
    if tmax <= 0:
        return np.zeros_like(thetas, dtype=np.float64)
    kx = np.sin(np.deg2rad(thetas)) / np.sin(np.deg2rad(tmax))
    x = np.abs(kx) ** order
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-8)


def lowpass_curve(thetas: np.ndarray, sigma_deg: float) -> np.ndarray:
    thetas = np.asarray(thetas, dtype=np.float64)
    x = np.exp(-(thetas ** 2) / max(float(sigma_deg) ** 2, 1e-8))
    return x / max(float(np.max(x)), 1e-8)


def best_lowpass_sigma(target_row: np.ndarray, thetas: np.ndarray, candidates: tuple[float, ...] = (8.0, 12.0, 16.0)) -> float:
    y = normalized_row(target_row)
    valid = np.isfinite(y)
    if not np.any(valid):
        return float(candidates[0])
    best_sigma = float(candidates[0])
    best_err = float("inf")
    for sigma in candidates:
        ref = lowpass_curve(thetas, sigma)
        err = float(np.mean(np.abs(y[valid] - ref[valid])))
        if err < best_err:
            best_err = err
            best_sigma = float(sigma)
    return best_sigma


def ideal_curve_for_case(case: TaskCase, thetas: np.ndarray, target_row: np.ndarray | None = None) -> np.ndarray | None:
    if case.task_key in {"p_second_order", "polarization_independent", "polarization_multiplexed"}:
        return second_like_curve(thetas, 2)
    if case.task_key == "fourth_order":
        return second_like_curve(thetas, 4)
    if case.task_key == "lowpass":
        sigma = best_lowpass_sigma(target_row, thetas) if target_row is not None else 12.0
        return lowpass_curve(thetas, sigma)
    return None


def use_raw_curve_display(case: TaskCase, channel_idx: int) -> bool:
    return case.task_key == "polarization_multiplexed" and int(channel_idx) == 1


def curve_display_row(case: TaskCase, channel_idx: int, row: np.ndarray) -> np.ndarray:
    arr = np.asarray(row, dtype=np.float64)
    if use_raw_curve_display(case, channel_idx):
        out = np.full_like(arr, np.nan, dtype=np.float64)
        valid = np.isfinite(arr)
        out[valid] = arr[valid]
        return out
    return normalized_row(arr)


def curve_ylim(case: TaskCase, channel_idx: int) -> tuple[float, float]:
    if use_raw_curve_display(case, channel_idx):
        return 0.0, 1.0
    return -0.05, 1.05


def curve_ylabel(case: TaskCase, channel_idx: int) -> str:
    if use_raw_curve_display(case, channel_idx):
        return "magnitude"
    return "normalized magnitude"


def show_ideal_curve(case: TaskCase, channel_idx: int) -> bool:
    return not (case.task_key == "polarization_multiplexed" and int(channel_idx) == 1)


def plot_target_lambda_curves(
    out_png: Path,
    case: TaskCase,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    thetas: np.ndarray,
    lambdas: np.ndarray,
    title: str,
    score_info: dict[str, float],
) -> None:
    lam_idx = int(np.argmin(np.abs(lambdas - float(case.target_lambda_nm))))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    names = [("tpp", 0, "#1f77b4"), ("tss", 1, "#ff7f0e")]
    for ax, (label, ch, color) in zip(axes, names):
        target_row = curve_display_row(case, ch, target_raw[ch, lam_idx])
        pred_row = curve_display_row(case, ch, pred_raw[ch, lam_idx])
        ideal_row = ideal_curve_for_case(case, thetas, target_raw[ch, lam_idx])
        theta40 = float(thetas[idx40])
        target40_raw = float(target_raw[ch, lam_idx, idx40])
        pred40_raw = float(pred_raw[ch, lam_idx, idx40]) if np.isfinite(pred_raw[ch, lam_idx, idx40]) else float("nan")
        target40 = float(target_row[idx40])
        pred40 = float(pred_row[idx40]) if np.isfinite(pred_row[idx40]) else float("nan")
        if ideal_row is not None and show_ideal_curve(case, ch):
            ax.plot(thetas, ideal_row, color="#2ca02c", ls=":", lw=1.6, label="ideal")
        ax.plot(thetas, target_row, "k--", lw=1.6, label="target")
        pred_valid = np.isfinite(pred_row)
        if np.any(pred_valid):
            ax.plot(thetas[pred_valid], pred_row[pred_valid], lw=1.9, color=color, label="pred")
        ax.axvline(theta40, color="0.5", ls=":", lw=1.0)
        ax.scatter([theta40], [target40], color="k", s=24, zorder=3)
        if np.isfinite(pred40):
            ax.scatter([theta40], [pred40], color=color, s=24, zorder=3)
        ax.set_ylim(*curve_ylim(case, ch))
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(curve_ylabel(case, ch))
        ax.set_title(
            f"{label} @ {float(case.target_lambda_nm):.0f}nm\n"
            f"target40={target40_raw:.3f} pred40={format_metric(pred40_raw)}"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        ax.annotate(
            f"40° target={target40_raw:.3f}",
            xy=(theta40, target40),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color="k",
            ha="left",
            va="bottom",
        )
        if np.isfinite(pred40):
            ax.annotate(
                f"40° pred={pred40_raw:.3f}",
                xy=(theta40, pred40),
                xytext=(8, -16),
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha="left",
                va="top",
            )
        else:
            ax.text(0.98, 0.03, "pred: not evaluated", transform=ax.transAxes, fontsize=8, color=color, ha="right", va="bottom")
        ax.text(
            0.02,
            0.03,
            f"score={float(score_info.get('task_score', np.nan)):.3f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ideal_pred_curves(
    out_png: Path,
    case: TaskCase,
    pred_raw: np.ndarray,
    thetas: np.ndarray,
    lambdas: np.ndarray,
    title: str,
    score_info: dict[str, float],
) -> None:
    if case.task_key == "st2":
        return
    lam_idx = int(np.argmin(np.abs(lambdas - float(case.target_lambda_nm))))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    names = [("tpp", 0, "#1f77b4"), ("tss", 1, "#ff7f0e")]
    for ax, (label, ch, color) in zip(axes, names):
        ideal_row = ideal_curve_for_case(case, thetas, pred_raw[ch, lam_idx])
        pred_row = curve_display_row(case, ch, pred_raw[ch, lam_idx])
        pred40_raw = float(pred_raw[ch, lam_idx, idx40]) if np.isfinite(pred_raw[ch, lam_idx, idx40]) else float("nan")
        pred40 = float(pred_row[idx40]) if np.isfinite(pred_row[idx40]) else float("nan")
        theta40 = float(thetas[idx40])
        if ideal_row is not None and show_ideal_curve(case, ch):
            ax.plot(thetas, ideal_row, color="#2ca02c", ls=":", lw=1.6, label="ideal")
        pred_valid = np.isfinite(pred_row)
        if np.any(pred_valid):
            ax.plot(thetas[pred_valid], pred_row[pred_valid], lw=1.9, color=color, label="pred")
        ax.axvline(theta40, color="0.5", ls=":", lw=1.0)
        if np.isfinite(pred40):
            ax.scatter([theta40], [pred40], color=color, s=24, zorder=3)
        ax.set_ylim(*curve_ylim(case, ch))
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(curve_ylabel(case, ch))
        ax.set_title(f"{label} @ {float(case.target_lambda_nm):.0f}nm\npred40={format_metric(pred40_raw)}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        if np.isfinite(pred40):
            ax.annotate(
                f"40° pred={pred40_raw:.3f}",
                xy=(theta40, pred40),
                xytext=(8, -16),
                textcoords="offset points",
                fontsize=8,
                color=color,
                ha="left",
                va="top",
            )
        else:
            ax.text(0.98, 0.03, "pred: not evaluated", transform=ax.transAxes, fontsize=8, color=color, ha="right", va="bottom")
        ax.text(0.02, 0.03, f"score={float(score_info.get('task_score', np.nan)):.3f}", transform=ax.transAxes, fontsize=8, ha="left", va="bottom")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def band_lambda_indices(case: TaskCase, lambdas: np.ndarray) -> list[int]:
    targets = [float(case.target_lambda_nm) - 50.0, float(case.target_lambda_nm), float(case.target_lambda_nm) + 50.0]
    return [int(np.argmin(np.abs(lambdas.astype(np.float64) - target))) for target in targets]


def band_lambda_labels() -> tuple[str, str, str]:
    return ("lambda_minus_50", "lambda_target", "lambda_plus_50")


def build_band_score_payload(case: TaskCase, pred_raw: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray, lam_indices: list[int]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    targets = [float(case.target_lambda_nm) - 50.0, float(case.target_lambda_nm), float(case.target_lambda_nm) + 50.0]
    for key, target_nm, lam_idx in zip(band_lambda_labels(), targets, lam_indices):
        details = dict(task_score_details_at_lambda(case, pred_raw, lambdas, thetas, lam_idx))
        details["lambda_nm_target"] = float(target_nm)
        details["lambda_nm_actual"] = float(lambdas[int(lam_idx)])
        payload[key] = details
    return payload


def plot_band_curves(
    out_png: Path,
    case: TaskCase,
    band_pred_rows: np.ndarray,
    lambdas_nm: np.ndarray,
    thetas: np.ndarray,
    title: str,
) -> None:
    if case.task_key == "st2":
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), constrained_layout=True)
    names = [("tpp", 0), ("tss", 1)]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    ideal_row = ideal_curve_for_case(case, thetas, band_pred_rows[0, 0])
    for ax, (label, ch) in zip(axes, names):
        if ideal_row is not None and show_ideal_curve(case, ch):
            ax.plot(thetas, ideal_row, color="#2ca02c", ls=":", lw=1.5, label="ideal")
        for i, color in enumerate(colors):
            pred_row = curve_display_row(case, ch, band_pred_rows[i, ch])
            valid = np.isfinite(pred_row)
            if np.any(valid):
                ax.plot(thetas[valid], pred_row[valid], color=color, lw=1.8, label=f"{float(lambdas_nm[i]):.0f}nm")
        ax.set_ylim(*curve_ylim(case, ch))
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel(curve_ylabel(case, ch))
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_full_map_top3_overview(
    out_png: Path,
    case: TaskCase,
    samples: np.ndarray,
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    ranked_idx: np.ndarray,
    topn: int,
) -> None:
    topn = min(topn, len(ranked_idx))
    if topn <= 0:
        return
    target_lambda = float(case.target_lambda_nm)
    lam_idx = int(np.argmin(np.abs(lambdas - target_lambda)))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    vmax_tpp = max(float(np.nanmax(target_raw[0])), float(np.nanmax(pred_raw[:, 0])) if pred_raw.size else 0.0, 1e-6)
    vmax_tss = max(float(np.nanmax(target_raw[1])), float(np.nanmax(pred_raw[:, 1])) if pred_raw.size else 0.0, 1e-6)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f2f2f2")
    fig, axes = plt.subplots(topn, 5, figsize=(22.0, max(2.8 * topn, 6.0)), gridspec_kw={"width_ratios": [0.7, 1.0, 1.0, 1.0, 1.0]}, constrained_layout=True)
    axes = np.atleast_2d(axes)
    hm_tpp = hm_tss = None
    ideal_row_tpp = ideal_curve_for_case(case, thetas, target_raw[0, lam_idx])
    ideal_row_tss = ideal_curve_for_case(case, thetas, target_raw[1, lam_idx])
    for r, idx in enumerate(ranked_idx[:topn]):
        a_struct, a_hmtpp, a_curve_tpp, a_hmtss, a_curve_tss = axes[r]
        i = int(idx)
        a_struct.imshow(samples[i, 0], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        a_struct.set_title(f"id={i}", fontsize=10)
        a_struct.axis("off")
        hm_tpp = a_hmtpp.imshow(np.ma.masked_invalid(pred_raw[r, 0]), cmap=cmap, aspect="auto", origin="lower", extent=extent, vmin=0.0, vmax=vmax_tpp, interpolation="bicubic")
        a_hmtpp.axhline(target_lambda, color="w", ls="--", lw=1.0)
        a_hmtpp.set_title("tpp full-map", fontsize=9)
        a_hmtpp.set_xlabel("theta")
        a_hmtpp.set_ylabel("lambda (nm)")
        if ideal_row_tpp is not None and show_ideal_curve(case, 0):
            a_curve_tpp.plot(thetas, ideal_row_tpp, color="#2ca02c", ls=":", lw=1.4, label="ideal")
        pred_tpp_row = curve_display_row(case, 0, pred_raw[r, 0, lam_idx])
        valid_tpp = np.isfinite(pred_tpp_row)
        if np.any(valid_tpp):
            a_curve_tpp.plot(thetas[valid_tpp], pred_tpp_row[valid_tpp], lw=1.8, color="#1f77b4", label="pred")
        a_curve_tpp.set_ylim(*curve_ylim(case, 0))
        a_curve_tpp.set_xlabel("theta")
        a_curve_tpp.set_title(f"tpp pred40={format_metric(pred_raw[r, 0, lam_idx, idx40])}", fontsize=9)
        a_curve_tpp.grid(alpha=0.25)
        if r == 0:
            a_curve_tpp.legend(fontsize=7, loc="lower right")
        hm_tss = a_hmtss.imshow(np.ma.masked_invalid(pred_raw[r, 1]), cmap=cmap, aspect="auto", origin="lower", extent=extent, vmin=0.0, vmax=vmax_tss, interpolation="bicubic")
        a_hmtss.axhline(target_lambda, color="w", ls="--", lw=1.0)
        a_hmtss.set_title("tss full-map", fontsize=9)
        a_hmtss.set_xlabel("theta")
        if ideal_row_tss is not None and show_ideal_curve(case, 1):
            a_curve_tss.plot(thetas, ideal_row_tss, color="#2ca02c", ls=":", lw=1.4, label="ideal")
        pred_tss_row = curve_display_row(case, 1, pred_raw[r, 1, lam_idx])
        valid_tss = np.isfinite(pred_tss_row)
        if np.any(valid_tss):
            a_curve_tss.plot(thetas[valid_tss], pred_tss_row[valid_tss], lw=1.8, color="#ff7f0e", label="pred")
        a_curve_tss.set_ylim(*curve_ylim(case, 1))
        a_curve_tss.set_xlabel("theta")
        a_curve_tss.set_title(f"tss pred40={format_metric(pred_raw[r, 1, lam_idx, idx40])}", fontsize=9)
        a_curve_tss.grid(alpha=0.25)
        if r == 0:
            a_curve_tss.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{case.case_label} Top-{topn} full-spectrum RCWA", fontsize=12)
    if hm_tpp is not None:
        fig.colorbar(hm_tpp, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp|")
    if hm_tss is not None:
        fig.colorbar(hm_tss, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label="|tss|")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ranked_overview(
    out_png: Path,
    case: TaskCase,
    samples: np.ndarray,
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    ranked_idx: np.ndarray,
    weighted_err: np.ndarray,
    global_err: np.ndarray,
    topk: int,
) -> None:
    if topk <= 0 or len(ranked_idx) == 0:
        return

    target_lambda = float(case.target_lambda_nm)
    lam_idx = int(np.argmin(np.abs(lambdas - target_lambda)))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))
    vmax_tpp = max(float(np.nanmax(target_raw[0])), float(np.nanmax(pred_raw[:, 0])) if pred_raw.size else 0.0, 1e-6)
    vmax_tss = max(float(np.nanmax(target_raw[1])), float(np.nanmax(pred_raw[:, 1])) if pred_raw.size else 0.0, 1e-6)
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]

    fig, axes = plt.subplots(
        topk,
        5,
        figsize=(22.0, max(2.8 * topk, 6.0)),
        gridspec_kw={"width_ratios": [0.7, 1.0, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    hm_tpp = hm_tss = None
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(color="#f2f2f2")

    target_tpp_row = curve_display_row(case, 0, target_raw[0, lam_idx])
    target_tss_row = curve_display_row(case, 1, target_raw[1, lam_idx])
    ideal_tpp_row = ideal_curve_for_case(case, thetas, target_raw[0, lam_idx])
    ideal_tss_row = ideal_curve_for_case(case, thetas, target_raw[1, lam_idx])

    for r, idx in enumerate(ranked_idx[:topk]):
        a_struct, a_hmtpp, a_curve_tpp, a_hmtss, a_curve_tss = axes[r]
        i = int(idx)
        a_struct.imshow(samples[i, 0], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        a_struct.set_title(f"id={i}", fontsize=10)
        a_struct.axis("off")

        hm_tpp = a_hmtpp.imshow(np.ma.masked_invalid(pred_raw[i, 0]), cmap=cmap, aspect="auto", origin="lower", extent=extent, vmin=0.0, vmax=vmax_tpp, interpolation="bicubic")
        a_hmtpp.axhline(target_lambda, color="w", ls="--", lw=1.0)
        a_hmtpp.set_title(f"werr={format_metric(weighted_err[i])}\ngerr={format_metric(global_err[i])}", fontsize=9)
        a_hmtpp.set_xlabel("theta")
        a_hmtpp.set_ylabel("lambda (nm)")

        if ideal_tpp_row is not None and show_ideal_curve(case, 0):
            a_curve_tpp.plot(thetas, ideal_tpp_row, color="#2ca02c", ls=":", lw=1.4, label="ideal")
        a_curve_tpp.plot(thetas, target_tpp_row, "k--", lw=1.5, label="target")
        pred_tpp_row = curve_display_row(case, 0, pred_raw[i, 0, lam_idx])
        pred_tpp_valid = np.isfinite(pred_tpp_row)
        if np.any(pred_tpp_valid):
            a_curve_tpp.plot(thetas[pred_tpp_valid], pred_tpp_row[pred_tpp_valid], lw=1.8, color="#1f77b4", label="pred")
        a_curve_tpp.set_ylim(*curve_ylim(case, 0))
        a_curve_tpp.set_xlabel("theta")
        a_curve_tpp.set_title(
            f"tpp t40={float(target_raw[0, lam_idx, idx40]):.3f} / p40={format_metric(pred_raw[i, 0, lam_idx, idx40])}",
            fontsize=9,
        )
        a_curve_tpp.grid(alpha=0.25)
        if r == 0:
            a_curve_tpp.legend(fontsize=7, loc="lower right")

        hm_tss = a_hmtss.imshow(np.ma.masked_invalid(pred_raw[i, 1]), cmap=cmap, aspect="auto", origin="lower", extent=extent, vmin=0.0, vmax=vmax_tss, interpolation="bicubic")
        a_hmtss.axhline(target_lambda, color="w", ls="--", lw=1.0)
        a_hmtss.set_title("tss", fontsize=9)
        a_hmtss.set_xlabel("theta")

        if ideal_tss_row is not None and show_ideal_curve(case, 1):
            a_curve_tss.plot(thetas, ideal_tss_row, color="#2ca02c", ls=":", lw=1.4, label="ideal")
        a_curve_tss.plot(thetas, target_tss_row, "k--", lw=1.5, label="target")
        pred_tss_row = curve_display_row(case, 1, pred_raw[i, 1, lam_idx])
        pred_tss_valid = np.isfinite(pred_tss_row)
        if np.any(pred_tss_valid):
            a_curve_tss.plot(thetas[pred_tss_valid], pred_tss_row[pred_tss_valid], lw=1.8, color="#ff7f0e", label="pred")
        a_curve_tss.set_ylim(*curve_ylim(case, 1))
        a_curve_tss.set_xlabel("theta")
        a_curve_tss.set_title(
            f"tss t40={float(target_raw[1, lam_idx, idx40]):.3f} / p40={format_metric(pred_raw[i, 1, lam_idx, idx40])}",
            fontsize=9,
        )
        a_curve_tss.grid(alpha=0.25)
        if r == 0:
            a_curve_tss.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"{case.case_label} Top-{min(topk, len(ranked_idx))} RCWA rerank", fontsize=12)
    if hm_tpp is not None:
        fig.colorbar(hm_tpp, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp|")
    if hm_tss is not None:
        fig.colorbar(hm_tss, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label="|tss|")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_case_visuals(
    save_dir: Path,
    case: TaskCase,
    target_structure: np.ndarray,
    target_raw: np.ndarray,
    all_samples: np.ndarray,
    topk_samples: np.ndarray,
    topk_pred_raw: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
) -> None:
    vmax = float(np.nanmax(target_raw)) if np.isfinite(target_raw).any() else 1.0
    vmax = max(vmax, 1e-6)
    plot_structure(save_dir / "target_structure.png", target_structure, f"{case.case_label} target structure")
    plot_map(save_dir / "target_tpp.png", target_raw, lambdas, thetas, f"{case.case_label} target tpp", vmax=vmax, channel_idx=0)
    plot_map(save_dir / "target_tss.png", target_raw, lambdas, thetas, f"{case.case_label} target tss", vmax=vmax, channel_idx=1)
    if len(all_samples):
        plot_structure(save_dir / "sample_00.png", all_samples[0], f"{case.case_label} sample0")
    if len(topk_samples):
        plot_structure(save_dir / "top1_structure.png", topk_samples[0], f"{case.case_label} top1")
        plot_map(
            save_dir / "top1_pred_tpp.png",
            topk_pred_raw[0],
            lambdas,
            thetas,
            f"{case.case_label} top1 RCWA tpp",
            vmax=vmax,
            channel_idx=0,
        )
        plot_map(
            save_dir / "top1_pred_tss.png",
            topk_pred_raw[0],
            lambdas,
            thetas,
            f"{case.case_label} top1 RCWA tss",
            vmax=vmax,
            channel_idx=1,
        )


def compute_topk_band_data(
    case: TaskCase,
    samples: torch.Tensor,
    topk_idx_np: np.ndarray,
    cond_channels: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    device: str,
    rcwa_orders: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, dict[str, float]]]]:
    lam_indices = band_lambda_indices(case, lambdas)
    topk_band_pred = np.full((len(topk_idx_np), len(lam_indices), cond_channels, len(thetas)), np.nan, dtype=np.float32)
    band_scores: list[dict[str, dict[str, float]]] = []
    for rank_pos, sample_idx in enumerate(topk_idx_np):
        if case.task_key == "st2":
            pred_for_score = simulate_full_map(
                samples[int(sample_idx) : int(sample_idx) + 1],
                cond_channels,
                lambdas,
                thetas,
                device,
                rcwa_orders,
            )
        else:
            pred_for_score = simulate_lambda_rows(
                samples[int(sample_idx) : int(sample_idx) + 1],
                cond_channels,
                lambdas,
                thetas,
                lam_indices,
                device,
                rcwa_orders,
            )
        for band_pos, lam_idx in enumerate(lam_indices):
            topk_band_pred[rank_pos, band_pos] = pred_for_score[:, lam_idx, :]
        band_scores.append(build_band_score_payload(case, pred_for_score, lambdas, thetas, lam_indices))
    return topk_band_pred, np.asarray([float(lambdas[idx]) for idx in lam_indices], dtype=np.float32), band_scores


def compute_top3_full_maps(
    samples: torch.Tensor,
    ranked_idx: np.ndarray,
    cond_channels: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    device: str,
    rcwa_orders: int,
    existing_pred_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    topn = min(3, len(ranked_idx))
    top_idx = np.asarray(ranked_idx[:topn], dtype=np.int64)
    if topn == 0:
        return top_idx, np.empty((0, cond_channels, len(lambdas), len(thetas)), dtype=np.float32)
    if existing_pred_raw is not None and existing_pred_raw.shape[1:] == (cond_channels, len(lambdas), len(thetas)) and np.isfinite(existing_pred_raw[top_idx]).all():
        return top_idx, existing_pred_raw[top_idx].astype(np.float32)
    preds = []
    for sample_idx in top_idx:
        preds.append(
            simulate_full_map(
                samples[int(sample_idx) : int(sample_idx) + 1],
                cond_channels,
                lambdas,
                thetas,
                device,
                rcwa_orders,
            )
        )
    return top_idx, np.stack(preds, axis=0).astype(np.float32)


@torch.no_grad()
def run_case(
    case: TaskCase,
    args,
    diffusion: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    train_npz: Path,
    root_save_dir: Path,
) -> dict:
    target_structure, target_raw, lambdas, thetas = load_target_from_dataset(train_npz, case.sample_idx)
    cond_channels = target_raw.shape[0]
    weight_np = build_case_weight(case, cond_channels, lambdas, thetas)
    selected_pairs = selected_pairs_from_weight(weight_np)

    target_norm = ((target_raw[None] - mean) / std).astype(np.float32)
    cond_batch = torch.from_numpy(target_norm).to(args.device).repeat(args.num_samples, 1, 1, 1)
    samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)

    rcwa_preds: list[np.ndarray] = []
    weighted_err = []
    global_err = []
    task_scores = []
    tpp_scores = []
    tss_scores = []
    task_details = []
    for idx in range(samples.shape[0]):
        print(f"[task_infer-rcwa] {case.case_label} sample {idx + 1}/{samples.shape[0]}", flush=True)
        if args.eval_mode == "full":
            pred = simulate_full_map(
                samples[idx : idx + 1],
                cond_channels,
                lambdas,
                thetas,
                args.device,
                args.rcwa_orders,
            )
        else:
            pred = simulate_selected_map(
                samples[idx : idx + 1],
                cond_channels,
                lambdas,
                thetas,
                selected_pairs,
                args.device,
                args.rcwa_orders,
            )
        rcwa_preds.append(pred)
        weighted_err.append(weighted_mae_numpy(pred, target_raw, weight_np))
        global_err.append(float(np.mean(np.abs(pred - target_raw))))
        details = task_score_details(case, pred, lambdas, thetas)
        task_details.append(details)
        if case.task_key == "st2":
            tpp_scores.append(float(details["tpp_score_total"]))
            tss_scores.append(float(details["tss_score_total"]))
            print(
                f"[task_infer-score] {case.case_label} sample {idx + 1}/{samples.shape[0]} "
                f"tpp_score={float(details['tpp_score_total']):.4f} "
                f"tss_score={float(details['tss_score_total']):.4f} "
                f"weighted_err={float(weighted_err[-1]):.4f}",
                flush=True,
            )
        else:
            task_scores.append(float(details["task_score"]))
            print(
                f"[task_infer-score] {case.case_label} sample {idx + 1}/{samples.shape[0]} "
                f"task_score={float(details['task_score']):.4f} weighted_err={float(weighted_err[-1]):.4f}",
                flush=True,
            )

    pred_raw = np.stack(rcwa_preds, axis=0).astype(np.float32)
    weighted_err_np = np.asarray(weighted_err, dtype=np.float32)
    global_err_np = np.asarray(global_err, dtype=np.float32)

    save_dir = case_output_dir(root_save_dir, case)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_samples = samples.cpu().numpy()

    np.save(save_dir / "target_cond_raw.npy", target_raw)
    np.save(save_dir / "target_cond_norm.npy", target_norm)
    np.save(save_dir / "target_structure.npy", target_structure)
    np.save(save_dir / "task_weight.npy", weight_np)
    np.save(save_dir / "all_samples.npy", all_samples)
    np.save(save_dir / "all_pred_cond_raw.npy", pred_raw)
    np.save(save_dir / "weighted_errors.npy", weighted_err_np)
    np.save(save_dir / "global_errors.npy", global_err_np)
    np.save(save_dir / "lambdas.npy", lambdas)
    np.save(save_dir / "thetas.npy", thetas)
    input_score = original_input_score(case, target_raw, lambdas, thetas)
    save_case_visuals(
        save_dir,
        case,
        target_structure,
        target_raw,
        all_samples,
        all_samples[:0],
        pred_raw[:0],
        lambdas,
        thetas,
    )
    summary = {
        "task_key": case.task_key,
        "task_label": case.task_label,
        "objective_key": case.objective_key,
        "score_source": case.score_source,
        "case_label": case.case_label,
        "sample_idx": int(case.sample_idx),
        "target_lambda_nm": float(case.target_lambda_nm),
        "note": case.note,
        "train_npz": str(train_npz),
        "diffusion_ckpt": str(args.diffusion_ckpt),
        "stats_path": str(args.stats_path),
        "rcwa_orders": int(args.rcwa_orders),
        "eval_mode": args.eval_mode,
        "num_samples": int(args.num_samples),
        "original_input_score": input_score,
    }

    if case.task_key == "st2":
        tpp_scores_np = np.asarray(tpp_scores, dtype=np.float32)
        tss_scores_np = np.asarray(tss_scores, dtype=np.float32)
        tpp_rank_np = np.argsort(tpp_scores_np)[::-1]
        tss_rank_np = np.argsort(tss_scores_np)[::-1]
        tpp_topk_idx_np = tpp_rank_np[: min(args.topk, args.num_samples)]
        tss_topk_idx_np = tss_rank_np[: min(args.topk, args.num_samples)]

        np.save(save_dir / "tpp_scores.npy", tpp_scores_np)
        np.save(save_dir / "tss_scores.npy", tss_scores_np)
        np.save(save_dir / "tpp_rank_indices.npy", tpp_rank_np)
        np.save(save_dir / "tss_rank_indices.npy", tss_rank_np)
        np.save(save_dir / "tpp_topk_indices.npy", tpp_topk_idx_np)
        np.save(save_dir / "tss_topk_indices.npy", tss_topk_idx_np)
        np.save(save_dir / "tpp_topk_samples.npy", all_samples[tpp_topk_idx_np])
        np.save(save_dir / "tss_topk_samples.npy", all_samples[tss_topk_idx_np])
        np.save(save_dir / "tpp_topk_pred_cond_raw.npy", pred_raw[tpp_topk_idx_np])
        np.save(save_dir / "tss_topk_pred_cond_raw.npy", pred_raw[tss_topk_idx_np])

        if len(tpp_topk_idx_np):
            plot_target_lambda_curves(
                save_dir / "target_vs_top1_tpp_curves.png",
                case,
                target_raw,
                pred_raw[int(tpp_topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} target vs top1 tpp-ranked",
                task_details[int(tpp_topk_idx_np[0])],
            )
            plot_ranked_overview(
                save_dir / f"tpp_top{len(tpp_topk_idx_np)}_overview.png",
                case,
                all_samples,
                pred_raw,
                target_raw,
                lambdas,
                thetas,
                tpp_rank_np,
                weighted_err_np,
                global_err_np,
                min(args.topk, len(tpp_rank_np)),
            )
        if len(tss_topk_idx_np):
            plot_target_lambda_curves(
                save_dir / "target_vs_top1_tss_curves.png",
                case,
                target_raw,
                pred_raw[int(tss_topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} target vs top1 tss-ranked",
                task_details[int(tss_topk_idx_np[0])],
            )
            plot_ranked_overview(
                save_dir / f"tss_top{len(tss_topk_idx_np)}_overview.png",
                case,
                all_samples,
                pred_raw,
                target_raw,
                lambdas,
                thetas,
                tss_rank_np,
                weighted_err_np,
                global_err_np,
                min(args.topk, len(tss_rank_np)),
            )

        summary.update(
            {
                "topk": int(min(args.topk, args.num_samples)),
                "best_tpp_score": float(tpp_scores_np[tpp_topk_idx_np[0]]) if len(tpp_topk_idx_np) else None,
                "best_tpp_weighted_error": float(weighted_err_np[tpp_topk_idx_np[0]]) if len(tpp_topk_idx_np) else None,
                "best_tpp_global_error": float(global_err_np[tpp_topk_idx_np[0]]) if len(tpp_topk_idx_np) else None,
                "best_tss_score": float(tss_scores_np[tss_topk_idx_np[0]]) if len(tss_topk_idx_np) else None,
                "best_tss_weighted_error": float(weighted_err_np[tss_topk_idx_np[0]]) if len(tss_topk_idx_np) else None,
                "best_tss_global_error": float(global_err_np[tss_topk_idx_np[0]]) if len(tss_topk_idx_np) else None,
                "tpp_rankings": [
                    {
                        "rank": int(r + 1),
                        "sample_rank_idx": int(tpp_topk_idx_np[r]),
                        "tpp_score": float(tpp_scores_np[tpp_topk_idx_np[r]]),
                        "weighted_error": float(weighted_err_np[tpp_topk_idx_np[r]]),
                        "global_error": float(global_err_np[tpp_topk_idx_np[r]]),
                        "task_details": task_details[int(tpp_topk_idx_np[r])],
                    }
                    for r in range(len(tpp_topk_idx_np))
                ],
                "tss_rankings": [
                    {
                        "rank": int(r + 1),
                        "sample_rank_idx": int(tss_topk_idx_np[r]),
                        "tss_score": float(tss_scores_np[tss_topk_idx_np[r]]),
                        "weighted_error": float(weighted_err_np[tss_topk_idx_np[r]]),
                        "global_error": float(global_err_np[tss_topk_idx_np[r]]),
                        "task_details": task_details[int(tss_topk_idx_np[r])],
                    }
                    for r in range(len(tss_topk_idx_np))
                ],
            }
        )
    else:
        task_scores_np = np.asarray(task_scores, dtype=np.float32)
        rank_np = np.argsort(task_scores_np)[::-1]
        topk_idx_np = rank_np[: min(args.topk, args.num_samples)]
        topk_samples = all_samples[topk_idx_np]
        topk_pred_raw = pred_raw[topk_idx_np]

        np.save(save_dir / "task_scores.npy", task_scores_np)
        np.save(save_dir / "rank_indices.npy", rank_np)
        np.save(save_dir / "topk_indices.npy", topk_idx_np)
        np.save(save_dir / "topk_samples.npy", topk_samples)
        np.save(save_dir / "topk_pred_cond_raw.npy", topk_pred_raw)

        save_case_visuals(save_dir, case, target_structure, target_raw, all_samples, topk_samples, topk_pred_raw, lambdas, thetas)
        if len(topk_idx_np):
            best_details = task_details[int(topk_idx_np[0])]
            plot_target_lambda_curves(
                save_dir / "target_vs_top1_curves.png",
                case,
                target_raw,
                pred_raw[int(topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} target vs top1",
                best_details,
            )
            plot_ideal_pred_curves(
                save_dir / "ideal_vs_top1_curves.png",
                case,
                pred_raw[int(topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} ideal vs top1",
                best_details,
            )
        plot_ranked_overview(
            save_dir / f"top{len(topk_idx_np)}_overview.png",
            case,
            all_samples,
            pred_raw,
            target_raw,
            lambdas,
            thetas,
            rank_np,
            weighted_err_np,
            global_err_np,
            min(args.topk, len(rank_np)),
        )

        topk_band_pred, topk_band_lambdas_nm, topk_band_scores = compute_topk_band_data(
            case,
            samples,
            topk_idx_np,
            cond_channels,
            lambdas,
            thetas,
            args.device,
            args.rcwa_orders,
        )
        np.save(save_dir / "topk_band_pred_rows.npy", topk_band_pred)
        np.save(save_dir / "topk_band_lambdas_nm.npy", topk_band_lambdas_nm)
        with (save_dir / "topk_band_scores_3pt.json").open("w", encoding="utf-8") as f:
            json.dump(topk_band_scores, f, ensure_ascii=False, indent=2)
        if len(topk_idx_np):
            plot_band_curves(
                save_dir / "top1_band_curves_3pt.png",
                case,
                topk_band_pred[0],
                topk_band_lambdas_nm,
                thetas,
                f"{case.case_label} top1 band curves",
            )

        top3_full_idx_np, top3_full_pred = compute_top3_full_maps(
            samples,
            rank_np,
            cond_channels,
            lambdas,
            thetas,
            args.device,
            args.rcwa_orders,
            existing_pred_raw=pred_raw if args.eval_mode == "full" else None,
        )
        np.save(save_dir / "top3_full_indices.npy", top3_full_idx_np)
        np.save(save_dir / "top3_full_pred_cond_raw.npy", top3_full_pred)
        plot_full_map_top3_overview(
            save_dir / "top3_full_overview.png",
            case,
            all_samples,
            top3_full_pred,
            target_raw,
            lambdas,
            thetas,
            top3_full_idx_np,
            len(top3_full_idx_np),
        )

        summary.update(
            {
                "topk": int(len(topk_idx_np)),
                "best_task_score": float(task_scores_np[topk_idx_np[0]]) if len(topk_idx_np) else None,
                "best_weighted_error": float(weighted_err_np[topk_idx_np[0]]) if len(topk_idx_np) else None,
                "best_global_error": float(global_err_np[topk_idx_np[0]]) if len(topk_idx_np) else None,
                "top3_full_indices": [int(i) for i in top3_full_idx_np],
                "topk_band_lambdas_nm": [float(x) for x in topk_band_lambdas_nm],
                "selected_rankings": [
                    {
                        "rank": int(r + 1),
                        "sample_rank_idx": int(topk_idx_np[r]),
                        "task_score": float(task_scores_np[topk_idx_np[r]]),
                        "weighted_error": float(weighted_err_np[topk_idx_np[r]]),
                        "global_error": float(global_err_np[topk_idx_np[r]]),
                        "task_details": task_details[int(topk_idx_np[r])],
                        "band_scores_3pt": topk_band_scores[r],
                    }
                    for r in range(len(topk_idx_np))
                ],
            }
        )
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run diffusion inference for the curated high-score task cases with RCWA reranking.")
    parser.add_argument("--train_npz", default=str(default_train_npz(ROOT)))
    parser.add_argument("--stats_path", default=str(resolve_default_stats_path(ROOT)))
    parser.add_argument("--diffusion_ckpt", default=str(resolve_default_diffusion_ckpt(ROOT)))
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale; 1.0 means disabled")
    parser.add_argument("--save_dir", default=str(ROOT / "samples" / "task_infer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rcwa_orders", type=int, default=7)
    parser.add_argument("--eval_mode", choices=["target_only", "full"], default="target_only")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Task selectors: task keys {list_task_keys()} or explicit case labels like 1100nm_id11339",
    )
    args = parser.parse_args()

    train_npz = Path(args.train_npz)
    mean, std = load_stats(args.stats_path)
    cond_channels = int(mean.shape[1])

    diffusion = GaussianDiffusion(ConditionalUNet(cond_channels).to(args.device), timesteps=1000, image_size=64).to(args.device)
    diffusion = load_model(args.diffusion_ckpt, diffusion, "diffusion", args.device)

    selected_cases = select_cases(args.tasks)
    run_root = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    results = []
    for case in selected_cases:
        print(
            f"[task_infer] {case.task_key} {case.case_label} sample_idx={case.sample_idx} "
            f"target_lambda={case.target_lambda_nm:.1f}nm cfg_scale={args.cfg_scale:.2f}",
            flush=True,
        )
        results.append(run_case(case, args, diffusion, mean, std, train_npz, run_root))

    with (run_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_npz": str(train_npz),
                "num_cases": len(results),
                "cases": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved_to: {run_root}")


if __name__ == "__main__":
    main()
