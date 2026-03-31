from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C0 = 299792458.0


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def resolve_default_out_dir(explicit: str | None = None) -> Path:
    if explicit:
        return resolve_from_root(explicit)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "data" / f"st2_template_match_{stamp}"


def robust_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(arr)))
    if scale < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / scale


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    required = ["structures", "tpp_mag", "tss_mag", "lambdas", "thetas"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing fields in {npz_path}: {missing}")
    return {
        "structures": np.asarray(data["structures"], dtype=np.float32),
        "tpp_mag": np.asarray(data["tpp_mag"], dtype=np.float32),
        "tss_mag": np.asarray(data["tss_mag"], dtype=np.float32),
        "lambdas": np.asarray(data["lambdas"], dtype=np.float64),
        "thetas": np.asarray(data["thetas"], dtype=np.float64),
    }


def find_lambda_index(lambdas: np.ndarray, target_lambda: float) -> int:
    return int(np.argmin(np.abs(lambdas - float(target_lambda))))


def theta_zero_band_mask(thetas: np.ndarray, width_deg: float) -> np.ndarray:
    return np.abs(thetas) <= float(width_deg)


def lambda_zero_band_mask(lambdas: np.ndarray, lambda0_nm: float, width_nm: float) -> np.ndarray:
    return np.abs(lambdas - float(lambda0_nm)) <= float(width_nm)


def work_region_mask(
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    lambda0_nm: float,
    lambda_window_nm: float,
    theta_max_deg: float,
) -> np.ndarray:
    lam = np.asarray(lambdas_nm, dtype=np.float64)[:, None]
    theta = np.asarray(thetas_deg, dtype=np.float64)[None, :]
    half_bw = 0.5 * float(lambda_window_nm)
    return (
        (np.abs(lam - float(lambda0_nm)) <= half_bw)
        & (np.abs(theta) <= float(theta_max_deg))
    )


def ideal_st2_map(
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    lambda0_nm: float,
    lambda_window_nm: float,
    theta_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lam_m = np.asarray(lambdas_nm, dtype=np.float64)[:, None] * 1e-9
    th = np.deg2rad(np.asarray(thetas_deg, dtype=np.float64))[None, :]
    kx2 = ((2.0 * np.pi / np.maximum(lam_m, 1e-20)) * np.sin(th)) ** 2
    omega = 2.0 * np.pi * C0 / np.maximum(lam_m, 1e-20)
    omega0 = 2.0 * np.pi * C0 / max(float(lambda0_nm) * 1e-9, 1e-20)
    om2 = (omega - omega0) ** 2
    raw = kx2 * om2
    mask = work_region_mask(lambdas_nm, thetas_deg, lambda0_nm, lambda_window_nm, theta_max_deg)
    ideal = np.zeros_like(raw, dtype=np.float64)
    if np.any(mask):
        masked = raw[mask]
        scale = float(np.max(np.abs(masked)))
        if scale > 1e-12:
            ideal[mask] = masked / scale
    return ideal, mask


def ideal_st2_map_dense(
    lambda0_nm: float,
    lambda_window_nm: float,
    theta_max_deg: float,
    n_lambda: int = 241,
    n_theta: int = 321,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambdas_dense = np.linspace(
        float(lambda0_nm) - 0.5 * float(lambda_window_nm),
        float(lambda0_nm) + 0.5 * float(lambda_window_nm),
        n_lambda,
        dtype=np.float64,
    )
    thetas_dense = np.linspace(
        -float(theta_max_deg),
        float(theta_max_deg),
        n_theta,
        dtype=np.float64,
    )
    lam_m = lambdas_dense[:, None] * 1e-9
    th = np.deg2rad(thetas_dense)[None, :]
    kx2 = ((2.0 * np.pi / np.maximum(lam_m, 1e-20)) * np.sin(th)) ** 2
    omega = 2.0 * np.pi * C0 / np.maximum(lam_m, 1e-20)
    omega0 = 2.0 * np.pi * C0 / max(float(lambda0_nm) * 1e-9, 1e-20)
    om2 = (omega - omega0) ** 2
    ideal_dense = robust_norm(kx2 * om2)
    return lambdas_dense, thetas_dense, ideal_dense


def projection_coeff_in_window(
    spec_map: np.ndarray,
    ideal_map: np.ndarray,
    work_mask: np.ndarray,
) -> tuple[float, float]:
    if not np.any(work_mask):
        return 0.0, 0.0
    y = robust_norm(spec_map)[work_mask]
    phi = ideal_map[work_mask]
    denom = float(np.dot(phi, phi))
    if denom <= 1e-12:
        return 0.0, 0.0
    coeff = float(np.dot(y, phi) / denom)
    corr = cosine_similarity(y, phi)
    return coeff, corr


def expansion_weight_k2omega2(
    spec_map: np.ndarray,
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    lambda0_nm: float,
    work_mask: np.ndarray,
) -> dict[str, float]:
    if not np.any(work_mask):
        return {
            "expansion_weight_k2omega2": 0.0,
            "expansion_coeff_k2": 0.0,
            "expansion_coeff_omega2": 0.0,
            "expansion_coeff_k2omega2": 0.0,
            "expansion_weight_k2": 0.0,
            "expansion_weight_omega2": 0.0,
        }

    lam_m = np.asarray(lambdas_nm, dtype=np.float64)[:, None] * 1e-9
    th = np.deg2rad(np.asarray(thetas_deg, dtype=np.float64))[None, :]
    kx2 = ((2.0 * np.pi / np.maximum(lam_m, 1e-20)) * np.sin(th)) ** 2
    omega = 2.0 * np.pi * C0 / np.maximum(lam_m, 1e-20)
    omega0 = 2.0 * np.pi * C0 / max(float(lambda0_nm) * 1e-9, 1e-20)
    om2 = (omega - omega0) ** 2
    om2 = np.broadcast_to(om2, kx2.shape)
    k2om2 = kx2 * om2

    basis_raw = {
        "k2": kx2,
        "omega2": om2,
        "k2omega2": k2om2,
    }

    basis = []
    names = []
    for name, arr in basis_raw.items():
        vec = np.asarray(arr[work_mask], dtype=np.float64)
        scale = float(np.max(np.abs(vec)))
        if scale > 1e-12:
            vec = vec / scale
        basis.append(vec)
        names.append(name)

    # Paper-style “weight” is best interpreted as basis-expansion share rather than a raw projection scale.
    # We remove the flat background first, then compute energy weights in an orthonormalized basis.
    y = robust_norm(spec_map)[work_mask]
    y = y - float(np.mean(y))

    q_list: list[np.ndarray] = []
    active_names: list[str] = []
    for name, vec in zip(names, basis):
        u = vec.copy()
        for q in q_list:
            u = u - np.dot(u, q) * q
        norm_u = float(np.linalg.norm(u))
        if norm_u > 1e-12:
            q_list.append(u / norm_u)
            active_names.append(name)

    if not q_list:
        return {
            "expansion_weight_k2omega2": 0.0,
            "expansion_coeff_k2": 0.0,
            "expansion_coeff_omega2": 0.0,
            "expansion_coeff_k2omega2": 0.0,
            "expansion_weight_k2": 0.0,
            "expansion_weight_omega2": 0.0,
        }

    coeffs = np.asarray([float(np.dot(y, q)) for q in q_list], dtype=np.float64)
    energies = coeffs ** 2
    total_energy = float(np.sum(energies))
    if total_energy <= 1e-12:
        weights = {name: 0.0 for name in active_names}
    else:
        weights = {name: float(e / total_energy) for name, e in zip(active_names, energies)}
    coeff_map = {name: float(c) for name, c in zip(active_names, coeffs)}

    return {
        "expansion_weight_k2omega2": float(weights.get("k2omega2", 0.0)),
        "expansion_coeff_k2": float(coeff_map.get("k2", 0.0)),
        "expansion_coeff_omega2": float(coeff_map.get("omega2", 0.0)),
        "expansion_coeff_k2omega2": float(coeff_map.get("k2omega2", 0.0)),
        "expansion_weight_k2": float(weights.get("k2", 0.0)),
        "expansion_weight_omega2": float(weights.get("omega2", 0.0)),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def mse_similarity(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))
    return float(1.0 / (1.0 + mse))


def score_zero_lines(
    spec_map: np.ndarray,
    theta0_mask: np.ndarray,
    lambda0_mask: np.ndarray,
    work_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    spec = robust_norm(spec_map)
    theta_region = work_mask & theta0_mask[None, :]
    lambda_region = work_mask & lambda0_mask[:, None]
    theta_zero_leak = float(np.mean(spec[theta_region])) if np.any(theta_region) else 1.0
    lambda_zero_leak = float(np.mean(spec[lambda_region])) if np.any(lambda_region) else 1.0
    theta_score = float(np.clip(1.0 - theta_zero_leak, 0.0, 1.0))
    lambda_score = float(np.clip(1.0 - lambda_zero_leak, 0.0, 1.0))
    score = 0.5 * theta_score + 0.5 * lambda_score
    return score, {
        "theta_zero_score": theta_score,
        "lambda_zero_score": lambda_score,
        "theta_zero_leak": theta_zero_leak,
        "lambda_zero_leak": lambda_zero_leak,
    }


def score_template(
    spec_map: np.ndarray,
    ideal_map: np.ndarray,
    work_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    spec = robust_norm(spec_map)
    cos = cosine_similarity(spec[work_mask], ideal_map[work_mask]) if np.any(work_mask) else 0.0
    mse_sim = mse_similarity(spec[work_mask], ideal_map[work_mask]) if np.any(work_mask) else 0.0
    score = 0.65 * cos + 0.35 * mse_sim
    return score, {
        "template_cosine": cos,
        "template_mse_similarity": mse_sim,
        "template_score": score,
    }


def build_error_weight_map(
    ideal_map: np.ndarray,
    theta0_mask: np.ndarray,
    lambda0_mask: np.ndarray,
    work_mask: np.ndarray,
    theta_zero_boost: float = 3.0,
    lambda_zero_boost: float = 3.0,
) -> np.ndarray:
    weights = np.zeros_like(ideal_map, dtype=np.float64)
    weights[work_mask] = 1.0 + ideal_map[work_mask]
    theta_region = work_mask & theta0_mask[None, :]
    lambda_region = work_mask & lambda0_mask[:, None]
    weights[theta_region] += float(theta_zero_boost)
    weights[lambda_region] += float(lambda_zero_boost)
    return weights


def score_weighted_error(
    spec_map: np.ndarray,
    ideal_map: np.ndarray,
    theta0_mask: np.ndarray,
    lambda0_mask: np.ndarray,
    work_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    if not np.any(work_mask):
        return 0.0, {
            "weighted_mae": 1.0,
            "weighted_rmse": 1.0,
            "weighted_error_score": 0.0,
        }
    spec = robust_norm(spec_map)
    weights = build_error_weight_map(ideal_map, theta0_mask, lambda0_mask, work_mask)
    diff = np.abs(spec - ideal_map)
    w = weights[work_mask]
    mae = float(np.sum(diff[work_mask] * w) / np.sum(w))
    rmse = float(np.sqrt(np.sum(((spec - ideal_map) ** 2)[work_mask] * w) / np.sum(w)))
    score = float(np.clip(1.0 - (0.7 * mae + 0.3 * rmse), 0.0, 1.0))
    return score, {
        "weighted_mae": mae,
        "weighted_rmse": rmse,
        "weighted_error_score": score,
    }


def score_corners(
    spec_map: np.ndarray,
    ideal_map: np.ndarray,
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    lambda0_nm: float,
    work_mask: np.ndarray,
    hi_q: float = 0.90,
    lo_q: float = 0.15,
) -> tuple[float, dict[str, float]]:
    spec = robust_norm(spec_map)
    active = ideal_map[work_mask] if np.any(work_mask) else ideal_map.ravel()
    hi_mask = work_mask & (ideal_map >= float(np.quantile(active, hi_q)))
    lo_mask = work_mask & (ideal_map <= float(np.quantile(active, lo_q)))
    hi_mean = float(np.mean(spec[hi_mask])) if np.any(hi_mask) else 0.0
    lo_mean = float(np.mean(spec[lo_mask])) if np.any(lo_mask) else 0.0
    contrast = float(np.clip((hi_mean - lo_mean) / max(hi_mean + lo_mean, 1e-8), 0.0, 1.0))
    hi_score = float(np.clip(hi_mean, 0.0, 1.0))
    theta_abs = np.abs(np.asarray(thetas_deg, dtype=np.float64))
    lam_abs = np.abs(np.asarray(lambdas_nm, dtype=np.float64) - float(lambda0_nm))
    active_lam = lam_abs[np.any(work_mask, axis=1)]
    half_bw = float(np.max(active_lam)) if active_lam.size else 0.0
    outer_lambda_thr = 0.5 * half_bw
    outer_lambda_mask = lam_abs >= outer_lambda_thr
    angle30_mask = theta_abs >= 30.0
    angle40_mask = theta_abs >= 37.5
    lobe30_region = work_mask & outer_lambda_mask[:, None] & angle30_mask[None, :]
    lobe40_region = work_mask & outer_lambda_mask[:, None] & angle40_mask[None, :]
    lobe30_mean = float(np.mean(spec[lobe30_region])) if np.any(lobe30_region) else 0.0
    lobe40_mean = float(np.mean(spec[lobe40_region])) if np.any(lobe40_region) else 0.0
    lobe30_score = float(np.clip(lobe30_mean, 0.0, 1.0))
    lobe40_score = float(np.clip(lobe40_mean, 0.0, 1.0))
    score = 0.30 * hi_score + 0.20 * contrast + 0.30 * lobe30_score + 0.20 * lobe40_score
    return score, {
        "corner_lobe_score": score,
        "corner_lobe_hi_score": hi_score,
        "corner_lobe_contrast": contrast,
        "corner_lobe_30deg_score": lobe30_score,
        "corner_lobe_40deg_score": lobe40_score,
        "corner_lobe_30deg_mean": lobe30_mean,
        "corner_lobe_40deg_mean": lobe40_mean,
        "ideal_high_region_mean": hi_mean,
        "ideal_low_region_mean": lo_mean,
    }


def score_channel(
    spec_map: np.ndarray,
    ideal_map: np.ndarray,
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    lambda0_nm: float,
    theta0_mask: np.ndarray,
    lambda0_mask: np.ndarray,
    work_mask: np.ndarray,
) -> dict[str, float]:
    zero_score, zero_details = score_zero_lines(spec_map, theta0_mask, lambda0_mask, work_mask)
    weighted_error_score, weighted_error_details = score_weighted_error(
        spec_map,
        ideal_map,
        theta0_mask,
        lambda0_mask,
        work_mask,
    )
    proj_coeff, proj_corr = projection_coeff_in_window(spec_map, ideal_map, work_mask)
    expansion = expansion_weight_k2omega2(spec_map, lambdas_nm, thetas_deg, lambda0_nm, work_mask)
    proj_score = 0.5 * float(expansion["expansion_weight_k2omega2"]) + 0.5 * proj_corr
    corner_score, corner_details = score_corners(spec_map, ideal_map, lambdas_nm, thetas_deg, lambda0_nm, work_mask)
    aux_score = 0.5 * weighted_error_score + 0.5 * proj_score
    total = 0.50 * zero_score + 0.40 * corner_score + 0.10 * aux_score
    return {
        "score_total": total,
        "score_zero_lines": zero_score,
        "score_weighted_error": weighted_error_score,
        "score_projection": proj_score,
        "score_corner_lobes": corner_score,
        "score_auxiliary": aux_score,
        "projection_coeff_k2omega2": proj_coeff,
        "projection_corr_k2omega2": proj_corr,
        **expansion,
        **zero_details,
        **weighted_error_details,
        **corner_details,
    }


def merge_channel_scores(tpp: dict[str, float], tss: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for key in sorted(set(tpp) | set(tss)):
        tv = float(tpp.get(key, np.nan))
        sv = float(tss.get(key, np.nan))
        if np.isfinite(tv) and np.isfinite(sv):
            merged[f"tpp_{key}"] = tv
            merged[f"tss_{key}"] = sv
            merged[key] = 0.5 * (tv + sv)
    return merged


def _draw_st2_guides(ax, lambda0_nm: float, theta_max_deg: float, lambda_window_nm: float) -> None:
    half_bw = 0.5 * float(lambda_window_nm)
    for theta in (-40.0, -30.0, 0.0, 30.0, 40.0):
        if abs(theta) > float(theta_max_deg) + 1e-6:
            continue
        style = "--" if abs(theta) in (0.0, 30.0, 40.0) else ":"
        color = "w" if abs(theta) in (30.0, 40.0) else "0.9"
        ax.axvline(theta, color=color, ls=style, lw=0.9, alpha=0.9)
    for delta in (-half_bw, -50.0, 0.0, 50.0, half_bw):
        lam = float(lambda0_nm + delta)
        if abs(delta) > half_bw + 1e-6:
            continue
        style = "--" if abs(delta) in (0.0, 50.0, half_bw) else ":"
        color = "w" if abs(delta) in (50.0, half_bw) else "0.9"
        ax.axhline(lam, color=color, ls=style, lw=0.9, alpha=0.9)


def plot_candidate(
    out_path: Path,
    structure: np.ndarray,
    tpp_map: np.ndarray,
    tss_map: np.ndarray,
    ideal_map: np.ndarray,
    work_mask: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lambda0_nm: float,
    theta_max_deg: float,
    lambda_window_nm: float,
    merged_score: float,
) -> None:
    lambda_idx = find_lambda_index(lambdas, lambda0_nm)
    theta0_idx = int(np.argmin(np.abs(thetas)))
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.ravel()

    ax0.imshow(structure, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax0.set_title("structure")
    ax0.axis("off")

    theta_min_vis = -float(theta_max_deg)
    theta_max_vis = float(theta_max_deg)
    lam_min_vis = float(lambda0_nm - 0.5 * lambda_window_nm)
    lam_max_vis = float(lambda0_nm + 0.5 * lambda_window_nm)

    hm0 = ax1.imshow(
        ideal_map,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _draw_st2_guides(ax1, lambda0_nm, theta_max_deg, lambda_window_nm)
    ax1.set_title(f"ideal ST2 @ {int(round(lambda0_nm))} nm")
    ax1.set_xlabel("theta (deg)")
    ax1.set_ylabel("lambda (nm)")
    ax1.set_xlim(theta_min_vis, theta_max_vis)
    ax1.set_ylim(lam_min_vis, lam_max_vis)

    hm1 = ax2.imshow(
        robust_norm(tpp_map),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _draw_st2_guides(ax2, lambda0_nm, theta_max_deg, lambda_window_nm)
    ax2.set_title(f"tpp | score={merged_score:.4f}")
    ax2.set_xlabel("theta (deg)")
    ax2.set_ylabel("lambda (nm)")
    ax2.set_xlim(theta_min_vis, theta_max_vis)
    ax2.set_ylim(lam_min_vis, lam_max_vis)

    hm2 = ax3.imshow(
        robust_norm(tss_map),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _draw_st2_guides(ax3, lambda0_nm, theta_max_deg, lambda_window_nm)
    ax3.set_title("tss")
    ax3.set_xlabel("theta (deg)")
    ax3.set_ylabel("lambda (nm)")
    ax3.set_xlim(theta_min_vis, theta_max_vis)
    ax3.set_ylim(lam_min_vis, lam_max_vis)

    ideal_row = ideal_map[lambda_idx]
    ax4.plot(thetas, ideal_row, "k--", lw=1.8, label="ideal row")
    ax4.plot(thetas, robust_norm(tpp_map[lambda_idx]), lw=2.0, label="tpp row")
    ax4.plot(thetas, robust_norm(tss_map[lambda_idx]), lw=2.0, label="tss row")
    for theta in (-40.0, -30.0, 30.0, 40.0):
        if abs(theta) <= float(theta_max_deg) + 1e-6:
            ax4.axvline(theta, color="0.6", ls=":", lw=0.8)
    ax4.set_title(f"row @ {int(round(lambda0_nm))} nm")
    ax4.set_xlabel("theta (deg)")
    ax4.set_ylabel("normalized amplitude")
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8)
    ax4.set_xlim(theta_min_vis, theta_max_vis)

    ideal_col = ideal_map[:, theta0_idx]
    ax5.plot(lambdas, ideal_col, "k--", lw=1.8, label="ideal theta=0")
    ax5.plot(lambdas, robust_norm(tpp_map[:, theta0_idx]), lw=2.0, label="tpp theta=0")
    ax5.plot(lambdas, robust_norm(tss_map[:, theta0_idx]), lw=2.0, label="tss theta=0")
    half_bw = 0.5 * float(lambda_window_nm)
    for lam in (lambda0_nm - half_bw, lambda0_nm - 50.0, lambda0_nm, lambda0_nm + 50.0, lambda0_nm + half_bw):
        if lam < float(np.min(lambdas)) - 1e-6 or lam > float(np.max(lambdas)) + 1e-6:
            continue
        ax5.axvline(float(lam), color="0.6", ls=":", lw=0.8)
    ax5.set_title("column @ theta=0")
    ax5.set_xlabel("lambda (nm)")
    ax5.set_ylabel("normalized amplitude")
    ax5.grid(alpha=0.25)
    ax5.legend(fontsize=8)
    ax5.set_xlim(lam_min_vis, lam_max_vis)

    fig.colorbar(hm2, ax=[ax1, ax2, ax3], shrink=0.82, pad=0.02, label="normalized magnitude")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_ranking_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sort_rows(rows: list[dict[str, float | int | str]], key: str) -> list[dict[str, float | int | str]]:
    return sorted(rows, key=lambda row: float(row.get(key, -1.0)), reverse=True)


def _extract_channel_rows(
    rows: list[dict[str, float | int | str]],
    score_key: str,
    prefix: str,
) -> list[dict[str, float | int | str]]:
    channel_rows: list[dict[str, float | int | str]] = []
    for row in rows:
        item: dict[str, float | int | str] = {
            "sample_idx": int(row["sample_idx"]),
            "target_lambda_nm": float(row["target_lambda_nm"]),
            "score_total": float(row[score_key]),
        }
        for key, value in row.items():
            if key.startswith(prefix):
                item[key[len(prefix):]] = value
        channel_rows.append(item)
    return channel_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Template-match train_data.npz against ideal k^2*Omega^2 spatiotemporal differentiator maps.")
    parser.add_argument("--npz", type=str, default="data/train_data.npz")
    parser.add_argument(
        "--target_lambdas",
        type=float,
        nargs="+",
        default=None,
        help="Target wavelengths to screen. Default: use all dataset lambdas (typically 50 nm spaced).",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--theta_zero_width_deg", type=float, default=2.5)
    parser.add_argument("--lambda_zero_width_nm", type=float, default=25.0)
    parser.add_argument("--lambda_window_nm", type=float, default=200.0, help="Working wavelength window centered at lambda0.")
    parser.add_argument("--theta_max_deg", type=float, default=None, help="Max abs(theta) used for ideal map and scoring window.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory. Default: create a new timestamped st2_template_match_* directory under data/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    out_dir = resolve_default_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(npz_path)
    structures = bundle["structures"]
    tpp = bundle["tpp_mag"]
    tss = bundle["tss_mag"]
    lambdas = bundle["lambdas"]
    thetas = bundle["thetas"]
    theta0_mask = theta_zero_band_mask(thetas, args.theta_zero_width_deg)
    theta_max_deg = float(args.theta_max_deg) if args.theta_max_deg is not None else float(np.max(np.abs(thetas)))
    target_lambdas = [float(x) for x in (args.target_lambdas if args.target_lambdas is not None else lambdas.tolist())]

    global_summary = []
    for target_lambda in target_lambdas:
        lambda_idx = find_lambda_index(lambdas, target_lambda)
        actual_lambda = float(lambdas[lambda_idx])
        lambda0_mask = lambda_zero_band_mask(lambdas, actual_lambda, args.lambda_zero_width_nm)
        ideal_map, work_mask = ideal_st2_map(lambdas, thetas, actual_lambda, args.lambda_window_nm, theta_max_deg)
        wave_dir = out_dir / f"lambda_{int(round(actual_lambda))}nm"
        wave_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(5.4, 4.2))
        plt.imshow(
            ideal_map,
            origin="lower",
            aspect="auto",
            extent=[float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])],
            cmap="turbo",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _draw_st2_guides(plt.gca(), actual_lambda, theta_max_deg, args.lambda_window_nm)
        plt.title(f"ideal ST2 map @ {int(round(actual_lambda))} nm")
        plt.xlabel("theta (deg)")
        plt.ylabel("lambda (nm)")
        plt.colorbar(label="normalized magnitude")
        plt.xlim(-theta_max_deg, theta_max_deg)
        plt.ylim(actual_lambda - 0.5 * args.lambda_window_nm, actual_lambda + 0.5 * args.lambda_window_nm)
        plt.tight_layout()
        plt.savefig(wave_dir / "ideal_map.png", dpi=180)
        plt.close()

        ranking_rows: list[dict[str, float | int | str]] = []
        for sample_idx in range(structures.shape[0]):
            tpp_score = score_channel(tpp[sample_idx], ideal_map, lambdas, thetas, actual_lambda, theta0_mask, lambda0_mask, work_mask)
            tss_score = score_channel(tss[sample_idx], ideal_map, lambdas, thetas, actual_lambda, theta0_mask, lambda0_mask, work_mask)
            merged = merge_channel_scores(tpp_score, tss_score)
            merged["sample_idx"] = int(sample_idx)
            merged["target_lambda_nm"] = actual_lambda
            ranking_rows.append(merged)

        merged_rows = _sort_rows(ranking_rows, "score_total")
        tpp_rows = _sort_rows(_extract_channel_rows(ranking_rows, "tpp_score_total", "tpp_"), "score_total")
        tss_rows = _sort_rows(_extract_channel_rows(ranking_rows, "tss_score_total", "tss_"), "score_total")

        save_ranking_csv(wave_dir / "merged_ranking.csv", merged_rows)
        save_ranking_csv(wave_dir / "tpp_ranking.csv", tpp_rows)
        save_ranking_csv(wave_dir / "tss_ranking.csv", tss_rows)
        save_ranking_csv(wave_dir / "ranking.csv", merged_rows)

        summary = {
            "target_lambda_nm": actual_lambda,
            "requested_lambda_nm": float(target_lambda),
            "merged_top_score": float(merged_rows[0]["score_total"]) if merged_rows else None,
            "merged_top_sample_idx": int(merged_rows[0]["sample_idx"]) if merged_rows else None,
            "tpp_top_score": float(tpp_rows[0]["score_total"]) if tpp_rows else None,
            "tpp_top_sample_idx": int(tpp_rows[0]["sample_idx"]) if tpp_rows else None,
            "tss_top_score": float(tss_rows[0]["score_total"]) if tss_rows else None,
            "tss_top_sample_idx": int(tss_rows[0]["sample_idx"]) if tss_rows else None,
            "num_samples": len(merged_rows),
        }
        global_summary.append(summary)

        for rank, row in enumerate(merged_rows[: max(1, args.topk)], start=1):
            sample_idx = int(row["sample_idx"])
            plot_candidate(
                wave_dir / f"rank_{rank:03d}_sample_{sample_idx}.png",
                structures[sample_idx],
                tpp[sample_idx],
                tss[sample_idx],
                ideal_map,
                work_mask,
                lambdas,
                thetas,
                actual_lambda,
                theta_max_deg,
                args.lambda_window_nm,
                float(row["score_total"]),
            )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_npz": str(npz_path),
                "out_dir": str(out_dir),
                "target_lambdas": target_lambdas,
                "theta_zero_width_deg": float(args.theta_zero_width_deg),
                "lambda_zero_width_nm": float(args.lambda_zero_width_nm),
                "lambda_window_nm": float(args.lambda_window_nm),
                "theta_max_deg": theta_max_deg,
                "results": global_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_to: {out_dir}")
    for item in global_summary:
        print(
            f"lambda={item['target_lambda_nm']:.1f}nm "
            f"merged={item['merged_top_sample_idx']}:{item['merged_top_score']:.4f} "
            f"tpp={item['tpp_top_sample_idx']}:{item['tpp_top_score']:.4f} "
            f"tss={item['tss_top_sample_idx']}:{item['tss_top_score']:.4f}"
        )


if __name__ == "__main__":
    main()
