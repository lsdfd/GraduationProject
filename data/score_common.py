from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


def load_bundle(npz_path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    data = np.load(npz_path)
    if "structures" not in data.files:
        raise ValueError(f"structures not found in {npz_path}")
    if field not in data.files:
        raise ValueError(f"{field} not found. available: {list(data.files)}")

    structures = np.asarray(data["structures"], dtype=np.float32)
    spec = np.asarray(data[field], dtype=np.float32)
    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures should be [N,64,64], got {structures.shape}")
    if spec.ndim != 3:
        raise ValueError(f"{field} should be [N,L,T], got {spec.shape}")
    if structures.shape[0] != spec.shape[0]:
        raise ValueError(f"structures/spec sample mismatch: {structures.shape[0]} vs {spec.shape[0]}")

    lambdas = np.asarray(data["lambdas"], dtype=np.float32) if "lambdas" in data.files else np.arange(spec.shape[1], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32) if "thetas" in data.files else np.arange(spec.shape[2], dtype=np.float32)
    if spec.shape[1] != len(lambdas) or spec.shape[2] != len(thetas):
        raise ValueError("spec shape and lambdas/thetas mismatch")

    return structures, spec, lambdas, thetas


def load_companion_spec(npz_path: Path, field: str) -> tuple[np.ndarray | None, str | None]:
    if field == "tpp_mag":
        other = "tss_mag"
    elif field == "tss_mag":
        other = "tpp_mag"
    else:
        return None, None

    data = np.load(npz_path)
    if other not in data.files:
        return None, None
    spec = np.asarray(data[other], dtype=np.float32)
    if spec.ndim != 3:
        return None, None
    return spec, other


def target_profile(thetas_deg: np.ndarray) -> np.ndarray:
    tmax = float(np.max(np.abs(thetas_deg)))
    if tmax <= 0:
        return np.zeros_like(thetas_deg, dtype=np.float32)
    kx = np.sin(np.deg2rad(thetas_deg)) / np.sin(np.deg2rad(tmax))
    x = np.abs(kx) ** 2
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    return x.astype(np.float32)


def _row_3term_score(
    y: np.ndarray,
    x: np.ndarray,
    denom: float,
    center_idx: int,
    edge_mask: np.ndarray,
    global_scale: float,
    w_center: float,
    w_shape: float,
    w_edge: float,
) -> tuple[float, float, float, float, float, float]:
    if not np.isfinite(y).all():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    y_norm = y / max(float(np.max(y)), 1e-8)
    a = float(np.sum(x * y_norm) / denom)
    y_fit = a * x
    ss_res = float(np.sum((y_norm - y_fit) ** 2))
    ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
    this_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    edge_mean = float(np.mean(y[edge_mask]))
    center_val = float(y[center_idx])
    c = float(np.clip(1.0 - center_val / max(edge_mean, 1e-8), 0.0, 1.0))
    s = float(np.clip(this_r2, 0.0, 1.0)) if a >= 0 else 0.0
    e = float(np.clip(edge_mean / global_scale, 0.0, 1.0))
    return w_center * c + w_shape * s + w_edge * e, c, s, e, a, this_r2


def score_spectra(
    spec: np.ndarray,
    thetas_deg: np.ndarray,
    w_center: float,
    w_shape: float,
    w_edge: float,
    w_bandwidth: float = 0.2,
) -> dict[str, np.ndarray]:
    n, l, _ = spec.shape
    x = target_profile(thetas_deg).astype(np.float64)
    center_idx = int(np.argmin(np.abs(thetas_deg)))
    edge_mask = np.abs(thetas_deg) >= 0.85 * float(np.max(np.abs(thetas_deg)))
    if not edge_mask.any():
        edge_mask[[0, -1]] = True

    global_scale = max(float(np.nanquantile(spec, 0.99)), 1e-8)

    score = np.full((n, l), np.nan, dtype=np.float32)
    center_s = np.full_like(score, np.nan)
    shape_s = np.full_like(score, np.nan)
    edge_s = np.full_like(score, np.nan)
    bandwidth_s = np.full_like(score, np.nan)
    coef_a = np.full_like(score, np.nan)
    fit_mse = np.full_like(score, np.nan)
    r2 = np.full_like(score, np.nan)

    denom = max(float(np.sum(x * x)), 1e-8)
    w3 = 1.0 - w_bandwidth

    for i in range(n):
        for j in range(l):
            y = spec[i, j].astype(np.float64)
            main, c, s, e, a, this_r2 = _row_3term_score(
                y, x, denom, center_idx, edge_mask, global_scale, w_center, w_shape, w_edge
            )
            if np.isnan(main):
                continue

            bw_scores = []
            for dj in (-1, 1):
                jj = j + dj
                if 0 <= jj < l:
                    yy = spec[i, jj].astype(np.float64)
                    nb, *_ = _row_3term_score(
                        yy, x, denom, center_idx, edge_mask, global_scale, w_center, w_shape, w_edge
                    )
                    if np.isfinite(nb):
                        bw_scores.append(nb)
            bw = float(np.mean(bw_scores)) if bw_scores else main

            score[i, j] = w3 * main + w_bandwidth * bw
            center_s[i, j] = c
            shape_s[i, j] = s
            edge_s[i, j] = e
            bandwidth_s[i, j] = bw
            coef_a[i, j] = a
            fit_mse[i, j] = float(np.mean((y / max(float(np.max(y)), 1e-8) - a * x) ** 2))
            r2[i, j] = this_r2

    return {
        "score": score,
        "center_score": center_s,
        "shape_score": shape_s,
        "edge_score": edge_s,
        "bandwidth_score": bandwidth_s,
        "coef_a": coef_a,
        "fit_mse": fit_mse,
        "r2": r2,
    }


def topk_per_lambda(score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n, l = score.shape
    k = min(k, n)
    idx = np.full((l, k), -1, dtype=np.int32)
    val = np.full((l, k), np.nan, dtype=np.float32)
    for j in range(l):
        col = score[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        order = np.argsort(col[valid])[::-1]
        sel = np.where(valid)[0][order[:k]]
        idx[j, : len(sel)] = sel
        val[j, : len(sel)] = col[sel]
    return idx, val
