from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TaskCase:
    task_key: str
    task_label: str
    objective_key: str
    sample_idx: int
    target_lambda_nm: float
    score_source: str
    case_label: str
    note: str = ""

    @property
    def selector(self) -> str:
        return f"{self.task_key}:{self.case_label}"


TASK_CASES: tuple[TaskCase, ...] = (
    TaskCase("p_second_order", "p polarization second-order", "row_tpp", 4279, 1050.0, "data/second_order_scores/tpp_mag_summary.json", "1050nm_id4279"),
    TaskCase("p_second_order", "p polarization second-order", "row_tpp", 4637, 1100.0, "data/second_order_scores/tpp_mag_summary.json", "1100nm_id4637"),
    TaskCase("p_second_order", "p polarization second-order", "row_tpp", 212, 1100.0, "data/second_order_scores/tpp_mag_summary.json", "1100nm_id0212"),
    TaskCase("p_second_order", "p polarization second-order", "row_tpp", 3138, 1150.0, "data/second_order_scores/tpp_mag_summary.json", "1150nm_id3138"),
    TaskCase("polarization_independent", "polarization independent", "row_both", 5817, 1000.0, "data/polarization_independent_scores/joint_summary.json", "1000nm_id5817"),
    TaskCase("polarization_independent", "polarization independent", "row_both", 4279, 1050.0, "data/polarization_independent_scores/joint_summary.json", "1050nm_id4279", "40 degree amplitudes differ but joint score is high"),
    TaskCase("polarization_independent", "polarization independent", "row_both", 2191, 1100.0, "data/polarization_independent_scores/joint_summary.json", "1100nm_id2191", "very good"),
    TaskCase("polarization_multiplexed", "polarization multiplexed", "row_both", 11339, 1100.0, "data/polarization_multiplexed_scores/p_active_s_silent/p_active_s_silent_summary.json", "1100nm_id11339"),
    TaskCase("fourth_order", "fourth-order differentiation", "row_tpp", 3886, 1050.0, "data/fourth_order_scores/tpp_mag_fourth_order_summary.json", "1050nm_id3886"),
    TaskCase("fourth_order", "fourth-order differentiation", "row_tpp", 3186, 1150.0, "data/fourth_order_scores/tpp_mag_fourth_order_summary.json", "1150nm_id3186", "t40 nearly 1"),
    TaskCase("lowpass", "lowpass", "row_tpp", 18249, 1100.0, "data/optica_lowpass_scores/tpp_mag_optica_lowpass_summary.json", "1100nm_id18249"),
    TaskCase("st2", "spatiotemporal differentiation", "map_window_both", 18959, 950.0, "data/st2_profiles/sample_18959/profile_summary.json", "0950nm_idx18959"),
)

TASK_INDEX = {case.selector: case for case in TASK_CASES}


def list_task_keys() -> list[str]:
    return sorted({case.task_key for case in TASK_CASES})


def select_cases(selectors: list[str] | None = None) -> list[TaskCase]:
    if not selectors:
        return list(TASK_CASES)
    picked: list[TaskCase] = []
    seen: set[str] = set()
    task_keys = set(list_task_keys())
    for raw in selectors:
        key = raw.strip()
        if not key:
            continue
        if key in task_keys:
            for case in TASK_CASES:
                if case.task_key == key and case.selector not in seen:
                    picked.append(case)
                    seen.add(case.selector)
            continue
        matches = [case for case in TASK_CASES if case.case_label == key]
        if len(matches) == 1:
            case = matches[0]
            if case.selector not in seen:
                picked.append(case)
                seen.add(case.selector)
            continue
        if len(matches) > 1:
            raise KeyError(f"Ambiguous case label: {key}. Use one of {[case.selector for case in matches]}")
        if key in TASK_INDEX and key not in seen:
            picked.append(TASK_INDEX[key])
            seen.add(key)
            continue
        raise KeyError(f"Unknown task selector: {key}")
    return picked


def default_train_npz(root: Path) -> Path:
    candidates = [root / "data" / "train_data_20000.npz", root / "data" / "train_data.npz", root / "train_data.npz"]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_case_weight(case: TaskCase, cond_ch: int, lambdas: np.ndarray, thetas: np.ndarray) -> np.ndarray:
    weight = np.zeros((cond_ch, len(lambdas), len(thetas)), dtype=np.float32)
    lam_idx = int(np.argmin(np.abs(lambdas.astype(np.float64) - float(case.target_lambda_nm))))
    if case.objective_key == "row_tpp":
        weight[0, lam_idx, :] = 1.0
        return weight
    if case.objective_key == "row_both":
        weight[: min(cond_ch, 2), lam_idx, :] = 1.0
        return weight
    if case.objective_key == "map_window_both":
        lam_mask = np.abs(lambdas.astype(np.float64) - float(case.target_lambda_nm)) <= 50.0
        if not np.any(lam_mask):
            lam_mask[lam_idx] = True
        weight[: min(cond_ch, 2), lam_mask, :] = 1.0
        return weight
    raise ValueError(f"Unsupported objective_key: {case.objective_key}")


def case_output_dir(base_dir: Path, case: TaskCase) -> Path:
    return base_dir / case.task_key / case.case_label


def _second_like_target(thetas_deg: np.ndarray, order: int) -> np.ndarray:
    tmax = float(np.max(np.abs(thetas_deg)))
    if tmax <= 0:
        return np.zeros_like(thetas_deg, dtype=np.float64)
    kx = np.sin(np.deg2rad(thetas_deg.astype(np.float64))) / np.sin(np.deg2rad(tmax))
    x = np.abs(kx) ** order
    return (x - x.min()) / max(float(x.max() - x.min()), 1e-8)


def _row_three_term_score(y: np.ndarray, target: np.ndarray, thetas_deg: np.ndarray, w_center: float, w_shape: float, w_edge: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    if (not np.isfinite(y).all()) or float(np.max(y)) <= 0:
        return {"score": -1.0, "center_score": 0.0, "shape_score": 0.0, "edge_score": 0.0, "r2": -1.0}
    y_norm = y / max(float(np.max(y)), 1e-8)
    denom = max(float(np.sum(target * target)), 1e-8)
    a = float(np.sum(target * y_norm) / denom)
    y_fit = a * target
    ss_res = float(np.sum((y_norm - y_fit) ** 2))
    ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    center_idx = int(np.argmin(np.abs(thetas_deg)))
    edge_mask = np.abs(thetas_deg) >= 0.85 * float(np.max(np.abs(thetas_deg)))
    if not edge_mask.any():
        edge_mask[[0, -1]] = True
    edge_mean = float(np.mean(y[edge_mask]))
    center_val = float(y[center_idx])
    center_score = float(np.clip(1.0 - center_val / max(edge_mean, 1e-8), 0.0, 1.0))
    shape_score = float(np.clip(r2, 0.0, 1.0)) if a >= 0 else 0.0
    edge_score = float(np.clip(edge_mean / max(float(np.max(y)), 1e-8), 0.0, 1.0))
    return {
        "score": float(w_center * center_score + w_shape * shape_score + w_edge * edge_score),
        "center_score": center_score,
        "shape_score": shape_score,
        "edge_score": edge_score,
        "r2": float(r2),
    }


def _second_order_row_details(y: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    return _row_three_term_score(y, _second_like_target(thetas, 2), thetas, 0.6, 0.3, 0.1)


def _fourth_order_row_details(y: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    return _row_three_term_score(y, _second_like_target(thetas, 4), thetas, 0.6, 0.3, 0.1)


def _lowpass_target(thetas_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    target = np.exp(-(thetas_deg.astype(np.float64) ** 2) / max(float(sigma_deg) ** 2, 1e-8))
    return target / max(float(np.max(target)), 1e-8)


def _lowpass_row_details(y: np.ndarray, thetas: np.ndarray, sigmas_deg: tuple[float, ...] = (8.0, 12.0, 16.0)) -> dict[str, float]:
    center_idx = int(np.argmin(np.abs(thetas)))
    edge_mask = np.abs(thetas) >= 0.75 * float(np.max(np.abs(thetas)))
    best = None
    for sigma in sigmas_deg:
        target = _lowpass_target(thetas, sigma)
        y_norm = y.astype(np.float64) / max(float(np.max(y)), 1e-8)
        denom = max(float(np.sum(target * target)), 1e-8)
        a = max(float(np.sum(target * y_norm) / denom), 0.0)
        y_fit = a * target
        ss_res = float(np.sum((y_norm - y_fit) ** 2))
        ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        center_val = float(y[center_idx])
        edge_mean = float(np.mean(y[edge_mask]))
        center_score = float(np.clip(center_val / max(float(np.max(y)), 1e-8), 0.0, 1.0))
        shape_score = float(np.clip(r2, 0.0, 1.0))
        edge_reject_score = float(np.clip(1.0 - edge_mean / max(center_val, 1e-8), 0.0, 1.0))
        score = 0.35 * center_score + 0.35 * shape_score + 0.15 * edge_reject_score
        candidate = {
            "task_score": score,
            "best_sigma_deg": float(sigma),
            "center_score": center_score,
            "shape_score": shape_score,
            "edge_reject_score": edge_reject_score,
            "bandwidth_score": score,
        }
        if best is None or candidate["task_score"] > best["task_score"]:
            best = candidate
    return best if best is not None else {"task_score": -1.0}


def _score_match_at_40(tpp_row: np.ndarray, tss_row: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    idx_p = int(np.argmin(np.abs(thetas - 40.0)))
    idx_m = int(np.argmin(np.abs(thetas + 40.0)))
    tpp_p = float(tpp_row[idx_p])
    tss_p = float(tss_row[idx_p])
    tpp_m = float(tpp_row[idx_m])
    tss_m = float(tss_row[idx_m])
    scale = max(float(np.max(np.abs(tpp_row))), float(np.max(np.abs(tss_row))), 1e-8)
    diff = 0.5 * (abs(tpp_p - tss_p) + abs(tpp_m - tss_m))
    return {
        "match40_score": float(np.clip(1.0 - diff / scale, 0.0, 1.0)),
        "tpp_+40": tpp_p,
        "tss_+40": tss_p,
        "tpp_-40": tpp_m,
        "tss_-40": tss_m,
    }


def _st2_ideal_map(lambdas_nm: np.ndarray, thetas_deg: np.ndarray, lambda0_nm: float, lambda_window_nm: float, theta_max_deg: float) -> tuple[np.ndarray, np.ndarray]:
    c0 = 299792458.0
    lam_m = np.asarray(lambdas_nm, dtype=np.float64)[:, None] * 1e-9
    th = np.deg2rad(np.asarray(thetas_deg, dtype=np.float64))[None, :]
    kx2 = ((2.0 * np.pi / np.maximum(lam_m, 1e-20)) * np.sin(th)) ** 2
    omega = 2.0 * np.pi * c0 / np.maximum(lam_m, 1e-20)
    omega0 = 2.0 * np.pi * c0 / max(float(lambda0_nm) * 1e-9, 1e-20)
    om2 = (omega - omega0) ** 2
    raw = kx2 * om2
    mask = (np.abs(lambdas_nm[:, None] - float(lambda0_nm)) <= 0.5 * float(lambda_window_nm)) & (np.abs(thetas_deg[None, :]) <= float(theta_max_deg))
    ideal = np.zeros_like(raw, dtype=np.float64)
    if np.any(mask):
        scale = float(np.max(np.abs(raw[mask])))
        if scale > 1e-12:
            ideal[mask] = raw[mask] / scale
    return ideal, mask


def _robust_norm(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr / max(float(np.max(np.abs(arr))), 1e-12)


def _st2_channel_score(spec_map: np.ndarray, ideal_map: np.ndarray, work_mask: np.ndarray, lambdas_nm: np.ndarray, thetas_deg: np.ndarray, lambda0_nm: float) -> dict[str, float]:
    spec = _robust_norm(spec_map)
    theta0_mask = np.abs(thetas_deg) <= 2.5
    lambda0_mask = np.abs(lambdas_nm - float(lambda0_nm)) <= 30.0
    theta_region = work_mask & theta0_mask[None, :]
    lambda_region = work_mask & lambda0_mask[:, None]
    theta_zero_leak = float(np.mean(spec[theta_region])) if np.any(theta_region) else 1.0
    lambda_zero_leak = float(np.mean(spec[lambda_region])) if np.any(lambda_region) else 1.0
    zero_score = 0.5 * float(np.clip(1.0 - theta_zero_leak, 0.0, 1.0)) + 0.5 * float(np.clip(1.0 - lambda_zero_leak, 0.0, 1.0))
    if np.any(work_mask):
        weights = np.zeros_like(ideal_map, dtype=np.float64)
        weights[work_mask] = 1.0 + ideal_map[work_mask]
        weights[theta_region] += 3.0
        weights[lambda_region] += 3.0
        diff = np.abs(spec - ideal_map)
        w = weights[work_mask]
        mae = float(np.sum(diff[work_mask] * w) / np.sum(w))
        rmse = float(np.sqrt(np.sum(((spec - ideal_map) ** 2)[work_mask] * w) / np.sum(w)))
        weighted_error_score = float(np.clip(1.0 - (0.7 * mae + 0.3 * rmse), 0.0, 1.0))
        phi = ideal_map[work_mask]
        y = spec[work_mask]
        proj_corr = float(np.dot(y, phi) / max(float(np.linalg.norm(y) * np.linalg.norm(phi)), 1e-12))
    else:
        weighted_error_score = 0.0
        proj_corr = 0.0
    total = 0.65 * zero_score + 0.25 * weighted_error_score + 0.10 * proj_corr
    return {"score_total": total, "score_zero_lines": zero_score, "score_weighted_error": weighted_error_score, "score_projection": proj_corr}


def task_score_details(case: TaskCase, pred_raw: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_raw, dtype=np.float32)
    lam_idx = int(np.argmin(np.abs(lambdas.astype(np.float64) - float(case.target_lambda_nm))))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))

    if case.task_key == "p_second_order":
        details = _second_order_row_details(pred[0, lam_idx], thetas)
        details["task_score"] = details["score"]
        details["tpp_at_40"] = float(pred[0, lam_idx, idx40])
        return details

    if case.task_key == "polarization_independent":
        tpp = _second_order_row_details(pred[0, lam_idx], thetas)
        tss = _second_order_row_details(pred[1, lam_idx], thetas)
        match = _score_match_at_40(pred[0, lam_idx], pred[1, lam_idx], thetas)
        base_joint = min(float(tpp["score"]), float(tss["score"]))
        return {
            "task_score": 0.7 * base_joint + 0.3 * float(match["match40_score"]),
            "base_joint_score": base_joint,
            "tpp_score": float(tpp["score"]),
            "tss_score": float(tss["score"]),
            **match,
        }

    if case.task_key == "polarization_multiplexed":
        active = _second_order_row_details(pred[0, lam_idx], thetas)
        passive_row = pred[1, lam_idx]
        passive_max = float(np.max(passive_row))
        passive_mean = float(np.mean(passive_row))
        silent = float(np.clip(1.0 - passive_max / 0.1, 0.0, 1.0))
        idxm40 = int(np.argmin(np.abs(thetas + 40.0)))
        return {
            "task_score": 0.5 * float(active["score"]) + 0.5 * silent,
            "active_shape_score": float(active["score"]),
            "silent_score": silent,
            "passive_max": passive_max,
            "passive_mean": passive_mean,
            "active_+40": float(pred[0, lam_idx, idx40]),
            "active_-40": float(pred[0, lam_idx, idxm40]),
            "passive_+40": float(pred[1, lam_idx, idx40]),
            "passive_-40": float(pred[1, lam_idx, idxm40]),
        }

    if case.task_key == "fourth_order":
        details = _fourth_order_row_details(pred[0, lam_idx], thetas)
        details["task_score"] = details["score"]
        details["tpp_at_40"] = float(pred[0, lam_idx, idx40])
        return details

    if case.task_key == "lowpass":
        details = _lowpass_row_details(pred[0, lam_idx], thetas)
        details["tpp_at_40"] = float(pred[0, lam_idx, idx40])
        return details

    if case.task_key == "st2":
        actual_lambda = float(lambdas[lam_idx])
        ideal_map, work_mask = _st2_ideal_map(lambdas.astype(np.float64), thetas.astype(np.float64), actual_lambda, 100.0, float(np.max(np.abs(thetas))))
        tpp = _st2_channel_score(pred[0], ideal_map, work_mask, lambdas.astype(np.float64), thetas.astype(np.float64), actual_lambda)
        tss = _st2_channel_score(pred[1], ideal_map, work_mask, lambdas.astype(np.float64), thetas.astype(np.float64), actual_lambda)
        return {
            "task_score": 0.5 * (float(tpp["score_total"]) + float(tss["score_total"])),
            "score_total": 0.5 * (float(tpp["score_total"]) + float(tss["score_total"])),
            "tpp_score_total": float(tpp["score_total"]),
            "tss_score_total": float(tss["score_total"]),
            "score_zero_lines": 0.5 * (float(tpp["score_zero_lines"]) + float(tss["score_zero_lines"])),
            "score_weighted_error": 0.5 * (float(tpp["score_weighted_error"]) + float(tss["score_weighted_error"])),
            "score_projection": 0.5 * (float(tpp["score_projection"]) + float(tss["score_projection"])),
        }

    raise ValueError(f"Unsupported task key: {case.task_key}")


def original_input_score(case: TaskCase, target_raw: np.ndarray, lambdas: np.ndarray, thetas: np.ndarray) -> dict[str, float]:
    return task_score_details(case, target_raw, lambdas, thetas)
