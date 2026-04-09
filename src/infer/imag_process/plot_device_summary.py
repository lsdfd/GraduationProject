from __future__ import annotations

import argparse
import math
import re
import string
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from infer.imag_process.scan_kspace import (  # noqa: E402
    DATA_PATH,
    LAMBDAS,
    NA,
    THETAS,
    build_2d_kspace,
    load_structure_from_npy,
    scan_phi,
)
from infer.task_library import (  # noqa: E402
    TASK_CASES,
    TaskCase,
    _lowpass_target,
    _second_like_target,
    default_train_npz,
    task_score_details_at_lambda,
)


BASE_DEFAULT_PANELS = (
    "structure_unit",
    "theta_curve_tpp",
    "lambda_theta_tpp",
    "kspace_tpp",
    "summary_table",
)
TSS_PANELS = ("theta_curve_tss", "lambda_theta_tss", "kspace_tss")
DEFAULT_PANELS = BASE_DEFAULT_PANELS + TSS_PANELS
OPTIONAL_PANELS = ("score_box", "cross_sections", "structure_hist")
ALL_PANELS = DEFAULT_PANELS + OPTIONAL_PANELS

TASK_DIR_TO_KEY = {
    "p": "p_second_order",
    "sp-all": "polarization_independent",
    "sp-mutiplex": "polarization_multiplexed",
    "fourth": "fourth_order",
    "lowpass": "lowpass",
    "st2": "st2",
}


@dataclass
class SummaryData:
    structure: np.ndarray
    case: TaskCase | None
    task_key: str | None
    task_label: str
    lambda_nm: float
    lambdas_nm: np.ndarray
    thetas_deg: np.ndarray
    tpp_lambda_theta: np.ndarray | None
    tss_lambda_theta: np.ndarray | None
    tpp_row: np.ndarray | None
    tss_row: np.ndarray | None
    tpp_ideal_dense: np.ndarray | None
    tss_ideal_dense: np.ndarray | None
    theta_dense: np.ndarray
    theta_sample: np.ndarray
    tpp_row_sample: np.ndarray | None
    tss_row_sample: np.ndarray | None
    kx_norm: np.ndarray | None
    ky_norm: np.ndarray | None
    T_pp: np.ndarray | None
    T_ss: np.ndarray | None
    metrics: dict[str, float] | None
    score_text: str | None
    summary_table_rows: list[tuple[str, str]]
    case_title: str
    lowpass_alpha_deg: float | None


@dataclass(frozen=True)
class TaskDisplayPolicy:
    show_tss: bool
    normalize_tss_theta: bool
    normalize_tss_kspace: bool
    normalize_tpp_kspace: bool
    draw_tss_ideal: bool
    tss_ideal_mode: str  # "task" or "zero"


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _task_label_from_key(task_key: str | None) -> str:
    mapping = {
        "p_second_order": "2nd-order",
        "polarization_independent": "Polarization-independent",
        "polarization_multiplexed": "Polarization-multiplexed",
        "fourth_order": "4th-order",
        "lowpass": "Low-pass",
        "st2": "Spatiotemporal",
    }
    return mapping.get(task_key, "Device summary")


def _guess_task_key_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part in TASK_DIR_TO_KEY:
            return TASK_DIR_TO_KEY[part]
    return None


def _match_case(structure_path: Path, lambda_nm: float, task_key: str | None) -> TaskCase | None:
    path_text = str(structure_path)
    filtered = list(TASK_CASES)
    if task_key:
        filtered = [case for case in filtered if case.task_key == task_key]
    filtered = [case for case in filtered if abs(float(case.target_lambda_nm) - float(lambda_nm)) < 1e-6]
    for case in filtered:
        if case.case_label in path_text:
            return case
    m = re.search(r"id=?0*([0-9]+)", path_text)
    if m is not None:
        wanted_id = int(m.group(1))
        for case in filtered:
            if int(case.sample_idx) == wanted_id:
                return case
    if len(filtered) == 1:
        return filtered[0]
    return None


def _panel_list_from_arg(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip().lower() in {"", "default"}:
        return DEFAULT_PANELS
    if raw.strip().lower() == "all":
        return ALL_PANELS
    parts = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in parts if item not in ALL_PANELS]
    if unknown:
        raise ValueError(f"Unknown panels: {unknown}. Available: {ALL_PANELS}")
    return parts


def _choose_theta_sample(thetas: np.ndarray, n_points: int = 17) -> np.ndarray:
    if int(n_points) >= len(thetas):
        return np.asarray(thetas, dtype=np.float64)
    idx = np.unique(np.round(np.linspace(0, len(thetas) - 1, n_points)).astype(int))
    return np.asarray(thetas, dtype=np.float64)[idx]


def _interp_row(row: np.ndarray, thetas: np.ndarray, theta_dense: np.ndarray) -> np.ndarray:
    return np.interp(theta_dense, np.asarray(thetas, dtype=np.float64), np.asarray(row, dtype=np.float64))


def _value_at(theta_axis: np.ndarray, row: np.ndarray, theta_deg: float) -> float:
    idx = int(np.argmin(np.abs(np.asarray(theta_axis, dtype=np.float64) - float(theta_deg))))
    return float(np.asarray(row, dtype=np.float64)[idx])


def _ideal_rows(task_key: str | None, theta_dense: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    zeros = np.zeros_like(theta_dense, dtype=np.float64)
    if task_key == "p_second_order":
        return _second_like_target(theta_dense, 2), zeros
    if task_key == "polarization_independent":
        ideal = _second_like_target(theta_dense, 2)
        return ideal.copy(), ideal.copy()
    if task_key == "polarization_multiplexed":
        return _second_like_target(theta_dense, 2), zeros
    if task_key == "fourth_order":
        ideal = _second_like_target(theta_dense, 4)
        return ideal.copy(), ideal.copy()
    if task_key == "lowpass":
        return _lowpass_target(theta_dense, 12.0), zeros
    return _second_like_target(theta_dense, 2), zeros


def _fit_lowpass_alpha_deg(theta_deg: np.ndarray, actual_row: np.ndarray) -> float:
    theta = np.asarray(theta_deg, dtype=np.float64)
    actual = np.asarray(actual_row, dtype=np.float64)
    peak = float(np.max(actual))
    if peak <= 1e-12:
        return 12.0
    target = actual / peak
    alpha_grid = np.linspace(4.0, 40.0, 721, dtype=np.float64)
    errors = []
    for alpha in alpha_grid:
        model = np.exp(-((theta / alpha) ** 2))
        errors.append(float(np.mean((model - target) ** 2)))
    return float(alpha_grid[int(np.argmin(np.asarray(errors)))])


def _task_display_policy(task_key: str | None) -> TaskDisplayPolicy:
    if task_key == "polarization_independent":
        return TaskDisplayPolicy(
            show_tss=True,
            normalize_tss_theta=True,
            normalize_tss_kspace=True,
            normalize_tpp_kspace=True,
            draw_tss_ideal=True,
            tss_ideal_mode="task",
        )
    if task_key == "polarization_multiplexed":
        return TaskDisplayPolicy(
            show_tss=True,
            normalize_tss_theta=False,
            normalize_tss_kspace=False,
            normalize_tpp_kspace=False,
            draw_tss_ideal=True,
            tss_ideal_mode="zero",
        )
    if task_key == "fourth_order":
        return TaskDisplayPolicy(
            show_tss=False,
            normalize_tss_theta=False,
            normalize_tss_kspace=False,
            normalize_tpp_kspace=False,
            draw_tss_ideal=True,
            tss_ideal_mode="task",
        )
    if task_key in {"p_second_order", "lowpass"}:
        return TaskDisplayPolicy(
            show_tss=False,
            normalize_tss_theta=False,
            normalize_tss_kspace=False,
            normalize_tpp_kspace=False,
            draw_tss_ideal=False,
            tss_ideal_mode="zero",
        )
    return TaskDisplayPolicy(
        show_tss=True,
        normalize_tss_theta=False,
        normalize_tss_kspace=False,
        normalize_tpp_kspace=False,
        draw_tss_ideal=False,
        tss_ideal_mode="zero",
    )


def _default_panels_for_task(task_key: str | None) -> tuple[str, ...]:
    if task_key == "polarization_independent":
        return (
            "structure_unit",
            "theta_curve_tpp",
            "summary_table",
            "theta_curve_tss",
        )
    policy = _task_display_policy(task_key)
    panels = list(BASE_DEFAULT_PANELS)
    if task_key == "lowpass" and "kspace_tpp" in panels:
        panels.remove("kspace_tpp")
    if policy.show_tss:
        panels.insert(2, "theta_curve_tss")
        panels.insert(4, "lambda_theta_tss")
        panels.insert(6, "kspace_tss")
    return tuple(panels)


def _scale_ideal_at_theta(ideal_row: np.ndarray, theta_dense: np.ndarray, actual_value: float, theta_ref_deg: float) -> np.ndarray:
    ideal = np.asarray(ideal_row, dtype=np.float64).copy()
    if np.allclose(ideal, 0.0):
        return ideal
    ref = max(_value_at(theta_dense, ideal, float(theta_ref_deg)), 1e-8)
    return ideal * (float(actual_value) / ref)


def _scale_ideal_by_task(task_key: str | None, ideal_row: np.ndarray, theta_dense: np.ndarray, actual_row: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual_row, dtype=np.float64)
    if task_key == "lowpass":
        return _scale_ideal_at_theta(ideal_row, theta_dense, _value_at(np.asarray(THETAS, dtype=np.float64), actual, 0.0), 0.0)
    return _scale_ideal_at_theta(ideal_row, theta_dense, _value_at(np.asarray(THETAS, dtype=np.float64), actual, 40.0), 40.0)


def _channel_has_explicit_ideal(task_key: str | None, channel: str) -> bool:
    policy = _task_display_policy(task_key)
    if channel == "tpp":
        return True
    return policy.show_tss and policy.draw_tss_ideal


def _score_summary(case: TaskCase | None, lambdas_nm: np.ndarray, thetas_deg: np.ndarray, tpp_map: np.ndarray, tss_map: np.ndarray, lambda_nm: float) -> tuple[dict[str, float] | None, str | None]:
    if case is None:
        return None, None
    pred = np.stack([tpp_map, tss_map], axis=0).astype(np.float32)
    lam_idx = int(np.argmin(np.abs(np.asarray(lambdas_nm, dtype=np.float64) - float(lambda_nm))))
    details = task_score_details_at_lambda(case, pred, np.asarray(lambdas_nm, dtype=np.float64), np.asarray(thetas_deg, dtype=np.float64), lam_idx)
    parts = [f"task_score={float(details.get('task_score', -1.0)):.3f}"]
    if "tpp_+40" in details and "tss_+40" in details:
        parts.append(f"tpp/tss@40={float(details['tpp_+40']):.3f}/{float(details['tss_+40']):.3f}")
    elif "tpp_at_40" in details:
        parts.append(f"tpp@40={float(details['tpp_at_40']):.3f}")
    elif "active_+40" in details and "passive_+40" in details:
        parts.append(f"active/passive@40={float(details['active_+40']):.3f}/{float(details['passive_+40']):.3f}")
    return details, "\n".join(parts)


def _score_at_target_lambda(
    case: TaskCase | None,
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    tpp_map: np.ndarray | None,
    tss_map: np.ndarray | None,
    target_lambda_nm: float,
) -> float | None:
    if case is None or tpp_map is None or tss_map is None:
        return None
    lam_idx = int(np.argmin(np.abs(np.asarray(lambdas_nm, dtype=np.float64) - float(target_lambda_nm))))
    nearest = float(np.asarray(lambdas_nm, dtype=np.float64)[lam_idx])
    if abs(nearest - float(target_lambda_nm)) > 1e-6:
        return None
    pred = np.stack([tpp_map, tss_map], axis=0).astype(np.float32)
    details = task_score_details_at_lambda(case, pred, np.asarray(lambdas_nm, dtype=np.float64), np.asarray(thetas_deg, dtype=np.float64), lam_idx)
    score = details.get("task_score", None)
    if score is None or not np.isfinite(float(score)):
        return None
    return float(score)


def _build_summary_table_rows(
    case: TaskCase | None,
    task_label: str,
    lambda_nm: float,
    metrics: dict[str, float] | None,
    tpp_row: np.ndarray | None,
    tss_row: np.ndarray | None,
    thetas_deg: np.ndarray,
    lambdas_nm: np.ndarray,
    tpp_lambda_theta: np.ndarray | None,
    tss_lambda_theta: np.ndarray | None,
    lowpass_alpha_deg: float | None,
) -> list[tuple[str, str]]:
    idx40 = int(np.argmin(np.abs(np.asarray(thetas_deg, dtype=np.float64) - 40.0)))
    tpp40 = float(np.asarray(tpp_row, dtype=np.float64)[idx40]) if tpp_row is not None else float("nan")
    tss_max = float(np.nanmax(np.asarray(tss_row, dtype=np.float64))) if tss_row is not None else float("nan")
    score = float(metrics.get("task_score", float("nan"))) if metrics is not None and "task_score" in metrics else float("nan")
    score_lo = _score_at_target_lambda(case, lambdas_nm, thetas_deg, tpp_lambda_theta, tss_lambda_theta, float(lambda_nm) - 50.0)
    score_hi = _score_at_target_lambda(case, lambdas_nm, thetas_deg, tpp_lambda_theta, tss_lambda_theta, float(lambda_nm) + 50.0)
    score_pm50 = (
        f"{score_lo:.3f}/{score_hi:.3f}"
        if score_lo is not None and score_hi is not None
        else "--/--"
    )
    rows = [
        ("Task", task_label),
        ("Lambda", f"{int(round(lambda_nm))} nm"),
        ("Score", f"{score:.3f}" if np.isfinite(score) else "--"),
        ("Tpp@40", f"{tpp40:.3f}" if np.isfinite(tpp40) else "--"),
        ("NA", f"{NA:.3f}"),
    ]
    if case is not None and case.task_key not in {"lowpass", "polarization_independent"}:
        rows.insert(3, ("Score (±50 nm)", score_pm50))
    if case is not None and case.task_key == "polarization_independent":
        rows[3] = ("T@40", f"{tpp40:.3f}" if np.isfinite(tpp40) else "--")
    if lowpass_alpha_deg is not None:
        insert_idx = 3 if case is None or case.task_key == "lowpass" else 4
        rows.insert(insert_idx, (r"$\alpha$", f"{lowpass_alpha_deg:.1f} deg"))
    if case is not None and case.task_key == "polarization_multiplexed":
        rows.insert(-1, ("Tss max", f"{tss_max:.3f}" if np.isfinite(tss_max) else "--"))
    return rows


def _normalize_if_needed(values: np.ndarray | None, enabled: bool) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).copy()
    if not enabled:
        return arr
    vmax = float(np.nanmax(np.abs(arr)))
    if vmax <= 1e-12:
        return arr
    return arr / vmax


def _build_output_dir(structure_path: Path, mode: str, lambda_nm: float, outdir: Path | None) -> Path:
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir
    base_dir = structure_path.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"run_device_summary_{_safe_tag(structure_path.stem)}_{int(round(lambda_nm))}nm_{mode}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_kspace_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        np.asarray(data["kx_norm"], dtype=np.float64),
        np.asarray(data["ky_norm"], dtype=np.float64),
        np.asarray(data["T_pp"], dtype=np.float64),
        np.asarray(data["T_ss"], dtype=np.float64),
    )


def _scan_lambda_theta(structure: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    tpp = np.zeros((len(LAMBDAS), len(THETAS)), dtype=np.float64)
    tss = np.zeros_like(tpp)
    for i, lam in enumerate(LAMBDAS):
        tss_row, tpp_row = scan_phi(structure, float(lam), 0.0, device=device)
        if tss_row is None or tpp_row is None:
            raise RuntimeError("torcwa is unavailable or phi=0 scan failed; cannot build T(lambda, theta).")
        tpp[i] = np.asarray(tpp_row, dtype=np.float64)
        tss[i] = np.asarray(tss_row, dtype=np.float64)
    return tpp, tss


def _build_kspace(structure: np.ndarray, lambda_nm: float, device: str, tpp_row: np.ndarray, tss_row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tss22, tpp22 = scan_phi(structure, lambda_nm, 22.5, device=device)
    tss45, tpp45 = scan_phi(structure, lambda_nm, 45.0, device=device)
    if tss22 is None or tpp22 is None or tss45 is None or tpp45 is None:
        raise RuntimeError("torcwa is unavailable or phi=22.5/45 scan failed; cannot build k-space.")
    kx_norm, ky_norm, T_pp = build_2d_kspace(np.asarray(tpp_row, dtype=np.float64), np.asarray(tpp22, dtype=np.float64), np.asarray(tpp45, dtype=np.float64))
    _, _, T_ss = build_2d_kspace(np.asarray(tss_row, dtype=np.float64), np.asarray(tss22, dtype=np.float64), np.asarray(tss45, dtype=np.float64))
    return kx_norm, ky_norm, T_pp, T_ss


@contextmanager
def _paper_style():
    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        yield


def _draw_structure(ax: plt.Axes, data: np.ndarray, title: str) -> None:
    ax.imshow(np.asarray(data, dtype=np.float64), cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _draw_theta_curve(ax: plt.Axes, summary: SummaryData, channel: str) -> None:
    if summary.tpp_row is None or summary.tss_row is None or summary.tpp_ideal_dense is None or summary.tss_ideal_dense is None:
        ax.axis("off")
        return
    policy = _task_display_policy(summary.task_key)
    if channel == "tpp":
        line_color = "#4b5563"
        star_color = "#d55e00"
        ideal = np.asarray(summary.tpp_ideal_dense, dtype=np.float64)
        sample = np.asarray(summary.tpp_row_sample, dtype=np.float64)
        actual = np.asarray(summary.tpp_row, dtype=np.float64)
        title = r"$t_{pp}(\theta)$"
        label = r"$t_{pp}$"
        if summary.task_key == "lowpass" and summary.lowpass_alpha_deg is not None:
            ref_label = rf"Fit ($\alpha$={summary.lowpass_alpha_deg:.1f}$^\circ$)"
        else:
            ref_label = f"Ideal {label}"
        show_ideal = True
    else:
        if not policy.show_tss:
            ax.axis("off")
            return
        line_color = "#6b7280"
        star_color = "#1f77b4"
        ideal = np.asarray(summary.tss_ideal_dense, dtype=np.float64)
        sample = np.asarray(summary.tss_row_sample, dtype=np.float64)
        actual = np.asarray(summary.tss_row, dtype=np.float64)
        if policy.normalize_tss_theta:
            actual = _normalize_if_needed(actual, True)
            sample = _normalize_if_needed(sample, True)
            ideal = _normalize_if_needed(ideal, True)
        elif policy.tss_ideal_mode == "zero":
            ideal = np.zeros_like(ideal)
        title = r"$t_{ss}(\theta)$"
        label = r"$t_{ss}$"
        ref_label = "Zero reference" if policy.tss_ideal_mode == "zero" else f"Ideal {label}"
        show_ideal = _channel_has_explicit_ideal(summary.task_key, "tss")
    if show_ideal:
        ax.plot(summary.theta_dense, ideal, color=line_color, linestyle="--", lw=1.8, label=ref_label)
    ax.plot(
        summary.theta_sample,
        sample,
        marker="*",
        linestyle="none",
        ms=8.5,
        color=star_color,
        markeredgecolor=star_color,
        label=f"Sample {label}",
    )
    ax.set_xlim(float(summary.thetas_deg[0]), float(summary.thetas_deg[-1]))
    ymax = 1.05 * max(float(np.max(np.abs(actual))), float(np.max(np.abs(ideal if show_ideal else 0.0))), 1e-3)
    ymin = -0.06 * ymax if (channel == "tss" and show_ideal and policy.tss_ideal_mode == "zero") else -0.02 * ymax
    if summary.task_key == "polarization_multiplexed" and channel == "tss":
        ymax = 1.0
        ymin = -0.02
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$|t|$")
    ax.set_title(title, pad=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=2, handlelength=2.0, columnspacing=1.1)


def _draw_lambda_theta(ax: plt.Axes, data: np.ndarray | None, lambdas_nm: np.ndarray, thetas_deg: np.ndarray, title: str, lambda_nm: float) -> None:
    if data is None:
        ax.axis("off")
        return
    im = ax.imshow(
        np.asarray(data, dtype=np.float64),
        origin="lower",
        aspect="auto",
        extent=[float(thetas_deg[0]), float(thetas_deg[-1]), float(lambdas_nm[0]), float(lambdas_nm[-1])],
        cmap="turbo",
        vmin=0.0,
        vmax=max(float(np.nanmax(data)), 1e-8),
        interpolation="bicubic",
    )
    ax.set_xlabel(r"Incidence angle $\theta$ (deg)")
    ax.set_ylabel(r"Wavelength $\lambda$ (nm)")
    ax.set_title(title, pad=8)
    ax.axhline(float(lambda_nm), color="white", lw=1.0, linestyle="--", alpha=0.95)
    ax.text(
        float(thetas_deg[0]) + 2.0,
        float(lambda_nm) + 8.0,
        r"$\lambda_0$",
        color="white",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox={"facecolor": (0, 0, 0, 0.18), "edgecolor": "none", "pad": 1.5},
    )
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"Amplitude $|t|$")


def _draw_kspace(
    ax: plt.Axes,
    kx_norm: np.ndarray | None,
    ky_norm: np.ndarray | None,
    data: np.ndarray | None,
    title: str,
    cax: plt.Axes | None = None,
    vmax_override: float | None = None,
) -> None:
    if kx_norm is None or ky_norm is None or data is None:
        ax.axis("off")
        return
    vmax = float(vmax_override) if vmax_override is not None else max(float(np.nanmax(data)), 1e-8)
    im = ax.imshow(
        np.asarray(data, dtype=np.float64),
        origin="lower",
        extent=[float(kx_norm[0]), float(kx_norm[-1]), float(ky_norm[0]), float(ky_norm[-1])],
        cmap="turbo",
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
    )
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(NA * np.cos(tt), NA * np.sin(tt), color="white", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.set_xlabel(r"$k_x/k_0$")
    ax.set_ylabel(r"$k_y/k_0$")
    ax.set_title(title, pad=8)
    ax.set_aspect("equal")
    cb = plt.colorbar(im, cax=cax, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|t|$")


def _shared_kspace_vmax(summary: SummaryData, policy: TaskDisplayPolicy) -> float:
    tpp_data = _normalize_if_needed(summary.T_pp, policy.normalize_tpp_kspace)
    tss_data = _normalize_if_needed(summary.T_ss, policy.normalize_tss_kspace)
    return max(
        float(np.nanmax(tpp_data)) if tpp_data is not None else 0.0,
        float(np.nanmax(tss_data)) if tss_data is not None else 0.0,
        1e-8,
    )


def _draw_score_box(ax: plt.Axes, text: str | None) -> None:
    ax.axis("off")
    if not text:
        ax.text(0.03, 0.95, "No task score available", ha="left", va="top", fontsize=10)
        return
    ax.text(
        0.04,
        0.94,
        text,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.5,
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#b5b5b5", "boxstyle": "round,pad=0.35"},
    )
    ax.set_title("Score summary", pad=8)


def _draw_summary_table(ax: plt.Axes, rows: list[tuple[str, str]]) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.45, 0.43],
        edges="closed",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.6)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor("#6b7280")
        if r == 0:
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(weight="bold", ha="center")
        else:
            cell.set_facecolor("white")
            if c == 0:
                cell.set_text_props(weight="bold")
    ax.set_title("Summary", pad=8)


def _draw_cross_sections(ax: plt.Axes, summary: SummaryData) -> None:
    if summary.tpp_lambda_theta is None or summary.tss_lambda_theta is None:
        ax.axis("off")
        return
    lam_idx = int(np.argmin(np.abs(summary.lambdas_nm - float(summary.lambda_nm))))
    idx0 = int(np.argmin(np.abs(summary.thetas_deg)))
    idx40 = int(np.argmin(np.abs(summary.thetas_deg - 40.0)))
    p_color = "#d55e00"
    s_color = "#1f77b4"
    ax.plot(summary.lambdas_nm, summary.tpp_lambda_theta[:, idx0], color=p_color, lw=1.6, label=r"$t_{pp}@0^\circ$")
    ax.plot(summary.lambdas_nm, summary.tpp_lambda_theta[:, idx40], color=p_color, lw=1.2, linestyle="--", label=r"$t_{pp}@40^\circ$")
    ax.plot(summary.lambdas_nm, summary.tss_lambda_theta[:, idx0], color=s_color, lw=1.6, label=r"$t_{ss}@0^\circ$")
    ax.plot(summary.lambdas_nm, summary.tss_lambda_theta[:, idx40], color=s_color, lw=1.2, linestyle="--", label=r"$t_{ss}@40^\circ$")
    ax.axvline(float(summary.lambda_nm), color="black", lw=0.9, alpha=0.5)
    ax.set_xlabel(r"$\lambda$ (nm)")
    ax.set_ylabel(r"$|t|$")
    ax.set_title("Spectral cross-sections", pad=8)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(alpha=0.2, linewidth=0.6)


def _draw_structure_hist(ax: plt.Axes, structure: np.ndarray) -> None:
    vals = np.asarray(structure, dtype=np.float64).ravel()
    ax.hist(vals, bins=np.linspace(0.0, 1.0, 21), color="#6f6f6f", edgecolor="white", linewidth=0.6)
    ax.axvline(float(np.mean(vals)), color="#d55e00", lw=1.4, linestyle="--", label=f"mean={float(np.mean(vals)):.3f}")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    ax.set_title("Structure histogram", pad=8)
    ax.legend(frameon=False, loc="upper center")


def _panel_dispatch(ax: plt.Axes, panel: str, summary: SummaryData) -> None:
    policy = _task_display_policy(summary.task_key)
    if panel == "structure_unit":
        _draw_structure(ax, summary.structure, "Unit cell")
    elif panel == "summary_table":
        _draw_summary_table(ax, summary.summary_table_rows)
    elif panel == "theta_curve_tpp":
        _draw_theta_curve(ax, summary, "tpp")
    elif panel == "theta_curve_tss":
        _draw_theta_curve(ax, summary, "tss")
    elif panel == "lambda_theta_tpp":
        _draw_lambda_theta(ax, summary.tpp_lambda_theta, summary.lambdas_nm, summary.thetas_deg, r"$T_{pp}(\lambda,\theta)$", summary.lambda_nm)
    elif panel == "lambda_theta_tss":
        _draw_lambda_theta(ax, summary.tss_lambda_theta, summary.lambdas_nm, summary.thetas_deg, r"$T_{ss}(\lambda,\theta)$", summary.lambda_nm)
    elif panel == "kspace_tpp":
        tpp_data = _normalize_if_needed(summary.T_pp, policy.normalize_tpp_kspace)
        vmax_override = None
        if summary.task_key != "polarization_multiplexed" and policy.show_tss and not policy.normalize_tss_kspace and not policy.normalize_tpp_kspace:
            vmax_override = _shared_kspace_vmax(summary, policy)
        _draw_kspace(ax, summary.kx_norm, summary.ky_norm, tpp_data, r"$T_{pp}(k_x,k_y)$", vmax_override=vmax_override)
    elif panel == "kspace_tss":
        if not policy.show_tss:
            ax.axis("off")
            return
        tss_data = _normalize_if_needed(summary.T_ss, policy.normalize_tss_kspace)
        vmax_override = None
        if summary.task_key == "polarization_multiplexed":
            vmax_override = 1.0
        elif not policy.normalize_tss_kspace and not policy.normalize_tpp_kspace:
            vmax_override = _shared_kspace_vmax(summary, policy)
        _draw_kspace(ax, summary.kx_norm, summary.ky_norm, tss_data, r"$T_{ss}(k_x,k_y)$", vmax_override=vmax_override)
    elif panel == "score_box":
        _draw_score_box(ax, summary.score_text)
    elif panel == "cross_sections":
        _draw_cross_sections(ax, summary)
    elif panel == "structure_hist":
        _draw_structure_hist(ax, summary.structure)
    else:
        raise ValueError(f"Unsupported panel: {panel}")


def _is_default_task_layout(summary: SummaryData, panels: tuple[str, ...]) -> bool:
    return panels == _default_panels_for_task(summary.task_key)


def _draw_figure(summary: SummaryData, panels: tuple[str, ...], mode: str, outdir: Path) -> list[Path]:
    paths: list[Path] = []
    policy = _task_display_policy(summary.task_key)
    if mode == "main":
        if _is_default_task_layout(summary, panels):
            if summary.task_key == "polarization_independent":
                fig = plt.figure(figsize=(12.8, 8.0), constrained_layout=True)
                outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.05])
                axes = {
                    "structure_unit": fig.add_subplot(outer[0, 0]),
                    "theta_curve_tpp": fig.add_subplot(outer[0, 1]),
                    "summary_table": fig.add_subplot(outer[1, 0]),
                    "theta_curve_tss": fig.add_subplot(outer[1, 1]),
                }
            elif summary.task_key == "lowpass":
                fig = plt.figure(figsize=(12.8, 8.0), constrained_layout=True)
                outer = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.05])
                axes = {
                    "structure_unit": fig.add_subplot(outer[0, 0]),
                    "theta_curve_tpp": fig.add_subplot(outer[0, 1]),
                    "lambda_theta_tpp": fig.add_subplot(outer[1, 0]),
                    "summary_table": fig.add_subplot(outer[1, 1]),
                }
            elif policy.show_tss:
                fig = plt.figure(figsize=(20.8, 8.0), constrained_layout=True)
                outer = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.05, 1.05, 1.0])
                axes = {
                    "structure_unit": fig.add_subplot(outer[0, 0]),
                    "theta_curve_tpp": fig.add_subplot(outer[0, 1]),
                    "theta_curve_tss": fig.add_subplot(outer[0, 2]),
                    "lambda_theta_tpp": fig.add_subplot(outer[0, 3]),
                    "lambda_theta_tss": fig.add_subplot(outer[1, 3]),
                    "summary_table": fig.add_subplot(outer[1, 2]),
                }
                left = outer[1, 0].subgridspec(1, 2, width_ratios=[24, 0.75], wspace=0.025)
                mid = outer[1, 1].subgridspec(1, 2, width_ratios=[24, 0.75], wspace=0.025)
                axes["kspace_tpp"] = fig.add_subplot(left[0, 0])
                axes["_kspace_tpp_cbar"] = fig.add_subplot(left[0, 1])
                axes["kspace_tss"] = fig.add_subplot(mid[0, 0])
                axes["_kspace_tss_cbar"] = fig.add_subplot(mid[0, 1])
            else:
                fig = plt.figure(figsize=(16.8, 8.0), constrained_layout=True)
                outer = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.05, 1.05, 1.0, 1.0])
                axes = {
                    "structure_unit": fig.add_subplot(outer[0, 0:2]),
                    "theta_curve_tpp": fig.add_subplot(outer[0, 2:4]),
                    "lambda_theta_tpp": fig.add_subplot(outer[0, 4:6]),
                    "summary_table": fig.add_subplot(outer[1, 3:6]),
                }
                left = outer[1, 0:3].subgridspec(1, 2, width_ratios=[24, 0.75], wspace=0.025)
                axes["kspace_tpp"] = fig.add_subplot(left[0, 0])
                axes["_kspace_tpp_cbar"] = fig.add_subplot(left[0, 1])
        else:
            n_cols = min(3, max(1, int(math.ceil(math.sqrt(len(panels))))))
            n_rows = int(math.ceil(len(panels) / n_cols))
            fig, axarr = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.6 * n_rows), constrained_layout=True, squeeze=False)
            axes = {}
            flat = axarr.ravel()
            for idx, panel in enumerate(panels):
                axes[panel] = flat[idx]
            for ax in flat[len(panels) :]:
                ax.axis("off")
    elif mode == "board":
        if summary.task_key in {"lowpass", "polarization_independent"} and _is_default_task_layout(summary, panels):
            fig, axarr = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True, squeeze=False)
            if summary.task_key == "polarization_independent":
                axes = {
                    "structure_unit": axarr[0, 0],
                    "theta_curve_tpp": axarr[0, 1],
                    "summary_table": axarr[1, 0],
                    "theta_curve_tss": axarr[1, 1],
                }
            else:
                axes = {
                    "structure_unit": axarr[0, 0],
                    "theta_curve_tpp": axarr[0, 1],
                    "lambda_theta_tpp": axarr[1, 0],
                    "summary_table": axarr[1, 1],
                }
        elif not policy.show_tss and _is_default_task_layout(summary, panels):
            fig = plt.figure(figsize=(16.8, 8.0), constrained_layout=True)
            outer = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0, 1.05, 1.05, 1.0, 1.0])
            axes = {
                "structure_unit": fig.add_subplot(outer[0, 0:2]),
                "theta_curve_tpp": fig.add_subplot(outer[0, 2:4]),
                "lambda_theta_tpp": fig.add_subplot(outer[0, 4:6]),
                "summary_table": fig.add_subplot(outer[1, 3:6]),
            }
            left = outer[1, 0:3].subgridspec(1, 2, width_ratios=[24, 0.75], wspace=0.025)
            axes["kspace_tpp"] = fig.add_subplot(left[0, 0])
            axes["_kspace_tpp_cbar"] = fig.add_subplot(left[0, 1])
        else:
            n_cols = min(3, max(1, len(panels)))
            n_rows = int(math.ceil(len(panels) / n_cols))
            fig, axarr = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), constrained_layout=True, squeeze=False)
            axes = {}
            flat = axarr.ravel()
            for idx, panel in enumerate(panels):
                axes[panel] = flat[idx]
            for ax in flat[len(panels) :]:
                ax.axis("off")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    for panel in panels:
        ax = axes[panel]
        if mode in {"main", "board"} and _is_default_task_layout(summary, panels) and panel == "kspace_tpp" and "_kspace_tpp_cbar" in axes:
            tpp_data = _normalize_if_needed(summary.T_pp, policy.normalize_tpp_kspace)
            if summary.task_key != "polarization_multiplexed" and policy.show_tss and not policy.normalize_tss_kspace:
                shared_kspace_vmax = max(
                    float(np.nanmax(tpp_data)) if tpp_data is not None else 0.0,
                    float(np.nanmax(summary.T_ss)) if summary.T_ss is not None else 0.0,
                    1e-8,
                )
            else:
                shared_kspace_vmax = max(float(np.nanmax(tpp_data)) if tpp_data is not None else 0.0, 1e-8)
            _draw_kspace(
                ax,
                summary.kx_norm,
                summary.ky_norm,
                tpp_data,
                r"$T_{pp}(k_x,k_y)$",
                cax=axes["_kspace_tpp_cbar"],
                vmax_override=shared_kspace_vmax,
            )
        elif mode in {"main", "board"} and _is_default_task_layout(summary, panels) and panel == "kspace_tss" and "_kspace_tss_cbar" in axes:
            tss_data = _normalize_if_needed(summary.T_ss, policy.normalize_tss_kspace)
            if summary.task_key == "polarization_multiplexed":
                shared_kspace_vmax = 1.0
            elif policy.normalize_tss_kspace:
                shared_kspace_vmax = max(float(np.nanmax(tss_data)) if tss_data is not None else 0.0, 1e-8)
            else:
                shared_kspace_vmax = max(
                    float(np.nanmax(_normalize_if_needed(summary.T_pp, policy.normalize_tpp_kspace))) if summary.T_pp is not None else 0.0,
                    float(np.nanmax(tss_data)) if tss_data is not None else 0.0,
                    1e-8,
                )
            _draw_kspace(
                ax,
                summary.kx_norm,
                summary.ky_norm,
                tss_data,
                r"$T_{ss}(k_x,k_y)$",
                cax=axes["_kspace_tss_cbar"],
                vmax_override=shared_kspace_vmax,
            )
        else:
            _panel_dispatch(ax, panel, summary)

    fig.suptitle(summary.case_title, fontsize=12, fontweight="bold", y=1.02)
    for ext in ("png", "pdf"):
        path = outdir / f"device_summary_{mode}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    for panel in panels:
        single_fig = plt.figure(figsize=(5.8, 4.6), constrained_layout=True)
        ax = single_fig.add_subplot(111)
        _panel_dispatch(ax, panel, summary)
        for ext in ("png", "pdf"):
            panel_path = outdir / f"{panel}.{ext}"
            single_fig.savefig(panel_path, dpi=300 if ext == "png" else None, bbox_inches="tight")
            paths.append(panel_path)
        plt.close(single_fig)
    return paths


def _prepare_summary(args: argparse.Namespace) -> SummaryData:
    structure_path = Path(args.structure_npy)
    if not structure_path.is_absolute():
        structure_path = PROJECT_ROOT / structure_path
    structure = load_structure_from_npy(structure_path, args.structure_index)

    task_key = args.task_key or _guess_task_key_from_path(structure_path)
    case = _match_case(structure_path, args.lambda_nm, task_key)
    if task_key is None and case is not None:
        task_key = case.task_key
    task_label = _task_label_from_key(task_key)

    theta_dense = np.linspace(float(THETAS[0]), float(THETAS[-1]), 401, dtype=np.float64)
    theta_sample = _choose_theta_sample(np.asarray(THETAS, dtype=np.float64), n_points=17)

    tpp_lambda_theta = None
    tss_lambda_theta = None
    need_lambda_theta = any(panel in args.panels for panel in ("lambda_theta_tpp", "lambda_theta_tss", "cross_sections"))
    need_theta_only = any(panel in args.panels for panel in ("theta_curve_tpp", "theta_curve_tss", "score_box", "summary_table"))

    if need_lambda_theta:
        tpp_lambda_theta, tss_lambda_theta = _scan_lambda_theta(structure, args.device)

    tpp_row = None
    tss_row = None
    tpp_ideal_dense = None
    tss_ideal_dense = None
    tpp_row_sample = None
    tss_row_sample = None
    metrics = None
    score_text = None
    lowpass_alpha_deg = None
    if tpp_lambda_theta is not None and tss_lambda_theta is not None:
        lam_idx = int(np.argmin(np.abs(np.asarray(LAMBDAS, dtype=np.float64) - float(args.lambda_nm))))
        tpp_row = np.asarray(tpp_lambda_theta[lam_idx], dtype=np.float64)
        tss_row = np.asarray(tss_lambda_theta[lam_idx], dtype=np.float64)
        ideal_tpp_raw, ideal_tss_raw = _ideal_rows(task_key, theta_dense)
        if task_key == "lowpass":
            lowpass_alpha_deg = _fit_lowpass_alpha_deg(np.asarray(THETAS, dtype=np.float64), tpp_row)
            ideal_tpp_raw = _lowpass_target(theta_dense, lowpass_alpha_deg)
        tpp_ideal_dense = _scale_ideal_by_task(task_key, ideal_tpp_raw, theta_dense, tpp_row)
        tss_ideal_dense = _scale_ideal_by_task(task_key, ideal_tss_raw, theta_dense, tss_row)
        tpp_row_sample = np.interp(theta_sample, np.asarray(THETAS, dtype=np.float64), tpp_row)
        tss_row_sample = np.interp(theta_sample, np.asarray(THETAS, dtype=np.float64), tss_row)
        metrics, score_text = _score_summary(case, np.asarray(LAMBDAS, dtype=np.float64), np.asarray(THETAS, dtype=np.float64), tpp_lambda_theta, tss_lambda_theta, args.lambda_nm)
    elif need_theta_only:
        tss_row_scan, tpp_row_scan = scan_phi(structure, float(args.lambda_nm), 0.0, device=args.device)
        if tss_row_scan is None or tpp_row_scan is None:
            raise RuntimeError("torcwa is unavailable or phi=0 scan failed; cannot build theta-response panels.")
        tpp_row = np.asarray(tpp_row_scan, dtype=np.float64)
        tss_row = np.asarray(tss_row_scan, dtype=np.float64)
        ideal_tpp_raw, ideal_tss_raw = _ideal_rows(task_key, theta_dense)
        if task_key == "lowpass":
            lowpass_alpha_deg = _fit_lowpass_alpha_deg(np.asarray(THETAS, dtype=np.float64), tpp_row)
            ideal_tpp_raw = _lowpass_target(theta_dense, lowpass_alpha_deg)
        tpp_ideal_dense = _scale_ideal_by_task(task_key, ideal_tpp_raw, theta_dense, tpp_row)
        tss_ideal_dense = _scale_ideal_by_task(task_key, ideal_tss_raw, theta_dense, tss_row)
        tpp_row_sample = np.interp(theta_sample, np.asarray(THETAS, dtype=np.float64), tpp_row)
        tss_row_sample = np.interp(theta_sample, np.asarray(THETAS, dtype=np.float64), tss_row)
        if case is not None and task_key != "st2":
            pred_theta_only = np.stack([tpp_row[None, :], tss_row[None, :]], axis=0).astype(np.float32)
            metrics = task_score_details_at_lambda(case, pred_theta_only, np.array([float(args.lambda_nm)], dtype=np.float64), np.asarray(THETAS, dtype=np.float64), 0)
            score_text = "\n".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, (int, float, np.floating)) and np.isfinite(v))

    kx_norm = None
    ky_norm = None
    T_pp = None
    T_ss = None
    if any(panel in args.panels for panel in ("kspace_tpp", "kspace_tss")):
        if args.kspace_npz:
            kspace_path = Path(args.kspace_npz)
            if not kspace_path.is_absolute():
                kspace_path = PROJECT_ROOT / kspace_path
            kx_norm, ky_norm, T_pp, T_ss = _load_kspace_npz(kspace_path)
        else:
            if tpp_row is None or tss_row is None:
                raise RuntimeError("K-space generation without --kspace_npz requires phi=0 scan data.")
            kx_norm, ky_norm, T_pp, T_ss = _build_kspace(structure, args.lambda_nm, args.device, tpp_row, tss_row)

    case_title = f"{task_label} | {int(round(args.lambda_nm))} nm"
    summary_table_rows = _build_summary_table_rows(
        case,
        task_label,
        float(args.lambda_nm),
        metrics,
        tpp_row,
        tss_row,
        np.asarray(THETAS, dtype=np.float64),
        np.asarray(LAMBDAS, dtype=np.float64),
        tpp_lambda_theta,
        tss_lambda_theta,
        lowpass_alpha_deg,
    )

    return SummaryData(
        structure=structure,
        case=case,
        task_key=task_key,
        task_label=task_label,
        lambda_nm=float(args.lambda_nm),
        lambdas_nm=np.asarray(LAMBDAS, dtype=np.float64),
        thetas_deg=np.asarray(THETAS, dtype=np.float64),
        tpp_lambda_theta=tpp_lambda_theta,
        tss_lambda_theta=tss_lambda_theta,
        tpp_row=tpp_row,
        tss_row=tss_row,
        tpp_ideal_dense=tpp_ideal_dense,
        tss_ideal_dense=tss_ideal_dense,
        theta_dense=theta_dense,
        theta_sample=theta_sample,
        tpp_row_sample=tpp_row_sample,
        tss_row_sample=tss_row_sample,
        kx_norm=kx_norm,
        ky_norm=ky_norm,
        T_pp=T_pp,
        T_ss=T_ss,
        metrics=metrics,
        score_text=score_text,
        summary_table_rows=summary_table_rows,
        case_title=case_title,
        lowpass_alpha_deg=lowpass_alpha_deg,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper-style summary figures for one structure sample.")
    parser.add_argument("--structure_npy", required=True, help="Path to a top-k structure npy file.")
    parser.add_argument("--structure_index", type=int, default=0, help="Index inside the top-k file.")
    parser.add_argument("--lambda_nm", type=float, required=True, help="Target wavelength in nm.")
    parser.add_argument("--device", default="cpu", help="RCWA device, e.g. cpu or cuda.")
    parser.add_argument("--task_key", default=None, help="Optional explicit task key.")
    parser.add_argument("--mode", choices=("main", "board", "both"), default="both")
    parser.add_argument("--panels", default="default", help="Comma-separated panel list, or default/all.")
    parser.add_argument("--kspace_npz", default=None, help="Optional existing kspace_result.npz path.")
    parser.add_argument("--outdir", default=None, help="Optional output directory.")
    parser.add_argument("--no_tss_theta", action="store_true", help="Disable the tss(theta) panel.")
    parser.add_argument("--no_lambda_theta_tss", action="store_true", help="Disable the Tss(lambda,theta) panel.")
    parser.add_argument("--no_kspace_tss", action="store_true", help="Disable the Tss(kx,ky) panel.")
    return parser.parse_args()


def _resolve_task_key(args: argparse.Namespace) -> str | None:
    structure_path = Path(args.structure_npy)
    if not structure_path.is_absolute():
        structure_path = PROJECT_ROOT / structure_path
    task_key = args.task_key or _guess_task_key_from_path(structure_path)
    if task_key is not None:
        return task_key
    case = _match_case(structure_path, args.lambda_nm, None)
    return case.task_key if case is not None else None


def main() -> None:
    args = _parse_args()
    resolved_task_key = _resolve_task_key(args)
    if args.panels is None or args.panels.strip().lower() in {"", "default"}:
        panels = list(_default_panels_for_task(resolved_task_key))
    else:
        panels = list(_panel_list_from_arg(args.panels))
    policy = _task_display_policy(resolved_task_key)
    if not policy.show_tss:
        panels = [panel for panel in panels if panel not in TSS_PANELS]
    if args.no_tss_theta and "theta_curve_tss" in panels:
        panels.remove("theta_curve_tss")
    if args.no_lambda_theta_tss and "lambda_theta_tss" in panels:
        panels.remove("lambda_theta_tss")
    if args.no_kspace_tss and "kspace_tss" in panels:
        panels.remove("kspace_tss")
    args.panels = tuple(panels)
    if args.task_key is None:
        args.task_key = resolved_task_key
    outdir = _build_output_dir(Path(args.structure_npy), args.mode, args.lambda_nm, Path(args.outdir) if args.outdir else None)
    with _paper_style():
        summary = _prepare_summary(args)
        modes = ("main", "board") if args.mode == "both" else (args.mode,)
        saved: list[Path] = []
        for mode in modes:
            saved.extend(_draw_figure(summary, args.panels, mode, outdir))
    for path in saved:
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
