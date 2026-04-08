from __future__ import annotations

import argparse
import math
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


DEFAULT_PANELS = (
    "structure_unit",
    "structure_tile",
    "theta_curve_tpp",
    "theta_curve_tss",
    "lambda_theta_tpp",
    "lambda_theta_tss",
    "kspace_tpp",
    "kspace_tss",
)
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
    structure_tile: np.ndarray
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
    case_title: str


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
        return _second_like_target(theta_dense, 4), zeros
    if task_key == "lowpass":
        return _lowpass_target(theta_dense, 12.0), zeros
    return _second_like_target(theta_dense, 2), zeros


def _scale_ideal_to_40(ideal_row: np.ndarray, theta_dense: np.ndarray, actual_at_40: float) -> np.ndarray:
    ideal = np.asarray(ideal_row, dtype=np.float64).copy()
    if np.allclose(ideal, 0.0):
        return ideal
    ref = max(_value_at(theta_dense, ideal, 40.0), 1e-8)
    return ideal * (float(actual_at_40) / ref)


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


def _panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.01,
        0.99,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.8},
    )


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
    if channel == "tpp":
        color = "#d55e00"
        ideal = summary.tpp_ideal_dense
        sample = summary.tpp_row_sample
        actual = summary.tpp_row
        title = r"$t_{pp}(\theta)$ at $\phi=0^\circ$"
        label = r"$t_{pp}$"
    else:
        color = "#1f77b4"
        ideal = summary.tss_ideal_dense
        sample = summary.tss_row_sample
        actual = summary.tss_row
        title = r"$t_{ss}(\theta)$ at $\phi=0^\circ$"
        label = r"$t_{ss}$"
    ax.plot(summary.theta_dense, ideal, color=color, linestyle="--", lw=1.6, label=f"Ideal {label}")
    ax.plot(
        summary.theta_sample,
        sample,
        marker="*",
        linestyle="none",
        ms=8.5,
        color=color,
        markeredgecolor=color,
        label=f"Sample {label}",
    )
    ax.set_xlim(float(summary.thetas_deg[0]), float(summary.thetas_deg[-1]))
    ax.set_ylim(0.0, 1.05 * max(float(np.max(actual)), float(np.max(ideal)), 1e-6))
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$|t|$")
    ax.set_title(title, pad=8)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, handlelength=2.0)


def _draw_lambda_theta(ax: plt.Axes, data: np.ndarray | None, lambdas_nm: np.ndarray, thetas_deg: np.ndarray, title: str) -> None:
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
    ax.set_xlabel(r"$\theta$ (deg)")
    ax.set_ylabel(r"$\lambda$ (nm)")
    ax.set_title(title, pad=8)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|t|$")


def _draw_kspace(ax: plt.Axes, kx_norm: np.ndarray | None, ky_norm: np.ndarray | None, data: np.ndarray | None, title: str) -> None:
    if kx_norm is None or ky_norm is None or data is None:
        ax.axis("off")
        return
    im = ax.imshow(
        np.asarray(data, dtype=np.float64),
        origin="lower",
        extent=[float(kx_norm[0]), float(kx_norm[-1]), float(ky_norm[0]), float(ky_norm[-1])],
        cmap="hot",
        vmin=0.0,
        vmax=max(float(np.nanmax(data)), 1e-8),
        interpolation="bilinear",
    )
    tt = np.linspace(0, 2 * np.pi, 400)
    ax.plot(NA * np.cos(tt), NA * np.sin(tt), color="white", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.set_xlabel(r"$k_x/k_0$")
    ax.set_ylabel(r"$k_y/k_0$")
    ax.set_title(title, pad=8)
    ax.set_aspect("equal")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$|t|$")


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
    if panel == "structure_unit":
        _draw_structure(ax, summary.structure, "Unit cell")
    elif panel == "structure_tile":
        _draw_structure(ax, summary.structure_tile, "10x10 periodic tiling")
    elif panel == "theta_curve_tpp":
        _draw_theta_curve(ax, summary, "tpp")
    elif panel == "theta_curve_tss":
        _draw_theta_curve(ax, summary, "tss")
    elif panel == "lambda_theta_tpp":
        _draw_lambda_theta(ax, summary.tpp_lambda_theta, summary.lambdas_nm, summary.thetas_deg, r"$T_{pp}(\lambda,\theta)$")
    elif panel == "lambda_theta_tss":
        _draw_lambda_theta(ax, summary.tss_lambda_theta, summary.lambdas_nm, summary.thetas_deg, r"$T_{ss}(\lambda,\theta)$")
    elif panel == "kspace_tpp":
        _draw_kspace(ax, summary.kx_norm, summary.ky_norm, summary.T_pp, r"$T_{pp}(k_x,k_y)$")
    elif panel == "kspace_tss":
        _draw_kspace(ax, summary.kx_norm, summary.ky_norm, summary.T_ss, r"$T_{ss}(k_x,k_y)$")
    elif panel == "score_box":
        _draw_score_box(ax, summary.score_text)
    elif panel == "cross_sections":
        _draw_cross_sections(ax, summary)
    elif panel == "structure_hist":
        _draw_structure_hist(ax, summary.structure)
    else:
        raise ValueError(f"Unsupported panel: {panel}")


def _main_positions(panels: tuple[str, ...]) -> dict[str, tuple[int, int, int, int]]:
    default = DEFAULT_PANELS
    if panels == default:
        return {
            "structure_unit": (0, 1, 0, 3),
            "structure_tile": (0, 1, 3, 6),
            "theta_curve_tpp": (0, 1, 6, 9),
            "theta_curve_tss": (0, 1, 9, 12),
            "lambda_theta_tpp": (1, 2, 0, 6),
            "lambda_theta_tss": (1, 2, 6, 12),
            "kspace_tpp": (2, 3, 0, 6),
            "kspace_tss": (2, 3, 6, 12),
        }
    grouped = [list(panels[i : i + 3]) for i in range(0, len(panels), 3)]
    mapping: dict[str, tuple[int, int, int, int]] = {}
    for r, row in enumerate(grouped):
        spans = 12 // len(row)
        for i, panel in enumerate(row):
            c0 = i * spans
            c1 = 12 if i == len(row) - 1 else (i + 1) * spans
            mapping[panel] = (r, r + 1, c0, c1)
    return mapping


def _draw_figure(summary: SummaryData, panels: tuple[str, ...], mode: str, outdir: Path) -> list[Path]:
    paths: list[Path] = []
    if mode == "main":
        if panels == DEFAULT_PANELS:
            layout = _main_positions(panels)
            n_rows = max(pos[1] for pos in layout.values())
            fig = plt.figure(figsize=(18.0, 10.0 if n_rows >= 3 else 7.4), constrained_layout=True)
            gs = fig.add_gridspec(
                n_rows,
                12,
                height_ratios=[1.0, 1.05, 1.05][:n_rows],
                width_ratios=[1.0] * 12,
            )
            axes = {}
            for panel in panels:
                r0, r1, c0, c1 = layout[panel]
                axes[panel] = fig.add_subplot(gs[r0:r1, c0:c1])
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

    for idx, panel in enumerate(panels):
        ax = axes[panel]
        _panel_dispatch(ax, panel, summary)
        _panel_label(ax, f"({string.ascii_lowercase[idx]})")

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
        _panel_label(ax, "(a)")
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
    structure_tile = np.tile(structure, (10, 10))

    task_key = args.task_key or _guess_task_key_from_path(structure_path)
    case = _match_case(structure_path, args.lambda_nm, task_key)
    if task_key is None and case is not None:
        task_key = case.task_key
    task_label = _task_label_from_key(task_key)

    theta_dense = np.linspace(float(THETAS[0]), float(THETAS[-1]), 401, dtype=np.float64)
    theta_sample = _choose_theta_sample(np.asarray(THETAS, dtype=np.float64), n_points=17)

    tpp_lambda_theta = None
    tss_lambda_theta = None
    if any(panel in args.panels for panel in ("theta_curve_tpp", "theta_curve_tss", "lambda_theta_tpp", "lambda_theta_tss", "score_box", "cross_sections")):
        tpp_lambda_theta, tss_lambda_theta = _scan_lambda_theta(structure, args.device)

    tpp_row = None
    tss_row = None
    tpp_ideal_dense = None
    tss_ideal_dense = None
    tpp_row_sample = None
    tss_row_sample = None
    metrics = None
    score_text = None
    if tpp_lambda_theta is not None and tss_lambda_theta is not None:
        lam_idx = int(np.argmin(np.abs(np.asarray(LAMBDAS, dtype=np.float64) - float(args.lambda_nm))))
        tpp_row = np.asarray(tpp_lambda_theta[lam_idx], dtype=np.float64)
        tss_row = np.asarray(tss_lambda_theta[lam_idx], dtype=np.float64)
        ideal_tpp_raw, ideal_tss_raw = _ideal_rows(task_key, theta_dense)
        tpp_ideal_dense = _scale_ideal_to_40(ideal_tpp_raw, theta_dense, _value_at(np.asarray(THETAS, dtype=np.float64), tpp_row, 40.0))
        tss_ideal_dense = _scale_ideal_to_40(ideal_tss_raw, theta_dense, _value_at(np.asarray(THETAS, dtype=np.float64), tss_row, 40.0))
        tpp_row_sample = np.interp(theta_sample, np.asarray(THETAS, dtype=np.float64), tpp_row)
        tss_row_sample = np.interp(theta_sample, np.asarray(THETAS, dtype=np.float64), tss_row)
        metrics, score_text = _score_summary(case, np.asarray(LAMBDAS, dtype=np.float64), np.asarray(THETAS, dtype=np.float64), tpp_lambda_theta, tss_lambda_theta, args.lambda_nm)

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

    case_title_parts = [task_label]
    if case is not None:
        case_title_parts.append(case.case_label)
    case_title_parts.append(f"{int(round(args.lambda_nm))} nm")
    case_title = " | ".join(case_title_parts)

    return SummaryData(
        structure=structure,
        structure_tile=structure_tile,
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
        case_title=case_title,
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.panels = _panel_list_from_arg(args.panels)
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
