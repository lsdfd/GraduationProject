from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from infer.st.common import (  # noqa: E402
    AxisConfig,
    DeviceConfig,
    build_axes,
    build_ideal_otf_on_device_grid,
    device_frequency_axes,
    envelope_carrier_metadata,
    load_device_otf,
    matched_compute_axes_from_display,
)
from infer.st.run_virtual_experiments import run_case  # noqa: E402


PEAK_MATCH_FACTOR = 0.646


def build_line_blink(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    length_lambda: float,
    duration_t1e4: float,
    center_x_lambda: float = -75.0,
    center_t_t1e4: float = 0.855,
) -> np.ndarray:
    x = np.asarray(x_lambda, dtype=np.float64)
    t = np.asarray(t_t1e4, dtype=np.float64)
    spatial = (np.abs(x - float(center_x_lambda)) <= 0.5 * float(length_lambda))[None, :]
    temporal = (np.abs(t - float(center_t_t1e4)) <= 0.5 * float(duration_t1e4))[:, None]
    return (spatial & temporal).astype(np.float64)


def recommend_lengths_from_device(device_otf_info: dict) -> dict[str, float]:
    k0 = float(device_otf_info["k0"])
    omega0 = float(device_otf_info["omega0"])
    kx_axis = np.asarray(device_otf_info["kx_axis"], dtype=np.float64)
    omega_axis = np.asarray(device_otf_info["omega_axis"], dtype=np.float64)
    otf = np.asarray(device_otf_info["otf"], dtype=np.float64)

    kappa_axis = kx_axis / max(k0, 1e-18)
    nu_axis = omega_axis / max(omega0, 1e-18)
    kappa_grid, nu_grid = np.meshgrid(kappa_axis, nu_axis)

    support_kappa_max = float(np.max(np.abs(kappa_axis)))
    support_nu_max = float(np.max(np.abs(nu_axis)))

    ideal_like = (kappa_grid ** 2) * (nu_grid ** 2)
    effective = otf * ideal_like
    positive = (kappa_grid > 0.0) & (nu_grid > 0.0)
    if np.any(positive):
        peak_idx = np.unravel_index(int(np.argmax(np.where(positive, effective, -np.inf))), effective.shape)
    else:
        peak_idx = np.unravel_index(int(np.argmax(effective)), effective.shape)

    kappa_peak = float(np.abs(kappa_grid[peak_idx]))
    nu_peak = float(np.abs(nu_grid[peak_idx]))

    return {
        "support_kappa_max": support_kappa_max,
        "support_nu_max": support_nu_max,
        "effective_kappa_peak": kappa_peak,
        "effective_nu_peak": nu_peak,
        "length_lambda_edge_match": 1.0 / max(support_kappa_max, 1e-12),
        "duration_t1e4_edge_match": 1.0 / max(1.0e4 * support_nu_max, 1e-12),
        "length_lambda_peak_match": PEAK_MATCH_FACTOR / max(kappa_peak, 1e-12),
        "duration_t1e4_peak_match": PEAK_MATCH_FACTOR / max(1.0e4 * nu_peak, 1e-12),
    }


def discretize_extent(target: float, step: float, min_pixels: int = 4) -> float:
    return max(float(target), float(min_pixels) * float(step))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze how a flashing 1D line should be sized to match the ST device OTF support.")
    parser.add_argument("--train_npz", default=str(ROOT / "data" / "train_data_20000.npz"))
    parser.add_argument("--sample_idx", type=int, default=18959)
    parser.add_argument("--channel", choices=["tpp", "tss"], default="tss")
    parser.add_argument("--lambda0_nm", type=float, default=950.0)
    parser.add_argument("--lambda_step_nm", type=float, default=10.0)
    parser.add_argument("--save_dir", default=str(ROOT / "samples" / "st_line_match"))
    parser.add_argument("--pad_factor", type=float, default=2.0)
    args = parser.parse_args()

    device_cfg = DeviceConfig(
        sample_idx=args.sample_idx,
        channel=args.channel,
        lambda0_nm=args.lambda0_nm,
        lambda_min_nm=900.0,
        lambda_max_nm=1050.0,
        lambda_step_nm=args.lambda_step_nm,
    )
    device_otf_info = load_device_otf(args.train_npz, device_cfg, theta_limit_deg=40.0)
    rec = recommend_lengths_from_device(device_otf_info)

    run_root = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    fig3_axes = AxisConfig(-160.0, 160.0, 0.0, 2.0, 256, 256)
    x_lambda, t_t1e4, _, _ = build_axes(fig3_axes, args.lambda0_nm)
    dx_lambda = float(np.mean(np.diff(x_lambda)))
    dt_t1e4 = float(np.mean(np.diff(t_t1e4)))
    x_m, t_s, scaling = matched_compute_axes_from_display(
        x_lambda,
        t_t1e4,
        args.lambda0_nm,
        rec["support_kappa_max"],
        rec["support_nu_max"],
    )
    rec.update(
        {
            "grid_dx_lambda": dx_lambda,
            "grid_dt_t1e4": dt_t1e4,
            "length_lambda_edge_match_discrete": discretize_extent(rec["length_lambda_edge_match"], dx_lambda),
            "duration_t1e4_edge_match_discrete": discretize_extent(rec["duration_t1e4_edge_match"], dt_t1e4),
            "length_lambda_peak_match_discrete": discretize_extent(rec["length_lambda_peak_match"], dx_lambda),
            "duration_t1e4_peak_match_discrete": discretize_extent(rec["duration_t1e4_peak_match"], dt_t1e4),
        }
    )

    cases = [
        {
            "name": "original_single_line",
            "length_lambda": 100.0,
            "duration_t1e4": 0.35,
            "center_x_lambda": -75.0,
            "center_t_t1e4": 0.855,
        },
        {
            "name": "edge_matched_single_line_discrete",
            "length_lambda": rec["length_lambda_edge_match_discrete"],
            "duration_t1e4": rec["duration_t1e4_edge_match_discrete"],
            "center_x_lambda": 0.0,
            "center_t_t1e4": 1.0,
        },
        {
            "name": "peak_matched_single_line_discrete",
            "length_lambda": rec["length_lambda_peak_match_discrete"],
            "duration_t1e4": rec["duration_t1e4_peak_match_discrete"],
            "center_x_lambda": 0.0,
            "center_t_t1e4": 1.0,
        },
    ]

    summaries: list[dict] = []
    for case in cases:
        signal_xt = build_line_blink(
            x_lambda,
            t_t1e4,
            length_lambda=case["length_lambda"],
            duration_t1e4=case["duration_t1e4"],
            center_x_lambda=case["center_x_lambda"],
            center_t_t1e4=case["center_t_t1e4"],
        )
        summary = run_case(
            case["name"],
            signal_xt,
            None,
            x_lambda,
            t_t1e4,
            x_m,
            t_s,
            device_otf_info,
            run_root,
            args.pad_factor,
        )
        summary.update(case)
        summaries.append(summary)

    device_grid_ideal = build_ideal_otf_on_device_grid(device_otf_info)
    kx_axis_dev, omega_axis_dev = device_frequency_axes(
        np.asarray(device_otf_info["display_lambda_nm"], dtype=np.float64),
        np.asarray(device_otf_info["thetas_deg"], dtype=np.float64),
        args.lambda0_nm,
    )
    device_meta = {
        **asdict(device_cfg),
        **rec,
        "carrier_envelope_model": envelope_carrier_metadata(args.lambda0_nm),
        "scaling": scaling,
        "device_grid": {
            "kappa_axis_min": float(np.min(kx_axis_dev / float(device_otf_info["k0"]))),
            "kappa_axis_max": float(np.max(kx_axis_dev / float(device_otf_info["k0"]))),
            "nu_axis_min": float(np.min(omega_axis_dev / float(device_otf_info["omega0"]))),
            "nu_axis_max": float(np.max(omega_axis_dev / float(device_otf_info["omega0"]))),
            "display_otf_peak": float(np.max(np.asarray(device_otf_info["display_otf"], dtype=np.float64))),
            "ideal_otf_peak": float(np.max(np.asarray(device_grid_ideal, dtype=np.float64))),
        },
    }

    with (run_root / "line_match_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"device": device_meta, "cases": summaries}, f, ensure_ascii=False, indent=2)
    print(f"saved_to: {run_root}")


if __name__ == "__main__":
    main()
