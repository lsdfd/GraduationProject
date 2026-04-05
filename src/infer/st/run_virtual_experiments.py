from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from infer.st.build_inputs import (  # noqa: E402
    build_breathing_segment,
    build_piecewise_motion,
    build_switching_segments,
    build_switching_segments_scaled,
    build_targeted_pulse,
    build_uniform_motion,
)
from infer.st.common import (  # noqa: E402
    AxisConfig,
    DeviceConfig,
    build_axes,
    build_device_otf_on_fft_grid,
    envelope_amplitude,
    envelope_intensity,
    build_ideal_otf,
    build_ideal_otf_on_device_grid,
    envelope_carrier_metadata,
    fft2_xt,
    ideal_gain_from_device,
    input_spectral_overlap_metrics,
    lambda0_to_period_s,
    load_device_otf,
    matched_compute_axes_from_display,
    remove_dc_component,
    response_metrics,
    reconstruct_real_field,
    signal_region_masks,
    spectral_debug_metrics,
    run_filter,
    spatial_temporal_frequency_axes,
    zero_pad_signal_xt,
)
from infer.st.plot_prl_style import (  # noqa: E402
    plot_fig4_compare,
    plot_fig3_triptych,
    plot_fig4_frequency_and_velocity,
    plot_fig4_main,
    plot_input_spectral_support,
    plot_scale_scan_grid,
    plot_transfer_compare,
)


def save_arrays(base: Path, **arrays: np.ndarray) -> None:
    for name, arr in arrays.items():
        np.save(base / f"{name}.npy", np.asarray(arr))


def velocity_in_lambda_per_t1e4(v_m_s: float, lambda0_nm: float) -> float:
    lambda0_m = float(lambda0_nm) * 1e-9
    t_unit = 1.0e4 * lambda0_to_period_s(lambda0_nm)
    return v_m_s * t_unit / lambda0_m


def fig4_axis_config_from_true_v0(v0_lambda_per_t1e4: float) -> AxisConfig:
    t_min = 0.0
    t_max = 3.0
    x0 = -150.0
    width = 55.0
    vmax_factor = 2.0
    x_max = x0 + vmax_factor * float(v0_lambda_per_t1e4) * t_max + 4.0 * width
    x_min = x0 - 3.0 * width
    return AxisConfig(float(x_min), float(x_max), t_min, t_max, 4096, 256)


def run_case(
    case_name: str,
    signal_xt: np.ndarray,
    filter_signal_xt: np.ndarray | None,
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    x_m: np.ndarray,
    t_s: np.ndarray,
    device_otf_info: dict,
    save_dir: Path,
    pad_factor: float,
) -> dict:
    signal_for_filter = np.asarray(signal_xt if filter_signal_xt is None else filter_signal_xt, dtype=np.float64)
    signal_pad, x_pad, t_pad, crop, pad_info = zero_pad_signal_xt(signal_for_filter, x_m, t_s, pad_factor=pad_factor)
    kx_fft, omega_fft = spatial_temporal_frequency_axes(x_pad, t_pad)
    device_otf_fft, support_mask = build_device_otf_on_fft_grid(kx_fft, omega_fft, device_otf_info)
    kx_mesh, omega_mesh = np.meshgrid(kx_fft, omega_fft)
    ideal_gain = ideal_gain_from_device(
        device_otf_fft,
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        support_mask=support_mask,
    )
    ideal_otf_fft = build_ideal_otf(
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        gain=ideal_gain,
        support_mask=support_mask,
    )

    device_filtered_kw, device_pad = run_filter(signal_pad, device_otf_fft)
    ideal_filtered_kw, ideal_pad = run_filter(signal_pad, ideal_otf_fft)
    device_xt = device_pad[crop]
    ideal_xt = ideal_pad[crop]
    device_intensity = envelope_intensity(device_xt)
    ideal_intensity = envelope_intensity(ideal_xt)
    device_field = reconstruct_real_field(device_xt, t_s, float(device_otf_info["omega0"]))
    ideal_field = reconstruct_real_field(ideal_xt, t_s, float(device_otf_info["omega0"]))
    metrics = response_metrics(signal_xt, device_xt, ideal_xt)
    edge_mask, interior_mask, background_mask = signal_region_masks(signal_xt)
    input_spec_kw = fft2_xt(signal_pad)
    device_spec_metrics, device_spec_arrays = spectral_debug_metrics(
        input_spec_kw,
        device_otf_fft,
        kx_fft,
        omega_fft,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        float(device_otf_info["na_s"]),
        float(device_otf_info["na_t"]),
    )
    ideal_spec_metrics, ideal_spec_arrays = spectral_debug_metrics(
        input_spec_kw,
        ideal_otf_fft,
        kx_fft,
        omega_fft,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        float(device_otf_info["na_s"]),
        float(device_otf_info["na_t"]),
    )
    overlap_metrics, overlap_arrays = input_spectral_overlap_metrics(
        input_spec_kw,
        kx_fft,
        omega_fft,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        support_mask,
    )

    case_dir = save_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    save_arrays(
        case_dir,
        input_xt=signal_xt,
        filter_input_xt=signal_for_filter,
        device_xt=device_xt,
        ideal_xt=ideal_xt,
        device_intensity_xt=device_intensity,
        ideal_intensity_xt=ideal_intensity,
        device_field_xt=device_field,
        ideal_field_xt=ideal_field,
        device_otf_fft=device_otf_fft,
        ideal_otf_fft=ideal_otf_fft,
        input_spec_abs_padded=device_spec_arrays["spec_abs"],
        input_spec_energy_padded=device_spec_arrays["spec_energy"],
        input_support_mask_padded=overlap_arrays["support_mask"],
        device_weighted_abs_padded=device_spec_arrays["weighted_abs"],
        device_weighted_energy_padded=device_spec_arrays["weighted_energy"],
        ideal_weighted_abs_padded=ideal_spec_arrays["weighted_abs"],
        ideal_weighted_energy_padded=ideal_spec_arrays["weighted_energy"],
        kappa_grid_padded=device_spec_arrays["kappa_grid"],
        nu_grid_padded=device_spec_arrays["nu_grid"],
        spectral_center_mask_padded=device_spec_arrays["center_mask"],
        spectral_edge_mask_padded=device_spec_arrays["edge_mask"],
        device_filtered_kw_abs_padded=np.abs(device_filtered_kw),
        ideal_filtered_kw_abs_padded=np.abs(ideal_filtered_kw),
        x_lambda=x_lambda,
        t_t1e4=t_t1e4,
        kx_fft=kx_fft,
        omega_fft=omega_fft,
        kappa_grid_padded_input=overlap_arrays["kappa_grid"],
        nu_grid_padded_input=overlap_arrays["nu_grid"],
        input_xt_padded=signal_pad,
        device_xt_padded=device_pad,
        ideal_xt_padded=ideal_pad,
        edge_mask=edge_mask.astype(np.uint8),
        interior_mask=interior_mask.astype(np.uint8),
        background_mask=background_mask.astype(np.uint8),
    )
    plot_transfer_compare(
        case_dir / "transfer_compare.png",
        np.asarray(device_otf_info["display_otf"], dtype=np.float64),
        build_ideal_otf_on_device_grid(device_otf_info),
        np.asarray(device_otf_info["thetas_deg"], dtype=np.float64),
        np.asarray(device_otf_info["display_lambda_nm"], dtype=np.float64),
        f"{case_name} transfer functions",
    )
    plot_input_spectral_support(
        case_dir / "input_spectrum_support.png",
        overlap_arrays["input_energy"],
        overlap_arrays["support_mask"],
        overlap_arrays["kappa_grid"][0],
        overlap_arrays["nu_grid"][:, 0] * 1e3,
        f"{case_name} envelope spectrum vs device support @ lambda0={float(device_otf_info['lambda0_nm']):.0f}nm",
    )
    debug_summary = {
        "case": case_name,
        "carrier_envelope_model": envelope_carrier_metadata(float(device_otf_info["lambda0_nm"])),
        "input_overlap_metrics": overlap_metrics,
        "device_spectral_metrics": device_spec_metrics,
        "ideal_spectral_metrics": ideal_spec_metrics,
    }
    with (case_dir / "spectral_debug.json").open("w", encoding="utf-8") as f:
        json.dump(debug_summary, f, ensure_ascii=False, indent=2)
    return {
        "case": case_name,
        "ideal_gain": float(ideal_gain),
        "filter_input_mean": float(np.mean(signal_for_filter)),
        "filter_input_abs_mean": float(np.mean(np.abs(signal_for_filter))),
        "device_peak_envelope": float(np.max(envelope_amplitude(device_xt))),
        "ideal_peak_envelope": float(np.max(envelope_amplitude(ideal_xt))),
        **{f"input_{k}": v for k, v in overlap_metrics.items()},
        **pad_info,
        **metrics,
        **{f"device_{k}": v for k, v in device_spec_metrics.items()},
    }


def run_scale_scan(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    x_m: np.ndarray,
    t_s: np.ndarray,
    device_otf_info: dict,
    save_dir: Path,
    pad_factor: float,
) -> dict:
    width_scales = [1.0, 0.6, 0.35]
    time_scales = [1.0, 0.6, 0.35]
    row_labels = [f"space x{scale:.2f}" for scale in width_scales]
    col_labels = [f"time x{scale:.2f}" for scale in time_scales]

    pad_signal, x_pad, t_pad, crop, pad_info = zero_pad_signal_xt(np.zeros((len(t_t1e4), len(x_lambda)), dtype=np.float64), x_m, t_s, pad_factor=pad_factor)
    kx_fft, omega_fft = spatial_temporal_frequency_axes(x_pad, t_pad)
    device_otf_fft, support_mask = build_device_otf_on_fft_grid(kx_fft, omega_fft, device_otf_info)
    kx_mesh, omega_mesh = np.meshgrid(kx_fft, omega_fft)
    ideal_gain = ideal_gain_from_device(
        device_otf_fft,
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        support_mask=support_mask,
    )
    ideal_otf_fft = build_ideal_otf(
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        gain=ideal_gain,
        support_mask=support_mask,
    )

    input_maps = np.zeros((len(width_scales), len(time_scales), len(t_t1e4), len(x_lambda)), dtype=np.float64)
    device_maps = np.zeros_like(input_maps)
    ideal_maps = np.zeros_like(input_maps)
    metrics: list[dict[str, float | str]] = []

    for r, width_scale in enumerate(width_scales):
        for c, time_scale in enumerate(time_scales):
            sig = build_breathing_segment(x_lambda, t_t1e4, width_scale=width_scale, time_scale=time_scale)
            signal_pad, _, _, crop_local, _ = zero_pad_signal_xt(sig.signal, x_m, t_s, pad_factor=pad_factor)
            _, device_pad = run_filter(signal_pad, device_otf_fft)
            _, ideal_pad = run_filter(signal_pad, ideal_otf_fft)
            device_xt = device_pad[crop_local]
            ideal_xt = ideal_pad[crop_local]
            input_maps[r, c] = sig.signal
            device_maps[r, c] = envelope_amplitude(device_xt)
            ideal_maps[r, c] = envelope_amplitude(ideal_xt)
            item = response_metrics(sig.signal, device_xt, ideal_xt)
            item.update(
                {
                    "width_scale": float(width_scale),
                    "time_scale": float(time_scale),
                }
            )
            metrics.append(item)

    scan_dir = save_dir / "scale_scan"
    scan_dir.mkdir(parents=True, exist_ok=True)
    save_arrays(
        scan_dir,
        input_maps=input_maps,
        device_maps=device_maps,
        ideal_maps=ideal_maps,
        x_lambda=x_lambda,
        t_t1e4=t_t1e4,
        width_scales=np.asarray(width_scales, dtype=np.float64),
        time_scales=np.asarray(time_scales, dtype=np.float64),
    )
    plot_scale_scan_grid(
        scan_dir / "input_grid.png",
        input_maps,
        x_lambda,
        t_t1e4,
        row_labels,
        col_labels,
        "Scale Scan Inputs",
        panel_prefix="(a",
    )
    plot_scale_scan_grid(
        scan_dir / "device_grid.png",
        device_maps,
        x_lambda,
        t_t1e4,
        row_labels,
        col_labels,
        "Scale Scan Device Outputs",
        panel_prefix="(b",
    )
    plot_scale_scan_grid(
        scan_dir / "ideal_grid.png",
        ideal_maps,
        x_lambda,
        t_t1e4,
        row_labels,
        col_labels,
        "Scale Scan Ideal Outputs",
        panel_prefix="(c",
    )
    with (scan_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return {"case": "scale_scan", "num_settings": len(metrics), "ideal_gain": float(ideal_gain), **pad_info}


def run_targeted_spectral_scan(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    x_m: np.ndarray,
    t_s: np.ndarray,
    device_otf_info: dict,
    save_dir: Path,
    pad_factor: float,
    dc_removed: bool = False,
) -> dict:
    width_values = [120.0, 80.0, 40.0, 20.0, 10.0]
    duration_values = [0.45, 0.22, 0.10, 0.05]
    row_labels = [f"w={w:.0f}" for w in width_values]
    col_labels = [f"dt={d:.2f}" for d in duration_values]

    zero_template, x_pad, t_pad, _, pad_info = zero_pad_signal_xt(np.zeros((len(t_t1e4), len(x_lambda)), dtype=np.float64), x_m, t_s, pad_factor=pad_factor)
    kx_fft, omega_fft = spatial_temporal_frequency_axes(x_pad, t_pad)
    device_otf_fft, support_mask = build_device_otf_on_fft_grid(kx_fft, omega_fft, device_otf_info)
    kx_mesh, omega_mesh = np.meshgrid(kx_fft, omega_fft)
    ideal_gain = ideal_gain_from_device(
        device_otf_fft,
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        support_mask=support_mask,
    )
    ideal_otf_fft = build_ideal_otf(
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        gain=ideal_gain,
        support_mask=support_mask,
    )

    input_maps = np.zeros((len(width_values), len(duration_values), len(t_t1e4), len(x_lambda)), dtype=np.float64)
    device_maps = np.zeros_like(input_maps)
    ideal_maps = np.zeros_like(input_maps)
    metrics: list[dict[str, float]] = []

    for r, width in enumerate(width_values):
        for c, duration in enumerate(duration_values):
            sig = build_targeted_pulse(x_lambda, t_t1e4, width_lambda=width, duration_t1e4=duration)
            signal_for_filter = remove_dc_component(sig.signal) if dc_removed else sig.signal
            signal_pad, _, _, crop_local, _ = zero_pad_signal_xt(signal_for_filter, x_m, t_s, pad_factor=pad_factor)
            input_spec_kw = fft2_xt(signal_pad)
            _, device_pad = run_filter(signal_pad, device_otf_fft)
            _, ideal_pad = run_filter(signal_pad, ideal_otf_fft)
            device_xt = device_pad[crop_local]
            ideal_xt = ideal_pad[crop_local]
            input_maps[r, c] = sig.signal
            device_maps[r, c] = envelope_amplitude(device_xt)
            ideal_maps[r, c] = envelope_amplitude(ideal_xt)
            response_item = response_metrics(sig.signal, device_xt, ideal_xt)
            spec_item, _ = spectral_debug_metrics(
                input_spec_kw,
                device_otf_fft,
                kx_fft,
                omega_fft,
                float(device_otf_info["k0"]),
                float(device_otf_info["omega0"]),
                float(device_otf_info["na_s"]),
                float(device_otf_info["na_t"]),
            )
            response_item.update(spec_item)
            response_item.update({
                "width_lambda": float(width),
                "duration_t1e4": float(duration),
                "dc_removed": bool(dc_removed),
                "filter_input_mean": float(np.mean(signal_for_filter)),
                "filter_input_abs_mean": float(np.mean(np.abs(signal_for_filter))),
            })
            metrics.append(response_item)

    metrics_sorted = sorted(metrics, key=lambda item: (
        item.get("weighted_energy_edge_frac", 0.0),
        item.get("device_edge_to_interior", 0.0),
        -item.get("input_energy_center_frac", 1.0),
    ), reverse=True)

    scan_dir = save_dir / ("targeted_spectral_scan_dc_removed" if dc_removed else "targeted_spectral_scan")
    scan_dir.mkdir(parents=True, exist_ok=True)
    save_arrays(
        scan_dir,
        input_maps=input_maps,
        device_maps=device_maps,
        ideal_maps=ideal_maps,
        processed_input_maps=np.asarray([
            [
                remove_dc_component(build_targeted_pulse(x_lambda, t_t1e4, width_lambda=width, duration_t1e4=duration).signal)
                if dc_removed
                else build_targeted_pulse(x_lambda, t_t1e4, width_lambda=width, duration_t1e4=duration).signal
                for duration in duration_values
            ]
            for width in width_values
        ], dtype=np.float64),
        x_lambda=x_lambda,
        t_t1e4=t_t1e4,
        width_values=np.asarray(width_values, dtype=np.float64),
        duration_values=np.asarray(duration_values, dtype=np.float64),
    )
    plot_scale_scan_grid(
        scan_dir / "input_grid.png",
        input_maps,
        x_lambda,
        t_t1e4,
        row_labels,
        col_labels,
        "Targeted Spectral Inputs" + (" (DC Removed for Filtering)" if dc_removed else ""),
        panel_prefix="(a",
    )
    plot_scale_scan_grid(
        scan_dir / "device_grid.png",
        device_maps,
        x_lambda,
        t_t1e4,
        row_labels,
        col_labels,
        "Targeted Spectral Device Outputs" + (" (DC Removed)" if dc_removed else ""),
        panel_prefix="(b",
    )
    plot_scale_scan_grid(
        scan_dir / "ideal_grid.png",
        ideal_maps,
        x_lambda,
        t_t1e4,
        row_labels,
        col_labels,
        "Targeted Spectral Ideal Outputs" + (" (DC Removed)" if dc_removed else ""),
        panel_prefix="(c",
    )
    with (scan_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with (scan_dir / "summary_sorted.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_sorted, f, ensure_ascii=False, indent=2)
    best = metrics_sorted[0] if metrics_sorted else {}
    return {
        "case": "targeted_spectral_scan_dc_removed" if dc_removed else "targeted_spectral_scan",
        "num_settings": len(metrics),
        "ideal_gain": float(ideal_gain),
        "best": best,
        **pad_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRL-style virtual space-time experiments for the ST metasurface case.")
    parser.add_argument("--train_npz", default=str(ROOT / "data" / "train_data_20000.npz"))
    parser.add_argument("--sample_idx", type=int, default=18959)
    parser.add_argument("--channel", choices=["tpp", "tss"], default="tss")
    parser.add_argument("--lambda0_nm", type=float, default=950.0)
    parser.add_argument("--lambda_step_nm", type=float, default=10.0)
    parser.add_argument("--pad_factor", type=float, default=2.0)
    parser.add_argument("--save_dir", default=str(ROOT / "samples" / "st_virtual"))
    parser.add_argument("--fig3_match_width_scale", type=float, default=0.05)
    parser.add_argument("--fig3_match_time_scale", type=float, default=0.08)
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
    target_kappa_max = float(np.max(np.abs(np.asarray(device_otf_info["kx_axis"], dtype=np.float64))) / float(device_otf_info["k0"]))
    target_nu_max = float(np.max(np.abs(np.asarray(device_otf_info["omega_axis"], dtype=np.float64))) / float(device_otf_info["omega0"]))

    run_root = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    scaling_info: dict[str, dict[str, float]] = {}

    fig3a_axes = AxisConfig(-160.0, 160.0, 0.0, 2.0, 256, 256)
    x_lambda, t_t1e4, _, _ = build_axes(fig3a_axes, args.lambda0_nm)
    x_m, t_s, scaling_info["fig3_switch"] = matched_compute_axes_from_display(x_lambda, t_t1e4, args.lambda0_nm, target_kappa_max, target_nu_max)
    sig = build_switching_segments(x_lambda, t_t1e4)
    summary = run_case(sig.name, sig.signal, None, x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor)
    plot_fig3_triptych(run_root / sig.name / "fig3_prl_style.png", sig.signal, np.load(run_root / sig.name / "device_xt.npy"), np.load(run_root / sig.name / "ideal_xt.npy"), x_lambda, t_t1e4, ("Input", "Device", "Ideal"))
    summaries.append(summary)

    sig_matched = build_switching_segments_scaled(
        x_lambda,
        t_t1e4,
        width_scale=args.fig3_match_width_scale,
        time_scale=args.fig3_match_time_scale,
    )
    matched_name = f"{sig_matched.name}_matched"
    summary_matched = run_case(matched_name, sig_matched.signal, None, x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor)
    summary_matched.update(
        {
            "width_scale": float(args.fig3_match_width_scale),
            "time_scale": float(args.fig3_match_time_scale),
        }
    )
    plot_fig3_triptych(
        run_root / matched_name / "fig3_prl_style.png",
        sig_matched.signal,
        np.load(run_root / matched_name / "device_xt.npy"),
        np.load(run_root / matched_name / "ideal_xt.npy"),
        x_lambda,
        t_t1e4,
        ("Input", "Device", "Ideal"),
    )
    summaries.append(summary_matched)
    summaries.append(run_targeted_spectral_scan(x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor))
    summaries.append(run_targeted_spectral_scan(x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor, dc_removed=True))

    fig3d_axes = AxisConfig(-500.0, 500.0, 0.0, 7.4, 256, 256)
    x_lambda, t_t1e4, _, _ = build_axes(fig3d_axes, args.lambda0_nm)
    x_m, t_s, scaling_info["fig3_breath"] = matched_compute_axes_from_display(x_lambda, t_t1e4, args.lambda0_nm, target_kappa_max, target_nu_max)
    sig = build_breathing_segment(x_lambda, t_t1e4)
    summary = run_case(sig.name, sig.signal, None, x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor)
    plot_fig3_triptych(run_root / sig.name / "fig3_prl_style.png", sig.signal, np.load(run_root / sig.name / "device_xt.npy"), np.load(run_root / sig.name / "ideal_xt.npy"), x_lambda, t_t1e4, ("Input", "Device", "Ideal"))
    summaries.append(summary)
    summaries.append(run_scale_scan(x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor))

    v0_lambda_per_t1e4 = velocity_in_lambda_per_t1e4(float(device_otf_info["v0_m_per_s"]), args.lambda0_nm)
    fig4_axes = fig4_axis_config_from_true_v0(v0_lambda_per_t1e4)
    x_lambda, t_t1e4, _, _ = build_axes(fig4_axes, args.lambda0_nm)
    x_m, t_s, scaling_info["fig4_motion"] = matched_compute_axes_from_display(x_lambda, t_t1e4, args.lambda0_nm, target_kappa_max, target_nu_max)
    sig = build_piecewise_motion(x_lambda, t_t1e4, v0_lambda_per_t1e4)
    summary = run_case(sig.name, sig.signal, None, x_lambda, t_t1e4, x_m, t_s, device_otf_info, run_root, args.pad_factor)
    device_xt = np.load(run_root / sig.name / "device_xt.npy")
    ideal_xt = np.load(run_root / sig.name / "ideal_xt.npy")
    plot_fig4_main(run_root / sig.name / "fig4_ab.png", sig.signal, device_xt, x_lambda, t_t1e4)
    plot_fig4_compare(run_root / sig.name / "fig4_compare.png", sig.signal, device_xt, ideal_xt, x_lambda, t_t1e4)
    summaries.append(summary)

    vel_factors = np.asarray([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], dtype=np.float64)
    zero_pad_template, x_pad, t_pad, _, _ = zero_pad_signal_xt(np.zeros((len(t_t1e4), len(x_lambda)), dtype=np.float64), x_m, t_s, pad_factor=args.pad_factor)
    kx_fft, omega_fft = spatial_temporal_frequency_axes(x_pad, t_pad)
    device_otf_fft, support_mask = build_device_otf_on_fft_grid(kx_fft, omega_fft, device_otf_info)
    kx_mesh, omega_mesh = np.meshgrid(kx_fft, omega_fft)
    ideal_gain = ideal_gain_from_device(
        device_otf_fft,
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        support_mask=support_mask,
    )
    ideal_otf_fft = build_ideal_otf(
        kx_mesh,
        omega_mesh,
        float(device_otf_info["k0"]),
        float(device_otf_info["omega0"]),
        gain=ideal_gain,
        support_mask=support_mask,
    )
    responses = []
    velocity_metrics = []
    masks = []
    labels = []
    colors = ["#2ca02c", "#ff00ff", "#1f77b4"]
    for idx, factor in enumerate(vel_factors):
        sig_v = build_uniform_motion(x_lambda, t_t1e4, factor * v0_lambda_per_t1e4)
        signal_pad, _, _, crop, _ = zero_pad_signal_xt(sig_v.signal, x_m, t_s, pad_factor=args.pad_factor)
        _, device_pad = run_filter(signal_pad, device_otf_fft)
        _, ideal_pad = run_filter(signal_pad, ideal_otf_fft)
        device_xt = device_pad[crop]
        ideal_xt = ideal_pad[crop]
        metric = response_metrics(sig_v.signal, device_xt, ideal_xt)
        metric.update({"velocity_factor": float(factor)})
        velocity_metrics.append(metric)
        responses.append(float(np.max(envelope_amplitude(device_xt))))
        if factor in (0.5, 1.0, 1.5):
            spec = np.abs(fft2_xt(sig_v.signal))
            mask = spec >= 0.015 * max(float(np.max(spec)), 1e-12)
            masks.append(mask)
            labels.append(f"{factor:.1f}$v_0$")
    responses = np.asarray(responses, dtype=np.float64)
    responses = responses / max(float(np.max(responses)), 1e-12)
    velocities_km_s = vel_factors * float(device_otf_info["v0_m_per_s"]) / 1000.0
    plot_fig4_frequency_and_velocity(
        run_root / "fig4_velocity_scan.png",
        kx_fft / float(device_otf_info["k0"]),
        (omega_fft / float(device_otf_info["omega0"])) * 1e3,
        masks,
        colors[: len(masks)],
        labels,
        velocities_km_s,
        responses,
        float(device_otf_info["v0_m_per_s"]) / 1000.0,
    )
    save_arrays(run_root, velocity_factors=vel_factors, velocities_km_s=velocities_km_s, velocity_response=responses)
    with (run_root / "velocity_scan_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(velocity_metrics, f, ensure_ascii=False, indent=2)

    summary = {
        "device": {
            **asdict(device_cfg),
            "t0_fs": lambda0_to_period_s(args.lambda0_nm) * 1e15,
            "na_s": float(device_otf_info["na_s"]),
            "na_t": float(device_otf_info["na_t"]),
            "v0_km_s": float(device_otf_info["v0_m_per_s"]) / 1000.0,
            "target_kappa_max": target_kappa_max,
            "target_nu_max": target_nu_max,
        },
        "carrier_envelope_model": envelope_carrier_metadata(args.lambda0_nm),
        "scaling_info": scaling_info,
        "cases": summaries,
    }
    with (run_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved_to: {run_root}")


if __name__ == "__main__":
    main()
