from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


C0 = 299_792_458.0


@dataclass(frozen=True)
class AxisConfig:
    x_lambda_min: float
    x_lambda_max: float
    t_t1e4_min: float
    t_t1e4_max: float
    nx: int
    nt: int


@dataclass(frozen=True)
class DeviceConfig:
    sample_idx: int = 18959
    channel: str = "tss"
    lambda0_nm: float = 950.0
    lambda_min_nm: float = 900.0
    lambda_max_nm: float = 1050.0
    lambda_step_nm: float = 10.0


def lambda0_to_period_s(lambda0_nm: float) -> float:
    return float(lambda0_nm) * 1e-9 / C0


def t1e4_to_seconds(t_t1e4: np.ndarray, lambda0_nm: float) -> np.ndarray:
    t0 = lambda0_to_period_s(lambda0_nm)
    return np.asarray(t_t1e4, dtype=np.float64) * 1.0e4 * t0


def lambda_units_to_meters(x_lambda: np.ndarray, lambda0_nm: float) -> np.ndarray:
    return np.asarray(x_lambda, dtype=np.float64) * float(lambda0_nm) * 1e-9


def build_axes(cfg: AxisConfig, lambda0_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_lambda = np.linspace(cfg.x_lambda_min, cfg.x_lambda_max, cfg.nx, dtype=np.float64)
    t_t1e4 = np.linspace(cfg.t_t1e4_min, cfg.t_t1e4_max, cfg.nt, dtype=np.float64)
    x_m = lambda_units_to_meters(x_lambda, lambda0_nm)
    t_s = t1e4_to_seconds(t_t1e4, lambda0_nm)
    return x_lambda, t_t1e4, x_m, t_s


def matched_compute_axes_from_display(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    lambda0_nm: float,
    target_kappa_max: float,
    target_nu_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_lambda = np.asarray(x_lambda, dtype=np.float64)
    t_t1e4 = np.asarray(t_t1e4, dtype=np.float64)
    x_nominal = lambda_units_to_meters(x_lambda, lambda0_nm)
    t_nominal = t1e4_to_seconds(t_t1e4, lambda0_nm)
    kx_nominal, omega_nominal = spatial_temporal_frequency_axes(x_nominal, t_nominal)

    lambda0_m = float(lambda0_nm) * 1e-9
    t0_s = lambda0_to_period_s(lambda0_nm)
    k0 = 2.0 * np.pi / lambda0_m
    omega0 = 2.0 * np.pi / t0_s

    current_kappa_max = float(np.max(np.abs(kx_nominal)) / k0)
    current_nu_max = float(np.max(np.abs(omega_nominal)) / omega0)

    scale_x = current_kappa_max / max(float(target_kappa_max), 1e-12)
    scale_t = current_nu_max / max(float(target_nu_max), 1e-12)

    x_matched = x_nominal * scale_x
    t_matched = t_nominal * scale_t
    return x_matched, t_matched, {
        "current_kappa_max": current_kappa_max,
        "target_kappa_max": float(target_kappa_max),
        "x_scale_factor": float(scale_x),
        "current_nu_max": current_nu_max,
        "target_nu_max": float(target_nu_max),
        "t_scale_factor": float(scale_t),
    }


def spatial_temporal_frequency_axes(x_m: np.ndarray, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = float(np.mean(np.diff(x_m)))
    dt = float(np.mean(np.diff(t_s)))
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(x_m), d=dx))
    omega = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(len(t_s), d=dt))
    return kx.astype(np.float64), omega.astype(np.float64)


def padded_axis(axis: np.ndarray, pad_each: int) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if pad_each <= 0:
        return axis.copy()
    step = float(np.mean(np.diff(axis)))
    start = float(axis[0]) - pad_each * step
    return start + step * np.arange(len(axis) + 2 * pad_each, dtype=np.float64)


def zero_pad_signal_xt(
    signal_xt: np.ndarray,
    x_m: np.ndarray,
    t_s: np.ndarray,
    pad_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[slice, slice], dict[str, int | float]]:
    signal = np.asarray(signal_xt, dtype=np.float64)
    nt, nx = signal.shape
    factor = max(float(pad_factor), 1.0)
    pad_x = max(int(np.ceil((factor - 1.0) * nx / 2.0)), 0)
    pad_t = max(int(np.ceil((factor - 1.0) * nt / 2.0)), 0)
    padded = np.pad(signal, ((pad_t, pad_t), (pad_x, pad_x)), mode="constant", constant_values=0.0)
    x_pad = padded_axis(x_m, pad_x)
    t_pad = padded_axis(t_s, pad_t)
    crop = (slice(pad_t, pad_t + nt), slice(pad_x, pad_x + nx))
    return padded, x_pad, t_pad, crop, {
        "pad_factor": factor,
        "pad_x": pad_x,
        "pad_t": pad_t,
        "padded_nx": int(padded.shape[1]),
        "padded_nt": int(padded.shape[0]),
    }


def omega_from_lambda_nm(lambda_nm: np.ndarray) -> np.ndarray:
    lam_m = np.asarray(lambda_nm, dtype=np.float64) * 1e-9
    return 2.0 * np.pi * C0 / np.maximum(lam_m, 1e-18)


def device_frequency_axes(lambda_nm: np.ndarray, thetas_deg: np.ndarray, lambda0_nm: float) -> tuple[np.ndarray, np.ndarray]:
    lambda0_m = float(lambda0_nm) * 1e-9
    k0 = 2.0 * np.pi / lambda0_m
    kx = k0 * np.sin(np.deg2rad(np.asarray(thetas_deg, dtype=np.float64)))
    omega0 = 2.0 * np.pi * C0 / lambda0_m
    omega = omega_from_lambda_nm(np.asarray(lambda_nm, dtype=np.float64))
    return kx.astype(np.float64), (omega - omega0).astype(np.float64)


def interp1d_with_fill(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    x_src = np.asarray(x_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    x_dst = np.asarray(x_dst, dtype=np.float64)
    out = np.interp(x_dst, x_src, y_src, left=np.nan, right=np.nan)
    out = np.where(np.isfinite(out), out, fill_value)
    return out.astype(np.float64)


def interp2_regular(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    values: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    yq = np.asarray(yq, dtype=np.float64)

    if values.shape != (len(y_axis), len(x_axis)):
        raise ValueError(f"values shape should be {(len(y_axis), len(x_axis))}, got {values.shape}")

    out = np.full_like(xq, fill_value, dtype=np.float64)
    valid = (
        (xq >= x_axis[0]) & (xq <= x_axis[-1]) &
        (yq >= y_axis[0]) & (yq <= y_axis[-1])
    )
    if not np.any(valid):
        return out

    xv = xq[valid]
    yv = yq[valid]
    ix = np.clip(np.searchsorted(x_axis, xv, side="right") - 1, 0, len(x_axis) - 2)
    iy = np.clip(np.searchsorted(y_axis, yv, side="right") - 1, 0, len(y_axis) - 2)

    x0 = x_axis[ix]
    x1 = x_axis[ix + 1]
    y0 = y_axis[iy]
    y1 = y_axis[iy + 1]

    wx = np.where(x1 > x0, (xv - x0) / (x1 - x0), 0.0)
    wy = np.where(y1 > y0, (yv - y0) / (y1 - y0), 0.0)

    v00 = values[iy, ix]
    v01 = values[iy, ix + 1]
    v10 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]

    interp = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )
    out[valid] = interp
    return out


def normalize_to_unit(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(arr)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros_like(arr, dtype=np.float64)
    if not np.any(mask):
        return out
    scale = float(np.max(np.abs(arr[mask])))
    if scale <= 1e-12:
        return out
    out[mask] = arr[mask] / scale
    return out


def load_device_otf(
    train_npz: str | Path,
    cfg: DeviceConfig,
    theta_limit_deg: float = 40.0,
) -> dict[str, np.ndarray | float]:
    data = np.load(train_npz)
    lambdas_raw = np.asarray(data["lambdas"], dtype=np.float64)
    thetas_raw = np.asarray(data["thetas"], dtype=np.float64)
    channel_key = "tss_mag" if cfg.channel == "tss" else "tpp_mag"
    spec_raw = np.asarray(data[channel_key][cfg.sample_idx], dtype=np.float64)

    theta_mask = np.abs(thetas_raw) <= float(theta_limit_deg)
    thetas_sel = thetas_raw[theta_mask]
    lambda_mask = (lambdas_raw >= float(cfg.lambda_min_nm)) & (lambdas_raw <= float(cfg.lambda_max_nm))
    lambdas_sel = lambdas_raw[lambda_mask]
    spec_sel = spec_raw[lambda_mask][:, theta_mask]
    # Display the device map in intensity-style form so it matches how we
    # inspect the sample heatmaps, while keeping the filtering OTF in
    # amplitude form below.
    display_otf = (spec_sel ** 2).astype(np.float64)

    lambda_fine = np.arange(cfg.lambda_min_nm, cfg.lambda_max_nm + 0.5 * cfg.lambda_step_nm, cfg.lambda_step_nm, dtype=np.float64)
    spec_fine = np.zeros((len(lambda_fine), len(thetas_sel)), dtype=np.float64)
    for j in range(len(thetas_sel)):
        spec_fine[:, j] = interp1d_with_fill(lambdas_sel, spec_sel[:, j], lambda_fine, fill_value=0.0)

    kx_axis, omega_axis = device_frequency_axes(lambda_fine, thetas_sel, cfg.lambda0_nm)
    order = np.argsort(omega_axis)
    omega_axis = omega_axis[order]
    spec_fine = spec_fine[order]

    lambda0_m = float(cfg.lambda0_nm) * 1e-9
    omega0 = 2.0 * np.pi * C0 / lambda0_m
    k0 = 2.0 * np.pi / lambda0_m
    na_s = float(np.max(np.abs(kx_axis)) / k0)
    na_t = float(np.max(np.abs(omega_axis)) / omega0)
    v0 = C0 * na_t / max(na_s, 1e-12)

    otf = spec_fine.astype(np.float64)
    return {
        "lambda0_nm": float(cfg.lambda0_nm),
        "lambda_nm": lambda_fine.astype(np.float64),
        "thetas_deg": thetas_sel.astype(np.float64),
        "display_lambda_nm": lambdas_sel.astype(np.float64),
        "display_otf": display_otf.astype(np.float64),
        "kx_axis": kx_axis.astype(np.float64),
        "omega_axis": omega_axis.astype(np.float64),
        "otf": otf.astype(np.float64),
        "omega0": float(omega0),
        "k0": float(k0),
        "na_s": float(na_s),
        "na_t": float(na_t),
        "v0_m_per_s": float(v0),
    }


def build_ideal_otf(
    kx_grid: np.ndarray,
    omega_grid: np.ndarray,
    k0: float,
    omega0: float,
    gain: float = 1.0,
    support_mask: np.ndarray | None = None,
) -> np.ndarray:
    kappa = np.asarray(kx_grid, dtype=np.float64) / max(float(k0), 1e-18)
    nu = np.asarray(omega_grid, dtype=np.float64) / max(float(omega0), 1e-18)
    ideal = float(gain) * (kappa ** 2) * (nu ** 2)
    if support_mask is not None:
        ideal = np.where(np.asarray(support_mask, dtype=bool), ideal, 0.0)
    return ideal.astype(np.float64)


def ideal_gain_from_device(
    device_otf: np.ndarray,
    kx_grid: np.ndarray,
    omega_grid: np.ndarray,
    k0: float,
    omega0: float,
    support_mask: np.ndarray | None = None,
) -> float:
    device = np.asarray(device_otf, dtype=np.float64)
    kappa = np.asarray(kx_grid, dtype=np.float64) / max(float(k0), 1e-18)
    nu = np.asarray(omega_grid, dtype=np.float64) / max(float(omega0), 1e-18)
    ideal_base = (kappa ** 2) * (nu ** 2)
    if support_mask is not None:
        mask = np.asarray(support_mask, dtype=bool)
    else:
        mask = np.isfinite(device) & np.isfinite(ideal_base)
    mask &= np.isfinite(device) & np.isfinite(ideal_base)
    if not np.any(mask):
        return 1.0
    dev_max = float(np.max(device[mask]))
    ideal_max = float(np.max(ideal_base[mask]))
    if ideal_max <= 1e-18:
        return 1.0
    return dev_max / ideal_max


def build_ideal_otf_on_device_grid(device_otf: dict[str, np.ndarray | float]) -> np.ndarray:
    thetas = np.asarray(device_otf["thetas_deg"], dtype=np.float64)
    lambdas = np.asarray(device_otf["display_lambda_nm"], dtype=np.float64)
    lambda0_nm = 2.0 * np.pi * C0 / float(device_otf["omega0"]) * 1e9
    kx_axis, omega_axis = device_frequency_axes(lambdas, thetas, lambda0_nm)
    kx_grid = np.broadcast_to(kx_axis[None, :], (len(lambdas), len(kx_axis)))
    omega_grid = np.broadcast_to(omega_axis[:, None], (len(lambdas), len(kx_axis)))
    gain = ideal_gain_from_device(
        np.asarray(device_otf["display_otf"], dtype=np.float64),
        kx_grid,
        omega_grid,
        float(device_otf["k0"]),
        float(device_otf["omega0"]),
    )
    return build_ideal_otf(kx_grid, omega_grid, float(device_otf["k0"]), float(device_otf["omega0"]), gain=gain)


def build_device_otf_on_fft_grid(
    kx_fft: np.ndarray,
    omega_fft: np.ndarray,
    device_otf: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    kx_mesh, omega_mesh = np.meshgrid(kx_fft, omega_fft)
    otf = interp2_regular(
        np.asarray(device_otf["kx_axis"], dtype=np.float64),
        np.asarray(device_otf["omega_axis"], dtype=np.float64),
        np.asarray(device_otf["otf"], dtype=np.float64),
        kx_mesh,
        omega_mesh,
        fill_value=0.0,
    )
    support_mask = (
        (np.abs(kx_mesh) <= float(np.max(np.abs(device_otf["kx_axis"])))) &
        (np.abs(omega_mesh) <= float(np.max(np.abs(device_otf["omega_axis"]))))
    )
    return otf.astype(np.float64), support_mask


def fft2_xt(signal_xt: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.asarray(signal_xt, dtype=np.float64)))


def ifft2_xt(spec_kw: np.ndarray) -> np.ndarray:
    return np.fft.ifft2(np.fft.ifftshift(np.asarray(spec_kw, dtype=np.complex128)))


def run_filter(signal_xt: np.ndarray, otf_kw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spec = fft2_xt(signal_xt)
    filtered = spec * np.asarray(otf_kw, dtype=np.float64)
    response_env = ifft2_xt(filtered)
    return filtered, response_env


def envelope_amplitude(response_env: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(response_env, dtype=np.complex128)).astype(np.float64)


def envelope_intensity(response_env: np.ndarray) -> np.ndarray:
    arr = np.asarray(response_env, dtype=np.complex128)
    return (np.abs(arr) ** 2).astype(np.float64)


def reconstruct_real_field(
    response_env: np.ndarray,
    t_s: np.ndarray,
    omega0: float,
    phase0: float = 0.0,
) -> np.ndarray:
    env = np.asarray(response_env, dtype=np.complex128)
    t = np.asarray(t_s, dtype=np.float64)
    carrier = np.exp(-1j * (float(omega0) * t[:, None] - float(phase0)))
    return np.real(env * carrier).astype(np.float64)


def envelope_carrier_metadata(lambda0_nm: float) -> dict[str, float | str]:
    lambda0_m = float(lambda0_nm) * 1e-9
    omega0 = 2.0 * np.pi * C0 / lambda0_m
    k0 = 2.0 * np.pi / lambda0_m
    return {
        "lambda0_nm": float(lambda0_nm),
        "lambda0_m": lambda0_m,
        "omega0_rad_per_s": float(omega0),
        "k0_rad_per_m": float(k0),
        "model": "carrier-envelope",
        "field_expression": "E(x,t)=Re{f(x,t) exp[i(k0 z-omega0 t)]}",
        "envelope_expression": "f(x,t)=integral F(kx,Omega) exp[-i(kx x-Omega t)] dkx dOmega",
    }


def _energy_bbox(
    energy: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    frac: float,
) -> dict[str, float]:
    arr = np.asarray(energy, dtype=np.float64)
    arr = np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)
    total = float(np.sum(arr))
    if total <= 1e-18:
        return {
            "energy_frac": float(frac),
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
        }

    flat = arr.ravel()
    order = np.argsort(flat)[::-1]
    csum = np.cumsum(flat[order])
    keep = order[csum <= frac * total]
    if keep.size == 0:
        keep = order[:1]
    mask = np.zeros_like(flat, dtype=bool)
    mask[keep] = True
    yy, xx = np.where(mask.reshape(arr.shape))
    return {
        "energy_frac": float(frac),
        "x_min": float(np.asarray(x_axis, dtype=np.float64)[int(np.min(xx))]),
        "x_max": float(np.asarray(x_axis, dtype=np.float64)[int(np.max(xx))]),
        "y_min": float(np.asarray(y_axis, dtype=np.float64)[int(np.min(yy))]),
        "y_max": float(np.asarray(y_axis, dtype=np.float64)[int(np.max(yy))]),
    }


def input_spectral_overlap_metrics(
    spec_kw: np.ndarray,
    kx_fft: np.ndarray,
    omega_fft: np.ndarray,
    k0: float,
    omega0: float,
    support_mask: np.ndarray,
) -> tuple[dict[str, float | dict[str, float]], dict[str, np.ndarray]]:
    spec = np.asarray(spec_kw, dtype=np.complex128)
    energy = np.abs(spec) ** 2
    support = np.asarray(support_mask, dtype=bool)

    kappa = np.asarray(kx_fft, dtype=np.float64) / max(float(k0), 1e-18)
    nu = np.asarray(omega_fft, dtype=np.float64) / max(float(omega0), 1e-18)
    kappa_grid, nu_grid = np.meshgrid(kappa, nu)

    total_energy = float(np.sum(energy))
    inside_energy = float(np.sum(energy[support])) if np.any(support) else 0.0
    outside_energy = max(total_energy - inside_energy, 0.0)

    if total_energy > 1e-18:
        peak_idx = np.unravel_index(int(np.argmax(energy)), energy.shape)
        mean_kappa = float(np.sum(kappa_grid * energy) / total_energy)
        mean_nu = float(np.sum(nu_grid * energy) / total_energy)
        rms_kappa = float(np.sqrt(np.sum((kappa_grid ** 2) * energy) / total_energy))
        rms_nu = float(np.sqrt(np.sum((nu_grid ** 2) * energy) / total_energy))
        if inside_energy > 1e-18:
            mean_kappa_inside = float(np.sum(kappa_grid[support] * energy[support]) / inside_energy)
            mean_nu_inside = float(np.sum(nu_grid[support] * energy[support]) / inside_energy)
        else:
            mean_kappa_inside = 0.0
            mean_nu_inside = 0.0
    else:
        peak_idx = (0, 0)
        mean_kappa = mean_nu = rms_kappa = rms_nu = 0.0
        mean_kappa_inside = mean_nu_inside = 0.0

    metrics = {
        "input_energy_total": total_energy,
        "input_energy_inside_support": inside_energy,
        "input_energy_outside_support": outside_energy,
        "input_energy_inside_frac": inside_energy / max(total_energy, 1e-18),
        "input_energy_outside_frac": outside_energy / max(total_energy, 1e-18),
        "peak_kappa": float(kappa_grid[peak_idx]),
        "peak_nu": float(nu_grid[peak_idx]),
        "mean_kappa": mean_kappa,
        "mean_nu": mean_nu,
        "rms_kappa": rms_kappa,
        "rms_nu": rms_nu,
        "mean_kappa_inside_support": mean_kappa_inside,
        "mean_nu_inside_support": mean_nu_inside,
        "bbox_50pct_energy": _energy_bbox(energy, kappa, nu, 0.50),
        "bbox_80pct_energy": _energy_bbox(energy, kappa, nu, 0.80),
        "bbox_95pct_energy": _energy_bbox(energy, kappa, nu, 0.95),
    }
    arrays = {
        "input_energy": energy.astype(np.float64),
        "support_mask": support.astype(np.uint8),
        "kappa_grid": kappa_grid.astype(np.float64),
        "nu_grid": nu_grid.astype(np.float64),
    }
    return metrics, arrays


def spectral_debug_metrics(
    spec_kw: np.ndarray,
    otf_kw: np.ndarray,
    kx_fft: np.ndarray,
    omega_fft: np.ndarray,
    k0: float,
    omega0: float,
    na_s: float,
    na_t: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    spec = np.asarray(spec_kw, dtype=np.complex128)
    otf = np.asarray(otf_kw, dtype=np.float64)
    kx = np.asarray(kx_fft, dtype=np.float64)
    omega = np.asarray(omega_fft, dtype=np.float64)

    spec_abs = np.abs(spec)
    spec_energy = spec_abs ** 2
    weighted_abs = spec_abs * otf
    weighted_energy = spec_energy * (otf ** 2)

    kappa = kx / max(float(k0), 1e-18)
    nu = omega / max(float(omega0), 1e-18)
    kappa_grid, nu_grid = np.meshgrid(kappa, nu)

    center_mask = (np.abs(kappa_grid) <= 0.1) & (np.abs(nu_grid) <= 0.02)
    edge_mask = (np.abs(kappa_grid) >= 0.5 * float(na_s)) & (np.abs(nu_grid) >= 0.6 * float(na_t))
    support_mask = np.isfinite(otf) & (otf > 0)

    def _sum(mask: np.ndarray, arr: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)
        if not np.any(mask):
            return 0.0
        return float(np.sum(arr[mask]))

    def _mean(mask: np.ndarray, arr: np.ndarray) -> float:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)
        if not np.any(mask):
            return float("nan")
        return float(np.mean(arr[mask]))

    total_input_energy = _sum(support_mask, spec_energy)
    total_weighted_energy = _sum(support_mask, weighted_energy)
    peak_idx = np.unravel_index(int(np.argmax(spec_energy)), spec_energy.shape)
    peak_weighted_idx = np.unravel_index(int(np.argmax(weighted_energy)), weighted_energy.shape)

    metrics = {
        "input_energy_total": total_input_energy,
        "weighted_energy_total": total_weighted_energy,
        "input_energy_center_frac": _sum(center_mask, spec_energy) / max(total_input_energy, 1e-18),
        "input_energy_edge_frac": _sum(edge_mask, spec_energy) / max(total_input_energy, 1e-18),
        "weighted_energy_center_frac": _sum(center_mask, weighted_energy) / max(total_weighted_energy, 1e-18),
        "weighted_energy_edge_frac": _sum(edge_mask, weighted_energy) / max(total_weighted_energy, 1e-18),
        "otf_center_mean": _mean(center_mask, otf),
        "otf_edge_mean": _mean(edge_mask, otf),
        "peak_input_kappa": float(kappa_grid[peak_idx]),
        "peak_input_nu": float(nu_grid[peak_idx]),
        "peak_weighted_kappa": float(kappa_grid[peak_weighted_idx]),
        "peak_weighted_nu": float(nu_grid[peak_weighted_idx]),
    }
    arrays = {
        "spec_abs": spec_abs.astype(np.float64),
        "spec_energy": spec_energy.astype(np.float64),
        "weighted_abs": weighted_abs.astype(np.float64),
        "weighted_energy": weighted_energy.astype(np.float64),
        "kappa_grid": kappa_grid.astype(np.float64),
        "nu_grid": nu_grid.astype(np.float64),
        "center_mask": center_mask.astype(np.uint8),
        "edge_mask": edge_mask.astype(np.uint8),
    }
    return metrics, arrays


def response_intensity(response_map: np.ndarray) -> np.ndarray:
    return envelope_intensity(response_map)


def remove_dc_component(signal_xt: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal_xt, dtype=np.float64)
    return signal - float(np.mean(signal))


def signal_region_masks(signal_xt: np.ndarray, edge_width: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal = np.asarray(signal_xt, dtype=np.float64) > 0.5
    edge = np.zeros_like(signal, dtype=bool)

    edge[:-1, :] |= signal[:-1, :] != signal[1:, :]
    edge[1:, :] |= signal[1:, :] != signal[:-1, :]
    edge[:, :-1] |= signal[:, :-1] != signal[:, 1:]
    edge[:, 1:] |= signal[:, 1:] != signal[:, :-1]

    edge_band = edge.copy()
    for _ in range(max(int(edge_width) - 1, 0)):
        grown = edge_band.copy()
        grown[:-1, :] |= edge_band[1:, :]
        grown[1:, :] |= edge_band[:-1, :]
        grown[:, :-1] |= edge_band[:, 1:]
        grown[:, 1:] |= edge_band[:, :-1]
        edge_band = grown

    interior = signal & (~edge_band)
    background = ~signal
    return edge_band, interior, background


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(arr)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(arr[mask]))


def response_metrics(signal_xt: np.ndarray, device_map: np.ndarray, ideal_map: np.ndarray) -> dict[str, float]:
    a = envelope_amplitude(device_map).ravel()
    b = envelope_amplitude(ideal_map).ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"corrcoef": float("nan"), "mae": float("nan"), "nmae": float("nan")}
    a = a[mask]
    b = b[mask]
    if a.size > 1 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = float("nan")
    mae = float(np.mean(np.abs(a - b)))
    denom = max(float(np.max(np.abs(b))), 1e-12)
    device_int = response_intensity(device_map)
    ideal_int = response_intensity(ideal_map)
    edge_mask, interior_mask, background_mask = signal_region_masks(signal_xt)

    device_edge_mean = _masked_mean(device_int, edge_mask)
    device_interior_mean = _masked_mean(device_int, interior_mask)
    device_background_mean = _masked_mean(device_int, background_mask)
    ideal_edge_mean = _masked_mean(ideal_int, edge_mask)
    ideal_interior_mean = _masked_mean(ideal_int, interior_mask)
    ideal_background_mean = _masked_mean(ideal_int, background_mask)

    def _safe_ratio(num: float, den: float) -> float:
        if not np.isfinite(num) or not np.isfinite(den):
            return float("nan")
        return float(num / max(abs(den), 1e-18))

    return {
        "corrcoef": corr,
        "mae": mae,
        "nmae": mae / denom,
        "device_peak_intensity": float(np.max(device_int)),
        "ideal_peak_intensity": float(np.max(ideal_int)),
        "device_edge_mean": device_edge_mean,
        "device_interior_mean": device_interior_mean,
        "device_background_mean": device_background_mean,
        "device_edge_to_interior": _safe_ratio(device_edge_mean, device_interior_mean),
        "device_edge_to_background": _safe_ratio(device_edge_mean, device_background_mean),
        "ideal_edge_mean": ideal_edge_mean,
        "ideal_interior_mean": ideal_interior_mean,
        "ideal_background_mean": ideal_background_mean,
        "ideal_edge_to_interior": _safe_ratio(ideal_edge_mean, ideal_interior_mean),
        "ideal_edge_to_background": _safe_ratio(ideal_edge_mean, ideal_background_mean),
    }
