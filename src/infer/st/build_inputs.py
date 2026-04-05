from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalXT:
    name: str
    signal: np.ndarray
    x_lambda: np.ndarray
    t_t1e4: np.ndarray


def indicator_segment(x_lambda: np.ndarray, x1: float, x2: float) -> np.ndarray:
    x = np.asarray(x_lambda, dtype=np.float64)
    return ((x >= x1) & (x <= x2)).astype(np.float64)


def build_switching_segments(x_lambda: np.ndarray, t_t1e4: np.ndarray) -> SignalXT:
    return build_switching_segments_scaled(x_lambda, t_t1e4)


def build_switching_segments_scaled(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    width_scale: float = 1.0,
    time_scale: float = 1.0,
    center_x_lambda: float = 0.0,
    center_t_t1e4: float = 1.055,
) -> SignalXT:
    x = np.asarray(x_lambda, dtype=np.float64)
    t = np.asarray(t_t1e4, dtype=np.float64)
    signal = np.zeros((len(t), len(x)), dtype=np.float64)
    segments = [
        (-75.0, 100.0, 0.855, 0.35),
        (-75.0, 100.0, 1.40, 0.16),
        (40.0, 60.0, 0.825, 0.39),
        (40.0, 60.0, 1.39, 0.18),
        (115.0, 30.0, 0.825, 0.39),
        (115.0, 30.0, 1.39, 0.18),
    ]
    for x_center, width, t_center, duration in segments:
        width_s = float(width) * float(width_scale)
        duration_s = float(duration) * float(time_scale)
        x_center_s = float(center_x_lambda) + (float(x_center) - float(center_x_lambda)) * float(width_scale)
        t_center_s = float(center_t_t1e4) + (float(t_center) - float(center_t_t1e4)) * float(time_scale)
        spatial = (np.abs(x - x_center_s) <= 0.5 * width_s)[None, :]
        temporal = (np.abs(t - t_center_s) <= 0.5 * duration_s).astype(np.float64)[:, None]
        signal = np.maximum(signal, temporal * spatial)
    return SignalXT("fig3_switch", signal, x_lambda=x, t_t1e4=t)


def build_breathing_segment(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    width_scale: float = 1.0,
    time_scale: float = 1.0,
) -> SignalXT:
    x = np.asarray(x_lambda, dtype=np.float64)
    t = np.asarray(t_t1e4, dtype=np.float64)
    # Control points chosen to match the hourglass-like contour in Fig. 3(d).
    t_ctrl = np.array([0.0, 1.0, 2.2, 3.5, 5.2, 6.4, 7.4], dtype=np.float64)
    w_ctrl = np.array([980.0, 980.0, 520.0, 260.0, 520.0, 980.0, 980.0], dtype=np.float64)
    t_ctrl = t_ctrl * float(time_scale)
    w_ctrl = w_ctrl * float(width_scale)
    width = np.interp(t, t_ctrl, w_ctrl, left=w_ctrl[0], right=w_ctrl[-1])
    signal = (np.abs(x[None, :]) <= 0.5 * width[:, None]).astype(np.float64)
    return SignalXT("fig3_breath", signal, x_lambda=x, t_t1e4=t)


def build_piecewise_motion(x_lambda: np.ndarray, t_t1e4: np.ndarray, v0_lambda_per_t1e4: float) -> SignalXT:
    x = np.asarray(x_lambda, dtype=np.float64)
    t = np.asarray(t_t1e4, dtype=np.float64)
    width = 55.0
    center = np.full_like(t, -95.0, dtype=np.float64)

    t1, t2, t3, t4, t5 = 0.45, 0.95, 1.45, 1.95, 3.0
    v0_true = float(v0_lambda_per_t1e4)
    v_half = 0.5 * v0_true
    v_one = 1.0 * v0_true
    v_one_half = 1.5 * v0_true

    m1 = (t >= t1) & (t < t2)
    center[m1] = -95.0 + v_half * (t[m1] - t1)
    x_t2 = -95.0 + v_half * (t2 - t1)

    m2 = (t >= t2) & (t < t3)
    center[m2] = x_t2 + v_one * (t[m2] - t2)
    x_t3 = x_t2 + v_one * (t3 - t2)

    m3 = (t >= t3) & (t < t4)
    center[m3] = x_t3 + v_one_half * (t[m3] - t3)
    x_t4 = x_t3 + v_one_half * (t4 - t3)

    # Final stage: a visible parabolic bend-back on the true v0 scale.
    m4 = t >= t4
    tau = np.clip((t[m4] - t4) / max(t5 - t4, 1e-8), 0.0, 1.0)
    x_end = x_t4 - 0.85 * v0_true * (t5 - t4)
    center[m4] = x_t4 - (x_t4 - x_end) * (tau ** 2)

    signal = (np.abs(x[None, :] - center[:, None]) <= 0.5 * width).astype(np.float64)
    return SignalXT("fig4_motion", signal, x_lambda=x, t_t1e4=t)


def build_uniform_motion(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    velocity_lambda_per_t1e4: float,
    width_lambda: float = 55.0,
    x0_lambda: float = -120.0,
) -> SignalXT:
    x = np.asarray(x_lambda, dtype=np.float64)
    t = np.asarray(t_t1e4, dtype=np.float64)
    center = x0_lambda + velocity_lambda_per_t1e4 * t
    signal = (np.abs(x[None, :] - center[:, None]) <= 0.5 * width_lambda).astype(np.float64)
    return SignalXT("uniform_motion", signal, x_lambda=x, t_t1e4=t)


def build_targeted_pulse(
    x_lambda: np.ndarray,
    t_t1e4: np.ndarray,
    width_lambda: float,
    duration_t1e4: float,
    center_x_lambda: float = 0.0,
    center_t_t1e4: float = 1.0,
) -> SignalXT:
    x = np.asarray(x_lambda, dtype=np.float64)
    t = np.asarray(t_t1e4, dtype=np.float64)
    spatial = (np.abs(x - float(center_x_lambda)) <= 0.5 * float(width_lambda))[None, :]
    temporal = (np.abs(t - float(center_t_t1e4)) <= 0.5 * float(duration_t1e4))[:, None]
    signal = (spatial & temporal).astype(np.float64)
    return SignalXT("targeted_pulse", signal, x_lambda=x, t_t1e4=t)
