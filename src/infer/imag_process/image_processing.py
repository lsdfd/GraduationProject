# -*- coding: utf-8 -*-
"""
Fourier optics imaging simulation for metagen project.

Usage
-----
  # 理想二阶微分传递函数（无需任何 RCWA 数据，直接验证仿真流程）
  python image_processing.py --ideal

  # 用训练数据的某个样本（1D 角度数据，各向同性近似）
  python image_processing.py --sample 0

  # 用 laplas 推理结果的最优样本
  python image_processing.py --from_infer

  # 用 scan_kspace.py 生成的精确 2D 传递函数
  python image_processing.py --from_kspace

  # 改偏振 / 改波长
  python image_processing.py --ideal --pol x45 --lambda_nm 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d, RegularGridInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "train_data.npz"
ASSETS_DIR   = _HERE / "assets"
DEFAULT_THU_IMAGE_PATH = ASSETS_DIR / "fc728b57c87df1ef1c97e5fa08a6434f.png"
DEFAULT_CUSTOM_IMAGE_PATH = ASSETS_DIR / "581575372c297e43f81babaa2dd7e9a7.png"

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
THETA_MAX = 40.0
NA        = np.sin(np.radians(THETA_MAX))    # ≈ 0.6428
LAMBDAS   = np.arange(800.0, 1300.1, 50.0)  # [11]
THETAS    = np.arange(-40.0, 40.1,  5.0)    # [17]
# 与 scan_kspace.py 保持一致，避免额外的 512 -> 502 重采样。
N_GRID    = 512
NX, NY    = N_GRID, N_GRID

POL_MAP: dict[str, np.ndarray] = {
    "x":    np.array([1.0,  0.0]),
    "y":    np.array([0.0,  1.0]),
    "x45":  np.array([1.0,  1.0]) / np.sqrt(2),
    "x-45": np.array([1.0, -1.0]) / np.sqrt(2),
    "RCP":  np.array([1.0, -1j])  / np.sqrt(2),
    "LCP":  np.array([1.0,  1j])  / np.sqrt(2),
}
POL_CHOICES = list(POL_MAP.keys()) + ["all"]

DEFAULT_PATTERN_SET = ("square", "circle", "checkerboard", "thu", "custom")
PATTERN_CHOICES = DEFAULT_PATTERN_SET + ("random16", "random64", "square_thu", "all")


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _task_tag_from_path(path: Path | None) -> str:
    if path is None:
        return "general"
    parts = list(path.parts)
    if "results" in parts:
        i = parts.index("results")
        if i + 2 < len(parts):
            return _safe_tag(f"{parts[i + 1]}_{parts[i + 2]}")
    if "run_kspace" in path.name:
        stem = path.stem.replace("kspace_result_", "")
        return _safe_tag(stem)
    return _safe_tag(path.stem)


def _results_case_dir_from_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    parts = list(path.parts)
    if "results" not in parts:
        return None
    i = parts.index("results")
    if i + 2 >= len(parts):
        return None
    return Path(*parts[: i + 3])


def infer_task_family(path: Path | None, fallback_tag: str | None = None) -> str:
    if path is not None:
        parts = list(path.parts)
        if "results" in parts:
            i = parts.index("results")
            if i + 1 < len(parts):
                bucket = parts[i + 1]
                if bucket == "p":
                    return "p_second_order"
                if bucket == "sp-all":
                    return "polarization_independent"
                if bucket in {"sp-mutiplex", "sp-multiplex"}:
                    return "polarization_multiplexed"
                if bucket == "fourth":
                    return "fourth_order"
                if bucket == "lowpass":
                    return "lowpass"
                if bucket == "st":
                    return "st2"
    if fallback_tag == "ideal":
        return "ideal"
    return "generic"


def make_run_output_dir(base_dir: Path, task_tag: str, source_tag: str, lambda_nm: float, polarization: str, pattern: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_tag = _safe_tag(task_tag)
    src_tag = _safe_tag(source_tag)
    pol_tag = _safe_tag(polarization)
    pattern_tag = _safe_tag(pattern)
    lam_tag = f"{int(round(lambda_nm))}nm"
    out_dir = base_dir / f"run_imaging_{task_tag}_{lam_tag}_{src_tag}_{pol_tag}_{pattern_tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def latest_kspace_npz() -> Path | None:
    candidates = sorted(_HERE.rglob("kspace_result*.npz"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def edge_mask_from_input(I_in: np.ndarray, dilation_px: int = 4) -> np.ndarray:
    mask = I_in > 0.5
    # 二值输入下，边缘可由 4 邻域不一致位置近似得到。
    edge = (
        (mask != np.roll(mask, 1, axis=0)) |
        (mask != np.roll(mask, -1, axis=0)) |
        (mask != np.roll(mask, 1, axis=1)) |
        (mask != np.roll(mask, -1, axis=1))
    ) & mask
    if dilation_px > 0:
        edge = binary_dilation(edge, iterations=dilation_px)
    return edge


def compute_imaging_metrics(I_in: np.ndarray, I_out: np.ndarray, edge_mask: np.ndarray) -> dict[str, float]:
    peak_in = max(float(np.max(I_in)), 1e-12)
    eta_peak = float(np.max(I_out) / peak_in)
    eta_avg = float(np.mean(I_out[edge_mask]) / peak_in) if np.any(edge_mask) else 0.0
    return {
        "eta_peak": eta_peak,
        "eta_avg": eta_avg,
        "input_peak": float(np.max(I_in)),
        "output_peak": float(np.max(I_out)),
        "edge_pixels": int(np.sum(edge_mask)),
    }


def compute_center_cut_summary(I_out: np.ndarray) -> dict[str, object]:
    cy, cx = I_out.shape[0] // 2, I_out.shape[1] // 2
    x_cut = np.asarray(I_out[cy, :], dtype=np.float64)
    y_cut = np.asarray(I_out[:, cx], dtype=np.float64)
    x_peak = max(float(np.max(x_cut)), 1e-12)
    y_peak = max(float(np.max(y_cut)), 1e-12)
    return {
        "cut_row": int(cy),
        "cut_col": int(cx),
        "x_cut_peak": x_peak,
        "y_cut_peak": y_peak,
        "x_cut_center_value": float(x_cut[cx]),
        "y_cut_center_value": float(y_cut[cy]),
        "x_cut_norm": (x_cut / x_peak).tolist(),
        "y_cut_norm": (y_cut / y_peak).tolist(),
    }


def save_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# ===========================================================================
# 1. 传递函数加载
# ===========================================================================

def make_ideal_transfer(KX: np.ndarray, KY: np.ndarray,
                        K_MAX: float) -> tuple[np.ndarray, np.ndarray]:
    """
    理想二阶微分传递函数：T = (k_rho / k_max)^2，各向同性，圆外为 0。
    tss = tpp，任意偏振都得到 Laplacian 效果。
    """
    k_rho = np.sqrt(KX**2 + KY**2)
    T = (k_rho / K_MAX) ** 2
    T[k_rho > K_MAX] = 0.0
    return T.copy(), T.copy()   # T_ss, T_pp


def make_task_ideal_transfer(
    task_family: str,
    KX: np.ndarray,
    KY: np.ndarray,
    K0: float,
    K_MAX: float,
) -> tuple[np.ndarray, np.ndarray]:
    k_rho = np.sqrt(KX**2 + KY**2)
    theta_deg = np.degrees(np.arcsin(np.clip(k_rho / max(K0, 1e-12), 0.0, 1.0)))

    if task_family in {"ideal", "generic", "polarization_independent"}:
        return make_ideal_transfer(KX, KY, K_MAX)

    if task_family in {"p_second_order", "polarization_multiplexed"}:
        T = np.zeros_like(k_rho, dtype=np.float64)
        inside = k_rho <= K_MAX
        T[inside] = (k_rho[inside] / max(K_MAX, 1e-12)) ** 2
        return np.zeros_like(T), T

    if task_family == "fourth_order":
        T = np.zeros_like(k_rho, dtype=np.float64)
        inside = k_rho <= K_MAX
        T[inside] = (k_rho[inside] / max(K_MAX, 1e-12)) ** 4
        return np.zeros_like(T), T

    if task_family == "lowpass":
        sigma_deg = 12.0
        T = np.exp(-(theta_deg ** 2) / max(sigma_deg ** 2, 1e-12))
        T[k_rho > K_MAX] = 0.0
        return np.zeros_like(T), T

    if task_family == "st2":
        T = np.zeros_like(k_rho, dtype=np.float64)
        inside = k_rho <= K_MAX
        T[inside] = (k_rho[inside] / max(K_MAX, 1e-12)) ** 2
        return T.copy(), T.copy()

    return make_ideal_transfer(KX, KY, K_MAX)


def load_from_training(sample_idx: int,
                       lam_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (t_ss_1d, t_pp_1d)，shape [17]。"""
    data    = np.load(DATA_PATH)
    tpp_all = data["tpp_mag"]   # [N, 11, 17]
    tss_all = data["tss_mag"]
    n = tpp_all.shape[0]
    if not (0 <= sample_idx < n):
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {n})")
    return (
        tss_all[sample_idx, lam_idx, :].astype(np.float64),
        tpp_all[sample_idx, lam_idx, :].astype(np.float64),
    )


def load_from_infer(infer_dir: Path,
                    lam_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """从 laplas 推理目录加载最优样本的 1D 传递函数。"""
    d = Path(infer_dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    if not (d / "all_pred_cond_raw.npy").exists():
        raise FileNotFoundError(f"找不到 all_pred_cond_raw.npy in {d}")
    pred   = np.load(d / "all_pred_cond_raw.npy")   # [N, 2, 11, 17]
    errors = np.load(d / "all_errors.npy")
    best   = int(np.argmin(errors))
    print(f"[infer] best={best}, error={errors[best]:.4f}")
    return (
        pred[best, 1, lam_idx, :].astype(np.float64),
        pred[best, 0, lam_idx, :].astype(np.float64),
    )


def load_from_kspace(npz_path: Path,
                     KX: np.ndarray, KY: np.ndarray,
                     K0: float, K_MAX: float) -> tuple[np.ndarray, np.ndarray]:
    """从 scan_kspace.py 生成的 npz 加载 2D 传递函数并插值到当前网格。"""
    d       = np.load(npz_path)
    kx_norm = d["kx_norm"]                        # [N] = kx/k0
    ky_norm = d["ky_norm"]                        # [N]
    T_pp_2d = np.nan_to_num(d["T_pp"], nan=0.0)  # [N, N]
    T_ss_2d = np.nan_to_num(d["T_ss"], nan=0.0)

    interp_pp = RegularGridInterpolator(
        (ky_norm, kx_norm), T_pp_2d,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    interp_ss = RegularGridInterpolator(
        (ky_norm, kx_norm), T_ss_2d,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    pts  = np.stack([KY.ravel() / K0, KX.ravel() / K0], axis=-1)
    T_pp = interp_pp(pts).reshape(KX.shape)
    T_ss = interp_ss(pts).reshape(KX.shape)

    k_rho = np.sqrt(KX**2 + KY**2)
    T_pp[k_rho > K_MAX] = 0.0
    T_ss[k_rho > K_MAX] = 0.0
    print(f"[kspace] 从 {npz_path.name} 加载，插值到 {KX.shape}")
    return T_ss, T_pp


# ===========================================================================
# 2. 1D → 2D 插值（各向同性近似）
# ===========================================================================

def build_2d_transfer(t_1d: np.ndarray,
                      KX: np.ndarray, KY: np.ndarray,
                      K0: float, K_MAX: float) -> np.ndarray:
    """将 1D 角度扫描（shape [17]）插值为 2D 传递函数（各向同性假设）。"""
    center = 8   # THETAS 中 theta=0 的 index
    n_half = 8
    t_sym  = np.empty(n_half + 1)
    t_sym[0] = t_1d[center]
    for i in range(1, n_half + 1):
        t_sym[i] = 0.5 * (t_1d[center - i] + t_1d[center + i])
    thetas_pos    = np.arange(0.0, 40.1, 5.0)
    k_rho_samples = K0 * np.sin(np.radians(thetas_pos))

    interp = interp1d(k_rho_samples, t_sym,
                      kind="linear", bounds_error=False, fill_value=0.0)
    k_rho = np.sqrt(KX**2 + KY**2)
    T_2d  = interp(k_rho)
    T_2d[k_rho > K_MAX] = 0.0
    return T_2d


# ===========================================================================
# 3. 输入图案 + 傅里叶光学仿真
# ===========================================================================

def _draw_rect(img: np.ndarray, x0: int, x1: int, y0: int, y1: int, value: float = 1.0) -> None:
    img[max(0, y0):min(img.shape[0], y1), max(0, x0):min(img.shape[1], x1)] = value


def _draw_disk(img: np.ndarray, cx: int, cy: int, radius: int, value: float = 1.0) -> None:
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img[mask] = value


def _make_random_binary_tile(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, n)) > 0.5).astype(np.float64)


def build_builtin_input(pattern: str) -> np.ndarray:
    img = np.zeros((NY, NX), dtype=np.float64)
    cx, cy = NX // 2, NY // 2

    if pattern in {"square", "square_thu"}:
        sq = max(8, int(min(NX, NY) * 0.7))
        hs = sq // 2
        _draw_rect(img, cx - hs, cx + hs, cy - hs, cy + hs)
    elif pattern == "circle":
        radius = max(8, int(min(NX, NY) * 0.36))
        _draw_disk(img, cx, cy, radius)
    elif pattern == "checkerboard":
        tile = max(8, min(NX, NY) // 8)
        n_rows = int(np.ceil(NY / tile))
        n_cols = int(np.ceil(NX / tile))
        for row in range(n_rows):
            for col in range(n_cols):
                if (row + col) % 2 == 0:
                    _draw_rect(
                        img,
                        col * tile,
                        (col + 1) * tile,
                        row * tile,
                        (row + 1) * tile,
                    )
    elif pattern == "random16":
        tile = _make_random_binary_tile(16, 20260408)
        img = np.asarray(
            Image.fromarray((tile * 255).astype(np.uint8), mode="L").resize((NX, NY), Image.Resampling.NEAREST),
            dtype=np.float64,
        ) / 255.0
    elif pattern == "random64":
        tile = _make_random_binary_tile(64, 20260408)
        img = np.asarray(
            Image.fromarray((tile * 255).astype(np.uint8), mode="L").resize((NX, NY), Image.Resampling.NEAREST),
            dtype=np.float64,
        ) / 255.0

    return img


def _auto_binarize_mask(gray: np.ndarray) -> np.ndarray:
    values = gray.ravel()
    if np.allclose(values.min(), values.max()):
        raise ValueError("输入图片亮度几乎恒定，无法区分黑底和白字。")

    hist, edges = np.histogram(values, bins=256, range=(0.0, 1.0))
    total = values.size
    sum_total = np.dot(hist, 0.5 * (edges[:-1] + edges[1:]))
    sum_bg = 0.0
    weight_bg = 0.0
    best_var = -1.0
    threshold = 0.5

    for idx, count in enumerate(hist):
        weight_bg += count
        if weight_bg == 0 or weight_bg == total:
            sum_bg += count * (0.5 * (edges[idx] + edges[idx + 1]))
            continue
        bin_center = 0.5 * (edges[idx] + edges[idx + 1])
        sum_bg += count * bin_center
        mean_bg = sum_bg / weight_bg
        weight_fg = total - weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > best_var:
            best_var = between
            threshold = bin_center

    mask = gray >= threshold
    fill_ratio = float(np.mean(mask))
    if fill_ratio < 1e-4 or fill_ratio > 0.9999:
        raise ValueError(
            f"自动阈值失败，白字区域占比={fill_ratio:.4f}，请检查图片是否为黑底白字。"
        )
    return mask.astype(np.float64)


def load_image_input(image_path: Path, nx: int, ny: int) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(f"找不到输入图片：{image_path}")
    with Image.open(image_path) as pil_img:
        rgba_src = pil_img.convert("RGBA")
        src_w, src_h = rgba_src.size
        scale = min(nx / src_w, ny / src_h)
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))
        rgba_resized = rgba_src.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        rgba = Image.new("RGBA", (nx, ny), (0, 0, 0, 255))
        off_x = (nx - resized_w) // 2
        off_y = (ny - resized_h) // 2
        rgba.paste(rgba_resized, (off_x, off_y), rgba_resized)
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4]
    rgb_on_black = rgb * alpha
    gray = 0.2126 * rgb_on_black[..., 0] + 0.7152 * rgb_on_black[..., 1] + 0.0722 * rgb_on_black[..., 2]
    return _auto_binarize_mask(gray)


def build_input_image(pattern: str, input_image: str | None = None) -> tuple[np.ndarray, str]:
    source_desc = pattern
    builtin = build_builtin_input(pattern)

    image_path: Path | None = None
    if input_image:
        image_path = Path(input_image)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        source_desc = f"{pattern} + {image_path.name}" if pattern == "square_thu" else image_path.name
    elif pattern in {"thu", "square_thu"}:
        image_path = DEFAULT_THU_IMAGE_PATH
        source_desc = f"{pattern} + {image_path.name}" if pattern == "square_thu" else image_path.name
    elif pattern == "custom":
        image_path = DEFAULT_CUSTOM_IMAGE_PATH
        source_desc = f"{pattern} + {image_path.name}" if pattern == "square_thu" else image_path.name

    if image_path is None:
        return builtin, source_desc

    image_mask = load_image_input(image_path, NX, NY)
    if pattern == "square_thu":
        return np.maximum(builtin, image_mask), source_desc
    return image_mask, source_desc

def run_fourier_optics(T_ss: np.ndarray, T_pp: np.ndarray,
                       e_in: np.ndarray,
                       KX: np.ndarray, KY: np.ndarray,
                       pattern: str,
                       K0: float,
                       input_image: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """
    角谱法 Jones 矩阵仿真。
    返回 (I_out [NY,NX], I_in [NY,NX], input_label)。
    """
    I_in, input_label = build_input_image(pattern, input_image=input_image)
    f_in        = np.sqrt(I_in)

    # k 空间角度
    k_rho     = np.sqrt(KX**2 + KY**2)
    sin_theta = np.clip(k_rho / K0, 0.0, 1.0)
    theta_k   = np.arcsin(sin_theta)
    phi_k     = np.arctan2(KY, KX)

    # 正变换
    F_in = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f_in)))

    # 分解 s/p
    E_p_in = (e_in[0] * np.cos(phi_k) + e_in[1] * np.sin(phi_k)) * F_in
    E_s_in = (np.cos(theta_k)
              * (e_in[1] * np.cos(phi_k) - e_in[0] * np.sin(phi_k))
              * F_in)

    # 施加传递函数
    E_s_out = T_ss * E_s_in
    E_p_out = T_pp * E_p_in

    # 转回 x/y
    cos_theta = np.cos(theta_k)
    cos_theta[cos_theta == 0] = np.finfo(float).eps
    E_x_out = np.cos(phi_k) * E_p_out - (np.sin(phi_k) / cos_theta) * E_s_out
    E_y_out = np.sin(phi_k) * E_p_out + (np.cos(phi_k) / cos_theta) * E_s_out

    # 逆变换
    E_x_sp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_x_out)))
    E_y_sp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(E_y_out)))
    I_out  = np.abs(E_x_sp)**2 + np.abs(E_y_sp)**2
    return I_out, I_in, input_label


def run_ideal_reference(
    task_family: str,
    KX: np.ndarray,
    KY: np.ndarray,
    K0: float,
    K_MAX: float,
    pattern: str,
    input_image: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, object]]:
    T_ideal_ss, T_ideal_pp = make_task_ideal_transfer(task_family, KX, KY, K0, K_MAX)
    I_out, I_in, input_label = run_fourier_optics(
        T_ideal_ss,
        T_ideal_pp,
        POL_MAP["x45"],
        KX,
        KY,
        pattern,
        K0,
        input_image=input_image,
    )
    edge_mask = edge_mask_from_input(I_in)
    metrics = compute_imaging_metrics(I_in, I_out, edge_mask)
    metrics.update(compute_center_cut_summary(I_out))
    return I_out, I_in, input_label, metrics


# ===========================================================================
# 4. 可视化
# ===========================================================================

def plot_results(I_in: np.ndarray, I_out: np.ndarray,
                 lambda_nm: float, polarization: str, pattern: str, input_label: str,
                 metrics: dict[str, float], save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    # 仅保留输入图和输出图，结果页更聚焦。
    im0 = axes[0].imshow(I_in, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    axes[0].set_title(f"Input image ({input_label})")
    axes[0].set_xlabel("x (pixel)")
    axes[0].set_ylabel("y (pixel)")
    plt.colorbar(im0, ax=axes[0])

    # 输出强度
    vmax = float(np.percentile(I_out, 99)) or 1.0
    im1  = axes[1].imshow(I_out, cmap="inferno",
                           origin="upper", vmin=0, vmax=vmax)
    axes[1].set_title(f"Output intensity  (pol={polarization})")
    axes[1].set_xlabel("x (pixel)")
    axes[1].set_ylabel("y (pixel)")
    plt.colorbar(im1, ax=axes[1])

    fig.suptitle(
        f"Fourier optics simulation  |  "
        f"λ = {lambda_nm:.0f} nm,  NA = {NA:.4f}  (θ_max = {THETA_MAX:.0f}°),  input = {pattern}"
    )
    fig.text(
        0.5, 0.02,
        f"eta_peak = {metrics['eta_peak']:.4f}    eta_avg = {metrics['eta_avg']:.4f}    edge_pixels = {metrics['edge_pixels']}",
        ha="center", va="bottom", fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 4.0},
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


def plot_multi_pol_results(
    I_in: np.ndarray,
    I_ideal: np.ndarray,
    ideal_metrics: dict[str, object],
    result_map: dict[str, tuple[np.ndarray, dict[str, object]]],
    lambda_nm: float,
    pattern: str,
    input_label: str,
    save_path: Path,
) -> None:
    pol_order = [pol for pol in POL_MAP.keys() if pol in result_map]
    fig, axes = plt.subplots(len(pol_order), 5, figsize=(22.0, 3.2 * len(pol_order)))
    if len(pol_order) == 1:
        axes = np.expand_dims(axes, axis=0)

    vmax = max(float(np.percentile(I_out, 99)) for I_out, _ in result_map.values())
    vmax = vmax or 1.0

    for row_idx, pol in enumerate(pol_order):
        ax_in, ax_ideal, ax_out, ax_xcut, ax_ycut = axes[row_idx]
        I_out, metrics = result_map[pol]
        x_cut_norm = np.asarray(metrics["x_cut_norm"], dtype=np.float64)
        y_cut_norm = np.asarray(metrics["y_cut_norm"], dtype=np.float64)

        im0 = ax_in.imshow(I_in, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
        ax_in.set_title(f"{pol} | Input")
        ax_in.set_xlabel("x (pixel)")
        ax_in.set_ylabel("y (pixel)")
        plt.colorbar(im0, ax=ax_in, fraction=0.046, pad=0.04)

        im_ideal = ax_ideal.imshow(I_ideal, cmap="inferno", origin="upper", vmin=0.0, vmax=vmax)
        ax_ideal.set_title(
            f"Ideal output\neta_avg={ideal_metrics['eta_avg']:.4f}, eta_peak={ideal_metrics['eta_peak']:.4f}"
        )
        ax_ideal.set_xlabel("x (pixel)")
        ax_ideal.set_ylabel("y (pixel)")
        plt.colorbar(im_ideal, ax=ax_ideal, fraction=0.046, pad=0.04)

        im1 = ax_out.imshow(I_out, cmap="inferno", origin="upper", vmin=0.0, vmax=vmax)
        ax_out.set_title(
            f"{pol} | Output\neta_avg={metrics['eta_avg']:.4f}, eta_peak={metrics['eta_peak']:.4f}"
        )
        ax_out.set_xlabel("x (pixel)")
        ax_out.set_ylabel("y (pixel)")
        plt.colorbar(im1, ax=ax_out, fraction=0.046, pad=0.04)

        ax_xcut.plot(np.arange(x_cut_norm.size), x_cut_norm, color="tab:red", lw=1.6)
        ax_xcut.set_title(f"{pol} | X cut")
        ax_xcut.set_xlabel("x (pixel)")
        ax_xcut.set_ylabel("normalized intensity")
        ax_xcut.set_xlim(0, x_cut_norm.size - 1)
        ax_xcut.set_ylim(0.0, 1.05)
        ax_xcut.grid(True, alpha=0.25)

        ax_ycut.plot(np.arange(y_cut_norm.size), y_cut_norm, color="tab:blue", lw=1.6)
        ax_ycut.set_title(f"{pol} | Y cut")
        ax_ycut.set_xlabel("y (pixel)")
        ax_ycut.set_ylabel("normalized intensity")
        ax_ycut.set_xlim(0, y_cut_norm.size - 1)
        ax_ycut.set_ylim(0.0, 1.05)
        ax_ycut.grid(True, alpha=0.25)

    fig.suptitle(
        f"Fourier optics simulation across polarizations | "
        f"λ = {lambda_nm:.0f} nm, NA = {NA:.4f}, input = {pattern}, source = {input_label}"
    )
    fig.text(0.5, 0.02, "Columns: input, ideal output, output, centered x-cut, centered y-cut", ha="center", va="bottom", fontsize=11)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.show()


def run_single_pattern(
    pattern: str,
    pol: str,
    task_family: str,
    T_ss: np.ndarray,
    T_pp: np.ndarray,
    KX: np.ndarray,
    KY: np.ndarray,
    K0: float,
    K_MAX: float,
    lambda_nm: float,
    input_image: str | None,
    out_path: Path,
) -> dict[str, object] | dict[str, dict[str, object]]:
    if pol == "all":
        result_map: dict[str, tuple[np.ndarray, dict[str, object]]] = {}
        I_in_ref = None
        input_label_ref = ""
        I_ideal_ref = None
        ideal_metrics_ref = None
        for pol_name, e_in in POL_MAP.items():
            I_out, I_in, input_label = run_fourier_optics(
                T_ss, T_pp, e_in, KX, KY, pattern, K0, input_image=input_image
            )
            edge_mask = edge_mask_from_input(I_in)
            metrics = compute_imaging_metrics(I_in, I_out, edge_mask)
            metrics.update(compute_center_cut_summary(I_out))
            print(
                f"[metrics:{pattern}:{pol_name}] eta_peak={metrics['eta_peak']:.6f} "
                f"eta_avg={metrics['eta_avg']:.6f} edge_pixels={metrics['edge_pixels']}"
            )
            result_map[pol_name] = (I_out, metrics)
            if I_in_ref is None:
                I_in_ref = I_in
                input_label_ref = input_label
                I_ideal_ref, _, _, ideal_metrics_ref = run_ideal_reference(
                    task_family, KX, KY, K0, K_MAX, pattern, input_image=input_image
                )
        assert I_in_ref is not None
        assert I_ideal_ref is not None
        assert ideal_metrics_ref is not None
        plot_multi_pol_results(I_in_ref, I_ideal_ref, ideal_metrics_ref, result_map, lambda_nm, pattern, input_label_ref, out_path)
        return {pol_name: metric for pol_name, (_, metric) in result_map.items()}

    e_in = POL_MAP[pol]
    I_out, I_in, input_label = run_fourier_optics(
        T_ss, T_pp, e_in, KX, KY, pattern, K0, input_image=input_image
    )
    edge_mask = edge_mask_from_input(I_in)
    metrics = compute_imaging_metrics(I_in, I_out, edge_mask)
    metrics.update(compute_center_cut_summary(I_out))
    print(
        f"[metrics:{pattern}] eta_peak={metrics['eta_peak']:.6f} "
        f"eta_avg={metrics['eta_avg']:.6f} edge_pixels={metrics['edge_pixels']}"
    )
    plot_results(I_in, I_out, lambda_nm, pol, pattern, input_label, metrics, out_path)
    return metrics


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Fourier optics imaging simulation")
    parser.add_argument("--pol",         default="all",
                        choices=POL_CHOICES,
                        help="入射偏振；默认 all，一次输出 6 种偏振的组图")
    parser.add_argument("--lambda_nm",   type=float, default=1000.0,
                        help="工作波长 nm（默认 1000）")
    parser.add_argument("--ideal",       action="store_true",
                        help="使用理想二阶传递函数 T=(k_rho/k_max)²，验证仿真流程")
    parser.add_argument("--sample",      type=int, default=0,
                        help="训练样本索引（默认 0）")
    parser.add_argument("--from_infer",  action="store_true",
                        help="从 laplas 推理结果加载最优样本")
    parser.add_argument("--infer_dir",   default=None,
                        help="laplas 输出目录；不指定则自动选最新")
    parser.add_argument("--from_kspace", action="store_true",
                        help="从 scan_kspace.py 生成的 kspace_result.npz 加载 2D 传递函数")
    parser.add_argument("--kspace_npz",  default=None,
                        help="kspace_result.npz 路径；不指定则找同目录下的文件")
    parser.add_argument("--out",         default=None,
                        help="输出 PNG 路径；不指定时自动新建结果文件夹")
    parser.add_argument("--pattern",     default="all", choices=PATTERN_CHOICES,
                        help="输入图案；默认 all，一次输出 square / circle / checkerboard 三张组图")
    parser.add_argument("--input_image", default=None,
                        help="输入图片路径（PNG/JPG）；指定后优先使用图片像素生成输入图")
    args = parser.parse_args()

    # --- 波长相关参数 -------------------------------------------------------
    lambda_nm = args.lambda_nm
    K0        = 2 * np.pi / (lambda_nm * 1e-9)
    K_MAX     = K0 * NA
    lam_idx   = int(np.argmin(np.abs(LAMBDAS - lambda_nm)))
    print(f"λ = {lambda_nm:.0f} nm  →  LAMBDAS[{lam_idx}] = {LAMBDAS[lam_idx]:.0f} nm")
    print(f"NA = {NA:.4f}  (θ_max = {THETA_MAX}°)")

    # --- k 空间网格（所有分支共用）-----------------------------------------
    kx_arr = np.linspace(-K0, K0, NX)
    ky_arr = np.linspace(-K0, K0, NY)
    KX, KY = np.meshgrid(kx_arr, ky_arr)

    # --- 传递函数 -----------------------------------------------------------
    if args.ideal:
        T_ss, T_pp = make_ideal_transfer(KX, KY, K_MAX)
        src_label  = "ideal  T = (k_rho / k_max)²"
        task_tag = "ideal"
        task_family = "ideal"
        source_tag = "ideal_laplacian"
        output_base_dir = _HERE

    elif args.from_kspace:
        npz_path = Path(args.kspace_npz) if args.kspace_npz else latest_kspace_npz()
        if npz_path is None:
            print(f"找不到 {_HERE / 'kspace_result*.npz'}，请先运行 scan_kspace.py", file=sys.stderr)
            sys.exit(1)
        if not npz_path.exists():
            print(f"找不到 {npz_path}，请先运行 scan_kspace.py", file=sys.stderr)
            sys.exit(1)
        T_ss, T_pp = load_from_kspace(npz_path, KX, KY, K0, K_MAX)
        src_label  = f"kspace: {npz_path.name}"
        with np.load(npz_path) as d:
            structure_source = str(d["structure_source"][0]) if "structure_source" in d.files and len(d["structure_source"]) else ""
        structure_source_path = Path(structure_source) if structure_source else None
        task_tag = _task_tag_from_path(structure_source_path) if structure_source else _task_tag_from_path(npz_path)
        task_family = infer_task_family(structure_source_path if structure_source else npz_path, task_tag)
        source_tag = "from_kspace"
        output_base_dir = _results_case_dir_from_path(structure_source_path) or _HERE

    elif args.from_infer:
        if args.infer_dir is None:
            laplas_root = PROJECT_ROOT / "samples" / "laplas"
            runs = sorted(laplas_root.iterdir()) if laplas_root.exists() else []
            if not runs:
                print("找不到推理结果，请用 --sample", file=sys.stderr)
                sys.exit(1)
            args.infer_dir = runs[-1]
            print(f"[auto] infer_dir = {args.infer_dir}")
        t_ss_1d, t_pp_1d = load_from_infer(Path(args.infer_dir), lam_idx)
        T_ss = build_2d_transfer(t_ss_1d, KX, KY, K0, K_MAX)
        T_pp = build_2d_transfer(t_pp_1d, KX, KY, K0, K_MAX)
        src_label = f"infer: {Path(args.infer_dir).name}"
        task_tag = _task_tag_from_path(Path(args.infer_dir))
        task_family = infer_task_family(Path(args.infer_dir), task_tag)
        source_tag = f"infer_{Path(args.infer_dir).name}"
        output_base_dir = _results_case_dir_from_path(Path(args.infer_dir)) or _HERE

    else:
        t_ss_1d, t_pp_1d = load_from_training(args.sample, lam_idx)
        T_ss = build_2d_transfer(t_ss_1d, KX, KY, K0, K_MAX)
        T_pp = build_2d_transfer(t_pp_1d, KX, KY, K0, K_MAX)
        src_label = f"train sample {args.sample}"
        task_tag = "train"
        task_family = "generic"
        source_tag = f"sample_{args.sample}"
        output_base_dir = _HERE

    print(f"Source : {src_label}")

    pattern_list = list(DEFAULT_PATTERN_SET) if args.pattern == "all" else [args.pattern]
    if args.input_image is not None and args.pattern == "all":
        print("[warn] 指定 --input_image 时，默认批量图案模式将退化为单图输入。")
        pattern_list = ["square"]

    metrics_payload: dict[str, object] = {}

    # --- 仿真 + 画图 --------------------------------------------------------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(pattern_list) != 1:
            print("当使用 --out 时，--pattern 不能为 all。请指定单个 pattern。", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = make_run_output_dir(output_base_dir, task_tag, source_tag, lambda_nm, args.pol, args.pattern)
        print(f"[output_dir] {run_dir}")

    for pattern_name in pattern_list:
        if args.out:
            pattern_out_path = out_path
        else:
            suffix = "imaging_result_all.png" if args.pol == "all" else "imaging_result.png"
            pattern_out_path = run_dir / f"{pattern_name}_{suffix}"
        metrics_payload[pattern_name] = run_single_pattern(
            pattern_name, args.pol, task_family, T_ss, T_pp, KX, KY, K0, K_MAX, lambda_nm, args.input_image, pattern_out_path
        )

    if not args.out:
        save_metrics(run_dir / "metrics.json", metrics_payload if len(pattern_list) > 1 else metrics_payload[pattern_list[0]])


if __name__ == "__main__":
    main()
