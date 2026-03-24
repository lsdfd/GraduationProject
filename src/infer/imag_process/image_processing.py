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
  python image_processing.py --ideal --pol x --lambda_nm 1000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import json
from scipy.ndimage import binary_dilation
from scipy.interpolate import interp1d, RegularGridInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]
DATA_PATH    = PROJECT_ROOT / "data" / "train_data.npz"

# ---------------------------------------------------------------------------
# Fixed constants
# ---------------------------------------------------------------------------
THETA_MAX = 40.0
NA        = np.sin(np.radians(THETA_MAX))    # ≈ 0.6428
LAMBDAS   = np.arange(800.0, 1300.1, 50.0)  # [11]
THETAS    = np.arange(-40.0, 40.1,  5.0)    # [17]
NX, NY    = 502, 502

POL_MAP: dict[str, np.ndarray] = {
    "x":    np.array([1.0,  0.0]),
    "y":    np.array([0.0,  1.0]),
    "x45":  np.array([1.0,  1.0]) / np.sqrt(2),
    "x-45": np.array([1.0, -1.0]) / np.sqrt(2),
    "RCP":  np.array([1.0, -1j])  / np.sqrt(2),
    "LCP":  np.array([1.0,  1j])  / np.sqrt(2),
}

PATTERN_CHOICES = ("square", "thu", "square_thu")


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def make_run_output_dir(src_label: str, lambda_nm: float, polarization: str, pattern: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_tag = _safe_tag(src_label)
    pol_tag = _safe_tag(polarization)
    pattern_tag = _safe_tag(pattern)
    lam_tag = f"{int(round(lambda_nm))}nm"
    out_dir = _HERE / f"run_{src_tag}_{lam_tag}_{pol_tag}_{pattern_tag}_{stamp}"
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


def build_input_image(pattern: str) -> np.ndarray:
    img = np.zeros((NY, NX), dtype=np.float64)
    cx, cy = NX // 2, NY // 2

    # 保留原始白色方块，便于和历史结果直接对比。
    sq = min(NX, NY) // 16
    hs = sq // 2
    _draw_rect(img, cx - hs, cx + hs, cy - hs, cy + hs)

    if pattern in {"thu", "square_thu"}:
        stroke = max(14, NX // 26)
        letter_h = max(120, NY // 2)
        letter_w = max(52, NX // 9)
        gap = max(18, NX // 30)
        margin = max(20, NX // 18)
        top = cy - letter_h // 2
        bottom = top + letter_h
        total_w = 3 * letter_w + 2 * gap
        left_t = cx - total_w // 2
        left_h = left_t + letter_w + gap
        left_u = left_h + letter_w + gap

        # T
        _draw_rect(img, left_t, left_t + letter_w, top, top + stroke)
        _draw_rect(img, left_t + letter_w // 2 - stroke // 2, left_t + letter_w // 2 + (stroke + 1) // 2, top, bottom)

        # H
        _draw_rect(img, left_h, left_h + stroke, top, bottom)
        _draw_rect(img, left_h + letter_w - stroke, left_h + letter_w, top, bottom)
        _draw_rect(img, left_h, left_h + letter_w, cy - stroke // 2, cy + (stroke + 1) // 2)

        # U
        _draw_rect(img, left_u, left_u + stroke, top, bottom - margin)
        _draw_rect(img, left_u + letter_w - stroke, left_u + letter_w, top, bottom - margin)
        _draw_rect(img, left_u, left_u + letter_w, bottom - stroke, bottom)

    return img

def run_fourier_optics(T_ss: np.ndarray, T_pp: np.ndarray,
                       e_in: np.ndarray,
                       KX: np.ndarray, KY: np.ndarray,
                       pattern: str,
                       K0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    角谱法 Jones 矩阵仿真。
    返回 (I_out [NY,NX], I_in [NY,NX])。
    """
    I_in = build_input_image(pattern)
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
    return I_out, f_in


# ===========================================================================
# 4. 可视化
# ===========================================================================

def plot_results(I_in: np.ndarray, I_out: np.ndarray,
                 T_ss: np.ndarray, T_pp: np.ndarray,
                 KX: np.ndarray, K_MAX: float,
                 lambda_nm: float, polarization: str, pattern: str,
                 edge_mask: np.ndarray, metrics: dict[str, float],
                 save_path: Path) -> None:
    roi  = slice(150, 352)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # 输入图像
    im0 = axes[0].imshow(I_in[roi, roi], cmap="gray", origin="upper")
    axes[0].set_title(f"Input image ({pattern})")
    axes[0].set_xlabel("x (pixel)")
    axes[0].set_ylabel("y (pixel)")
    plt.colorbar(im0, ax=axes[0])

    # 期望边缘区域
    im_edge = axes[1].imshow(edge_mask[roi, roi].astype(float), cmap="viridis", origin="upper", vmin=0.0, vmax=1.0)
    axes[1].set_title("Expected edge region")
    axes[1].set_xlabel("x (pixel)")
    axes[1].set_ylabel("y (pixel)")
    plt.colorbar(im_edge, ax=axes[1])

    # 输出强度
    vmax = float(np.percentile(I_out[roi, roi], 99)) or 1.0
    im1  = axes[2].imshow(I_out[roi, roi], cmap="inferno",
                           origin="upper", vmin=0, vmax=vmax)
    axes[2].set_title(f"Output intensity  (pol={polarization})")
    axes[2].set_xlabel("x (pixel)")
    axes[2].set_ylabel("y (pixel)")
    plt.colorbar(im1, ax=axes[2])

    # 传递函数截面（沿 kx，归一化到 k_max）
    half    = NX // 2
    kx_norm = KX[NY // 2, half:] / K_MAX
    axes[3].plot(kx_norm, T_ss[NY // 2, half:], "b-",  lw=1.5, label=r"$|t_{ss}|$")
    axes[3].plot(kx_norm, T_pp[NY // 2, half:], "r--", lw=1.5, label=r"$|t_{pp}|$")
    axes[3].axvline(1.0, color="gray", lw=0.8, ls=":", label="NA boundary")
    axes[3].set_xlabel(r"$k_x\,/\,k_\mathrm{max}$")
    axes[3].set_ylabel("Transmission amplitude")
    axes[3].set_xlim([0, 1.3])
    axes[3].set_ylim([0, None])
    axes[3].legend()
    axes[3].grid(True, alpha=0.4)
    axes[3].set_title("Transfer function (kx cross-section)")

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


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Fourier optics imaging simulation")
    parser.add_argument("--pol",         default="x",
                        choices=list(POL_MAP.keys()),
                        help="入射偏振（默认 x = p 偏振）")
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
    parser.add_argument("--pattern",     default="square_thu", choices=PATTERN_CHOICES,
                        help="输入图案：square / thu / square_thu（默认 square_thu）")
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

    else:
        t_ss_1d, t_pp_1d = load_from_training(args.sample, lam_idx)
        T_ss = build_2d_transfer(t_ss_1d, KX, KY, K0, K_MAX)
        T_pp = build_2d_transfer(t_pp_1d, KX, KY, K0, K_MAX)
        src_label = f"train sample {args.sample}"

    print(f"Source : {src_label}")

    # --- 仿真 ---------------------------------------------------------------
    e_in  = POL_MAP[args.pol]
    I_out, I_in = run_fourier_optics(T_ss, T_pp, e_in, KX, KY, args.pattern, K0)
    edge_mask = edge_mask_from_input(I_in)
    metrics = compute_imaging_metrics(I_in, I_out, edge_mask)
    print(f"[metrics] eta_peak={metrics['eta_peak']:.6f} eta_avg={metrics['eta_avg']:.6f} edge_pixels={metrics['edge_pixels']}")

    # --- 画图 ---------------------------------------------------------------
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = make_run_output_dir(src_label, lambda_nm, args.pol, args.pattern)
        out_path = run_dir / "imaging_result.png"
        save_metrics(run_dir / "metrics.json", metrics)
        print(f"[output_dir] {run_dir}")
    plot_results(I_in, I_out, T_ss, T_pp, KX, K_MAX,
                 lambda_nm, args.pol, args.pattern, edge_mask, metrics, out_path)


if __name__ == "__main__":
    main()
