# -*- coding: utf-8 -*-
"""
scan_kspace.py — 生成结构的 2D k 空间传递函数图（文献标准图）

原理
----
RCWA 只扫了 phi=0 平面（θ ∈ [-60°,+60°]）。
利用结构的 C4 + σx 对称性，只需额外扫 phi=45° 一条线，
再在 [0°,45°] 之间插值，就能重建完整的 1/8 扇区，
然后通过对称操作填满整个圆盘。

Usage
-----
  # 从 train_data.npz 取第0个样本，λ=1000nm
  python scan_kspace.py --sample 0

  # 从 laplas 推理结果取最优样本
  python scan_kspace.py --from_infer

  # 指定波长（nm）
  python scan_kspace.py --sample 0 --lambda_nm 1000

输出
----
  kspace_result.npz   — T_pp, T_ss [N_grid, N_grid] + 坐标轴
  kspace_tpp.png      — |t_pp| 文献风格图
  kspace_tss.png      — |t_ss| 文献风格图
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_PATH = PROJECT_ROOT / "data" / "train_data.npz"

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
LAMBDAS = np.arange(800.0, 1300.1, 50.0)   # [11]
THETAS  = np.arange(-60.0, 60.1, 10.0)     # [13]
THETA_MAX = 60.0
NA        = np.sin(np.radians(THETA_MAX))   # ≈ 0.8660

# k 空间图分辨率（越大越细腻，但插值更慢）
N_GRID = 300


def _safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def make_output_paths(label: str, lambda_nm: float) -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_tag = _safe_tag(label)
    lam_tag = f"{int(round(lambda_nm))}nm"
    run_dir = _HERE / f"run_kspace_{label_tag}_{lam_tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{label_tag}_{lam_tag}_{stamp}"
    return (
        run_dir / f"kspace_result_{prefix}.npz",
        run_dir / f"kspace_tpp_{prefix}.png",
        run_dir / f"kspace_tss_{prefix}.png",
    )


# ===========================================================================
# 1. 从现有数据 / 推理结果加载 phi=0 的传递函数
# ===========================================================================

def load_phi0_from_training(sample_idx: int, lambda_nm: float):
    """返回 t_ss_phi0, t_pp_phi0，shape [17]，对应 THETAS。"""
    data = np.load(DATA_PATH)
    lam_idx = int(np.argmin(np.abs(LAMBDAS - lambda_nm)))
    tpp = data["tpp_mag"][sample_idx, lam_idx, :].astype(np.float64)
    tss = data["tss_mag"][sample_idx, lam_idx, :].astype(np.float64)
    print(f"[phi=0] 从 train_data 加载 sample={sample_idx}, λ={LAMBDAS[lam_idx]:.0f}nm")
    return tss, tpp


def load_phi0_from_infer(infer_dir: Path, lambda_nm: float):
    """从 laplas 推理结果加载最优样本的 phi=0 数据。"""
    lam_idx = int(np.argmin(np.abs(LAMBDAS - lambda_nm)))
    pred = np.load(infer_dir / "all_pred_cond_raw.npy")   # [N, 2, 11, 17]
    errors = np.load(infer_dir / "all_errors.npy")
    best = int(np.argmin(errors))
    tss = pred[best, 1, lam_idx, :].astype(np.float64)
    tpp = pred[best, 0, lam_idx, :].astype(np.float64)
    print(f"[phi=0] 从推理结果加载 best={best}, error={errors[best]:.4f}, λ={LAMBDAS[lam_idx]:.0f}nm")
    return tss, tpp


# ===========================================================================
# 2. （可选）新跑 RCWA：phi=45° 扫描
# ===========================================================================

def scan_phi45(structure: np.ndarray, lambda_nm: float, device: str = "cpu"):
    """
    对给定结构在 phi=45° 扫描 17 个 θ，返回 t_ss_phi45, t_pp_phi45 [17]。
    需要 torcwa 可用。
    """
    try:
        from dataset.rcwa.rcwa import torcwa_simulation
        import torch
    except ImportError:
        print("[phi=45] torcwa 不可用，将用 phi=0 数据近似（各向同性假设）")
        return None, None

    import torch
    layer = torch.from_numpy(structure.astype(np.float32)).to(device)
    tpp45 = np.full(len(THETAS), np.nan)
    tss45 = np.full(len(THETAS), np.nan)

    for j, theta in enumerate(THETAS):
        try:
            out = torcwa_simulation(
                {
                    "periodicity": 500.0, "h": 500.0,
                    "lam": float(lambda_nm),
                    "tet": float(theta), "phi": 45.0,
                    "angle_unit": "deg", "angle_layer": "input",
                    "input_medium": "air", "output_medium": "SiO2",
                    "structure": "Si",
                },
                layer, rcwa_orders=7, project=False, device=device,
            )
            tpp45[j] = float(out["tpp_mag"].detach().cpu().item())
            tss45[j] = float(out["tss_mag"].detach().cpu().item())
        except Exception as e:
            print(f"  [phi=45, θ={theta}°] 失败: {e}")

    print(f"[phi=45] RCWA 完成，tpp 范围=[{np.nanmin(tpp45):.3f}, {np.nanmax(tpp45):.3f}]")
    return tss45, tpp45


# ===========================================================================
# 3. 重建 2D k 空间传递函数
# ===========================================================================

def _get_1d_interp(t_1d: np.ndarray):
    """
    从 [17] 的角度扫描数据构建 t(θ) 插值器。
    输入: θ ∈ [-60,+60]，输出: 对任意 θ 插值（超出范围置0）。
    """
    from scipy.interpolate import interp1d
    return interp1d(
        THETAS, t_1d, kind="linear",
        bounds_error=False, fill_value=0.0,
    )


def build_2d_kspace(
    t_phi0: np.ndarray,     # [17]，phi=0° 传递函数
    t_phi45: np.ndarray,    # [17]，phi=45° 传递函数（若为 None 则用 phi=0 近似）
    n_grid: int = N_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (kx_norm, ky_norm, T_2d)。
    kx_norm / ky_norm: 归一化坐标 kx/k0，范围 [-NA, NA]
    T_2d: [n_grid, n_grid]，圆外为 nan
    """
    interp0  = _get_1d_interp(t_phi0)
    interp45 = _get_1d_interp(t_phi45) if t_phi45 is not None else interp0

    # 归一化坐标网格
    ax = np.linspace(-NA, NA, n_grid)
    KXn, KYn = np.meshgrid(ax, ax)          # kx/k0, ky/k0

    k_rho_norm = np.sqrt(KXn**2 + KYn**2)   # k_rho / k0
    phi_rad    = np.arctan2(KYn, KXn)        # [-π, π]
    phi_deg    = np.degrees(phi_rad)          # [-180, 180]

    # 利用 C4+σx 对称性，折叠到 [0°, 45°]
    # C4: 每 90° 旋转对称 → 折叠到 [0°, 90°]
    # σx: kx 轴镜像      → 折叠到 [0°, 45°]（因为 45° 是 [0,90] 的中线）
    phi_fold = np.abs(phi_deg % 90.0)         # 折叠到 [0°, 90°]
    phi_fold = np.where(phi_fold > 45.0, 90.0 - phi_fold, phi_fold)  # 折叠到 [0°, 45°]

    # θ（polar angle）从 k_rho 得到
    k_rho_clipped = np.clip(k_rho_norm, 0.0, 1.0)
    theta_deg = np.degrees(np.arcsin(k_rho_clipped))  # [0°, 90°]，但有效范围 [0°, 60°]

    t_at_phi0  = interp0(theta_deg)
    t_at_phi45 = interp45(theta_deg)

    # 用最简单的 C4 谐波重建角向响应，而不是在 0° 和 45° 之间做线性插值：
    # T(phi, theta) = A(theta) + B(theta) cos(4phi)
    # 其中：
    #   phi=0°  → cos(0)=1   → T = T_phi0
    #   phi=45° → cos(pi)=-1 → T = T_phi45
    # 这样能保持四重对称并让角向过渡更圆滑。
    phi_fold_rad = np.radians(phi_fold)
    cos4 = np.cos(4.0 * phi_fold_rad)
    a = 0.5 * (t_at_phi0 + t_at_phi45)
    b = 0.5 * (t_at_phi0 - t_at_phi45)
    T_2d = a + b * cos4
    T_2d = np.clip(T_2d, 0.0, 1.0)

    # 圆外（k_rho > NA）置 nan（mask 掉）
    T_2d[k_rho_norm > NA] = np.nan
    # k_rho > 1（倏逝波）置 nan（理论上和 >NA 重叠，但保险起见）
    T_2d[k_rho_norm > 1.0] = np.nan

    return ax, ax, T_2d


# ===========================================================================
# 4. 画文献标准图
# ===========================================================================

def plot_kspace(
    kx_norm: np.ndarray, ky_norm: np.ndarray,
    T_2d: np.ndarray,
    title: str, save_path: Path,
    cmap: str = "hot",
    vmin: float = 0.0, vmax: float = 1.0,
):
    fig, ax = plt.subplots(figsize=(5, 4.5))

    im = ax.imshow(
        T_2d,
        extent=[kx_norm[0], kx_norm[-1], ky_norm[0], ky_norm[-1]],
        origin="lower",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="bilinear",
    )

    # NA 圆边界
    theta_circle = np.linspace(0, 2 * np.pi, 500)
    ax.plot(NA * np.cos(theta_circle), NA * np.sin(theta_circle),
            "w--", lw=0.8, alpha=0.6)

    # 极坐标网格线（仿文献）
    for r_norm in [0.2, 0.4, 0.6]:
        if r_norm <= NA:
            ax.plot(r_norm * np.cos(theta_circle), r_norm * np.sin(theta_circle),
                    color="white", lw=0.4, alpha=0.3)
    for ang in np.arange(0, 360, 30):
        rad = np.radians(ang)
        ax.plot([0, NA * np.cos(rad)], [0, NA * np.sin(rad)],
                color="white", lw=0.4, alpha=0.3)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$|t|$", fontsize=12)
    cb.set_ticks([0, 0.5, 1.0])

    ax.set_xlabel(r"$k_x/k_0$", fontsize=13)
    ax.set_ylabel(r"$k_y/k_0$", fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=11)

    # 坐标轴刻度归一化
    ticks = [-NA, -NA/2, 0, NA/2, NA]
    labels = [f"{v:.2f}" for v in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close()


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="k 空间传递函数 2D 图")
    parser.add_argument("--sample",     type=int, default=0)
    parser.add_argument("--lambda_nm",  type=float, default=1000.0)
    parser.add_argument("--from_infer", action="store_true")
    parser.add_argument("--infer_dir",  default=None)
    parser.add_argument("--run_phi45",  action="store_true",
                        help="实际跑 phi=45° RCWA（需要 torcwa）；不加则用 phi=0 近似各向同性")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--cmap",       default="hot",
                        help="colormap，如 hot / inferno / turbo（默认 hot）")
    args = parser.parse_args()

    lam = args.lambda_nm

    # --- 加载 phi=0 数据 ---
    if args.from_infer:
        if args.infer_dir is None:
            laplas_root = PROJECT_ROOT / "samples" / "laplas"
            runs = sorted(laplas_root.iterdir()) if laplas_root.exists() else []
            if not runs:
                print("找不到推理结果，请用 --sample", file=sys.stderr); sys.exit(1)
            args.infer_dir = runs[-1]
            print(f"[auto] 使用推理目录: {args.infer_dir}")
        tss0, tpp0 = load_phi0_from_infer(Path(args.infer_dir), lam)
        structure_for_phi45 = None   # 推理模式下没有结构原始 npy，phi45 需另外处理
        label = f"infer λ={lam:.0f}nm"
    else:
        tss0, tpp0 = load_phi0_from_training(args.sample, lam)
        # 加载原始结构（用于跑 phi=45）
        data = np.load(DATA_PATH)
        structure_for_phi45 = data["structures"][args.sample]
        label = f"sample={args.sample} λ={lam:.0f}nm"

    # --- phi=45° 数据 ---
    if args.run_phi45 and structure_for_phi45 is not None:
        tss45, tpp45 = scan_phi45(structure_for_phi45, lam, args.device)
    else:
        if args.run_phi45:
            print("[phi=45] 推理模式下无原始结构，跳过，使用各向同性近似")
        else:
            print("[phi=45] 未指定 --run_phi45，使用 phi=0 数据近似（各向同性）")
        tss45, tpp45 = None, None

    # --- 构建 2D k 空间 ---
    kx_norm, ky_norm, T_pp = build_2d_kspace(tpp0, tpp45)
    _, _, T_ss = build_2d_kspace(tss0, tss45)

    # --- 保存 npz（供 image_processing.py 使用）---
    out_npz, out_tpp_png, out_tss_png = make_output_paths(label, lam)
    np.savez(
        out_npz,
        T_pp=T_pp, T_ss=T_ss,
        kx_norm=kx_norm, ky_norm=ky_norm,
        lambda_nm=np.array([lam]),
        NA=np.array([NA]),
    )
    print(f"Saved → {out_npz}")

    # --- 画图 ---
    plot_kspace(
        kx_norm, ky_norm, T_pp,
        title=rf"$|t_{{pp}}|$   {label}",
        save_path=out_tpp_png,
        cmap=args.cmap,
    )
    plot_kspace(
        kx_norm, ky_norm, T_ss,
        title=rf"$|t_{{ss}}|$   {label}",
        save_path=out_tss_png,
        cmap=args.cmap,
    )

    # --- 同时更新 image_processing.py 的传递函数（如果存在）---
    print("\n提示：可将 kspace_result.npz 中的 T_pp/T_ss 直接传入 image_processing.py")
    print("      替换 build_2d_transfer() 的输出，使用更精确的 2D 传递函数。")


if __name__ == "__main__":
    main()
