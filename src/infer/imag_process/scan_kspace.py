# -*- coding: utf-8 -*-
"""
scan_kspace.py — 生成结构的 2D k 空间传递函数图（文献标准图）

原理
----
RCWA 直接提供若干 phi 方向上的 1D 角度扫描（θ ∈ [-40°,+40°]）。
若结构满足 C4 + 镜面对称，则只需在 phi ∈ [0°,45°] 的扇区内
确定传递函数；其余方位可通过对称折叠得到。
当前默认使用 phi = 0° / 22.5° / 45° 三条扫描线来重建该扇区。

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
THETAS  = np.arange(-40.0, 40.1,  5.0)     # [17]
THETA_MAX = 40.0
NA        = np.sin(np.radians(THETA_MAX))   # ≈ 0.6428

# k 空间图分辨率（与 image_processing.py 的频域重采样解耦）
N_GRID = 512


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
    if "laplas" in parts:
        i = parts.index("laplas")
        if i + 1 < len(parts):
            return _safe_tag(f"laplas_{parts[i + 1]}")
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


def make_output_paths(base_dir: Path, task_tag: str, mode_tag: str, lambda_nm: float) -> tuple[Path, Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_tag = _safe_tag(task_tag)
    mode_tag = _safe_tag(mode_tag)
    lam_tag = f"{int(round(lambda_nm))}nm"
    run_dir = base_dir / f"run_kspace_{task_tag}_{lam_tag}_{mode_tag}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task_tag}_{lam_tag}_{mode_tag}_{stamp}"
    return (
        run_dir / f"kspace_result_{prefix}.npz",
        run_dir / f"kspace_tpp_{prefix}.png",
        run_dir / f"kspace_tss_{prefix}.png",
    )


# ===========================================================================
# 1. 从现有数据 / 推理结果加载 phi=0 的传递函数
# ===========================================================================

def load_phi0_from_training(sample_idx: int, lambda_nm: float, data_file: Path | str | None = None):
    """返回 t_ss_phi0, t_pp_phi0，shape [17]，对应 THETAS。"""
    if data_file is None:
        data_file = DATA_PATH
    else:
        data_file = Path(data_file)

    data = np.load(data_file)
    lam_idx = int(np.argmin(np.abs(LAMBDAS - lambda_nm)))
    tpp = data["tpp_mag"][sample_idx, lam_idx, :].astype(np.float64)
    tss = data["tss_mag"][sample_idx, lam_idx, :].astype(np.float64)
    print(f"[phi=0] 从 {data_file.name} 加载 sample={sample_idx}, λ={LAMBDAS[lam_idx]:.0f}nm")
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


def load_structure_from_npy(path: Path | str, index: int = 0) -> np.ndarray:
    """
    从单结构或 top-k 结构文件中读取一个 64x64 结构，统一返回 float32 [64, 64]。

    支持形状:
      [64, 64]
      [1, 64, 64]
      [N, 64, 64]
      [N, 1, 64, 64]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到结构文件: {path}")

    arr = np.asarray(np.load(path), dtype=np.float32)

    if arr.ndim == 2:
        structure = arr
    elif arr.ndim == 3:
        if arr.shape[0] == 1 and arr.shape[1:] == (64, 64):
            structure = arr[0]
        else:
            if not (0 <= index < arr.shape[0]):
                raise IndexError(f"structure_index={index} 超出范围 [0, {arr.shape[0]})")
            structure = arr[index]
    elif arr.ndim == 4:
        if arr.shape[1] != 1 or arr.shape[2:] != (64, 64):
            raise ValueError(f"不支持的结构数组形状: {arr.shape}，期望 [N,1,64,64]")
        if not (0 <= index < arr.shape[0]):
            raise IndexError(f"structure_index={index} 超出范围 [0, {arr.shape[0]})")
        structure = arr[index, 0]
    else:
        raise ValueError(f"不支持的结构数组维度: {arr.shape}")

    if structure.shape != (64, 64):
        raise ValueError(f"结构应为 [64,64]，实际得到 {structure.shape}")

    vmin = float(np.min(structure))
    vmax = float(np.max(structure))
    if vmin < 0.0 or vmax > 1.0:
        raise ValueError(f"结构值域应在 [0,1]，实际范围 [{vmin:.4f}, {vmax:.4f}]")

    return structure.astype(np.float32)


# ===========================================================================
# 2. （可选）新跑 RCWA：phi=22.5° / 45° 扫描
# ===========================================================================

def scan_phi(structure: np.ndarray, lambda_nm: float, phi_deg: float, device: str = "cpu"):
    """
    对给定结构在指定 phi 角度扫描 17 个 θ，返回 t_ss, t_pp [17]。
    需要 torcwa 可用。
    """
    try:
        from dataset.rcwa.rcwa import torcwa_simulation
        import torch
    except ImportError:
        print(f"[phi={phi_deg}°] torcwa 不可用")
        return None, None

    import torch
    layer = torch.from_numpy(structure.astype(np.float32)).to(device)
    tpp_phi = np.full(len(THETAS), np.nan)
    tss_phi = np.full(len(THETAS), np.nan)

    for j, theta in enumerate(THETAS):
        try:
            out = torcwa_simulation(
                {
                    "periodicity": 500.0, "h": 500.0,
                    "lam": float(lambda_nm),
                    "tet": float(theta), "phi": float(phi_deg),
                    "angle_unit": "deg", "angle_layer": "input",
                    "input_medium": "air", "output_medium": "SiO2",
                    "structure": "Si",
                },
                layer, rcwa_orders=7, project=False, device=device,
            )
            tpp_phi[j] = float(out["tpp_mag"].detach().cpu().item())
            tss_phi[j] = float(out["tss_mag"].detach().cpu().item())
        except Exception as e:
            print(f"  [phi={phi_deg}°, θ={theta}°] 失败: {e}")

    print(f"[phi={phi_deg}°] RCWA 完成，tpp 范围=[{np.nanmin(tpp_phi):.3f}, {np.nanmax(tpp_phi):.3f}]")
    return tss_phi, tpp_phi


def scan_phi45(structure: np.ndarray, lambda_nm: float, device: str = "cpu"):
    """向后兼容：调用 scan_phi(..., phi_deg=45.0)"""
    return scan_phi(structure, lambda_nm, 45.0, device)


# ===========================================================================
# 3. 重建 2D k 空间传递函数
# ===========================================================================

def _get_1d_interp(t_1d: np.ndarray):
    """
    从 [17] 的角度扫描数据构建 t(θ) 插值器。
    输入: θ ∈ [-40,+40]，输出: 对任意 θ 插值（超出范围置0）。
    """
    from scipy.interpolate import interp1d
    return interp1d(
        THETAS, t_1d, kind="quadratic",
        bounds_error=False, fill_value=0.0,
    )


def build_2d_kspace(
    t_phi0: np.ndarray,     # [17]，phi=0° 传递函数
    t_phi22_5: np.ndarray | None = None,  # [17]，phi=22.5° 传递函数（可选）
    t_phi45: np.ndarray | None = None,    # [17]，phi=45° 传递函数（可选）
    n_grid: int = N_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 (kx_norm, ky_norm, T_2d)。
    支持 1、2 或 3 个 phi 方向的数据进行插值重建。

    kx_norm / ky_norm: 归一化坐标 kx/k0，范围 [-NA, NA]
    T_2d: [n_grid, n_grid]，圆外为 nan
    """
    from scipy.interpolate import interp1d

    # 收集可用的 phi 数据点
    phi_data = {}  # {phi_deg: t_1d_array}
    phi_data[0.0] = t_phi0
    if t_phi22_5 is not None:
        phi_data[22.5] = t_phi22_5
    if t_phi45 is not None:
        phi_data[45.0] = t_phi45

    phi_angles = sorted(phi_data.keys())

    # 为每个 phi 构建 theta 插值器
    interp_dict = {}
    for phi_deg in phi_angles:
        interp_dict[phi_deg] = _get_1d_interp(phi_data[phi_deg])

    # 归一化坐标网格
    ax = np.linspace(-NA, NA, n_grid)
    KXn, KYn = np.meshgrid(ax, ax)          # kx/k0, ky/k0

    k_rho_norm = np.sqrt(KXn**2 + KYn**2)   # k_rho / k0
    phi_rad    = np.arctan2(KYn, KXn)        # [-π, π]
    phi_deg    = np.degrees(phi_rad)          # [-180, 180]

    # 利用 C4 + 镜面对称性，折叠到 [0°, 45°]
    # C4: 每 90° 旋转对称 → 折叠到 [0°, 90°]
    # 镜像: 再折到 [0°, 45°]（45° 是该区间的中线）
    phi_fold = np.abs(phi_deg % 90.0)         # 折叠到 [0°, 90°]
    phi_fold = np.where(phi_fold > 45.0, 90.0 - phi_fold, phi_fold)  # 折叠到 [0°, 45°]

    # θ（polar angle）从 k_rho 得到
    k_rho_clipped = np.clip(k_rho_norm, 0.0, 1.0)
    theta_deg = np.degrees(np.arcsin(k_rho_clipped))  # [0°, 90°]，但有效范围 [0°, 40°]

    # 在每个 phi_fold 位置插值得到 T(phi, theta)。
    # 先沿每个 phi 方向对 theta 做二次插值；再在固定 theta（等价于固定 k_rho）
    # 的同一半径上做角向重建，尽量保持以径向项为主、角向项为低阶修正。
    if len(phi_angles) == 1:
        # 只有 phi=0，直接用
        T_2d = interp_dict[phi_angles[0]](theta_deg)
    elif len(phi_angles) == 2:
        # 两个方向时，使用满足 C4 对称的最低阶角向项：
        # T(phi, theta) = a0(theta) + a1(theta) * cos(4phi)
        t_a = interp_dict[phi_angles[0]](theta_deg)
        t_b = interp_dict[phi_angles[1]](theta_deg)

        if abs(phi_angles[1] - 45.0) < 0.1:  # 是 phi=45°
            phi_fold_rad = np.radians(phi_fold)
            cos4 = np.cos(4.0 * phi_fold_rad)
            a = 0.5 * (t_a + t_b)
            b = 0.5 * (t_a - t_b)
            T_2d = a + b * cos4
        else:
            # 若只有 0° + 22.5°，只能在线性角向插值的同时外推到 45°。
            # 这条分支保留兼容，但默认流程应尽量提供 45° 数据。
            alpha = (phi_fold - phi_angles[0]) / (phi_angles[1] - phi_angles[0])
            T_2d = (1.0 - alpha) * t_a + alpha * t_b
    else:
        # 三个方向时，用 C4 对称谐波展开：
        # T(phi, theta) = a0(theta) + a1(theta) * cos(4phi) + a2(theta) * cos(8phi)
        # 其中 a0 是各向同性的径向主项，更贴近二阶 Laplacian 的目标形态。
        t0 = interp_dict[0.0](theta_deg)
        t22 = interp_dict[22.5](theta_deg)
        t45 = interp_dict[45.0](theta_deg)
        phi_fold_rad = np.radians(phi_fold)
        cos4 = np.cos(4.0 * phi_fold_rad)
        cos8 = np.cos(8.0 * phi_fold_rad)

        a0 = 0.25 * (t0 + 2.0 * t22 + t45)
        a1 = 0.5 * (t0 - t45)
        a2 = 0.25 * (t0 - 2.0 * t22 + t45)
        T_2d = a0 + a1 * cos4 + a2 * cos8

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
    parser.add_argument("--structure_npy", default=None,
                        help="单结构或 top-k 结构 .npy 路径；指定后优先读取其中一个结构做 RCWA")
    parser.add_argument("--structure_index", type=int, default=0,
                        help="当 --structure_npy 为候选批次时，选择第几个结构（默认 0）")
    parser.add_argument("--n_grid",     type=int, default=N_GRID,
                        help="输出 k-space 网格分辨率（默认 512）")
    parser.add_argument("--data_file",  default=None,
                        help="npz 数据文件路径（默认 train_data.npz，可用 train_data_20000.npz）")
    parser.add_argument("--from_infer", action="store_true")
    parser.add_argument("--infer_dir",  default=None)
    parser.add_argument("--skip_phi45", action="store_true",
                        help="跳过默认的 phi=45° RCWA 扫描")
    parser.add_argument("--skip_phi22_5", action="store_true",
                        help="跳过默认的 phi=22.5° RCWA 扫描")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--cmap",       default="hot",
                        help="colormap，如 hot / inferno / turbo（默认 hot）")
    args = parser.parse_args()

    lam = args.lambda_nm
    run_phi22_5 = not args.skip_phi22_5
    run_phi45 = not args.skip_phi45

    # --- 加载 phi 数据 / 结构 ---
    if args.structure_npy is not None:
        if args.from_infer:
            print("[input] 同时指定了 --structure_npy 和 --from_infer，将优先使用 --structure_npy")
        structure_path = Path(args.structure_npy)
        if not structure_path.is_absolute():
            structure_path = PROJECT_ROOT / structure_path
        structure_for_phi_scan = load_structure_from_npy(structure_path, args.structure_index)
        task_tag = _task_tag_from_path(structure_path)
        mode_tag = f"struct_{structure_path.stem}_idx{args.structure_index}"
        label = f"{task_tag} | {mode_tag} | λ={lam:.0f}nm"
        output_base_dir = _results_case_dir_from_path(structure_path) or _HERE
        print(f"[structure] 从 {structure_path.name} 读取结构 index={args.structure_index}, shape={structure_for_phi_scan.shape}")

        print("[phi=0] 使用 RCWA 直接计算结构的 phi=0° 扫描")
        tss0, tpp0 = scan_phi(structure_for_phi_scan, lam, 0.0, args.device)
        if tss0 is None or tpp0 is None:
            print("无法对输入结构执行 phi=0° RCWA 扫描，请确认 torcwa 环境可用。", file=sys.stderr)
            sys.exit(1)

    elif args.from_infer:
        if args.infer_dir is None:
            laplas_root = PROJECT_ROOT / "samples" / "laplas"
            runs = sorted(laplas_root.iterdir()) if laplas_root.exists() else []
            if not runs:
                print("找不到推理结果，请用 --sample", file=sys.stderr); sys.exit(1)
            args.infer_dir = runs[-1]
            print(f"[auto] 使用推理目录: {args.infer_dir}")
        infer_path = Path(args.infer_dir)
        tss0, tpp0 = load_phi0_from_infer(infer_path, lam)
        structure_for_phi_scan = None
        task_tag = _task_tag_from_path(infer_path)
        mode_tag = "infer_phi0"
        label = f"{task_tag} | {mode_tag} | λ={lam:.0f}nm"
        output_base_dir = _results_case_dir_from_path(infer_path) or _HERE
    else:
        tss0, tpp0 = load_phi0_from_training(args.sample, lam, args.data_file)
        # 加载原始结构（用于跑 phi=22.5/45）
        data_path = Path(args.data_file) if args.data_file else DATA_PATH
        data = np.load(data_path)
        structure_for_phi_scan = data["structures"][args.sample]
        task_tag = _task_tag_from_path(data_path)
        mode_tag = f"train_sample_{args.sample}"
        label = f"{task_tag} | {mode_tag} | λ={lam:.0f}nm"
        output_base_dir = _results_case_dir_from_path(data_path) or _HERE

    # --- 扫描 phi=22.5° 和 phi=45° ---
    tss22_5, tpp22_5 = None, None
    tss45, tpp45 = None, None

    if structure_for_phi_scan is not None:
        if run_phi22_5:
            tss22_5, tpp22_5 = scan_phi(structure_for_phi_scan, lam, 22.5, args.device)
        if run_phi45:
            tss45, tpp45 = scan_phi(structure_for_phi_scan, lam, 45.0, args.device)
    else:
        if run_phi22_5 or run_phi45:
            print("[phi scan] 推理模式下无原始结构，跳过 phi=22.5°/45° 扫描，仅用 phi=0 数据重建")

    if not run_phi22_5 and not run_phi45:
        print("[phi scan] 已跳过 phi=22.5°/45°，使用 phi=0 数据近似（各向同性）")

    # --- 构建 2D k 空间 ---
    kx_norm, ky_norm, T_pp = build_2d_kspace(tpp0, tpp22_5, tpp45, n_grid=args.n_grid)
    _, _, T_ss = build_2d_kspace(tss0, tss22_5, tss45, n_grid=args.n_grid)

    # --- 保存 npz（供 image_processing.py 使用）---
    out_npz, out_tpp_png, out_tss_png = make_output_paths(output_base_dir, task_tag, mode_tag, lam)
    phi_angles_deg = [0.0]
    if tpp22_5 is not None and tss22_5 is not None:
        phi_angles_deg.append(22.5)
    if tpp45 is not None and tss45 is not None:
        phi_angles_deg.append(45.0)
    np.savez(
        out_npz,
        T_pp=T_pp, T_ss=T_ss,
        kx_norm=kx_norm, ky_norm=ky_norm,
        lambda_nm=np.array([lam]),
        NA=np.array([NA]),
        n_grid=np.array([args.n_grid]),
        phi_angles_deg=np.array(phi_angles_deg, dtype=np.float64),
        theta_angles_deg=np.array(THETAS, dtype=np.float64),
        structure_source=np.array([str(args.structure_npy) if args.structure_npy is not None else ""]),
        structure_index=np.array([args.structure_index], dtype=np.int64),
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
