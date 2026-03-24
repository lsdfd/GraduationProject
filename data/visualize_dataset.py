from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_STRUCTURES = ROOT / "structures" / "structures.npy"
DEFAULT_TRAIN = ROOT / "train_data.npz"


def default_structures_path() -> Path:
    return DEFAULT_STRUCTURES


def default_train_path() -> Path:
    return DEFAULT_TRAIN


def resolve_existing_train_path(path: Path | None) -> Path | None:
    if path is not None and path.exists():
        return path
    return DEFAULT_TRAIN if DEFAULT_TRAIN.exists() else None


def load_structures(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 structures 文件: {path}\n"
            f"默认路径: {DEFAULT_STRUCTURES}"
        )
    data = np.load(path)
    if isinstance(data, np.ndarray):
        structures = data
    else:
        if "structures" not in data.files:
            raise ValueError(f"{path} 不包含 structures 字段")
        structures = data["structures"]

    structures = np.asarray(structures)
    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures 应为 [N,64,64]，实际得到 {structures.shape}")
    return structures.astype(np.float32)


def load_tpp_mag(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 train_data 文件: {path}\n"
            f"默认路径: {DEFAULT_TRAIN}"
        )
    data = np.load(path)
    if not hasattr(data, "files"):
        raise ValueError(f"{path} 不是 npz 文件，无法读取 tpp")

    if "tpp_mag" in data.files:
        tpp = data["tpp_mag"]
    elif "tpp_real" in data.files and "tpp_imag" in data.files:
        tpp = np.sqrt(data["tpp_real"] ** 2 + data["tpp_imag"] ** 2)
    elif "tpp_real" in data.files:
        tpp = np.abs(data["tpp_real"])
    else:
        raise ValueError(f"{path} 不包含 tpp_mag 或 tpp_real/tpp_imag 字段")

    tpp = np.asarray(tpp)
    if tpp.ndim == 2:
        tpp = tpp[:, None, :]
    elif tpp.ndim != 3:
        raise ValueError(f"tpp 应为 [N,H,W] 或 [N,W]，实际得到 {tpp.shape}")
    return tpp.astype(np.float32)


def load_tpp_component(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    if not hasattr(data, "files"):
        raise ValueError(f"{path} 不是 npz 文件，无法读取 {key}")
    if key not in data.files:
        raise ValueError(f"{path} 不包含 {key} 字段")
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim != 3:
        raise ValueError(f"{key} 应为 [N,H,W] 或 [N,W]，实际得到 {arr.shape}")
    return arr


def load_required_spectrum(path: Path, key: str) -> np.ndarray:
    data = np.load(path)
    if not hasattr(data, "files") or key not in data.files:
        raise ValueError(f"{path} 不包含 {key} 字段。当前字段: {list(getattr(data, 'files', []))}")
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, None, :]
    elif arr.ndim != 3:
        raise ValueError(f"{key} 应为 [N,H,W] 或 [N,W]，实际得到 {arr.shape}")
    return arr


def load_lambda_theta(path: Path, tpp_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if hasattr(data, "files") and "thetas" in data.files:
        if "lambdas" in data.files:
            lambdas = np.asarray(data["lambdas"], dtype=np.float32)
        elif "target_lambda" in data.files:
            lambdas = np.asarray([float(data["target_lambda"])], dtype=np.float32)
        else:
            lambdas = np.arange(tpp_shape[0], dtype=np.float32)
        thetas = np.asarray(data["thetas"], dtype=np.float32)
        return lambdas, thetas

    lambda_count, theta_count = tpp_shape
    if (lambda_count, theta_count) == (11, 17):
        lambdas = np.arange(800.0, 1300.1, 50.0, dtype=np.float32)
        thetas = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)
        return lambdas, thetas

    lambdas = np.arange(lambda_count, dtype=np.float32)
    thetas = np.arange(theta_count, dtype=np.float32)
    return lambdas, thetas


def select_indices(count: int, limit: int, mode: str, values: np.ndarray | None = None) -> np.ndarray:
    if limit <= 0:
        limit = count
    limit = min(limit, count)
    if limit <= 0:
        raise ValueError("没有可用样本")

    if mode == "first":
        return np.arange(limit)
    if mode == "random":
        rng = np.random.default_rng(42)
        return rng.choice(count, size=limit, replace=False)
    if mode == "fill":
        if values is None:
            raise ValueError("fill 排序需要 values")
        return np.argsort(values)[:limit]
    if mode == "energy":
        if values is None:
            raise ValueError("energy 排序需要 values")
        return np.argsort(values)[::-1][:limit]

    raise ValueError(f"不支持的模式: {mode}")


def grid_shape(n: int, max_cols: int = 20) -> tuple[int, int]:
    cols = min(max_cols, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    return rows, cols


def normalize_per_sample(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)
    mins = np.nanmin(flat, axis=1)[:, None, None]
    maxs = np.nanmax(flat, axis=1)[:, None, None]
    denom = np.maximum(maxs - mins, 1e-8)
    out = (images - mins) / denom
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def chunk_indices(indices: np.ndarray, page_size: int) -> list[np.ndarray]:
    return [indices[i:i + page_size] for i in range(0, len(indices), page_size)]


def plot_structure_grid(structures: np.ndarray, out_path: Path, title: str) -> None:
    rows, cols = grid_shape(len(structures))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.0), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, image in zip(axes, structures):
        ax.imshow(image, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.axis("off")

    for ax in axes[len(structures):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum_grid(
    spec: np.ndarray,
    out_path: Path,
    title: str,
    vmin: float,
    vmax: float,
    normalize_mode: str,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    global_label: str = "magnitude",
) -> None:
    rows, cols = grid_shape(len(spec))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.1), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    if normalize_mode == "per_sample":
        images = normalize_per_sample(spec)
        plot_vmin, plot_vmax = 0.0, 1.0
        cbar_label = f"normalized {global_label} (per sample)"
    else:
        images = spec
        plot_vmin, plot_vmax = vmin, vmax
        cbar_label = global_label

    im = None
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    for ax, image in zip(axes, images):
        im = ax.imshow(
            image,
            cmap="turbo",
            aspect="auto",
            origin="lower",
            vmin=plot_vmin,
            vmax=plot_vmax,
            extent=extent,
            interpolation="bicubic",
        )
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.55, pad=0.01, label=cbar_label)

    fig.suptitle(title, fontsize=14)
    fig.text(
        0.02,
        0.02,
        f"x-axis: theta (deg), y-axis: lambda (nm), colors: {cbar_label}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_structure_pages(
    structures: np.ndarray,
    ordered_indices: np.ndarray,
    out_dir: Path,
    prefix: str,
    page_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for page_id, page_indices in enumerate(chunk_indices(ordered_indices, page_size), start=1):
        plot_structure_grid(
            structures[page_indices],
            out_dir / f"page_{page_id:02d}.png",
            f"{prefix} page {page_id} ({len(page_indices)})",
        )


def save_spectrum_pages(
    spec: np.ndarray,
    ordered_indices: np.ndarray,
    out_dir: Path,
    prefix: str,
    page_size: int,
    normalize_mode: str,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    finite_mask = np.isfinite(spec)
    if finite_mask.any():
        global_vmin = float(np.nanmin(spec))
        global_vmax = float(np.nanmax(spec))
    else:
        global_vmin, global_vmax = 0.0, 1.0
        print(f"[warn] {prefix}: all values are NaN, plotting as empty maps")

    if global_vmax <= global_vmin:
        global_vmax = global_vmin + 1e-6

    # For transmission-magnitude visualization, keep the color scale in the
    # usual [0, 1] range and saturate rare outliers above 1.0.
    global_vmin = max(global_vmin, 0.0)
    global_vmax = min(global_vmax, 1.0)
    if global_vmax <= global_vmin:
        global_vmax = global_vmin + 1e-6

    for page_id, page_indices in enumerate(chunk_indices(ordered_indices, page_size), start=1):
        page_spec = np.nan_to_num(spec[page_indices], nan=global_vmin)
        page_spec = np.clip(page_spec, global_vmin, global_vmax)
        title = f"{prefix} page {page_id} ({len(page_indices)})"
        if normalize_mode == "global":
            title += f" | color: blue={global_vmin:.4g}, red={global_vmax:.4g}"
        else:
            title += " | color: blue=0, red=1 (per-sample)"
        plot_spectrum_grid(
            page_spec,
            out_dir / f"page_{page_id:02d}.png",
            title,
            vmin=global_vmin,
            vmax=global_vmax,
            normalize_mode=normalize_mode,
            lambdas=lambdas,
            thetas=thetas,
            global_label=label,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="拼图查看 structures / tpp / tss 数据集分布")
    parser.add_argument("--structures", type=Path, default=default_structures_path())
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=ROOT / "vis")
    parser.add_argument("--num_structures", type=int, default=1000, help="结构可视化数量；<=0 表示全部")
    parser.add_argument("--num_tpp", type=int, default=1000, help="tpp/tss 可视化数量；<=0 表示全部")
    parser.add_argument("--page_size", type=int, default=100)
    parser.add_argument("--tpp_norm", choices=["global", "per_sample"], default="global")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    structures_out = args.out_dir / "structures"
    tpp_out = args.out_dir / "tpp"
    tss_out = args.out_dir / "tss"
    args.train = resolve_existing_train_path(args.train)
    print(f"[input] structures: {args.structures}")
    print(f"[input] train: {args.train}")

    structures = load_structures(args.structures)
    fill_ratio = structures.mean(axis=(1, 2))

    idx_random = select_indices(len(structures), args.num_structures, mode="random")
    idx_fill = select_indices(len(structures), args.num_structures, mode="fill", values=fill_ratio)

    save_structure_pages(
        structures,
        idx_random,
        structures_out / "random",
        "structures_random_sample",
        args.page_size,
    )
    save_structure_pages(
        structures,
        idx_fill,
        structures_out / "sorted_by_fill",
        "structures_sorted_by_fill_ratio",
        args.page_size,
    )

    if args.num_tpp == 0:
        print("[info] --num_tpp=0，仅输出结构拼图，不读取 train_data.npz")
        return
    if args.train is None:
        print("[info] 未找到 train_data.npz，仅输出结构拼图")
        return

    tpp_mag = load_required_spectrum(args.train, "tpp_mag")
    tss_mag = load_required_spectrum(args.train, "tss_mag")
    lambdas, thetas = load_lambda_theta(args.train, tpp_mag.shape[1:])
    tpp_energy = np.nanmean(tpp_mag, axis=(1, 2))
    tss_energy = np.nanmean(tss_mag, axis=(1, 2))
    tpp_energy = np.nan_to_num(tpp_energy, nan=0.0)
    tss_energy = np.nan_to_num(tss_energy, nan=0.0)
    if len(structures) != len(tpp_mag):
        print(f"[warn] structures count ({len(structures)}) != tpp/tss count ({len(tpp_mag)})")

    idx_tpp_random = select_indices(len(tpp_mag), args.num_tpp, mode="random")
    idx_tpp_energy = select_indices(len(tpp_mag), args.num_tpp, mode="energy", values=tpp_energy)
    fill_count = min(len(fill_ratio), len(tpp_mag))
    idx_tpp_fill = select_indices(fill_count, args.num_tpp, mode="fill", values=fill_ratio[:fill_count])
    idx_tss_random = select_indices(len(tss_mag), args.num_tpp, mode="random")
    idx_tss_energy = select_indices(len(tss_mag), args.num_tpp, mode="energy", values=tss_energy)
    idx_tss_fill = select_indices(min(len(fill_ratio), len(tss_mag)), args.num_tpp, mode="fill", values=fill_ratio[: min(len(fill_ratio), len(tss_mag))])

    save_spectrum_pages(
        tpp_mag,
        idx_tpp_random,
        tpp_out / "random",
        "tpp_random_sample",
        args.page_size,
        args.tpp_norm,
        lambdas,
        thetas,
        "tpp magnitude",
    )
    save_spectrum_pages(
        tpp_mag,
        idx_tpp_energy,
        tpp_out / "sorted_by_mean",
        "tpp_sorted_by_mean_magnitude",
        args.page_size,
        args.tpp_norm,
        lambdas,
        thetas,
        "tpp magnitude",
    )
    save_spectrum_pages(
        tpp_mag,
        idx_tpp_fill,
        tpp_out / "sorted_by_fill",
        "tpp_sorted_by_structure_fill_ratio",
        args.page_size,
        args.tpp_norm,
        lambdas,
        thetas,
        "tpp magnitude",
    )
    save_spectrum_pages(
        tss_mag,
        idx_tss_random,
        tss_out / "random",
        "tss_random_sample",
        args.page_size,
        args.tpp_norm,
        lambdas,
        thetas,
        "tss magnitude",
    )
    save_spectrum_pages(
        tss_mag,
        idx_tss_energy,
        tss_out / "sorted_by_mean",
        "tss_sorted_by_mean_magnitude",
        args.page_size,
        args.tpp_norm,
        lambdas,
        thetas,
        "tss magnitude",
    )
    save_spectrum_pages(
        tss_mag,
        idx_tss_fill,
        tss_out / "sorted_by_fill",
        "tss_sorted_by_structure_fill_ratio",
        args.page_size,
        args.tpp_norm,
        lambdas,
        thetas,
        "tss magnitude",
    )

    print(f"structures file: {args.structures}")
    print(f"structures loaded: {len(structures)} | plotted: {len(idx_random)} random, {len(idx_fill)} sorted")
    print(f"train file: {args.train}")
    print(f"tpp loaded: {len(tpp_mag)} | plotted: {len(idx_tpp_random)} random, {len(idx_tpp_energy)} sorted")
    print(f"tss loaded: {len(tss_mag)} | plotted: {len(idx_tss_random)} random, {len(idx_tss_energy)} sorted")
    print(f"page size: {args.page_size}")
    print(f"saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
