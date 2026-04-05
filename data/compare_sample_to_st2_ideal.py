from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C0 = 299_792_458.0


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def robust_norm(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    scale = float(np.max(np.abs(x)))
    if scale <= eps:
        return np.zeros_like(x, dtype=np.float64)
    return x / scale


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na <= eps or nb <= eps:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def fit_scale_and_r2(y: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> tuple[float, float]:
    yy = np.asarray(y, dtype=np.float64).ravel()
    tt = np.asarray(target, dtype=np.float64).ravel()
    denom = float(np.dot(tt, tt))
    if denom <= eps:
        return 0.0, 0.0
    coef = max(float(np.dot(yy, tt) / denom), 0.0)
    fit = coef * tt
    ss_res = float(np.sum((yy - fit) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot <= eps:
        return coef, 0.0
    return coef, float(1.0 - ss_res / ss_tot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare one sample's local map against an ideal mixed second-order ST template.")
    parser.add_argument("--npz", type=str, default="data/train_data_20000.npz")
    parser.add_argument("--sample_idx", type=int, default=10546)
    parser.add_argument("--channel", choices=["tpp", "tss"], default="tpp")
    parser.add_argument("--lambda0_nm", type=float, default=1050.0)
    parser.add_argument("--lambda_window_nm", type=float, default=100.0)
    parser.add_argument("--theta_limit_deg", type=float, default=30.0)
    parser.add_argument("--out_png", type=str, default="data/sample_10546_tpp_vs_st2_ideal.png")
    parser.add_argument("--out_json", type=str, default="data/sample_10546_tpp_vs_st2_ideal.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    out_png = resolve_from_root(args.out_png)
    out_json = resolve_from_root(args.out_json)

    data = np.load(npz_path)
    channel_key = "tss_mag" if args.channel == "tss" else "tpp_mag"
    spec_all = np.asarray(data[channel_key], dtype=np.float64)
    lambdas = np.asarray(data["lambdas"], dtype=np.float64)
    thetas = np.asarray(data["thetas"], dtype=np.float64)

    spec = spec_all[args.sample_idx]
    lam_mask = np.abs(lambdas - float(args.lambda0_nm)) <= float(args.lambda_window_nm)
    theta_mask = np.abs(thetas) <= float(args.theta_limit_deg)
    local = spec[np.ix_(lam_mask, theta_mask)]
    local_lambdas = lambdas[lam_mask]
    local_thetas = thetas[theta_mask]

    lambda0_m = float(args.lambda0_nm) * 1e-9
    omega0 = 2.0 * np.pi * C0 / lambda0_m
    k0 = 2.0 * np.pi / lambda0_m
    omega = 2.0 * np.pi * C0 / np.maximum(local_lambdas[:, None] * 1e-9, 1e-18)
    kx = k0 * np.sin(np.deg2rad(local_thetas[None, :]))
    ideal = ((kx / k0) ** 2) * (((omega - omega0) / omega0) ** 2)

    local_norm = robust_norm(local)
    ideal_norm = robust_norm(ideal)
    scale, r2 = fit_scale_and_r2(local_norm, ideal_norm)
    cos = cosine_similarity(local_norm, ideal_norm)
    mae = float(np.mean(np.abs(local_norm - ideal_norm)))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    extent = [float(local_thetas[0]), float(local_thetas[-1]), float(local_lambdas[0]), float(local_lambdas[-1])]

    im0 = axes[0].imshow(local, origin="lower", aspect="auto", extent=extent, cmap="turbo", interpolation="nearest")
    axes[0].set_title(f"Sample {args.sample_idx} raw {args.channel}")
    axes[0].set_xlabel("theta (deg)")
    axes[0].set_ylabel("lambda (nm)")

    im1 = axes[1].imshow(local_norm, origin="lower", aspect="auto", extent=extent, cmap="turbo", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("Normalized sample")
    axes[1].set_xlabel("theta (deg)")
    axes[1].set_ylabel("lambda (nm)")

    im2 = axes[2].imshow(ideal_norm, origin="lower", aspect="auto", extent=extent, cmap="turbo", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[2].set_title("Ideal kx^2 * Omega^2")
    axes[2].set_xlabel("theta (deg)")
    axes[2].set_ylabel("lambda (nm)")

    for ax, arr in zip(axes, [local, local_norm, ideal_norm]):
        for i, lam in enumerate(local_lambdas):
            for j, theta in enumerate(local_thetas):
                ax.text(float(theta), float(lam), f"{float(arr[i, j]):.2f}", ha="center", va="center", fontsize=7, color="white")

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    fig.suptitle(
        f"sample={args.sample_idx} channel={args.channel} | lambda0={args.lambda0_nm:.0f}nm | "
        f"cos={cos:.3f}, r2={r2:.3f}, mae={mae:.3f}, fit_scale={scale:.3f}",
        fontsize=12,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "npz": str(npz_path),
        "sample_idx": int(args.sample_idx),
        "channel": args.channel,
        "lambda0_nm": float(args.lambda0_nm),
        "lambda_window_nm": float(args.lambda_window_nm),
        "theta_limit_deg": float(args.theta_limit_deg),
        "local_lambdas_nm": local_lambdas.tolist(),
        "local_thetas_deg": local_thetas.tolist(),
        "cosine_similarity": cos,
        "r2": r2,
        "mae": mae,
        "fit_scale": scale,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved_png: {out_png}")
    print(f"saved_json: {out_json}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
