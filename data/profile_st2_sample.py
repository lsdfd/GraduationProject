from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from screen_st2_template_match import (
    find_lambda_index,
    lambda_zero_band_mask,
    load_dataset,
    merge_channel_scores,
    resolve_from_root,
    score_channel,
    theta_zero_band_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one ST2 candidate sample in detail.")
    parser.add_argument("--npz", type=str, default="data/train_data.npz")
    parser.add_argument("--sample_idx", type=int, required=True)
    parser.add_argument("--target_lambdas", type=float, nargs="+", default=[900.0, 1000.0, 1100.0])
    parser.add_argument("--theta_zero_width_deg", type=float, default=2.5)
    parser.add_argument("--lambda_zero_width_nm", type=float, default=25.0)
    parser.add_argument("--lambda_window_nm", type=float, default=200.0)
    parser.add_argument("--lambda_lower_offset_nm", type=float, default=None, help="Lower-side offset from lambda0 for asymmetric window.")
    parser.add_argument("--lambda_upper_offset_nm", type=float, default=None, help="Upper-side offset from lambda0 for asymmetric window.")
    parser.add_argument("--theta_max_deg", type=float, default=40.0)
    parser.add_argument("--out_dir", type=str, default="data/st2_profiles")
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_lambda_bounds(
    lambda0_nm: float,
    lambda_window_nm: float,
    lambda_lower_offset_nm: float | None,
    lambda_upper_offset_nm: float | None,
) -> tuple[float, float]:
    if lambda_lower_offset_nm is not None and lambda_upper_offset_nm is not None:
        return float(lambda0_nm) - float(lambda_lower_offset_nm), float(lambda0_nm) + float(lambda_upper_offset_nm)
    half = 0.5 * float(lambda_window_nm)
    return float(lambda0_nm) - half, float(lambda0_nm) + half


def ideal_st2_map_custom(
    lambdas_nm: np.ndarray,
    thetas_deg: np.ndarray,
    lambda0_nm: float,
    lambda_min_nm: float,
    lambda_max_nm: float,
    theta_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lam_m = np.asarray(lambdas_nm, dtype=np.float64)[:, None] * 1e-9
    th = np.deg2rad(np.asarray(thetas_deg, dtype=np.float64))[None, :]
    kx2 = ((2.0 * np.pi / np.maximum(lam_m, 1e-20)) * np.sin(th)) ** 2
    omega = 2.0 * np.pi * 299792458.0 / np.maximum(lam_m, 1e-20)
    omega0 = 2.0 * np.pi * 299792458.0 / max(float(lambda0_nm) * 1e-9, 1e-20)
    om2 = (omega - omega0) ** 2
    raw = kx2 * om2
    lam = np.asarray(lambdas_nm, dtype=np.float64)[:, None]
    theta = np.asarray(thetas_deg, dtype=np.float64)[None, :]
    mask = (
        (lam >= float(lambda_min_nm))
        & (lam <= float(lambda_max_nm))
        & (np.abs(theta) <= float(theta_max_deg))
    )
    ideal = np.zeros_like(raw, dtype=np.float64)
    if np.any(mask):
        masked = raw[mask]
        scale = float(np.max(np.abs(masked)))
        if scale > 1e-12:
            ideal[mask] = masked / scale
    return ideal, mask


def ideal_st2_map_dense_custom(
    lambda0_nm: float,
    lambda_min_nm: float,
    lambda_max_nm: float,
    theta_max_deg: float,
    n_lambda: int = 241,
    n_theta: int = 321,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambdas_dense = np.linspace(float(lambda_min_nm), float(lambda_max_nm), n_lambda, dtype=np.float64)
    thetas_dense = np.linspace(-float(theta_max_deg), float(theta_max_deg), n_theta, dtype=np.float64)
    lam_m = lambdas_dense[:, None] * 1e-9
    th = np.deg2rad(thetas_dense)[None, :]
    kx2 = ((2.0 * np.pi / np.maximum(lam_m, 1e-20)) * np.sin(th)) ** 2
    omega = 2.0 * np.pi * 299792458.0 / np.maximum(lam_m, 1e-20)
    omega0 = 2.0 * np.pi * 299792458.0 / max(float(lambda0_nm) * 1e-9, 1e-20)
    om2 = (omega - omega0) ** 2
    ideal_dense = np.asarray(kx2 * om2, dtype=np.float64)
    scale = float(np.max(np.abs(ideal_dense)))
    if scale > 1e-12:
        ideal_dense = ideal_dense / scale
    else:
        ideal_dense = np.zeros_like(ideal_dense)
    return lambdas_dense, thetas_dense, ideal_dense


def plot_profile_page(
    out_path: Path,
    structure: np.ndarray,
    tpp_map: np.ndarray,
    tss_map: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lambda0_nm: float,
    lambda_min_nm: float,
    lambda_max_nm: float,
    theta_max_deg: float,
    merged: dict[str, float],
    tpp_scores: dict[str, float],
    tss_scores: dict[str, float],
) -> None:
    lambda_idx = find_lambda_index(lambdas, lambda0_nm)
    theta0_idx = int(np.argmin(np.abs(thetas)))
    lambdas_dense, thetas_dense, ideal_dense = ideal_st2_map_dense_custom(
        lambda0_nm,
        lambda_min_nm,
        lambda_max_nm,
        theta_max_deg,
    )

    theta_min_vis = -float(theta_max_deg)
    theta_max_vis = float(theta_max_deg)
    lam_min_vis = float(lambda_min_nm)
    lam_max_vis = float(lambda_max_nm)
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    extent_dense = [float(thetas_dense[0]), float(thetas_dense[-1]), float(lambdas_dense[0]), float(lambdas_dense[-1])]
    sample_vmin = float(min(np.min(tpp_map), np.min(tss_map)))
    sample_vmax = float(max(np.max(tpp_map), np.max(tss_map)))

    fig, axes = plt.subplots(4, 3, figsize=(15.0, 14.5), constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11 = axes.ravel()

    ax0.imshow(structure, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax0.set_title(f"structure | sample {int(merged['sample_idx'])}")
    ax0.axis("off")

    hm0 = ax1.imshow(
        ideal_dense,
        origin="lower",
        aspect="auto",
        extent=extent_dense,
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        interpolation="bicubic",
    )
    ax1.set_title(f"ideal ST2 @ {int(round(lambda0_nm))} nm")
    ax1.set_xlabel("theta (deg)")
    ax1.set_ylabel("lambda (nm)")
    ax1.set_xlim(theta_min_vis, theta_max_vis)
    ax1.set_ylim(lam_min_vis, lam_max_vis)

    hm1 = ax2.imshow(
        tpp_map,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=sample_vmin,
        vmax=sample_vmax,
        interpolation="bicubic",
    )
    ax2.set_title(f"tpp | merged score={merged['score_total']:.4f}")
    ax2.set_xlabel("theta (deg)")
    ax2.set_ylabel("lambda (nm)")
    ax2.set_xlim(theta_min_vis, theta_max_vis)
    ax2.set_ylim(lam_min_vis, lam_max_vis)

    hm2 = ax3.imshow(
        tss_map,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=sample_vmin,
        vmax=sample_vmax,
        interpolation="nearest",
    )
    ax3.set_title("tss")
    ax3.set_xlabel("theta (deg)")
    ax3.set_ylabel("lambda (nm)")
    ax3.set_xlim(theta_min_vis, theta_max_vis)
    ax3.set_ylim(lam_min_vis, lam_max_vis)

    hm3 = ax4.imshow(
        tpp_map,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=sample_vmin,
        vmax=sample_vmax,
        interpolation="nearest",
    )
    ax4.axvline(0.0, color="w", ls="--", lw=0.8)
    ax4.axhline(float(lambda0_nm), color="w", ls="--", lw=0.8)
    ax4.set_title("tpp full map (raw grid)")
    ax4.set_xlabel("theta (deg)")
    ax4.set_ylabel("lambda (nm)")

    hm4 = ax5.imshow(
        tss_map,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="turbo",
        vmin=sample_vmin,
        vmax=sample_vmax,
        interpolation="nearest",
    )
    ax5.axvline(0.0, color="w", ls="--", lw=0.8)
    ax5.axhline(float(lambda0_nm), color="w", ls="--", lw=0.8)
    ax5.set_title("tss full map (raw grid)")
    ax5.set_xlabel("theta (deg)")
    ax5.set_ylabel("lambda (nm)")

    ideal_row_dense = ideal_dense[np.argmin(np.abs(lambdas_dense - lambda0_nm))]
    ax6.plot(thetas_dense, ideal_row_dense, "k--", lw=1.8, label="ideal row")
    ax6.plot(thetas, tpp_map[lambda_idx], lw=2.0, label="tpp row")
    ax6.plot(thetas, tss_map[lambda_idx], lw=2.0, label="tss row")
    ax6.set_title(f"row @ {int(round(lambda0_nm))} nm")
    ax6.set_xlabel("theta (deg)")
    ax6.set_ylabel("magnitude")
    ax6.set_xlim(theta_min_vis, theta_max_vis)
    ax6.grid(alpha=0.25)
    ax6.legend(fontsize=8)

    ideal_col_dense = ideal_dense[:, np.argmin(np.abs(thetas_dense))]
    ax7.plot(lambdas_dense, ideal_col_dense, "k--", lw=1.8, label="ideal theta=0")
    ax7.plot(lambdas, tpp_map[:, theta0_idx], lw=2.0, label="tpp theta=0")
    ax7.plot(lambdas, tss_map[:, theta0_idx], lw=2.0, label="tss theta=0")
    ax7.set_title("column @ theta=0")
    ax7.set_xlabel("lambda (nm)")
    ax7.set_ylabel("magnitude")
    ax7.set_xlim(lam_min_vis, lam_max_vis)
    ax7.grid(alpha=0.25)
    ax7.legend(fontsize=8)

    metrics_text = [
        f"merged score_total       = {merged['score_total']:.4f}",
        f"merged score_zero_lines  = {merged['score_zero_lines']:.4f}",
        f"merged score_weighted    = {merged['score_weighted_error']:.4f}",
        f"merged score_projection  = {merged['score_projection']:.4f}",
        f"merged theta_zero_leak   = {merged['theta_zero_leak']:.4f}",
        f"merged lambda_zero_leak  = {merged['lambda_zero_leak']:.4f}",
        f"merged weighted_mae      = {merged['weighted_mae']:.4f}",
        f"merged weighted_rmse     = {merged['weighted_rmse']:.4f}",
        f"merged proj_coeff        = {merged['projection_coeff_k2omega2']:.4f}",
        f"merged proj_corr         = {merged['projection_corr_k2omega2']:.4f}",
        "",
        f"tpp score_total          = {tpp_scores['score_total']:.4f}",
        f"tss score_total          = {tss_scores['score_total']:.4f}",
        f"tpp zero_lines           = {tpp_scores['score_zero_lines']:.4f}",
        f"tss zero_lines           = {tss_scores['score_zero_lines']:.4f}",
    ]
    ax8.axis("off")
    ax8.set_title("metrics")
    ax8.text(0.0, 1.0, "\n".join(metrics_text), va="top", ha="left", family="monospace", fontsize=10)

    theta_vals = np.abs(thetas)
    ax9.plot(theta_vals, tpp_map[lambda_idx], "o-", lw=1.8, label="tpp row")
    ax9.plot(theta_vals, tss_map[lambda_idx], "o-", lw=1.8, label="tss row")
    ax9.set_title("row vs |theta|")
    ax9.set_xlabel("|theta| (deg)")
    ax9.set_ylabel("magnitude")
    ax9.grid(alpha=0.25)
    ax9.legend(fontsize=8)

    delta_lambda = np.abs(lambdas - lambda0_nm)
    ax10.plot(delta_lambda, tpp_map[:, theta0_idx], "o-", lw=1.8, label="tpp theta=0")
    ax10.plot(delta_lambda, tss_map[:, theta0_idx], "o-", lw=1.8, label="tss theta=0")
    ax10.set_title("column vs |lambda-lambda0|")
    ax10.set_xlabel("|lambda-lambda0| (nm)")
    ax10.set_ylabel("magnitude")
    ax10.grid(alpha=0.25)
    ax10.legend(fontsize=8)

    ax11.axis("off")

    for ax in (ax1, ax2, ax3):
        ax.axvline(0.0, color="w", ls="--", lw=0.8)
        ax.axhline(float(lambda0_nm), color="w", ls="--", lw=0.8)
        ax.axvline(-float(theta_max_deg), color="w", ls=":", lw=0.8)
        ax.axvline(float(theta_max_deg), color="w", ls=":", lw=0.8)
        ax.axhline(float(lambda_min_nm), color="w", ls=":", lw=0.8)
        ax.axhline(float(lambda_max_nm), color="w", ls=":", lw=0.8)

    fig.colorbar(hm0, ax=[ax1], shrink=0.82, pad=0.02, label="ideal weight (normalized)")
    fig.colorbar(hm2, ax=[ax2, ax3, ax4, ax5], shrink=0.82, pad=0.02, label="magnitude")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    out_root = resolve_from_root(args.out_dir) / f"sample_{args.sample_idx}"
    out_root.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(npz_path)
    structures = bundle["structures"]
    tpp_all = bundle["tpp_mag"]
    tss_all = bundle["tss_mag"]
    lambdas = bundle["lambdas"]
    thetas = bundle["thetas"]

    if args.sample_idx < 0 or args.sample_idx >= len(structures):
        raise IndexError(f"sample_idx out of range: {args.sample_idx}, dataset size={len(structures)}")

    theta0_mask = theta_zero_band_mask(thetas, args.theta_zero_width_deg)
    structure = structures[args.sample_idx]
    tpp_map = tpp_all[args.sample_idx]
    tss_map = tss_all[args.sample_idx]

    all_results: list[dict[str, object]] = []
    for target_lambda in args.target_lambdas:
        lambda_idx = find_lambda_index(lambdas, target_lambda)
        actual_lambda = float(lambdas[lambda_idx])
        lambda_min_nm, lambda_max_nm = resolve_lambda_bounds(
            actual_lambda,
            args.lambda_window_nm,
            args.lambda_lower_offset_nm,
            args.lambda_upper_offset_nm,
        )
        lambda0_mask = lambda_zero_band_mask(lambdas, actual_lambda, args.lambda_zero_width_nm)
        ideal_map, work_mask = ideal_st2_map_custom(
            lambdas,
            thetas,
            actual_lambda,
            lambda_min_nm,
            lambda_max_nm,
            args.theta_max_deg,
        )

        tpp_scores = score_channel(tpp_map, ideal_map, lambdas, thetas, actual_lambda, theta0_mask, lambda0_mask, work_mask)
        tss_scores = score_channel(tss_map, ideal_map, lambdas, thetas, actual_lambda, theta0_mask, lambda0_mask, work_mask)
        merged = merge_channel_scores(tpp_scores, tss_scores)
        merged["sample_idx"] = int(args.sample_idx)
        merged["target_lambda_nm"] = actual_lambda

        plot_profile_page(
            out_root / f"profile_{int(round(actual_lambda))}nm.png",
            structure=structure,
            tpp_map=tpp_map,
            tss_map=tss_map,
            lambdas=lambdas,
            thetas=thetas,
            lambda0_nm=actual_lambda,
            lambda_min_nm=lambda_min_nm,
            lambda_max_nm=lambda_max_nm,
            theta_max_deg=args.theta_max_deg,
            merged=merged,
            tpp_scores=tpp_scores,
            tss_scores=tss_scores,
        )

        all_results.append(
            {
                "requested_lambda_nm": float(target_lambda),
                "actual_lambda_nm": actual_lambda,
                "sample_idx": int(args.sample_idx),
                "merged": merged,
                "tpp": tpp_scores,
                "tss": tss_scores,
            }
        )

    payload = {
        "input_npz": str(npz_path),
        "sample_idx": int(args.sample_idx),
        "theta_zero_width_deg": float(args.theta_zero_width_deg),
        "lambda_zero_width_nm": float(args.lambda_zero_width_nm),
        "lambda_window_nm": float(args.lambda_window_nm),
        "lambda_lower_offset_nm": args.lambda_lower_offset_nm,
        "lambda_upper_offset_nm": args.lambda_upper_offset_nm,
        "theta_max_deg": float(args.theta_max_deg),
        "results": all_results,
    }
    save_json(out_root / "profile_summary.json", payload)
    print(f"saved_to: {out_root}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
