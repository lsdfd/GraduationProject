from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def robust_norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(arr)))
    if scale < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / scale


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    required = ["structures", "tpp_mag", "tss_mag", "lambdas", "thetas"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"Missing fields in {npz_path}: {missing}")

    bundle = {
        "structures": np.asarray(data["structures"], dtype=np.float32),
        "tpp_mag": np.asarray(data["tpp_mag"], dtype=np.float32),
        "tss_mag": np.asarray(data["tss_mag"], dtype=np.float32),
        "lambdas": np.asarray(data["lambdas"], dtype=np.float64),
        "thetas": np.asarray(data["thetas"], dtype=np.float64),
    }
    return bundle


def find_lambda_index(lambdas: np.ndarray, target_lambda: float) -> int:
    return int(np.argmin(np.abs(lambdas - float(target_lambda))))


def theta_zero_band_mask(thetas: np.ndarray, width_deg: float) -> np.ndarray:
    return np.abs(thetas) <= float(width_deg)


def lambda_neighborhood_indices(center_idx: int, radius: int, length: int) -> list[int]:
    idxs = []
    for offset in range(-radius, radius + 1):
        j = center_idx + offset
        if 0 <= j < length:
            idxs.append(j)
    return idxs


def ideal_theta_second_order(thetas_deg: np.ndarray) -> np.ndarray:
    target = np.sin(np.deg2rad(thetas_deg)) ** 2
    return robust_norm(target)


def ideal_lambda_second_order(lambdas_nm: np.ndarray, lambda0_nm: float) -> np.ndarray:
    inv_delta = (1.0 / np.asarray(lambdas_nm, dtype=np.float64)) - (1.0 / float(lambda0_nm))
    target = inv_delta ** 2
    return robust_norm(target)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(aa, bb) / (na * nb))


def fit_r2_nonnegative(y: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    yy = np.asarray(y, dtype=np.float64)
    tt = np.asarray(target, dtype=np.float64)
    denom = float(np.dot(tt, tt))
    if denom <= 1e-12:
        return 0.0, 0.0
    coef = max(float(np.dot(yy, tt) / denom), 0.0)
    fit = coef * tt
    ss_res = float(np.sum((yy - fit) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    if ss_tot <= 1e-12:
        return coef, 0.0
    return coef, float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def score_zero_lines(
    spec_map: np.ndarray,
    lambda_idx: int,
    theta0_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    spec = robust_norm(spec_map)
    theta_zero_leak = float(np.mean(spec[:, theta0_mask]))
    lambda_zero_leak = float(np.mean(spec[lambda_idx, :]))
    theta_score = float(np.clip(1.0 - theta_zero_leak, 0.0, 1.0))
    lambda_score = float(np.clip(1.0 - lambda_zero_leak, 0.0, 1.0))
    total = 0.5 * theta_score + 0.5 * lambda_score
    return total, {
        "theta_zero_score": theta_score,
        "lambda_zero_score": lambda_score,
        "theta_zero_leak": theta_zero_leak,
        "lambda_zero_leak": lambda_zero_leak,
    }


def score_second_order(
    spec_map: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lambda_idx: int,
    lambda0_nm: float,
    theta0_mask: np.ndarray,
    lambda_radius: int,
) -> tuple[float, dict[str, float]]:
    spec = robust_norm(spec_map)
    theta_target = ideal_theta_second_order(thetas)
    lambda_target = ideal_lambda_second_order(lambdas, lambda0_nm)

    theta_scores = []
    theta_r2s = []
    theta_cosines = []
    eval_rows = [j for j in lambda_neighborhood_indices(lambda_idx, lambda_radius, len(lambdas)) if j != lambda_idx]
    if not eval_rows:
        eval_rows = [lambda_idx]

    for j in eval_rows:
        row = robust_norm(spec[j, :])
        _, r2 = fit_r2_nonnegative(row, theta_target)
        cos = cosine_similarity(row, theta_target)
        theta_r2s.append(r2)
        theta_cosines.append(cos)
        theta_scores.append(0.5 * r2 + 0.5 * cos)

    theta_line = robust_norm(np.mean(spec[:, theta0_mask], axis=1))
    _, lambda_r2 = fit_r2_nonnegative(theta_line, lambda_target)
    lambda_cos = cosine_similarity(theta_line, lambda_target)
    lambda_score = 0.5 * lambda_r2 + 0.5 * lambda_cos

    theta_score = float(np.mean(theta_scores)) if theta_scores else 0.0
    total = 0.5 * theta_score + 0.5 * lambda_score
    return total, {
        "theta_second_score": theta_score,
        "theta_second_r2": float(np.mean(theta_r2s)) if theta_r2s else 0.0,
        "theta_second_cosine": float(np.mean(theta_cosines)) if theta_cosines else 0.0,
        "lambda_second_score": lambda_score,
        "lambda_second_r2": lambda_r2,
        "lambda_second_cosine": lambda_cos,
    }


def basis_features(lambdas_nm: np.ndarray, thetas_deg: np.ndarray, lambda0_nm: float) -> np.ndarray:
    lam = np.asarray(lambdas_nm, dtype=np.float64)[:, None]
    theta = np.asarray(thetas_deg, dtype=np.float64)[None, :]
    k = np.sin(np.deg2rad(theta)) / np.maximum(lam, 1e-8)
    omega = (1.0 / np.maximum(lam, 1e-8)) - (1.0 / float(lambda0_nm))
    k2 = robust_norm(k ** 2)
    om2 = robust_norm(np.broadcast_to(omega ** 2, k.shape))
    k2om2 = robust_norm((k ** 2) * np.broadcast_to(omega ** 2, k.shape))
    ones = np.ones_like(k2)
    return np.stack([ones, k2, om2, k2om2], axis=-1)


def score_basis_coefficient(
    spec_map: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lambda0_nm: float,
) -> tuple[float, dict[str, float]]:
    y = robust_norm(spec_map).reshape(-1)
    basis = basis_features(lambdas, thetas, lambda0_nm).reshape(-1, 4)
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    recon = basis @ coef
    corr = cosine_similarity(y, recon)
    abs_sum = float(np.sum(np.abs(coef))) + 1e-12
    k2om2_ratio = float(np.clip(max(float(coef[3]), 0.0) / abs_sum, 0.0, 1.0))
    score = 0.7 * k2om2_ratio + 0.3 * corr
    return score, {
        "basis_score": score,
        "basis_recon_cosine": corr,
        "coef_const": float(coef[0]),
        "coef_k2": float(coef[1]),
        "coef_omega2": float(coef[2]),
        "coef_k2omega2": float(coef[3]),
        "coef_k2omega2_ratio": k2om2_ratio,
    }


def score_channel(
    spec_map: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lambda_idx: int,
    lambda0_nm: float,
    theta0_mask: np.ndarray,
    lambda_radius: int,
) -> dict[str, float]:
    zero_score, zero_details = score_zero_lines(spec_map, lambda_idx, theta0_mask)
    second_score, second_details = score_second_order(
        spec_map, lambdas, thetas, lambda_idx, lambda0_nm, theta0_mask, lambda_radius
    )
    basis_score, basis_details = score_basis_coefficient(spec_map, lambdas, thetas, lambda0_nm)
    total = 0.55 * zero_score + 0.30 * second_score + 0.15 * basis_score
    return {
        "score_total": float(total),
        "score_zero_lines": float(zero_score),
        "score_second_order": float(second_score),
        "score_basis": float(basis_score),
        **zero_details,
        **second_details,
        **basis_details,
    }


def merge_channel_scores(tpp: dict[str, float], tss: dict[str, float]) -> dict[str, float]:
    merged = {}
    numeric_keys = sorted(set(tpp) | set(tss))
    for key in numeric_keys:
        tv = float(tpp.get(key, np.nan))
        sv = float(tss.get(key, np.nan))
        if np.isfinite(tv) and np.isfinite(sv):
            merged[f"tpp_{key}"] = tv
            merged[f"tss_{key}"] = sv
            merged[key] = 0.5 * (tv + sv)
    return merged


def plot_candidate(
    out_path: Path,
    structure: np.ndarray,
    tpp_map: np.ndarray,
    tss_map: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    lambda0_nm: float,
    row_indices: list[int],
    merged_score: float,
) -> None:
    lambda_idx = find_lambda_index(lambdas, lambda0_nm)
    theta_target = ideal_theta_second_order(thetas)
    lambda_target = ideal_lambda_second_order(lambdas, lambda0_nm)
    theta0_idx = int(np.argmin(np.abs(thetas)))

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8), constrained_layout=True)
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.ravel()

    ax0.imshow(structure, cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax0.set_title("structure")
    ax0.axis("off")

    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    hm1 = ax1.imshow(robust_norm(tpp_map), origin="lower", aspect="auto", extent=extent, cmap="turbo", vmin=0.0, vmax=1.0)
    ax1.axvline(0.0, color="w", ls="--", lw=0.8)
    ax1.axhline(float(lambdas[lambda_idx]), color="w", ls="--", lw=0.8)
    ax1.set_title(f"tpp | score={merged_score:.4f}")
    ax1.set_xlabel("theta (deg)")
    ax1.set_ylabel("lambda (nm)")

    hm2 = ax2.imshow(robust_norm(tss_map), origin="lower", aspect="auto", extent=extent, cmap="turbo", vmin=0.0, vmax=1.0)
    ax2.axvline(0.0, color="w", ls="--", lw=0.8)
    ax2.axhline(float(lambdas[lambda_idx]), color="w", ls="--", lw=0.8)
    ax2.set_title("tss")
    ax2.set_xlabel("theta (deg)")
    ax2.set_ylabel("lambda (nm)")

    for j in row_indices:
        ax3.plot(thetas, robust_norm(tpp_map[j]), lw=1.8, label=f"{int(round(lambdas[j]))} nm")
    ax3.plot(thetas, theta_target, "k--", lw=1.8, label="ideal ~ sin^2(theta)")
    ax3.set_title("theta-direction second-order")
    ax3.set_xlabel("theta (deg)")
    ax3.set_ylabel("normalized amplitude")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8)

    ax4.plot(lambdas, robust_norm(tpp_map[:, theta0_idx]), lw=2.0, label="tpp @ theta=0")
    ax4.plot(lambdas, robust_norm(tss_map[:, theta0_idx]), lw=2.0, label="tss @ theta=0")
    ax4.plot(lambdas, lambda_target, "k--", lw=1.8, label="ideal ~ Omega^2")
    ax4.axvline(float(lambda0_nm), color="k", ls="--", lw=0.8)
    ax4.set_title("lambda-direction second-order on theta=0")
    ax4.set_xlabel("lambda (nm)")
    ax4.set_ylabel("normalized amplitude")
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=8)

    zero_row = robust_norm(tpp_map[lambda_idx])
    zero_col = robust_norm(tpp_map[:, theta0_idx])
    ax5.plot(thetas, zero_row, lw=1.8, label=f"tpp @ {int(round(lambda0_nm))} nm")
    ax5.plot(lambdas, zero_col, lw=1.8, label="tpp @ theta=0")
    ax5.axhline(0.0, color="k", lw=0.8)
    ax5.set_title("zero-line diagnostics")
    ax5.grid(alpha=0.25)
    ax5.legend(fontsize=8)

    fig.colorbar(hm2, ax=[ax1, ax2], shrink=0.85, pad=0.02, label="normalized magnitude")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_ranking_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen train_data.npz for second-order spatiotemporal metasurface candidates.")
    parser.add_argument("--npz", type=str, default="data/train_data.npz", help="Dataset npz path.")
    parser.add_argument("--target_lambdas", type=float, nargs="+", default=[900.0, 1000.0, 1100.0], help="Working wavelengths to screen.")
    parser.add_argument("--topk", type=int, default=10, help="How many candidates to visualize per wavelength.")
    parser.add_argument("--theta_zero_width_deg", type=float, default=2.5, help="Theta=0 line half-width in degrees.")
    parser.add_argument("--lambda_radius", type=int, default=1, help="Neighboring lambda rows for theta-direction second-order scoring.")
    parser.add_argument("--out_dir", type=str, default="data/st2_screen", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    out_dir = resolve_from_root(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(npz_path)
    structures = bundle["structures"]
    tpp = bundle["tpp_mag"]
    tss = bundle["tss_mag"]
    lambdas = bundle["lambdas"]
    thetas = bundle["thetas"]
    theta0_mask = theta_zero_band_mask(thetas, args.theta_zero_width_deg)

    global_summary = []
    for target_lambda in args.target_lambdas:
        lambda_idx = find_lambda_index(lambdas, target_lambda)
        actual_lambda = float(lambdas[lambda_idx])
        row_indices = lambda_neighborhood_indices(lambda_idx, args.lambda_radius, len(lambdas))
        wave_dir = out_dir / f"lambda_{int(round(actual_lambda))}nm"
        wave_dir.mkdir(parents=True, exist_ok=True)

        ranking_rows: list[dict[str, float | int | str]] = []
        for sample_idx in range(structures.shape[0]):
            tpp_score = score_channel(tpp[sample_idx], lambdas, thetas, lambda_idx, actual_lambda, theta0_mask, args.lambda_radius)
            tss_score = score_channel(tss[sample_idx], lambdas, thetas, lambda_idx, actual_lambda, theta0_mask, args.lambda_radius)
            merged = merge_channel_scores(tpp_score, tss_score)
            merged["sample_idx"] = int(sample_idx)
            merged["target_lambda_nm"] = actual_lambda
            ranking_rows.append(merged)

        ranking_rows.sort(key=lambda row: float(row.get("score_total", -1.0)), reverse=True)
        save_ranking_csv(wave_dir / "ranking.csv", ranking_rows)

        summary = {
            "target_lambda_nm": actual_lambda,
            "requested_lambda_nm": float(target_lambda),
            "top_score": float(ranking_rows[0]["score_total"]) if ranking_rows else None,
            "top_sample_idx": int(ranking_rows[0]["sample_idx"]) if ranking_rows else None,
            "num_samples": len(ranking_rows),
        }
        global_summary.append(summary)

        for rank, row in enumerate(ranking_rows[: max(1, args.topk)], start=1):
            sample_idx = int(row["sample_idx"])
            plot_candidate(
                wave_dir / f"rank_{rank:03d}_sample_{sample_idx}.png",
                structures[sample_idx],
                tpp[sample_idx],
                tss[sample_idx],
                lambdas,
                thetas,
                actual_lambda,
                row_indices,
                float(row["score_total"]),
            )

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "input_npz": str(npz_path),
                "target_lambdas": [float(x) for x in args.target_lambdas],
                "theta_zero_width_deg": float(args.theta_zero_width_deg),
                "lambda_radius": int(args.lambda_radius),
                "results": global_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_to: {out_dir}")
    for item in global_summary:
        print(
            f"lambda={item['target_lambda_nm']:.1f}nm "
            f"top_sample={item['top_sample_idx']} "
            f"top_score={item['top_score']:.4f}"
        )


if __name__ == "__main__":
    main()
