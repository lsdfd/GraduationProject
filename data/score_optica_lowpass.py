from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_common import load_bundle, topk_per_lambda


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


def lowpass_target(thetas_deg: np.ndarray, sigma_deg: float) -> np.ndarray:
    theta = np.asarray(thetas_deg, dtype=np.float64)
    sigma = max(float(sigma_deg), 1e-8)
    target = np.exp(-(theta**2) / (sigma**2))
    scale = float(np.max(target))
    if scale <= 1e-12:
        return np.zeros_like(theta, dtype=np.float32)
    return np.asarray(target / scale, dtype=np.float32)


def row_lowpass_main_score(
    y: np.ndarray,
    target: np.ndarray,
    center_idx: int,
    edge_mask: np.ndarray,
    global_scale: float,
    w_center: float,
    w_shape: float,
    w_edge_reject: float,
) -> tuple[float, float, float, float, float, float]:
    if not np.isfinite(y).all():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    y_norm = y / max(float(np.max(y)), 1e-8)

    denom = float(np.sum(t * t))
    if denom <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    a = max(float(np.sum(t * y_norm) / denom), 0.0)
    y_fit = a * t
    ss_res = float(np.sum((y_norm - y_fit) ** 2))
    ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    center_val = float(y[center_idx])
    edge_mean = float(np.mean(y[edge_mask]))
    center_pass_score = float(np.clip(center_val / max(global_scale, 1e-8), 0.0, 1.0))
    shape_score = float(np.clip(r2, 0.0, 1.0))
    edge_reject_score = float(np.clip(1.0 - edge_mean / max(center_val, 1e-8), 0.0, 1.0))
    score = w_center * center_pass_score + w_shape * shape_score + w_edge_reject * edge_reject_score
    return score, center_pass_score, shape_score, edge_reject_score, a, r2


def score_lowpass_spectra(
    spec: np.ndarray,
    thetas_deg: np.ndarray,
    sigmas_deg: list[float],
    w_center: float,
    w_shape: float,
    w_edge_reject: float,
    w_bandwidth: float,
) -> dict[str, np.ndarray]:
    n, l, _ = spec.shape
    center_idx = int(np.argmin(np.abs(thetas_deg)))
    edge_mask = np.abs(thetas_deg) >= 0.75 * float(np.max(np.abs(thetas_deg)))
    if not edge_mask.any():
        edge_mask[[0, -1]] = True

    global_scale = max(float(np.nanquantile(spec, 0.99)), 1e-8)
    sigma_targets = {
        float(sigma): lowpass_target(thetas_deg, float(sigma)).astype(np.float64)
        for sigma in sigmas_deg
    }
    w3 = 1.0 - float(w_bandwidth)

    score = np.full((n, l), np.nan, dtype=np.float32)
    center_s = np.full_like(score, np.nan)
    shape_s = np.full_like(score, np.nan)
    edge_reject_s = np.full_like(score, np.nan)
    bandwidth_s = np.full_like(score, np.nan)
    coef_a = np.full_like(score, np.nan)
    r2 = np.full_like(score, np.nan)
    best_sigma = np.full_like(score, np.nan)

    for i in range(n):
        for j in range(l):
            best_total = -np.inf
            best_main = np.nan
            best_bw = np.nan
            best_c = np.nan
            best_s = np.nan
            best_e = np.nan
            best_a = np.nan
            best_r2 = np.nan
            best_sigma_j = np.nan

            for sigma in sigmas_deg:
                target = sigma_targets[float(sigma)]
                y = spec[i, j].astype(np.float64)
                main, c, s, e, a, this_r2 = row_lowpass_main_score(
                    y,
                    target,
                    center_idx,
                    edge_mask,
                    global_scale,
                    w_center,
                    w_shape,
                    w_edge_reject,
                )
                if not np.isfinite(main):
                    continue

                bw_scores = []
                for dj in (-1, 1):
                    jj = j + dj
                    if 0 <= jj < l:
                        yy = spec[i, jj].astype(np.float64)
                        nb, *_ = row_lowpass_main_score(
                            yy,
                            target,
                            center_idx,
                            edge_mask,
                            global_scale,
                            w_center,
                            w_shape,
                            w_edge_reject,
                        )
                        if np.isfinite(nb):
                            bw_scores.append(nb)
                bw = float(np.mean(bw_scores)) if bw_scores else float(main)
                total = w3 * float(main) + float(w_bandwidth) * bw

                if total > best_total:
                    best_total = total
                    best_main = main
                    best_bw = bw
                    best_c = c
                    best_s = s
                    best_e = e
                    best_a = a
                    best_r2 = this_r2
                    best_sigma_j = float(sigma)

            if np.isfinite(best_total):
                score[i, j] = float(best_total)
                center_s[i, j] = float(best_c)
                shape_s[i, j] = float(best_s)
                edge_reject_s[i, j] = float(best_e)
                bandwidth_s[i, j] = float(best_bw)
                coef_a[i, j] = float(best_a)
                r2[i, j] = float(best_r2)
                best_sigma[i, j] = float(best_sigma_j)

    return {
        "score": score,
        "center_score": center_s,
        "shape_score": shape_s,
        "edge_reject_score": edge_reject_s,
        "bandwidth_score": bandwidth_s,
        "coef_a": coef_a,
        "r2": r2,
        "best_sigma_deg": best_sigma,
    }


def save_scores_csv(path: Path, lambdas: np.ndarray, pack: dict[str, np.ndarray]) -> None:
    keys = [
        "score",
        "center_score",
        "shape_score",
        "edge_reject_score",
        "bandwidth_score",
        "coef_a",
        "r2",
        "best_sigma_deg",
    ]
    n, l = pack["score"].shape
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "lambda_idx", "lambda_nm", *keys])
        for i in range(n):
            for j in range(l):
                w.writerow([i, j, float(lambdas[j]), *(float(pack[k][i, j]) for k in keys)])


def save_topk_csv(path: Path, lambdas: np.ndarray, top_idx: np.ndarray, top_score: np.ndarray, pack: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lambda_idx",
                "lambda_nm",
                "rank",
                "sample_idx",
                "score",
                "best_sigma_deg",
                "center_score",
                "shape_score",
                "edge_reject_score",
                "bandwidth_score",
            ]
        )
        for j, lam in enumerate(lambdas):
            for r in range(top_idx.shape[1]):
                i = int(top_idx[j, r])
                s = float(top_score[j, r])
                if i < 0 or not np.isfinite(s):
                    continue
                w.writerow(
                    [
                        j,
                        float(lam),
                        r + 1,
                        i,
                        f"{s:.4f}",
                        f"{float(pack['best_sigma_deg'][i, j]):.2f}",
                        f"{float(pack['center_score'][i, j]):.4f}",
                        f"{float(pack['shape_score'][i, j]):.4f}",
                        f"{float(pack['edge_reject_score'][i, j]):.4f}",
                        f"{float(pack['bandwidth_score'][i, j]):.4f}",
                    ]
                )


def summary_json(lambdas: np.ndarray, pack: dict[str, np.ndarray], top_idx: np.ndarray, top_score: np.ndarray) -> list[dict]:
    out = []
    for j, lam in enumerate(lambdas):
        col = pack["score"][:, j]
        valid = col[np.isfinite(col)]
        top = []
        for r in range(top_idx.shape[1]):
            i = int(top_idx[j, r])
            if i < 0:
                continue
            top.append(
                {
                    "rank": r + 1,
                    "sample_idx": i,
                    "score": float(top_score[j, r]),
                    "best_sigma_deg": float(pack["best_sigma_deg"][i, j]),
                    "center_score": float(pack["center_score"][i, j]),
                    "shape_score": float(pack["shape_score"][i, j]),
                    "edge_reject_score": float(pack["edge_reject_score"][i, j]),
                    "bandwidth_score": float(pack["bandwidth_score"][i, j]),
                }
            )
        out.append(
            {
                "lambda_idx": int(j),
                "lambda_nm": float(lam),
                "num_valid": int(len(valid)),
                "score_mean": float(np.mean(valid)) if len(valid) else None,
                "score_p90": float(np.quantile(valid, 0.9)) if len(valid) else None,
                "score_max": float(np.max(valid)) if len(valid) else None,
                "top": top,
            }
        )
    return out


def plot_per_lambda(
    out_dir: Path,
    structures: np.ndarray,
    spec: np.ndarray,
    pack: dict[str, np.ndarray],
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_score: np.ndarray,
    topk: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    finite = spec[np.isfinite(spec)]
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    center_idx = int(np.argmin(np.abs(thetas)))

    for j, lam in enumerate(lambdas):
        fig, axes = plt.subplots(
            topk,
            3,
            figsize=(14.0, max(2.8 * topk, 5.5)),
            gridspec_kw={"width_ratios": [0.7, 1.1, 1.0]},
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        hm = None

        for r in range(topk):
            a_struct, a_hm, a_curve = axes[r]
            i = int(top_idx[j, r])
            if i < 0:
                for ax in axes[r]:
                    ax.axis("off")
                continue

            sigma = float(pack["best_sigma_deg"][i, j])
            target = lowpass_target(thetas, sigma)
            y = spec[i, j].astype(np.float64)
            yn = y / max(float(np.max(y)), 1e-8)
            t0 = float(y[center_idx])
            tp40 = float(y[np.argmin(np.abs(thetas - 40.0))])
            tm40 = float(y[np.argmin(np.abs(thetas + 40.0))])

            a_struct.imshow(structures[i], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
            a_struct.set_title(f"id={i}", fontsize=10)
            a_struct.axis("off")

            hm = a_hm.imshow(
                spec[i],
                cmap="turbo",
                aspect="auto",
                origin="lower",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                interpolation="bicubic",
            )
            a_hm.axhline(float(lam), color="w", ls="--", lw=1.0)
            a_hm.set_title(
                f"rank{r+1} score={float(top_score[j, r]):.3f} sigma={sigma:.1f}deg\n"
                f"C={float(pack['center_score'][i, j]):.3f} S={float(pack['shape_score'][i, j]):.3f} ER={float(pack['edge_reject_score'][i, j]):.3f}",
                fontsize=9,
            )
            a_hm.set_xlabel("theta (deg)")
            a_hm.set_ylabel("lambda (nm)")
            a_hm.text(
                0.98,
                0.03,
                f"|t|(0)={t0:.3f}\n|t|(+40)={tp40:.3f}\n|t|(-40)={tm40:.3f}",
                transform=a_hm.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
            )

            a_curve.plot(thetas, target, "k--", lw=1.7, label=f"ideal lp (sigma={sigma:.1f}deg)")
            a_curve.plot(thetas, yn, lw=1.9, color="#1f77b4", label="tpp row (normalized)")
            a_curve.set_ylim(-0.05, 1.05)
            a_curve.set_xlabel("theta (deg)")
            a_curve.set_ylabel("normalized |tpp|")
            a_curve.grid(alpha=0.25)
            if r == 0:
                a_curve.legend(fontsize=8, loc="lower right")
            a_curve.text(
                0.02,
                0.03,
                f"bandwidth={float(pack['bandwidth_score'][i, j]):.3f}\nr2={float(pack['r2'][i, j]):.3f}",
                transform=a_curve.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )

        valid = pack["score"][:, j][np.isfinite(pack["score"][:, j])]
        stats = (
            f"lambda={float(lam):.1f} nm | valid={len(valid)} | mean={float(np.mean(valid)):.3f} | p90={float(np.quantile(valid, 0.9)):.3f}"
            if len(valid)
            else f"lambda={float(lam):.1f} nm | valid=0"
        )
        fig.suptitle(f"Optica-style low-pass top-{topk} for tpp ({stats})", fontsize=12)
        if hm is not None:
            fig.colorbar(hm, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp|")
        fig.savefig(out_dir / f"lambda_{float(lam):.1f}nm_top{topk}.png", dpi=180)
        plt.close(fig)


def plot_overview(
    out_png: Path,
    spec: np.ndarray,
    pack: dict[str, np.ndarray],
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_score: np.ndarray,
    topk: int,
) -> None:
    n_lambda = len(lambdas)
    ncol = 4
    nrow = int(np.ceil(n_lambda / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.6, nrow * 3.3), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for j, lam in enumerate(lambdas):
        ax = axes[j]
        i0 = int(top_idx[j, 0])
        if i0 >= 0 and np.isfinite(top_score[j, 0]):
            sigma0 = float(pack["best_sigma_deg"][i0, j])
            ax.plot(thetas, lowpass_target(thetas, sigma0), "k--", lw=1.2, label="ideal")
        for r in range(topk):
            i = int(top_idx[j, r])
            if i < 0:
                continue
            y = spec[i, j].astype(np.float64)
            if not np.isfinite(y).all():
                continue
            ax.plot(thetas, y / max(float(np.max(y)), 1e-8), lw=1.0)
        ax.set_title(f"{float(lam):.0f}nm")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if i0 >= 0 and np.isfinite(top_score[j, 0]):
            ax.text(
                0.02,
                0.03,
                f"best={float(top_score[j, 0]):.3f}\nsigma={float(pack['best_sigma_deg'][i0, j]):.1f}",
                transform=ax.transAxes,
                fontsize=8,
            )
    for ax in axes[n_lambda:]:
        ax.axis("off")

    fig.suptitle("Optica-style low-pass tpp top spectra", fontsize=13)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score tpp rows against Optica-style Gaussian low-pass targets.")
    p.add_argument("--in_npz", type=Path, default=ROOT / "data" / "train_data.npz")
    p.add_argument("--field", type=str, default="tpp_mag")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "optica_lowpass_scores")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--plot_topk", type=int, default=5)
    p.add_argument("--sigmas_deg", type=float, nargs="+", default=[8.0, 12.0, 16.0])
    p.add_argument("--w_center", type=float, default=0.35)
    p.add_argument("--w_shape", type=float, default=0.35)
    p.add_argument("--w_edge_reject", type=float, default=0.15)
    p.add_argument("--w_bandwidth", type=float, default=0.15)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.in_npz = resolve_from_root(args.in_npz)
    args.out_dir = resolve_from_root(args.out_dir)

    structures, spec, lambdas, thetas = load_bundle(args.in_npz, args.field)
    pack = score_lowpass_spectra(
        spec,
        thetas,
        [float(x) for x in args.sigmas_deg],
        args.w_center,
        args.w_shape,
        args.w_edge_reject,
        args.w_bandwidth,
    )
    top_idx, top_score = topk_per_lambda(pack["score"], args.topk)
    summary = summary_json(lambdas, pack, top_idx[:, : max(1, args.plot_topk)], top_score[:, : max(1, args.plot_topk)])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"{args.field}_optica_lowpass_scores.npz", lambdas=lambdas, thetas=thetas, top_indices=top_idx, top_scores=top_score, **pack)
    save_scores_csv(args.out_dir / f"{args.field}_optica_lowpass_scores.csv", lambdas, pack)
    save_topk_csv(
        args.out_dir / f"{args.field}_optica_lowpass_top{args.plot_topk}_per_lambda.csv",
        lambdas,
        top_idx[:, : max(1, args.plot_topk)],
        top_score[:, : max(1, args.plot_topk)],
        pack,
    )
    with (args.out_dir / f"{args.field}_optica_lowpass_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.plot_topk > 0:
        k = min(args.plot_topk, spec.shape[0])
        plot_dir = args.out_dir / f"{args.field}_optica_lowpass_top{k}_plots"
        plot_per_lambda(plot_dir, structures, spec, pack, lambdas, thetas, top_idx[:, :k], top_score[:, :k], k)
        plot_overview(args.out_dir / f"{args.field}_optica_lowpass_top{k}_overview.png", spec, pack, lambdas, thetas, top_idx[:, :k], top_score[:, :k], k)

    print(f"input: {args.in_npz}")
    print(f"field: {args.field}")
    print(f"samples: {spec.shape[0]}, lambdas: {spec.shape[1]}, thetas: {spec.shape[2]}")
    print(f"sigmas_deg: {[float(x) for x in args.sigmas_deg]}")
    print(
        f"weights: center={args.w_center}, shape={args.w_shape}, "
        f"edge_reject={args.w_edge_reject}, bandwidth={args.w_bandwidth}"
    )
    print(f"plot_topk: {args.plot_topk}")
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
