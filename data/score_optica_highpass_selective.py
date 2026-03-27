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

from score_optica_highpass import highpass_target, score_highpass_spectra
from score_common import load_bundle, topk_per_lambda


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


def passive_silent_metrics(spec: np.ndarray, target_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_max = np.max(spec, axis=2).astype(np.float32)
    row_mean = np.mean(spec, axis=2).astype(np.float32)
    row_min = np.min(spec, axis=2).astype(np.float32)
    silent = np.clip(1.0 - row_max / max(float(target_max), 1e-8), 0.0, 1.0).astype(np.float32)
    return silent, row_min, row_mean, row_max


def combine_scores(
    active_score: np.ndarray,
    silent_score: np.ndarray,
    active_target_min: float,
    passive_target_max: float,
    passive_max: np.ndarray,
    w_active: float,
    w_silent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_active = active_score >= float(active_target_min)
    keep_passive = passive_max <= float(passive_target_max)
    raw_joint = (float(w_active) * active_score + float(w_silent) * silent_score) / max(float(w_active + w_silent), 1e-8)
    keep = keep_active & keep_passive
    joint = np.where(keep, raw_joint, np.nan)
    return joint.astype(np.float32), keep_active, keep_passive


def save_topk_csv(
    path: Path,
    lambdas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    active_pack: dict[str, np.ndarray],
    silent_score: np.ndarray,
    passive_min: np.ndarray,
    passive_mean: np.ndarray,
    passive_max: np.ndarray,
    keep_active: np.ndarray,
    keep_passive: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lambda_idx",
                "lambda_nm",
                "rank",
                "sample_idx",
                "joint_score",
                "active_score",
                "best_sigma_deg",
                "shape_score",
                "center_score",
                "edge_score",
                "silent_score",
                "passive_min",
                "passive_mean",
                "passive_max",
                "active_keep",
                "passive_keep",
            ]
        )
        for j, lam in enumerate(lambdas):
            for r in range(top_idx.shape[1]):
                i = int(top_idx[j, r])
                s = float(top_joint[j, r])
                if i < 0 or not np.isfinite(s):
                    continue
                w.writerow(
                    [
                        j,
                        float(lam),
                        r + 1,
                        i,
                        f"{s:.4f}",
                        f"{float(active_pack['score'][i, j]):.4f}",
                        f"{float(active_pack['best_sigma_deg'][i, j]):.2f}",
                        f"{float(active_pack['shape_score'][i, j]):.4f}",
                        f"{float(active_pack['center_score'][i, j]):.4f}",
                        f"{float(active_pack['edge_score'][i, j]):.4f}",
                        f"{float(silent_score[i, j]):.4f}",
                        f"{float(passive_min[i, j]):.4f}",
                        f"{float(passive_mean[i, j]):.4f}",
                        f"{float(passive_max[i, j]):.4f}",
                        int(bool(keep_active[i, j])),
                        int(bool(keep_passive[i, j])),
                    ]
                )


def summary_json(
    lambdas: np.ndarray,
    joint: np.ndarray,
    active_pack: dict[str, np.ndarray],
    silent_score: np.ndarray,
    passive_min: np.ndarray,
    passive_mean: np.ndarray,
    passive_max: np.ndarray,
    keep_active: np.ndarray,
    keep_passive: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
) -> list[dict]:
    out = []
    for j, lam in enumerate(lambdas):
        col = joint[:, j]
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
                    "joint_score": float(top_joint[j, r]),
                    "active_score": float(active_pack["score"][i, j]),
                    "best_sigma_deg": float(active_pack["best_sigma_deg"][i, j]),
                    "shape_score": float(active_pack["shape_score"][i, j]),
                    "center_score": float(active_pack["center_score"][i, j]),
                    "edge_score": float(active_pack["edge_score"][i, j]),
                    "silent_score": float(silent_score[i, j]),
                    "passive_min": float(passive_min[i, j]),
                    "passive_mean": float(passive_mean[i, j]),
                    "passive_max": float(passive_max[i, j]),
                    "active_keep": bool(keep_active[i, j]),
                    "passive_keep": bool(keep_passive[i, j]),
                }
            )
        out.append(
            {
                "lambda_idx": int(j),
                "lambda_nm": float(lam),
                "num_valid": int(len(valid)),
                "joint_mean": float(np.mean(valid)) if len(valid) else None,
                "joint_p90": float(np.quantile(valid, 0.9)) if len(valid) else None,
                "joint_max": float(np.max(valid)) if len(valid) else None,
                "top": top,
            }
        )
    return out


def plot_per_lambda(
    out_dir: Path,
    structures: np.ndarray,
    active_spec: np.ndarray,
    passive_spec: np.ndarray,
    active_pack: dict[str, np.ndarray],
    joint: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    silent_score: np.ndarray,
    passive_min: np.ndarray,
    passive_mean: np.ndarray,
    passive_max: np.ndarray,
    passive_target_max: float,
    topk: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _clim(spec: np.ndarray) -> tuple[float, float]:
        f = spec[np.isfinite(spec)]
        lo = float(np.quantile(f, 0.01)) if f.size else 0.0
        hi = float(np.quantile(f, 0.99)) if f.size else 1.0
        return lo, max(hi, lo + 1e-6)

    vmin_a, vmax_a = _clim(active_spec)
    vmin_p, vmax_p = _clim(passive_spec)
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    idx0 = int(np.argmin(np.abs(thetas)))
    idxp40 = int(np.argmin(np.abs(thetas - 40.0)))
    idxm40 = int(np.argmin(np.abs(thetas + 40.0)))

    for j, lam in enumerate(lambdas):
        fig, axes = plt.subplots(
            topk,
            5,
            figsize=(22.0, max(2.8 * topk, 6.0)),
            gridspec_kw={"width_ratios": [0.7, 1.0, 1.0, 1.0, 1.0]},
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        hm_a = hm_p = None

        for r in range(topk):
            a_st, a_hma, a_ca, a_hmp, a_cp = axes[r]
            i = int(top_idx[j, r])
            if i < 0:
                for ax in axes[r]:
                    ax.axis("off")
                continue

            sigma = float(active_pack["best_sigma_deg"][i, j])
            target = highpass_target(thetas, sigma)
            y_a = active_spec[i, j].astype(np.float64)
            y_p = passive_spec[i, j].astype(np.float64)
            yn_a = y_a / max(float(np.max(y_a)), 1e-8)

            a_st.imshow(structures[i], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
            a_st.set_title(f"id={i}", fontsize=10)
            a_st.axis("off")

            hm_a = a_hma.imshow(active_spec[i], cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=vmin_a, vmax=vmax_a, interpolation="bicubic")
            a_hma.axhline(float(lam), color="w", ls="--", lw=1.0)
            a_hma.set_title(
                f"tpp high-pass J={float(top_joint[j, r]):.3f}\n"
                f"A={float(active_pack['score'][i, j]):.3f} S={float(silent_score[i, j]):.3f} sigma={sigma:.1f}",
                fontsize=9,
            )
            a_hma.set_xlabel("theta (deg)")
            a_hma.set_ylabel("lambda (nm)")

            a_ca.plot(thetas, target, "k--", lw=1.5, label="ideal hp")
            a_ca.plot(thetas, yn_a, lw=1.8, color="#1f77b4", label="tpp")
            a_ca.set_ylim(-0.05, 1.05)
            a_ca.set_xlabel("theta (deg)")
            a_ca.set_ylabel("normalized |tpp|")
            a_ca.grid(alpha=0.25)
            if r == 0:
                a_ca.legend(fontsize=7, loc="lower right")

            hm_p = a_hmp.imshow(passive_spec[i], cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=vmin_p, vmax=vmax_p, interpolation="bicubic")
            a_hmp.axhline(float(lam), color="w", ls="--", lw=1.0)
            a_hmp.set_title("tss should stay low", fontsize=9)
            a_hmp.set_xlabel("theta (deg)")
            a_hmp.set_ylabel("lambda (nm)")

            a_cp.plot(thetas, y_p, lw=1.8, color="#ff7f0e", label="tss")
            a_cp.axhline(float(passive_target_max), color="k", ls="--", lw=1.0, label=f"target max={passive_target_max:.2f}")
            a_cp.set_xlabel("theta (deg)")
            a_cp.set_ylabel("|tss|")
            a_cp.grid(alpha=0.25)
            if r == 0:
                a_cp.legend(fontsize=7, loc="upper right")
            a_cp.text(
                0.02,
                0.03,
                f"min={float(passive_min[i, j]):.3f}\nmean={float(passive_mean[i, j]):.3f}\nmax={float(passive_max[i, j]):.3f}\n"
                f"0deg={float(y_p[idx0]):.3f}  +40={float(y_p[idxp40]):.3f}  -40={float(y_p[idxm40]):.3f}",
                transform=a_cp.transAxes,
                fontsize=8,
                va="bottom",
            )

        valid = joint[:, j][np.isfinite(joint[:, j])]
        stats = (
            f"lambda={float(lam):.0f}nm | valid={len(valid)} | mean={float(np.mean(valid)):.3f} | p90={float(np.quantile(valid, 0.9)):.3f}"
            if len(valid)
            else f"lambda={float(lam):.0f}nm | valid=0"
        )
        fig.suptitle(f"Polarization-selective high-pass Top-{topk} ({stats})", fontsize=12)
        if hm_a is not None:
            fig.colorbar(hm_a, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp|")
        if hm_p is not None:
            fig.colorbar(hm_p, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label="|tss|")
        fig.savefig(out_dir / f"lambda_{float(lam):.1f}nm_top{topk}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarization-selective Optica high-pass: tpp high-pass, tss low.")
    p.add_argument("--in_npz", type=Path, default=ROOT / "data" / "train_data.npz")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "optica_highpass_selective_scores")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--plot_topk", type=int, default=5)
    p.add_argument("--sigmas_deg", type=float, nargs="+", default=[8.0, 12.0, 16.0])
    p.add_argument("--w_center", type=float, default=0.35)
    p.add_argument("--w_shape", type=float, default=0.35)
    p.add_argument("--w_edge", type=float, default=0.15)
    p.add_argument("--w_bandwidth", type=float, default=0.15)
    p.add_argument("--w_active", type=float, default=1.0)
    p.add_argument("--w_silent", type=float, default=1.0)
    p.add_argument("--active_target_min", type=float, default=0.7)
    p.add_argument("--passive_target_max", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.in_npz = resolve_from_root(args.in_npz)
    args.out_dir = resolve_from_root(args.out_dir)

    structures, tpp, lambdas, thetas = load_bundle(args.in_npz, "tpp_mag")
    _, tss, _, _ = load_bundle(args.in_npz, "tss_mag")

    active_pack = score_highpass_spectra(
        tpp,
        thetas,
        [float(x) for x in args.sigmas_deg],
        args.w_center,
        args.w_shape,
        args.w_edge,
        args.w_bandwidth,
    )
    silent_score, passive_min, passive_mean, passive_max = passive_silent_metrics(tss, args.passive_target_max)
    joint, keep_active, keep_passive = combine_scores(
        active_pack["score"],
        silent_score,
        args.active_target_min,
        args.passive_target_max,
        passive_max,
        args.w_active,
        args.w_silent,
    )
    top_idx, top_joint = topk_per_lambda(joint, args.topk)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / "p_highpass_s_silent_scores.npz",
        lambdas=lambdas,
        thetas=thetas,
        top_indices=top_idx,
        top_scores=top_joint,
        joint_score=joint,
        silent_score=silent_score,
        passive_min=passive_min,
        passive_mean=passive_mean,
        passive_max=passive_max,
        keep_active=keep_active,
        keep_passive=keep_passive,
        **active_pack,
    )
    save_topk_csv(
        args.out_dir / f"p_highpass_s_silent_top{args.plot_topk}_per_lambda.csv",
        lambdas,
        top_idx[:, : max(1, args.plot_topk)],
        top_joint[:, : max(1, args.plot_topk)],
        active_pack,
        silent_score,
        passive_min,
        passive_mean,
        passive_max,
        keep_active,
        keep_passive,
    )
    with (args.out_dir / "p_highpass_s_silent_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            summary_json(
                lambdas,
                joint,
                active_pack,
                silent_score,
                passive_min,
                passive_mean,
                passive_max,
                keep_active,
                keep_passive,
                top_idx[:, : max(1, args.plot_topk)],
                top_joint[:, : max(1, args.plot_topk)],
            ),
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.plot_topk > 0:
        k = min(args.plot_topk, tpp.shape[0])
        plot_per_lambda(
            args.out_dir / f"top{k}_plots",
            structures,
            tpp,
            tss,
            active_pack,
            joint,
            lambdas,
            thetas,
            top_idx[:, :k],
            top_joint[:, :k],
            silent_score,
            passive_min,
            passive_mean,
            passive_max,
            args.passive_target_max,
            k,
        )

    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
