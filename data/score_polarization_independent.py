"""Polarization-independent second-order scoring.

Scores each structure by min(tpp_score, tss_score) per lambda,
so only structures with good second-order response in BOTH
polarizations rank high.

Usage:
    python data/score_polarization_independent.py
    python data/score_polarization_independent.py --topk 20 --plot_topk 5
"""
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

from score_common import (
    load_bundle,
    score_spectra,
    target_profile,
    topk_per_lambda,
)


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


# ── CSV / JSON helpers ──────────────────────────────────────────────

def save_topk_csv(
    path: Path,
    lambdas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    tpp_score: np.ndarray,
    tss_score: np.ndarray,
    match40_score: np.ndarray,
    tpp_p40: np.ndarray,
    tss_p40: np.ndarray,
    tpp_m40: np.ndarray,
    tss_m40: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda_idx", "lambda_nm", "rank", "sample_idx",
                     "joint_score", "tpp_score", "tss_score", "match40_score",
                     "tpp_+40", "tss_+40", "tpp_-40", "tss_-40"])
        for j, lam in enumerate(lambdas):
            for r in range(top_idx.shape[1]):
                i = int(top_idx[j, r])
                js = float(top_joint[j, r])
                if i < 0 or not np.isfinite(js):
                    continue
                w.writerow([
                    j, float(lam), r + 1, i, f"{js:.4f}",
                    f"{float(tpp_score[i, j]):.4f}",
                    f"{float(tss_score[i, j]):.4f}",
                    f"{float(match40_score[i, j]):.4f}",
                    f"{float(tpp_p40[i, j]):.4f}",
                    f"{float(tss_p40[i, j]):.4f}",
                    f"{float(tpp_m40[i, j]):.4f}",
                    f"{float(tss_m40[i, j]):.4f}",
                ])


def summary_json(
    lambdas: np.ndarray,
    joint: np.ndarray,
    tpp: np.ndarray,
    tss: np.ndarray,
    match40: np.ndarray,
    tpp_p40: np.ndarray,
    tss_p40: np.ndarray,
    tpp_m40: np.ndarray,
    tss_m40: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
) -> list[dict]:
    out = []
    for j, lam in enumerate(lambdas):
        col = joint[:, j]
        valid = col[np.isfinite(col)]
        top_details = []
        for r in range(top_idx.shape[1]):
            i = int(top_idx[j, r])
            if i < 0:
                continue
            top_details.append({
                "rank": r + 1,
                "sample_idx": i,
                "joint": float(top_joint[j, r]),
                "tpp": float(tpp[i, j]),
                "tss": float(tss[i, j]),
                "match40": float(match40[i, j]),
                "tpp_+40": float(tpp_p40[i, j]),
                "tss_+40": float(tss_p40[i, j]),
                "tpp_-40": float(tpp_m40[i, j]),
                "tss_-40": float(tss_m40[i, j]),
            })
        out.append({
            "lambda_idx": int(j),
            "lambda_nm": float(lam),
            "num_valid": int(len(valid)),
            "joint_mean": float(np.mean(valid)) if len(valid) else None,
            "joint_p90": float(np.quantile(valid, 0.9)) if len(valid) else None,
            "joint_max": float(np.max(valid)) if len(valid) else None,
            "top": top_details,
        })
    return out


# ── Visualization ───────────────────────────────────────────────────

def plot_per_lambda(
    out_dir: Path,
    structures: np.ndarray,
    tpp_spec: np.ndarray,
    tss_spec: np.ndarray,
    joint_score: np.ndarray,
    tpp_score: np.ndarray,
    tss_score: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    match40_score: np.ndarray,
    tpp_p40: np.ndarray,
    tss_p40: np.ndarray,
    tpp_m40: np.ndarray,
    tss_m40: np.ndarray,
    topk: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = target_profile(thetas)

    # shared color limits
    def _clim(spec):
        f = spec[np.isfinite(spec)]
        lo = float(np.quantile(f, 0.01)) if f.size else 0.0
        hi = float(np.quantile(f, 0.99)) if f.size else 1.0
        return lo, max(hi, lo + 1e-6)

    vmin_p, vmax_p = _clim(tpp_spec)
    vmin_s, vmax_s = _clim(tss_spec)
    extent = [float(thetas[0]), float(thetas[-1]),
              float(lambdas[0]), float(lambdas[-1])]

    for j, lam in enumerate(lambdas):
        # columns: struct | tpp heatmap | tpp curve | tss heatmap | tss curve
        ncols = 5
        fig, axes = plt.subplots(
            topk, ncols,
            figsize=(22.0, max(2.6 * topk, 5.0)),
            gridspec_kw={"width_ratios": [0.7, 1.0, 1.0, 1.0, 1.0]},
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        hm_p = hm_s = None

        for r in range(topk):
            a_st, a_hmp, a_cp, a_hms, a_cs = axes[r]
            i = int(top_idx[j, r])
            if i < 0:
                for ax in axes[r]:
                    ax.axis("off")
                continue

            # structure
            a_st.imshow(structures[i], cmap="gray_r",
                        interpolation="nearest", vmin=0.0, vmax=1.0)
            a_st.set_title(f"id={i}", fontsize=10)
            a_st.axis("off")

            # tpp heatmap
            hm_p = a_hmp.imshow(tpp_spec[i], cmap="turbo", aspect="auto",
                                origin="lower", extent=extent,
                                vmin=vmin_p, vmax=vmax_p, interpolation="bicubic")
            a_hmp.axhline(float(lam), color="w", ls="--", lw=1.0)
            jt = float(top_joint[j, r])
            tp = float(tpp_score[i, j])
            ts = float(tss_score[i, j])
            m40 = float(match40_score[i, j])
            pp = float(tpp_p40[i, j])
            sp = float(tss_p40[i, j])
            pm = float(tpp_m40[i, j])
            sm = float(tss_m40[i, j])
            a_hmp.set_title(
                f"tpp  J={jt:.3f} P={tp:.3f} S={ts:.3f}\n"
                f"M40={m40:.3f}  +40: {pp:.3f}/{sp:.3f}  -40: {pm:.3f}/{sm:.3f}",
                fontsize=9,
            )
            a_hmp.set_xlabel("theta")
            a_hmp.set_ylabel("lambda (nm)")

            # tpp curve
            y_p = tpp_spec[i, j].astype(np.float64)
            yn_p = y_p / max(float(np.max(y_p)), 1e-8)
            a_cp.plot(thetas, target, "k--", lw=1.5, label="|sin|^2")
            a_cp.plot(thetas, yn_p, lw=1.8, color="#1f77b4", label="tpp")
            a_cp.set_ylim(-0.05, 1.05)
            a_cp.set_xlabel("theta")
            a_cp.grid(alpha=0.25)
            if r == 0:
                a_cp.legend(fontsize=7, loc="lower right")

            # tss heatmap
            hm_s = a_hms.imshow(tss_spec[i], cmap="turbo", aspect="auto",
                                origin="lower", extent=extent,
                                vmin=vmin_s, vmax=vmax_s, interpolation="bicubic")
            a_hms.axhline(float(lam), color="w", ls="--", lw=1.0)
            a_hms.set_title("tss", fontsize=9)
            a_hms.set_xlabel("theta")

            # tss curve
            y_s = tss_spec[i, j].astype(np.float64)
            yn_s = y_s / max(float(np.max(y_s)), 1e-8)
            a_cs.plot(thetas, target, "k--", lw=1.5)
            a_cs.plot(thetas, yn_s, lw=1.8, color="#ff7f0e", label="tss")
            a_cs.set_ylim(-0.05, 1.05)
            a_cs.set_xlabel("theta")
            a_cs.grid(alpha=0.25)
            if r == 0:
                a_cs.legend(fontsize=7, loc="lower right")

        valid = joint_score[:, j][np.isfinite(joint_score[:, j])]
        stats = (f"lambda={float(lam):.0f}nm | valid={len(valid)} "
                 f"| mean={float(np.mean(valid)):.3f} "
                 f"| p90={float(np.quantile(valid, 0.9)):.3f}") if len(valid) else f"lambda={float(lam):.0f}nm | valid=0"
        fig.suptitle(f"Pol-independent Top-{topk} ({stats})", fontsize=12)
        if hm_p is not None:
            fig.colorbar(hm_p, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp|")
        if hm_s is not None:
            fig.colorbar(hm_s, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label="|tss|")
        fig.savefig(out_dir / f"lambda_{float(lam):.1f}nm_top{topk}.png", dpi=180)
        plt.close(fig)


def plot_overview(
    out_png: Path,
    tpp_spec: np.ndarray,
    tss_spec: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    match40_score: np.ndarray,
    topk: int,
) -> None:
    target = target_profile(thetas)
    n_lambda = len(lambdas)
    ncol = 4
    nrow = int(np.ceil(n_lambda / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 5.0, nrow * 3.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for j, lam in enumerate(lambdas):
        ax = axes[j]
        ax.plot(thetas, target, "k--", lw=1.2, label="|sin|^2")
        for r in range(topk):
            i = int(top_idx[j, r])
            if i < 0:
                continue
            yp = tpp_spec[i, j].astype(np.float64)
            ys = tss_spec[i, j].astype(np.float64)
            if not (np.isfinite(yp).all() and np.isfinite(ys).all()):
                continue
            ypn = yp / max(float(np.max(yp)), 1e-8)
            ysn = ys / max(float(np.max(ys)), 1e-8)
            c = plt.cm.tab10(r)
            ax.plot(thetas, ypn, lw=1.0, color=c, ls="-")
            ax.plot(thetas, ysn, lw=1.0, color=c, ls="--")
        ax.set_title(f"{float(lam):.0f}nm")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if np.isfinite(top_joint[j, 0]):
            i0 = int(top_idx[j, 0])
            m40 = float(match40_score[i0, j]) if i0 >= 0 else float("nan")
            ax.text(0.02, 0.03, f"best={float(top_joint[j, 0]):.3f}\nm40={m40:.3f}",
                    transform=ax.transAxes, fontsize=8)
    for ax in axes[n_lambda:]:
        ax.axis("off")

    fig.suptitle(f"Pol-independent Top-{topk} (solid=tpp, dashed=tss)", fontsize=13)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def score_match_at_angle(
    tpp_spec: np.ndarray,
    tss_spec: np.ndarray,
    thetas: np.ndarray,
    angle_deg: float = 40.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_p = int(np.argmin(np.abs(thetas - float(angle_deg))))
    idx_m = int(np.argmin(np.abs(thetas + float(angle_deg))))
    tpp_p = np.asarray(tpp_spec[:, :, idx_p], dtype=np.float64)
    tss_p = np.asarray(tss_spec[:, :, idx_p], dtype=np.float64)
    tpp_m = np.asarray(tpp_spec[:, :, idx_m], dtype=np.float64)
    tss_m = np.asarray(tss_spec[:, :, idx_m], dtype=np.float64)
    diff = 0.5 * (np.abs(tpp_p - tss_p) + np.abs(tpp_m - tss_m))
    scale = float(
        max(
            np.nanmax(np.abs(tpp_spec)),
            np.nanmax(np.abs(tss_spec)),
            1e-8,
        )
    )
    match = np.clip(1.0 - diff / scale, 0.0, 1.0)
    return match, tpp_p, tss_p, tpp_m, tss_m


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Polarization-independent scoring: joint = min(tpp, tss)")
    p.add_argument("--in_npz", type=Path,
                   default=ROOT / "data" / "train_data.npz")
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "data" / "polarization_independent_scores")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--w_center", type=float, default=0.6)
    p.add_argument("--w_shape", type=float, default=0.3)
    p.add_argument("--w_edge", type=float, default=0.1)
    p.add_argument("--w_bandwidth", type=float, default=0.2)
    p.add_argument("--w_match40", type=float, default=0.3)
    p.add_argument("--plot_topk", type=int, default=5)
    args = p.parse_args()

    args.in_npz = resolve_from_root(args.in_npz)
    args.out_dir = resolve_from_root(args.out_dir)

    # load both polarizations
    structures, tpp_spec, lambdas, thetas = load_bundle(args.in_npz, "tpp_mag")
    _, tss_spec, _, _ = load_bundle(args.in_npz, "tss_mag")

    # score each polarization independently
    tpp_pack = score_spectra(tpp_spec, thetas,
                             args.w_center, args.w_shape, args.w_edge,
                             args.w_bandwidth)
    tss_pack = score_spectra(tss_spec, thetas,
                             args.w_center, args.w_shape, args.w_edge,
                             args.w_bandwidth)
    tpp_s = tpp_pack["score"]
    tss_s = tss_pack["score"]
    match40_s, tpp_p40, tss_p40, tpp_m40, tss_m40 = score_match_at_angle(tpp_spec, tss_spec, thetas, angle_deg=40.0)

    # joint emphasizes both polarizations being good and also close at +-40deg
    base_joint = np.minimum(tpp_s, tss_s)
    joint = (1.0 - float(args.w_match40)) * base_joint + float(args.w_match40) * match40_s

    # top-k per lambda
    top_idx, top_joint = topk_per_lambda(joint, args.topk)

    # save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        args.out_dir / "joint_scores.npz",
        lambdas=lambdas, thetas=thetas,
        joint_score=joint, base_joint_score=base_joint, tpp_score=tpp_s, tss_score=tss_s,
        match40_score=match40_s,
        tpp_p40=tpp_p40, tss_p40=tss_p40, tpp_m40=tpp_m40, tss_m40=tss_m40,
        top_indices=top_idx, top_scores=top_joint,
    )

    k_csv = min(args.topk, structures.shape[0])
    save_topk_csv(
        args.out_dir / f"joint_top{k_csv}_per_lambda.csv",
        lambdas, top_idx[:, :k_csv], top_joint[:, :k_csv],
        tpp_s, tss_s, match40_s, tpp_p40, tss_p40, tpp_m40, tss_m40,
    )

    summ = summary_json(
        lambdas, joint, tpp_s, tss_s,
        match40_s, tpp_p40, tss_p40, tpp_m40, tss_m40,
        top_idx, top_joint,
    )
    with (args.out_dir / "joint_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)

    # plots
    if args.plot_topk > 0:
        k = min(args.plot_topk, structures.shape[0])
        plot_per_lambda(
            args.out_dir / f"joint_top{k}_plots",
            structures, tpp_spec, tss_spec,
            joint, tpp_s, tss_s,
            lambdas, thetas,
            top_idx[:, :k], top_joint[:, :k],
            match40_s, tpp_p40, tss_p40, tpp_m40, tss_m40,
            k,
        )
        plot_overview(
            args.out_dir / f"joint_top{k}_overview.png",
            tpp_spec, tss_spec,
            lambdas, thetas,
            top_idx[:, :k], top_joint[:, :k], match40_s, k,
        )

    print(f"input:   {args.in_npz}")
    print(f"samples: {tpp_spec.shape[0]}, lambdas: {len(lambdas)}, thetas: {len(thetas)}")
    print(f"scoring: joint = {(1.0 - float(args.w_match40)):.2f} * min(tpp_score, tss_score) + {float(args.w_match40):.2f} * match40")
    print(f"weights: center={args.w_center}, shape={args.w_shape}, "
          f"edge={args.w_edge}, bandwidth={args.w_bandwidth}")
    print(f"topk:    {args.topk}, plot_topk: {args.plot_topk}")
    print(f"saved:   {args.out_dir}")

    # print per-lambda best
    print("\n--- Per-lambda best (joint / tpp / tss / match40) ---")
    for j, lam in enumerate(lambdas):
        i = int(top_idx[j, 0])
        if i < 0:
            print(f"  {float(lam):7.1f} nm: no valid sample")
            continue
        print(f"  {float(lam):7.1f} nm: id={i:5d}  "
              f"joint={float(top_joint[j, 0]):.3f}  "
              f"tpp={float(tpp_s[i, j]):.3f}  "
              f"tss={float(tss_s[i, j]):.3f}  "
              f"m40={float(match40_s[i, j]):.3f}")


if __name__ == "__main__":
    main()
