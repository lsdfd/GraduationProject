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

from score_common import load_bundle, score_spectra, target_profile, topk_per_lambda


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


def passive_high_metrics(spec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_min = np.min(spec, axis=2)
    row_mean = np.mean(spec, axis=2)
    row_max = np.max(spec, axis=2)
    return (
        np.asarray(row_min, dtype=np.float32),
        np.asarray(row_mean, dtype=np.float32),
        np.asarray(row_max, dtype=np.float32),
    )


def apply_filters(
    active_second_order: np.ndarray,
    active_target_min: float,
    passive_min: np.ndarray,
    passive_target_min: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_active = active_second_order >= float(active_target_min)
    keep_passive = passive_min >= float(passive_target_min)
    keep = keep_active & keep_passive
    filtered = np.where(keep, active_second_order, np.nan).astype(np.float32)
    return filtered, keep, keep_active


def save_topk_csv(
    path: Path,
    lambdas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    active_score: np.ndarray,
    passive_min: np.ndarray,
    passive_mean: np.ndarray,
    passive_max: np.ndarray,
    keep_mask: np.ndarray,
    keep_active_mask: np.ndarray,
    mode: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "lambda_idx",
                "lambda_nm",
                "rank",
                "sample_idx",
                "joint_score",
                "active_second_order_score",
                "active_keep",
                "passive_keep",
                "passive_min",
                "passive_mean",
                "passive_max",
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
                        mode,
                        j,
                        float(lam),
                        r + 1,
                        i,
                        f"{s:.4f}",
                        f"{float(active_score[i, j]):.4f}",
                        int(bool(keep_active_mask[i, j])),
                        int(bool(keep_mask[i, j])),
                        f"{float(passive_min[i, j]):.4f}",
                        f"{float(passive_mean[i, j]):.4f}",
                        f"{float(passive_max[i, j]):.4f}",
                    ]
                )


def summary_json(
    mode: str,
    lambdas: np.ndarray,
    joint: np.ndarray,
    active_score: np.ndarray,
    passive_min: np.ndarray,
    passive_mean: np.ndarray,
    passive_max: np.ndarray,
    keep_mask: np.ndarray,
    keep_active_mask: np.ndarray,
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
                    "joint": float(top_joint[j, r]),
                    "active_second_order_score": float(active_score[i, j]),
                    "active_keep": bool(keep_active_mask[i, j]),
                    "passive_keep": bool(keep_mask[i, j]),
                    "passive_min": float(passive_min[i, j]),
                    "passive_mean": float(passive_mean[i, j]),
                    "passive_max": float(passive_max[i, j]),
                }
            )
        out.append(
            {
                "mode": mode,
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
    joint: np.ndarray,
    active_score: np.ndarray,
    passive_min: np.ndarray,
    passive_mean: np.ndarray,
    passive_max: np.ndarray,
    keep_mask: np.ndarray,
    keep_active_mask: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_joint: np.ndarray,
    topk: int,
    mode: str,
    active_label: str,
    passive_label: str,
    passive_target_min: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = target_profile(thetas)

    def _clim(spec: np.ndarray) -> tuple[float, float]:
        f = spec[np.isfinite(spec)]
        lo = float(np.quantile(f, 0.01)) if f.size else 0.0
        hi = float(np.quantile(f, 0.99)) if f.size else 1.0
        return lo, max(hi, lo + 1e-6)

    vmin_a, vmax_a = _clim(active_spec)
    vmin_p, vmax_p = _clim(passive_spec)
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]

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

            a_st.imshow(structures[i], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
            a_st.set_title(f"id={i}", fontsize=10)
            a_st.axis("off")

            hm_a = a_hma.imshow(
                active_spec[i],
                cmap="turbo",
                aspect="auto",
                origin="lower",
                extent=extent,
                vmin=vmin_a,
                vmax=vmax_a,
                interpolation="bicubic",
            )
            a_hma.axhline(float(lam), color="w", ls="--", lw=1.0)
            js = float(top_joint[j, r])
            act = float(active_score[i, j])
            pmin = float(passive_min[i, j])
            pmean = float(passive_mean[i, j])
            pmax = float(passive_max[i, j])
            keep = bool(keep_mask[i, j])
            keep_active = bool(keep_active_mask[i, j])
            a_hma.set_title(
                f"{active_label} joint={js:.3f}\n"
                f"second-order={act:.3f} active_ok={keep_active} keep={keep}",
                fontsize=9,
            )
            a_hma.set_xlabel("theta")
            a_hma.set_ylabel("lambda (nm)")

            y_a = active_spec[i, j].astype(np.float64)
            yn_a = y_a / max(float(np.max(y_a)), 1e-8)
            a_ca.plot(thetas, target, "k--", lw=1.5, label="|sin|^2")
            a_ca.plot(thetas, yn_a, lw=1.8, color="#1f77b4", label=active_label)
            a_ca.set_ylim(-0.05, 1.05)
            a_ca.set_xlabel("theta")
            a_ca.grid(alpha=0.25)
            if r == 0:
                a_ca.legend(fontsize=7, loc="lower right")

            hm_p = a_hmp.imshow(
                passive_spec[i],
                cmap="turbo",
                aspect="auto",
                origin="lower",
                extent=extent,
                vmin=vmin_p,
                vmax=vmax_p,
                interpolation="bicubic",
            )
            a_hmp.axhline(float(lam), color="w", ls="--", lw=1.0)
            a_hmp.set_title(f"{passive_label} high-trans", fontsize=9)
            a_hmp.set_xlabel("theta")
            a_hmp.set_ylabel("lambda (nm)")

            y_p = passive_spec[i, j].astype(np.float64)
            a_cp.plot(thetas, y_p, lw=1.8, color="#ff7f0e", label=passive_label)
            a_cp.axhline(float(passive_target_min), color="k", ls="--", lw=1.0, label=f"{passive_target_min:.1f} target")
            a_cp.set_xlabel("theta")
            a_cp.set_ylabel(f"|{passive_label}|")
            a_cp.grid(alpha=0.25)
            if r == 0:
                a_cp.legend(fontsize=7, loc="lower right")
            a_cp.text(
                0.02,
                0.03,
                f"min={pmin:.3f}\nmean={pmean:.3f}\nmax={pmax:.3f}",
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
        fig.suptitle(f"{mode} Top-{topk} ({stats})", fontsize=12)
        if hm_a is not None:
            fig.colorbar(hm_a, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label=f"|{active_label}|")
        if hm_p is not None:
            fig.colorbar(hm_p, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label=f"|{passive_label}|")
        fig.savefig(out_dir / f"{mode}_lambda_{float(lam):.1f}nm_top{topk}.png", dpi=180)
        plt.close(fig)


def run_mode(
    out_dir: Path,
    mode: str,
    active_label: str,
    passive_label: str,
    structures: np.ndarray,
    active_spec: np.ndarray,
    passive_spec: np.ndarray,
    active_score: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    topk: int,
    plot_topk: int,
    active_target_min: float,
    passive_target_min: float,
) -> None:
    passive_min, passive_mean, passive_max = passive_high_metrics(passive_spec)
    joint, keep_mask, keep_active_mask = apply_filters(
        active_score,
        active_target_min,
        passive_min,
        passive_target_min,
    )
    top_idx, top_joint = topk_per_lambda(joint, topk)

    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        mode_dir / f"{mode}_scores.npz",
        lambdas=lambdas,
        thetas=thetas,
        joint_score=joint,
        active_second_order_score=active_score,
        active_keep=keep_active_mask,
        passive_keep=keep_mask,
        passive_min=passive_min,
        passive_mean=passive_mean,
        passive_max=passive_max,
        top_indices=top_idx,
        top_scores=top_joint,
    )

    save_topk_csv(
        mode_dir / f"{mode}_top{topk}_per_lambda.csv",
        lambdas,
        top_idx,
        top_joint,
        active_score,
        passive_min,
        passive_mean,
        passive_max,
        keep_mask,
        keep_active_mask,
        mode,
    )

    summary = summary_json(
        mode,
        lambdas,
        joint,
        active_score,
        passive_min,
        passive_mean,
        passive_max,
        keep_mask,
        keep_active_mask,
        top_idx,
        top_joint,
    )
    with (mode_dir / f"{mode}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if plot_topk > 0:
        k = min(plot_topk, structures.shape[0])
        plot_per_lambda(
            mode_dir / f"{mode}_top{k}_plots",
            structures,
            active_spec,
            passive_spec,
            joint,
            active_score,
            passive_min,
            passive_mean,
            passive_max,
            keep_mask,
            keep_active_mask,
            lambdas,
            thetas,
            top_idx[:, :k],
            top_joint[:, :k],
            k,
            mode,
            active_label,
            passive_label,
            passive_target_min,
        )

    print(f"\n--- {mode} best per lambda ---")
    for j, lam in enumerate(lambdas):
        i = int(top_idx[j, 0])
        if i < 0:
            print(f"  {float(lam):7.1f} nm: no valid sample")
            continue
        print(
            f"  {float(lam):7.1f} nm: id={i:5d}  "
            f"joint={float(top_joint[j, 0]):.3f}  "
            f"active_second_order={float(active_score[i, j]):.3f}  "
            f"passive_min={float(passive_min[i, j]):.3f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Polarization multiplexed screening: one polarization high second-order, the other high transmission over all angles.")
    p.add_argument("--in_npz", type=Path, default=ROOT / "data" / "train_data.npz")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "polarization_multiplexed_hightrans_scores")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--plot_topk", type=int, default=5)
    p.add_argument("--w_center", type=float, default=0.6)
    p.add_argument("--w_shape", type=float, default=0.3)
    p.add_argument("--w_edge", type=float, default=0.1)
    p.add_argument("--w_bandwidth", type=float, default=0.2)
    p.add_argument("--active_target_min", type=float, default=0.8, help="Hard minimum second-order score for the active polarization.")
    p.add_argument("--passive_target_min", type=float, default=0.9, help="Desired minimum value across all theta for the passive high-trans polarization.")
    args = p.parse_args()

    args.in_npz = resolve_from_root(args.in_npz)
    args.out_dir = resolve_from_root(args.out_dir)

    structures, tpp_spec, lambdas, thetas = load_bundle(args.in_npz, "tpp_mag")
    _, tss_spec, _, _ = load_bundle(args.in_npz, "tss_mag")

    tpp_pack = score_spectra(tpp_spec, thetas, args.w_center, args.w_shape, args.w_edge, args.w_bandwidth)
    tss_pack = score_spectra(tss_spec, thetas, args.w_center, args.w_shape, args.w_edge, args.w_bandwidth)
    tpp_s = tpp_pack["score"]
    tss_s = tss_pack["score"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_mode(
        args.out_dir,
        mode="p_active_s_high",
        active_label="tpp",
        passive_label="tss",
        structures=structures,
        active_spec=tpp_spec,
        passive_spec=tss_spec,
        active_score=tpp_s,
        lambdas=lambdas,
        thetas=thetas,
        topk=args.topk,
        plot_topk=args.plot_topk,
        active_target_min=args.active_target_min,
        passive_target_min=args.passive_target_min,
    )

    run_mode(
        args.out_dir,
        mode="s_active_p_high",
        active_label="tss",
        passive_label="tpp",
        structures=structures,
        active_spec=tss_spec,
        passive_spec=tpp_spec,
        active_score=tss_s,
        lambdas=lambdas,
        thetas=thetas,
        topk=args.topk,
        plot_topk=args.plot_topk,
        active_target_min=args.active_target_min,
        passive_target_min=args.passive_target_min,
    )

    print(f"\ninput: {args.in_npz}")
    print(f"samples: {tpp_spec.shape[0]}, lambdas: {len(lambdas)}, thetas: {len(thetas)}")
    print(
        f"active second-order weights: center={args.w_center}, shape={args.w_shape}, "
        f"edge={args.w_edge}, bandwidth={args.w_bandwidth}"
    )
    print(f"active hard threshold: second-order >= {args.active_target_min}")
    print(f"passive hard threshold over all theta: min >= {args.passive_target_min}")
    print(f"topk: {args.topk}, plot_topk: {args.plot_topk}")
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
