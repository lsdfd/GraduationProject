from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


def load_bundle(npz_path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    data = np.load(npz_path)
    if "structures" not in data.files:
        raise ValueError(f"structures not found in {npz_path}")
    if field not in data.files:
        raise ValueError(f"{field} not found. available: {list(data.files)}")

    structures = np.asarray(data["structures"], dtype=np.float32)
    spec = np.asarray(data[field], dtype=np.float32)
    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures should be [N,64,64], got {structures.shape}")
    if spec.ndim != 3:
        raise ValueError(f"{field} should be [N,L,T], got {spec.shape}")
    if structures.shape[0] != spec.shape[0]:
        raise ValueError(f"structures/spec sample mismatch: {structures.shape[0]} vs {spec.shape[0]}")

    lambdas = np.asarray(data["lambdas"], dtype=np.float32) if "lambdas" in data.files else np.arange(spec.shape[1], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32) if "thetas" in data.files else np.arange(spec.shape[2], dtype=np.float32)
    if spec.shape[1] != len(lambdas) or spec.shape[2] != len(thetas):
        raise ValueError("spec shape and lambdas/thetas mismatch")

    return structures, spec, lambdas, thetas


def load_companion_spec(npz_path: Path, field: str) -> tuple[np.ndarray | None, str | None]:
    if field == "tpp_mag":
        other = "tss_mag"
    elif field == "tss_mag":
        other = "tpp_mag"
    else:
        return None, None

    data = np.load(npz_path)
    if other not in data.files:
        return None, None
    spec = np.asarray(data[other], dtype=np.float32)
    if spec.ndim != 3:
        return None, None
    return spec, other


def target_profile(thetas_deg: np.ndarray) -> np.ndarray:
    tmax = float(np.max(np.abs(thetas_deg)))
    if tmax <= 0:
        return np.zeros_like(thetas_deg, dtype=np.float32)
    kx = np.sin(np.deg2rad(thetas_deg)) / np.sin(np.deg2rad(tmax))
    x = np.abs(kx) ** 2
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-8)
    return x.astype(np.float32)


def score_spectra(
    spec: np.ndarray,
    thetas_deg: np.ndarray,
    w_center: float,
    w_shape: float,
    w_edge: float,
) -> dict[str, np.ndarray]:
    n, l, _ = spec.shape
    x = target_profile(thetas_deg).astype(np.float64)
    center_idx = int(np.argmin(np.abs(thetas_deg)))
    edge_mask = np.abs(thetas_deg) >= 0.85 * float(np.max(np.abs(thetas_deg)))
    if not edge_mask.any():
        edge_mask[[0, -1]] = True

    global_scale = max(float(np.nanquantile(spec, 0.99)), 1e-8)

    score = np.full((n, l), np.nan, dtype=np.float32)
    center_s = np.full_like(score, np.nan)
    shape_s = np.full_like(score, np.nan)
    edge_s = np.full_like(score, np.nan)
    coef_a = np.full_like(score, np.nan)
    fit_mse = np.full_like(score, np.nan)
    r2 = np.full_like(score, np.nan)

    denom = max(float(np.sum(x * x)), 1e-8)
    for i in range(n):
        for j in range(l):
            y = spec[i, j].astype(np.float64)
            if not np.isfinite(y).all():
                continue

            y_norm = y / max(float(np.max(y)), 1e-8)
            a = float(np.sum(x * y_norm) / denom)  # y ~ a*|kx|^2
            y_fit = a * x

            mse = float(np.mean((y_norm - y_fit) ** 2))
            ss_res = float(np.sum((y_norm - y_fit) ** 2))
            ss_tot = float(np.sum((y_norm - np.mean(y_norm)) ** 2))
            this_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

            edge_mean = float(np.mean(y[edge_mask]))
            center_val = float(y[center_idx])
            c_score = float(np.clip(1.0 - center_val / max(edge_mean, 1e-8), 0.0, 1.0))
            s_score = float(np.clip(this_r2, 0.0, 1.0)) if a >= 0 else 0.0
            e_score = float(np.clip(edge_mean / global_scale, 0.0, 1.0))

            score[i, j] = w_center * c_score + w_shape * s_score + w_edge * e_score
            center_s[i, j] = c_score
            shape_s[i, j] = s_score
            edge_s[i, j] = e_score
            coef_a[i, j] = a
            fit_mse[i, j] = mse
            r2[i, j] = this_r2

    return {
        "score": score,
        "center_score": center_s,
        "shape_score": shape_s,
        "edge_score": edge_s,
        "coef_a": coef_a,
        "fit_mse": fit_mse,
        "r2": r2,
    }


def topk_per_lambda(score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n, l = score.shape
    k = min(k, n)
    idx = np.full((l, k), -1, dtype=np.int32)
    val = np.full((l, k), np.nan, dtype=np.float32)
    for j in range(l):
        col = score[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        order = np.argsort(col[valid])[::-1]
        sel = np.where(valid)[0][order[:k]]
        idx[j, : len(sel)] = sel
        val[j, : len(sel)] = col[sel]
    return idx, val


def save_scores_csv(path: Path, lambdas: np.ndarray, pack: dict[str, np.ndarray]) -> None:
    keys = ["score", "center_score", "shape_score", "edge_score", "coef_a", "fit_mse", "r2"]
    n, l = pack["score"].shape
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "lambda_idx", "lambda_nm", *keys])
        for i in range(n):
            for j in range(l):
                w.writerow([i, j, float(lambdas[j]), *(float(pack[k][i, j]) for k in keys)])


def save_topk_csv(path: Path, lambdas: np.ndarray, top_idx: np.ndarray, top_score: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda_idx", "lambda_nm", "rank", "sample_idx", "score"])
        for j, lam in enumerate(lambdas):
            for r in range(top_idx.shape[1]):
                i = int(top_idx[j, r])
                s = float(top_score[j, r])
                if i >= 0 and np.isfinite(s):
                    w.writerow([j, float(lam), r + 1, i, s])


def summary_json(lambdas: np.ndarray, score: np.ndarray, top_idx: np.ndarray, top_score: np.ndarray) -> list[dict]:
    out = []
    for j, lam in enumerate(lambdas):
        col = score[:, j]
        valid = col[np.isfinite(col)]
        out.append(
            {
                "lambda_idx": int(j),
                "lambda_nm": float(lam),
                "num_valid": int(len(valid)),
                "score_mean": float(np.mean(valid)) if len(valid) else None,
                "score_p90": float(np.quantile(valid, 0.9)) if len(valid) else None,
                "top_indices": [int(x) for x in top_idx[j] if x >= 0],
                "top_scores": [float(x) for x in top_score[j] if np.isfinite(x)],
            }
        )
    return out


def plot_per_lambda(
    out_dir: Path,
    structures: np.ndarray,
    spec: np.ndarray,
    companion_spec: np.ndarray | None,
    companion_label: str | None,
    field_label: str,
    score: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_score: np.ndarray,
    topk: int,
    theta_ref: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = target_profile(thetas)
    t_ref_idx = int(np.argmin(np.abs(thetas - theta_ref)))
    t_ref_actual = float(thetas[t_ref_idx])

    finite = spec[np.isfinite(spec)]
    vmin = float(np.quantile(finite, 0.01)) if finite.size else 0.0
    vmax = float(np.quantile(finite, 0.99)) if finite.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]
    comp_vmin = comp_vmax = None
    if companion_spec is not None:
        comp_finite = companion_spec[np.isfinite(companion_spec)]
        comp_vmin = float(np.quantile(comp_finite, 0.01)) if comp_finite.size else 0.0
        comp_vmax = float(np.quantile(comp_finite, 0.99)) if comp_finite.size else 1.0
        if comp_vmax <= comp_vmin:
            comp_vmax = comp_vmin + 1e-6

    for j, lam in enumerate(lambdas):
        ncols = 5 if companion_spec is not None else 3
        width_ratios = [0.75, 1.05, 1.0, 1.05, 1.0] if companion_spec is not None else [0.75, 1.05, 1.0]
        fig, axes = plt.subplots(
            topk,
            ncols,
            figsize=(20.0 if companion_spec is not None else 13.2, max(2.6 * topk, 5.0)),
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        hm = None
        hm_comp = None
        for r in range(topk):
            if companion_spec is not None:
                a_struct, a_hm, a_curve, a_hm_comp, a_curve_comp = axes[r]
            else:
                a_struct, a_hm, a_curve = axes[r]
            i = int(top_idx[j, r])
            if i < 0:
                for ax in axes[r]:
                    ax.axis("off")
                continue

            full = spec[i].astype(np.float64)
            if not np.isfinite(full).all():
                for ax in axes[r]:
                    ax.axis("off")
                continue

            a_struct.imshow(structures[i], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
            a_struct.set_title(f"id={i}", fontsize=10)
            a_struct.axis("off")

            hm = a_hm.imshow(full, cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax, interpolation="bicubic")
            a_hm.axhline(float(lam), color="w", ls="--", lw=1.0)
            a_hm.set_title(f"rank{r+1} score={float(top_score[j, r]):.3f}", fontsize=10)
            a_hm.set_xlabel("theta (deg)")
            a_hm.set_ylabel(f"lambda (nm)\n{field_label}")

            y = spec[i, j].astype(np.float64)
            yn = y / max(float(np.max(y)), 1e-8)
            t_ref_val = float(y[t_ref_idx])
            a_curve.plot(thetas, target, "k--", lw=1.7, label="target ~ |sin(theta)|^2")
            a_curve.plot(thetas, yn, lw=1.9, color="#1f77b4", label="candidate (normalized)")
            a_curve.set_ylim(-0.05, 1.05)
            a_curve.set_xlabel("theta (deg)")
            a_curve.set_ylabel(f"normalized |{field_label}|")
            a_curve.grid(alpha=0.25)
            if r == 0:
                a_curve.legend(fontsize=8, loc="lower right")

            tag = f"|t|@{t_ref_actual:.1f}deg={t_ref_val:.3f}"
            a_hm.text(
                0.98, 0.03, tag, transform=a_hm.transAxes, ha="right", va="bottom", fontsize=8, color="white",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
            )
            a_curve.text(0.02, 0.03, tag, transform=a_curve.transAxes, ha="left", va="bottom", fontsize=8)

            if companion_spec is not None:
                full_comp = companion_spec[i].astype(np.float64)
                y_comp = companion_spec[i, j].astype(np.float64)
                yn_comp = y_comp / max(float(np.max(y_comp)), 1e-8)
                t_ref_val_comp = float(y_comp[t_ref_idx])

                hm_comp = a_hm_comp.imshow(
                    full_comp,
                    cmap="turbo",
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    vmin=comp_vmin,
                    vmax=comp_vmax,
                    interpolation="bicubic",
                )
                a_hm_comp.axhline(float(lam), color="w", ls="--", lw=1.0)
                a_hm_comp.set_title(f"{companion_label}", fontsize=10)
                a_hm_comp.set_xlabel("theta (deg)")
                a_hm_comp.set_ylabel(f"lambda (nm)\n{companion_label}")

                a_curve_comp.plot(thetas, yn_comp, lw=1.9, color="#ff7f0e", label=f"{companion_label} (normalized)")
                a_curve_comp.set_ylim(-0.05, 1.05)
                a_curve_comp.set_xlabel("theta (deg)")
                a_curve_comp.set_ylabel(f"normalized |{companion_label}|")
                a_curve_comp.grid(alpha=0.25)
                if r == 0:
                    a_curve_comp.legend(fontsize=8, loc="lower right")
                tag_comp = f"|{companion_label}|@{t_ref_actual:.1f}deg={t_ref_val_comp:.3f}"
                a_hm_comp.text(
                    0.98, 0.03, tag_comp, transform=a_hm_comp.transAxes, ha="right", va="bottom", fontsize=8, color="white",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
                )
                a_curve_comp.text(0.02, 0.03, tag_comp, transform=a_curve_comp.transAxes, ha="left", va="bottom", fontsize=8)

        valid = score[:, j][np.isfinite(score[:, j])]
        stats = f"lambda={float(lam):.1f} nm | valid={len(valid)} | mean={float(np.mean(valid)):.3f} | p90={float(np.quantile(valid, 0.9)):.3f}" if len(valid) else f"lambda={float(lam):.1f} nm | valid=0"
        extra = f" + {companion_label}" if companion_spec is not None else ""
        fig.suptitle(f"Top-{topk}: {field_label}{extra} ({stats})", fontsize=12)
        if hm is not None:
            fig.colorbar(hm, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|t|")
        if hm_comp is not None:
            fig.colorbar(hm_comp, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label=f"|{companion_label}|")
        fig.savefig(out_dir / f"lambda_{float(lam):.1f}nm_top{topk}.png", dpi=180)
        plt.close(fig)


def plot_overview(
    out_png: Path,
    spec: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    top_idx: np.ndarray,
    top_score: np.ndarray,
    topk: int,
) -> None:
    target = target_profile(thetas)
    n_lambda = len(lambdas)
    ncol = 4
    nrow = int(np.ceil(n_lambda / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.4, nrow * 3.2), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for j, lam in enumerate(lambdas):
        ax = axes[j]
        ax.plot(thetas, target, "k--", lw=1.2)
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
        if np.isfinite(top_score[j, 0]):
            ax.text(0.02, 0.03, f"best={float(top_score[j, 0]):.3f}", transform=ax.transAxes, fontsize=8)
    for ax in axes[n_lambda:]:
        ax.axis("off")

    fig.suptitle(f"Per-lambda Top-{topk} spectra (higher score is better)", fontsize=14)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Per-lambda second-order scoring (higher is better)")
    p.add_argument("--in_npz", type=Path, default=ROOT / "data" / "train_data.npz")
    p.add_argument("--field", default="tpp_mag")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "second_order_scores")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--w_center", type=float, default=0.6)
    p.add_argument("--w_shape", type=float, default=0.3)
    p.add_argument("--w_edge", type=float, default=0.1)
    p.add_argument("--plot_topk", type=int, default=5)
    p.add_argument("--theta_ref", type=float, default=40.0)
    args = p.parse_args()

    args.in_npz = resolve_from_root(args.in_npz)
    args.out_dir = resolve_from_root(args.out_dir)

    structures, spec, lambdas, thetas = load_bundle(args.in_npz, args.field)
    companion_spec, companion_label = load_companion_spec(args.in_npz, args.field)
    pack = score_spectra(spec, thetas, args.w_center, args.w_shape, args.w_edge)
    top_idx, top_score = topk_per_lambda(pack["score"], args.topk)
    summary = summary_json(lambdas, pack["score"], top_idx, top_score)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / f"{args.field}_scores.npz", lambdas=lambdas, thetas=thetas, top_indices=top_idx, top_scores=top_score, **pack)
    save_scores_csv(args.out_dir / f"{args.field}_scores.csv", lambdas, pack)
    save_topk_csv(args.out_dir / f"{args.field}_top{args.plot_topk}_per_lambda.csv", lambdas, top_idx[:, : max(1, args.plot_topk)], top_score[:, : max(1, args.plot_topk)])
    with (args.out_dir / f"{args.field}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.plot_topk > 0:
        k = min(args.plot_topk, spec.shape[0])
        plot_dir = args.out_dir / f"{args.field}_top{k}_plots"
        plot_per_lambda(
            plot_dir,
            structures,
            spec,
            companion_spec,
            companion_label,
            args.field,
            pack["score"],
            lambdas,
            thetas,
            top_idx[:, :k],
            top_score[:, :k],
            k,
            args.theta_ref,
        )
        plot_overview(args.out_dir / f"{args.field}_top{k}_overview.png", spec, lambdas, thetas, top_idx[:, :k], top_score[:, :k], k)

    print(f"input: {args.in_npz}")
    print(f"field: {args.field}")
    print(f"samples: {spec.shape[0]}, lambdas: {spec.shape[1]}, thetas: {spec.shape[2]}")
    print(f"weights: center={args.w_center}, shape={args.w_shape}, edge={args.w_edge}")
    print("score direction: higher is better")
    print(f"plot_topk: {args.plot_topk}, theta_ref: {args.theta_ref} deg")
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
