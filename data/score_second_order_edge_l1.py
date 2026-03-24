from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from score_second_order import load_companion_spec, plot_overview, plot_per_lambda


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


def ideal_second_order_from_edge(
    thetas_deg: np.ndarray,
    edge_value: float,
    theta_ref: float,
) -> np.ndarray:
    ref = max(abs(float(np.sin(np.deg2rad(theta_ref)))), 1e-8)
    x = np.abs(np.sin(np.deg2rad(thetas_deg.astype(np.float64)))) / ref
    x = np.clip(x, 0.0, None) ** 2
    return (x * float(edge_value)).astype(np.float32)


def score_spectra_edge_l1(
    spec: np.ndarray,
    thetas_deg: np.ndarray,
    theta_ref: float,
) -> dict[str, np.ndarray]:
    n, l, t = spec.shape
    ref_idx = int(np.argmin(np.abs(thetas_deg - float(theta_ref))))
    ref_actual = float(thetas_deg[ref_idx])

    l1_sum = np.full((n, l), np.nan, dtype=np.float32)
    l1_mean = np.full((n, l), np.nan, dtype=np.float32)
    edge_value = np.full((n, l), np.nan, dtype=np.float32)
    score = np.full((n, l), np.nan, dtype=np.float32)
    target = np.full((n, l, t), np.nan, dtype=np.float32)

    for i in range(n):
        for j in range(l):
            row = spec[i, j].astype(np.float32)
            if not np.isfinite(row).all():
                continue
            edge = float(row[ref_idx])
            ideal = ideal_second_order_from_edge(thetas_deg, edge, ref_actual)
            err = np.abs(row - ideal)
            err_sum = float(np.sum(err))
            err_mean = float(np.mean(err))

            target[i, j] = ideal
            edge_value[i, j] = edge
            l1_sum[i, j] = err_sum
            l1_mean[i, j] = err_mean
            score[i, j] = 1.0 / (1.0 + err_sum)

    return {
        "score": score,
        "l1_sum": l1_sum,
        "l1_mean": l1_mean,
        "edge_value": edge_value,
        "target_curve": target,
        "theta_ref_actual": np.array(ref_actual, dtype=np.float32),
    }


def topk_per_lambda_from_error(l1_sum: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n, l = l1_sum.shape
    k = min(k, n)
    idx = np.full((l, k), -1, dtype=np.int32)
    val = np.full((l, k), np.nan, dtype=np.float32)
    for j in range(l):
        col = l1_sum[:, j]
        valid = np.isfinite(col)
        if not valid.any():
            continue
        order = np.argsort(col[valid])
        sel = np.where(valid)[0][order[:k]]
        idx[j, : len(sel)] = sel
        val[j, : len(sel)] = col[sel]
    return idx, val


def save_scores_csv(path: Path, lambdas: np.ndarray, pack: dict[str, np.ndarray]) -> None:
    n, l = pack["score"].shape
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "lambda_idx", "lambda_nm", "score", "l1_sum", "l1_mean", "edge_value"])
        for i in range(n):
            for j in range(l):
                w.writerow([
                    i,
                    j,
                    float(lambdas[j]),
                    float(pack["score"][i, j]),
                    float(pack["l1_sum"][i, j]),
                    float(pack["l1_mean"][i, j]),
                    float(pack["edge_value"][i, j]),
                ])


def save_topk_csv(path: Path, lambdas: np.ndarray, top_idx: np.ndarray, top_l1: np.ndarray, score: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda_idx", "lambda_nm", "rank", "sample_idx", "l1_sum", "score"])
        for j, lam in enumerate(lambdas):
            for r in range(top_idx.shape[1]):
                i = int(top_idx[j, r])
                if i < 0 or not np.isfinite(top_l1[j, r]):
                    continue
                w.writerow([j, float(lam), r + 1, i, float(top_l1[j, r]), float(score[i, j])])


def summary_json(lambdas: np.ndarray, pack: dict[str, np.ndarray], top_idx: np.ndarray, top_l1: np.ndarray) -> list[dict]:
    out = []
    for j, lam in enumerate(lambdas):
        l1 = pack["l1_sum"][:, j]
        valid = l1[np.isfinite(l1)]
        out.append(
            {
                "lambda_idx": int(j),
                "lambda_nm": float(lam),
                "num_valid": int(len(valid)),
                "l1_sum_mean": float(np.mean(valid)) if len(valid) else None,
                "l1_sum_p10": float(np.quantile(valid, 0.1)) if len(valid) else None,
                "best_indices": [int(x) for x in top_idx[j] if x >= 0],
                "best_l1_sum": [float(x) for x in top_l1[j] if np.isfinite(x)],
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Second-order score by L1 error to edge-scaled ideal curve (smaller error is better)")
    p.add_argument("--in_npz", type=Path, default=ROOT / "data" / "train_data.npz")
    p.add_argument("--field", default="tpp_mag")
    p.add_argument("--out_dir", type=Path, default=ROOT / "data" / "second_order_scores_edge_l1")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--plot_topk", type=int, default=5)
    p.add_argument("--theta_ref", type=float, default=60.0)
    args = p.parse_args()

    args.in_npz = resolve_from_root(args.in_npz)
    args.out_dir = resolve_from_root(args.out_dir)

    structures, spec, lambdas, thetas = load_bundle(args.in_npz, args.field)
    companion_spec, companion_label = load_companion_spec(args.in_npz, args.field)
    pack = score_spectra_edge_l1(spec, thetas, args.theta_ref)
    top_idx, top_l1 = topk_per_lambda_from_error(pack["l1_sum"], args.topk)
    summary = summary_json(lambdas, pack, top_idx, top_l1)
    top_score = np.full_like(top_l1, np.nan)
    for j in range(top_idx.shape[0]):
        for r in range(top_idx.shape[1]):
            i = int(top_idx[j, r])
            if i >= 0:
                top_score[j, r] = pack["score"][i, j]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out_dir / f"{args.field}_edge_l1_scores.npz",
        lambdas=lambdas,
        thetas=thetas,
        top_indices=top_idx,
        top_l1_sum=top_l1,
        **pack,
    )
    save_scores_csv(args.out_dir / f"{args.field}_edge_l1_scores.csv", lambdas, pack)
    save_topk_csv(args.out_dir / f"{args.field}_edge_l1_top{args.topk}_per_lambda.csv", lambdas, top_idx, top_l1, pack["score"])
    with (args.out_dir / f"{args.field}_edge_l1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.plot_topk > 0:
        k = min(args.plot_topk, spec.shape[0])
        plot_dir = args.out_dir / f"{args.field}_edge_l1_top{k}_plots"
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
            float(pack["theta_ref_actual"]),
        )
        plot_overview(
            args.out_dir / f"{args.field}_edge_l1_top{k}_overview.png",
            spec,
            lambdas,
            thetas,
            top_idx[:, :k],
            top_score[:, :k],
            k,
        )

    print(f"input: {args.in_npz}")
    print(f"field: {args.field}")
    print(f"samples: {spec.shape[0]}, lambdas: {spec.shape[1]}, thetas: {spec.shape[2]}")
    print(f"theta_ref requested: {args.theta_ref} deg")
    print(f"theta_ref actual: {float(pack['theta_ref_actual']):.1f} deg")
    print("scoring: ideal curve is scaled by the row value at theta_ref; smaller l1_sum is better; score=1/(1+l1_sum)")
    print(f"plot_topk: {args.plot_topk}")
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
