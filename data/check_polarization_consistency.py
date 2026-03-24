from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: Path) -> Path:
    return path_like if path_like.is_absolute() else ROOT / path_like


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(f"NPZ not found: {path}")
    return np.load(path)


def finite_stats(arr: np.ndarray) -> dict:
    x = np.asarray(arr, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "shape": list(x.shape),
            "finite_count": 0,
            "nan_count": int(np.isnan(x).sum()),
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "shape": list(x.shape),
        "finite_count": int(finite.size),
        "nan_count": int(np.isnan(x).sum()),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def compare_pair(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(aa) & np.isfinite(bb)
    if not mask.any():
        return {
            "pair": f"{label_a}_vs_{label_b}",
            "finite_pair_count": 0,
            "abs_diff_mean": None,
            "abs_diff_median": None,
            "abs_diff_p90": None,
            "rel_diff_mean": None,
            "rel_diff_median": None,
            "rel_diff_p90": None,
            "ratio_mean": None,
            "ratio_median": None,
            "corr": None,
        }

    va = aa[mask]
    vb = bb[mask]
    abs_diff = np.abs(va - vb)
    rel_diff = abs_diff / np.maximum(0.5 * (np.abs(va) + np.abs(vb)), 1e-8)
    ratio = vb / np.maximum(va, 1e-8)
    corr = float(np.corrcoef(va, vb)[0, 1]) if va.size > 1 else None
    return {
        "pair": f"{label_a}_vs_{label_b}",
        "finite_pair_count": int(va.size),
        "abs_diff_mean": float(np.mean(abs_diff)),
        "abs_diff_median": float(np.median(abs_diff)),
        "abs_diff_p90": float(np.quantile(abs_diff, 0.9)),
        "rel_diff_mean": float(np.mean(rel_diff)),
        "rel_diff_median": float(np.median(rel_diff)),
        "rel_diff_p90": float(np.quantile(rel_diff, 0.9)),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_median": float(np.median(ratio)),
        "corr": corr,
    }


def compare_theta_curves_per_lambda(
    a: np.ndarray,
    b: np.ndarray,
    lambdas: np.ndarray,
    rel_thresholds: tuple[float, ...] = (0.03, 0.05, 0.1),
) -> dict:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape or aa.ndim != 3:
        raise ValueError(f"expected [N,L,T] paired spectra, got {aa.shape} and {bb.shape}")

    abs_diff = np.abs(aa - bb)
    rel_diff = abs_diff / np.maximum(0.5 * (np.abs(aa) + np.abs(bb)), 1e-8)
    per_lambda_rel_mean = rel_diff.mean(axis=2)  # [N,L]
    per_lambda_rel_median = np.median(rel_diff, axis=2)
    per_lambda_abs_mean = abs_diff.mean(axis=2)

    lambda_rows = []
    for j, lam in enumerate(lambdas):
        row = per_lambda_rel_mean[:, j]
        row_med = per_lambda_rel_median[:, j]
        row_abs = per_lambda_abs_mean[:, j]
        entry = {
            "lambda_idx": int(j),
            "lambda_nm": float(lam),
            "rel_mean_q10": float(np.quantile(row, 0.1)),
            "rel_mean_q50": float(np.quantile(row, 0.5)),
            "rel_mean_q90": float(np.quantile(row, 0.9)),
            "rel_median_q50": float(np.quantile(row_med, 0.5)),
            "abs_mean_q50": float(np.quantile(row_abs, 0.5)),
        }
        for th in rel_thresholds:
            entry[f"frac_rel_mean_lt_{th:.2f}"] = float(np.mean(row < th))
        lambda_rows.append(entry)

    sample_good_frac = np.mean(per_lambda_rel_mean < 0.1, axis=1)
    sample_strict_frac = np.mean(per_lambda_rel_mean < 0.05, axis=1)
    target_idx = int(np.argmin(np.abs(lambdas - 1250.0)))
    target_rel = per_lambda_rel_mean[:, target_idx]
    target_rel_med = per_lambda_rel_median[:, target_idx]

    return {
        "per_lambda_curve_similarity": lambda_rows,
        "sample_level_summary": {
            "fraction_of_structures_target1250_rel_mean_lt_0.03": float(np.mean(target_rel < 0.03)),
            "fraction_of_structures_target1250_rel_mean_lt_0.05": float(np.mean(target_rel < 0.05)),
            "fraction_of_structures_target1250_rel_mean_lt_0.10": float(np.mean(target_rel < 0.10)),
            "fraction_of_structures_target1250_rel_median_lt_0.05": float(np.mean(target_rel_med < 0.05)),
            "fraction_of_structures_all_lambda_majority_rel_mean_lt_0.10": float(np.mean(sample_good_frac >= 0.5)),
            "fraction_of_structures_all_lambda_majority_rel_mean_lt_0.05": float(np.mean(sample_strict_frac >= 0.5)),
            "fraction_of_structures_all_lambda_all_rel_mean_lt_0.10": float(np.mean(np.all(per_lambda_rel_mean < 0.10, axis=1))),
            "fraction_of_structures_all_lambda_all_rel_mean_lt_0.05": float(np.mean(np.all(per_lambda_rel_mean < 0.05, axis=1))),
        },
        "best_target1250_structure_indices_by_rel_mean": np.argsort(target_rel)[:20].astype(int).tolist(),
        "worst_target1250_structure_indices_by_rel_mean": np.argsort(target_rel)[-20:][::-1].astype(int).tolist(),
    }


def summarize_loaded_fields(data: np.lib.npyio.NpzFile) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in ("tpp_mag", "tss_mag", "tps_mag", "tsp_mag"):
        if key in data.files:
            out[key] = finite_stats(data[key])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether tpp/tss are similar and whether cross-polarization is small.")
    parser.add_argument("--npz", type=Path, default=ROOT / "data" / "train_data.npz")
    args = parser.parse_args()

    args.npz = resolve_from_root(args.npz)

    data = load_npz(args.npz)
    summary = {
        "input": str(args.npz),
        "files": list(data.files),
        "field_stats": summarize_loaded_fields(data),
        "pair_checks": [],
        "polarization_independence": None,
        "notes": [],
    }

    if "tpp_mag" in data.files and "tss_mag" in data.files:
        tpp = np.asarray(data["tpp_mag"])
        tss = np.asarray(data["tss_mag"])
        if tpp.ndim == 2:
            tpp = tpp[:, None, :]
            tss = tss[:, None, :]
        summary["pair_checks"].append(compare_pair(tpp, tss, "tpp_mag", "tss_mag"))
        if "lambdas" in data.files:
            lambdas = np.asarray(data["lambdas"], dtype=np.float64)
        elif "target_lambda" in data.files:
            lambdas = np.asarray([float(data["target_lambda"])], dtype=np.float64)
        else:
            lambdas = np.arange(tpp.shape[1], dtype=np.float64)
        summary["polarization_independence"] = compare_theta_curves_per_lambda(tpp, tss, lambdas)
    else:
        summary["notes"].append("tpp_mag/tss_mag pair is unavailable in this npz.")

    if "tpp_mag" in data.files and "tps_mag" in data.files:
        summary["pair_checks"].append(compare_pair(data["tpp_mag"], data["tps_mag"], "tpp_mag", "tps_mag"))
    else:
        summary["notes"].append("tps_mag is unavailable, so cross-polarization cannot be judged from this npz alone.")

    if "tpp_mag" in data.files and "tsp_mag" in data.files:
        summary["pair_checks"].append(compare_pair(data["tpp_mag"], data["tsp_mag"], "tpp_mag", "tsp_mag"))
    else:
        summary["notes"].append("tsp_mag is unavailable, so cross-polarization cannot be judged from this npz alone.")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
