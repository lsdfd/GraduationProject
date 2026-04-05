from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - float(target))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter samples across all possible lambda0 using angular and neighboring-lambda thresholds."
    )
    parser.add_argument("--npz", type=str, default="data/train_data_20000.npz")
    parser.add_argument("--channel", choices=["tpp", "tss", "both"], default="both")
    parser.add_argument("--theta_min_deg", type=float, default=-20.0)
    parser.add_argument("--theta_max_deg", type=float, default=20.0)
    parser.add_argument("--window_max", type=float, default=0.05)
    parser.add_argument("--theta0_max", type=float, default=0.05)
    parser.add_argument("--theta_target_deg", type=float, default=20.0)
    parser.add_argument("--theta_target_min", type=float, default=0.5)
    parser.add_argument("--lambda_step_nm", type=float, default=50.0)
    parser.add_argument("--out_csv", type=str, default="data/lambda0_window_candidates.csv")
    parser.add_argument("--out_json", type=str, default="data/lambda0_window_candidates.json")
    return parser.parse_args()


def run_for_channel(data: np.lib.npyio.NpzFile, npz_path: Path, channel_name: str, args: argparse.Namespace) -> dict[str, object]:
    channel_key = "tss_mag" if channel_name == "tss" else "tpp_mag"
    spec = np.asarray(data[channel_key], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float64)
    thetas = np.asarray(data["thetas"], dtype=np.float64)

    theta_window_mask = (thetas >= float(args.theta_min_deg)) & (thetas <= float(args.theta_max_deg))
    idx_theta0 = nearest_index(thetas, 0.0)
    idx_theta_target = nearest_index(thetas, args.theta_target_deg)
    step = float(args.lambda_step_nm)

    center_indices = [
        i for i, lam in enumerate(lambdas)
        if np.any(np.isclose(lambdas, lam - step)) and np.any(np.isclose(lambdas, lam + step))
    ]

    rows: list[dict[str, float | int]] = []
    per_lambda_counts: dict[str, int] = {}

    for center_idx in center_indices:
        lambda0 = float(lambdas[center_idx])
        idx_prev = nearest_index(lambdas, lambda0 - step)
        idx_next = nearest_index(lambdas, lambda0 + step)

        center_row = spec[:, center_idx, :]
        cond_window = np.all(center_row[:, theta_window_mask] <= float(args.window_max), axis=1)
        cond_theta0_triplet = (
            (spec[:, idx_prev, idx_theta0] <= float(args.theta0_max))
            & (spec[:, center_idx, idx_theta0] <= float(args.theta0_max))
            & (spec[:, idx_next, idx_theta0] <= float(args.theta0_max))
        )
        cond_theta_target = spec[:, idx_next, idx_theta_target] > float(args.theta_target_min)
        hit_mask = cond_window & cond_theta0_triplet & cond_theta_target
        hit_indices = np.where(hit_mask)[0]
        per_lambda_counts[f"{lambda0:.0f}"] = int(len(hit_indices))

        for sample_idx in hit_indices.tolist():
            rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "lambda0_nm": lambda0,
                    "theta_window_max_value": float(np.max(center_row[sample_idx, theta_window_mask])),
                    "theta0_lambda_minus_50": float(spec[sample_idx, idx_prev, idx_theta0]),
                    "theta0_lambda0": float(spec[sample_idx, center_idx, idx_theta0]),
                    "theta0_lambda_plus_50": float(spec[sample_idx, idx_next, idx_theta0]),
                    "theta20_lambda_plus_50": float(spec[sample_idx, idx_next, idx_theta_target]),
                }
            )

    base_csv = resolve_from_root(args.out_csv)
    base_json = resolve_from_root(args.out_json)
    out_csv = base_csv.with_name(f"{base_csv.stem}_{channel_name}.csv")
    out_json = base_json.with_name(f"{base_json.stem}_{channel_name}.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "lambda0_nm",
                "theta_window_max_value",
                "theta0_lambda_minus_50",
                "theta0_lambda0",
                "theta0_lambda_plus_50",
                "theta20_lambda_plus_50",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_npz": str(npz_path),
        "channel": channel_key,
        "theta_window_deg": [float(args.theta_min_deg), float(args.theta_max_deg)],
        "window_max": float(args.window_max),
        "theta0_max": float(args.theta0_max),
        "theta_target_deg": float(args.theta_target_deg),
        "theta_target_min": float(args.theta_target_min),
        "lambda_step_nm": step,
        "per_lambda_counts": per_lambda_counts,
        "matched_count_total": int(len(rows)),
        "csv_path": str(out_csv),
        "json_path": str(out_json),
        "rows": rows,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    data = np.load(npz_path)

    channel_names = ["tpp", "tss"] if args.channel == "both" else [args.channel]
    summaries = [run_for_channel(data, npz_path, name, args) for name in channel_names]

    for summary in summaries:
        print(f"channel: {summary['channel']}")
        print(f"matched_count_total: {summary['matched_count_total']}")
        print("per_lambda_counts:", json.dumps(summary["per_lambda_counts"], ensure_ascii=False))
        print(f"saved_csv: {summary['csv_path']}")
        print(f"saved_json: {summary['json_path']}")


if __name__ == "__main__":
    main()
