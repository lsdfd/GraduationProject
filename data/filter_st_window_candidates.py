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
        description="Filter 20k dataset samples by ST transmission constraints around 950 nm."
    )
    parser.add_argument("--npz", type=str, default="data/train_data_20000.npz")
    parser.add_argument("--channel", choices=["tpp", "tss", "both"], default="both")
    parser.add_argument("--theta_min_deg", type=float, default=-30.0)
    parser.add_argument("--theta_max_deg", type=float, default=30.0)
    parser.add_argument("--center_lambda_nm", type=float, default=950.0)
    parser.add_argument("--center_theta_max", type=float, default=0.01)
    parser.add_argument("--lambda0_low_nm", type=float, default=900.0)
    parser.add_argument("--lambda0_high_nm", type=float, default=1000.0)
    parser.add_argument("--theta0_max", type=float, default=0.01)
    parser.add_argument("--theta30_min", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default="data/st_window_candidates.csv")
    parser.add_argument("--out_json", type=str, default="data/st_window_candidates.json")
    return parser.parse_args()


def run_filter_for_channel(
    data: np.lib.npyio.NpzFile,
    npz_path: Path,
    channel_name: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    channel_key = "tss_mag" if channel_name == "tss" else "tpp_mag"
    if channel_key not in data.files:
        raise ValueError(f"Missing channel {channel_key} in {npz_path}")

    spec = np.asarray(data[channel_key], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float64)
    thetas = np.asarray(data["thetas"], dtype=np.float64)

    idx_lam_center = nearest_index(lambdas, args.center_lambda_nm)
    idx_lam_low = nearest_index(lambdas, args.lambda0_low_nm)
    idx_lam_high = nearest_index(lambdas, args.lambda0_high_nm)
    idx_theta0 = nearest_index(thetas, 0.0)
    idx_theta30 = nearest_index(thetas, 30.0)

    theta_mask = (thetas >= float(args.theta_min_deg)) & (thetas <= float(args.theta_max_deg))
    if not np.any(theta_mask):
        raise ValueError("No theta points fall inside the requested angular window.")

    center_row = spec[:, idx_lam_center, :]
    cond_center_window = np.all(center_row[:, theta_mask] < float(args.center_theta_max), axis=1)
    cond_theta0_low = spec[:, idx_lam_low, idx_theta0] <= float(args.theta0_max)
    cond_theta0_high = spec[:, idx_lam_high, idx_theta0] <= float(args.theta0_max)
    cond_theta30_high = spec[:, idx_lam_high, idx_theta30] > float(args.theta30_min)

    hit_mask = cond_center_window & cond_theta0_low & cond_theta0_high & cond_theta30_high
    hit_indices = np.where(hit_mask)[0]

    rows: list[dict[str, float | int]] = []
    for idx in hit_indices.tolist():
        rows.append(
            {
                "sample_idx": int(idx),
                "center_lambda_nm": float(lambdas[idx_lam_center]),
                "center_window_max": float(np.max(center_row[idx, theta_mask])),
                "center_window_min": float(np.min(center_row[idx, theta_mask])),
                "lambda900_theta0": float(spec[idx, idx_lam_low, idx_theta0]),
                "lambda1000_theta0": float(spec[idx, idx_lam_high, idx_theta0]),
                "lambda1000_theta30": float(spec[idx, idx_lam_high, idx_theta30]),
            }
        )

    base_csv = resolve_from_root(args.out_csv)
    base_json = resolve_from_root(args.out_json)
    stem_csv = base_csv.stem if base_csv.suffix else base_csv.name
    stem_json = base_json.stem if base_json.suffix else base_json.name
    out_csv = base_csv.with_name(f"{stem_csv}_{channel_name}.csv")
    out_json = base_json.with_name(f"{stem_json}_{channel_name}.json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_idx",
                "center_lambda_nm",
                "center_window_max",
                "center_window_min",
                "lambda900_theta0",
                "lambda1000_theta0",
                "lambda1000_theta30",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "input_npz": str(npz_path),
        "channel": channel_key,
        "theta_window_deg": [float(args.theta_min_deg), float(args.theta_max_deg)],
        "center_lambda_nm": float(lambdas[idx_lam_center]),
        "center_window_threshold": float(args.center_theta_max),
        "lambda900_theta0_threshold": float(args.theta0_max),
        "lambda1000_theta0_threshold": float(args.theta0_max),
        "lambda1000_theta30_threshold_min": float(args.theta30_min),
        "matched_count": int(len(rows)),
        "matched_indices": [int(i) for i in hit_indices.tolist()],
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
    summaries = [run_filter_for_channel(data, npz_path, channel_name, args) for channel_name in channel_names]

    for summary in summaries:
        print(f"channel: {summary['channel']}")
        print(f"matched_count: {summary['matched_count']}")
        print(f"saved_csv: {summary['csv_path']}")
        print(f"saved_json: {summary['json_path']}")
        if summary["rows"]:
            print("first_match:", json.dumps(summary["rows"][0], ensure_ascii=False))


if __name__ == "__main__":
    main()
