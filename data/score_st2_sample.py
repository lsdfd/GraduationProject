from __future__ import annotations

import argparse
import json
from pathlib import Path

from screen_st2_template_match import (
    find_lambda_index,
    ideal_st2_map,
    lambda_zero_band_mask,
    load_dataset,
    merge_channel_scores,
    resolve_from_root,
    score_channel,
    theta_zero_band_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one sample for ST2 screening settings.")
    parser.add_argument("--npz", type=str, default="data/train_data.npz")
    parser.add_argument("--sample_idx", type=int, default=11742)
    parser.add_argument("--target_lambdas", type=float, nargs="+", default=[900.0, 1000.0, 1100.0])
    parser.add_argument("--theta_zero_width_deg", type=float, default=2.5)
    parser.add_argument("--lambda_zero_width_nm", type=float, default=25.0)
    parser.add_argument("--lambda_window_nm", type=float, default=200.0)
    parser.add_argument("--theta_max_deg", type=float, default=40.0)
    parser.add_argument("--out_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = resolve_from_root(args.npz)
    bundle = load_dataset(npz_path)

    structures = bundle["structures"]
    tpp = bundle["tpp_mag"]
    tss = bundle["tss_mag"]
    lambdas = bundle["lambdas"]
    thetas = bundle["thetas"]

    if args.sample_idx < 0 or args.sample_idx >= len(structures):
        raise IndexError(f"sample_idx out of range: {args.sample_idx}, dataset size={len(structures)}")

    theta0_mask = theta_zero_band_mask(thetas, args.theta_zero_width_deg)
    results: list[dict[str, object]] = []

    for target_lambda in args.target_lambdas:
        lambda_idx = find_lambda_index(lambdas, target_lambda)
        actual_lambda = float(lambdas[lambda_idx])
        lambda0_mask = lambda_zero_band_mask(lambdas, actual_lambda, args.lambda_zero_width_nm)
        ideal_map, work_mask = ideal_st2_map(
            lambdas,
            thetas,
            actual_lambda,
            args.lambda_window_nm,
            args.theta_max_deg,
        )

        tpp_score = score_channel(
            tpp[args.sample_idx],
            ideal_map,
            theta0_mask,
            lambda0_mask,
            work_mask,
        )
        tss_score = score_channel(
            tss[args.sample_idx],
            ideal_map,
            theta0_mask,
            lambda0_mask,
            work_mask,
        )
        merged = merge_channel_scores(tpp_score, tss_score)
        results.append(
            {
                "requested_lambda_nm": float(target_lambda),
                "actual_lambda_nm": actual_lambda,
                "sample_idx": int(args.sample_idx),
                "merged": merged,
                "tpp": tpp_score,
                "tss": tss_score,
            }
        )

    payload = {
        "input_npz": str(npz_path),
        "sample_idx": int(args.sample_idx),
        "theta_zero_width_deg": float(args.theta_zero_width_deg),
        "lambda_zero_width_nm": float(args.lambda_zero_width_nm),
        "lambda_window_nm": float(args.lambda_window_nm),
        "theta_max_deg": float(args.theta_max_deg),
        "results": results,
    }

    if args.out_json:
        out_json = resolve_from_root(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_json: {out_json}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
