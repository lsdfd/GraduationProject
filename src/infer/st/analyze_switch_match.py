from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from infer.st.build_inputs import build_switching_segments_scaled  # noqa: E402
from infer.st.common import (  # noqa: E402
    AxisConfig,
    DeviceConfig,
    envelope_carrier_metadata,
    load_device_otf,
    matched_compute_axes_from_display,
)
from infer.st.plot_prl_style import plot_fig3_triptych  # noqa: E402
from infer.st.run_virtual_experiments import run_case  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original Fig.3-style switching input with a support-matched scaled version.")
    parser.add_argument("--train_npz", default=str(ROOT / "data" / "train_data_20000.npz"))
    parser.add_argument("--sample_idx", type=int, default=18959)
    parser.add_argument("--channel", choices=["tpp", "tss"], default="tss")
    parser.add_argument("--lambda0_nm", type=float, default=950.0)
    parser.add_argument("--lambda_step_nm", type=float, default=10.0)
    parser.add_argument("--save_dir", default=str(ROOT / "samples" / "st_switch_match"))
    parser.add_argument("--pad_factor", type=float, default=2.0)
    parser.add_argument("--width_scale_matched", type=float, default=0.05)
    parser.add_argument("--time_scale_matched", type=float, default=0.08)
    args = parser.parse_args()

    device_cfg = DeviceConfig(
        sample_idx=args.sample_idx,
        channel=args.channel,
        lambda0_nm=args.lambda0_nm,
        lambda_min_nm=900.0,
        lambda_max_nm=1050.0,
        lambda_step_nm=args.lambda_step_nm,
    )
    device_otf_info = load_device_otf(args.train_npz, device_cfg, theta_limit_deg=40.0)

    target_kappa_max = float(np.max(np.abs(np.asarray(device_otf_info["kx_axis"], dtype=np.float64))) / float(device_otf_info["k0"]))
    target_nu_max = float(np.max(np.abs(np.asarray(device_otf_info["omega_axis"], dtype=np.float64))) / float(device_otf_info["omega0"]))

    axes_cfg = AxisConfig(-160.0, 160.0, 0.0, 2.0, 256, 256)
    x_lambda = np.linspace(axes_cfg.x_lambda_min, axes_cfg.x_lambda_max, axes_cfg.nx, dtype=np.float64)
    t_t1e4 = np.linspace(axes_cfg.t_t1e4_min, axes_cfg.t_t1e4_max, axes_cfg.nt, dtype=np.float64)
    x_m, t_s, scaling = matched_compute_axes_from_display(
        x_lambda,
        t_t1e4,
        args.lambda0_nm,
        target_kappa_max,
        target_nu_max,
    )

    run_root = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "fig3_switch_original",
            "width_scale": 1.0,
            "time_scale": 1.0,
        },
        {
            "name": "fig3_switch_matched",
            "width_scale": float(args.width_scale_matched),
            "time_scale": float(args.time_scale_matched),
        },
    ]

    summaries: list[dict] = []
    for case in cases:
        sig = build_switching_segments_scaled(
            x_lambda,
            t_t1e4,
            width_scale=case["width_scale"],
            time_scale=case["time_scale"],
        )
        summary = run_case(
            case["name"],
            sig.signal,
            None,
            x_lambda,
            t_t1e4,
            x_m,
            t_s,
            device_otf_info,
            run_root,
            args.pad_factor,
        )
        case_dir = run_root / case["name"]
        device_xt = np.load(case_dir / "device_xt.npy")
        ideal_xt = np.load(case_dir / "ideal_xt.npy")
        plot_fig3_triptych(
            case_dir / "fig3_compare.png",
            sig.signal,
            device_xt,
            ideal_xt,
            x_lambda,
            t_t1e4,
            ("Input", "Device", "Ideal"),
        )
        summary.update(case)
        summaries.append(summary)

    with (run_root / "switch_match_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": {
                    **asdict(device_cfg),
                    "target_kappa_max": target_kappa_max,
                    "target_nu_max": target_nu_max,
                    "carrier_envelope_model": envelope_carrier_metadata(args.lambda0_nm),
                    "scaling": scaling,
                },
                "cases": summaries,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved_to: {run_root}")


if __name__ == "__main__":
    main()
