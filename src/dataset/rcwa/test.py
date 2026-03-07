import json
import math
import statistics
import time
from pathlib import Path

import torch

from rcwa import build_hollow_brick_mask, torcwa_simulation


# 论文二阶微分目标角谱
TARGET = [0.00, 0.04, 0.16, 0.36, 0.64, 1.00]

# 论文角度采样
ANGLES = [0, 5, 10, 15, 20, 25]


def run_case(nx, order, device="cpu"):

    layer = build_hollow_brick_mask(
        periodicity=611.0,
        L_outer=340.0,
        W_inner=153.0,
        nx=nx,
        device=device,
    )

    vals = []
    timings = []

    for ang in ANGLES:

        phy = {
            "periodicity": 611.0,
            "h": 450.0,
            "lam": 1250.0,

            "tet": float(ang),
            "phi": 0.0,
            "angle_unit": "deg",

            "angle_layer": "input",

            "input_medium": "air",
            "output_medium": "SiO2",

            "structure": "Si",

            # 使用论文材料参数
            "n_output": 1.45,
            "n_structure": 3.48,
        }

        t0 = time.perf_counter()

        out = torcwa_simulation(
            phy,
            layer,
            rcwa_orders=order,
            project=False,
            device=device,
        )

        timings.append(time.perf_counter() - t0)

        # 论文比较的是 |t_pp|
        tpp = out["t_matrix"][1, 1]
        vals.append(float(torch.abs(tpp).detach().cpu()))

    vmax = max(vals)
    norm = [v / vmax if vmax > 0 else 0.0 for v in vals]

    mae = sum(abs(a - b) for a, b in zip(norm, TARGET)) / len(TARGET)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(norm, TARGET)) / len(TARGET))

    return {
        "nx": nx,
        "order": order,
        "raw_abs_tpp": vals,
        "norm_abs_tpp": norm,
        "target": TARGET,
        "mae_vs_target": mae,
        "rmse_vs_target": rmse,
        "avg_time_s": statistics.mean(timings),
        "total_time_s": sum(timings),
    }


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # RCWA 收敛测试
    cases = [
        (64, 7),
        (128, 9),
        (128, 11),
    ]

    results = []

    for nx, order in cases:
        print(f"Running nx={nx}, order={order} on {device} ...", flush=True)
        results.append(run_case(nx, order, device=device))

    out_path = Path(__file__).resolve().parent / "run_tableS2_check_results.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")