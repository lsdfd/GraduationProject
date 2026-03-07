# -*- coding: utf-8 -*-
"""读取 structures.npy，批量做 RCWA 仿真。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from rcwa import torcwa_simulation
except ModuleNotFoundError as exc:
    torcwa_simulation, IMPORT_ERROR = None, exc
else:
    IMPORT_ERROR = None

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRUCTURES = ROOT / "data" / "structures" / "structures.npy"
DEFAULT_OUT = ROOT / "data" / "train_data.npz"
DEFAULT_LOG = ROOT / "data" / "rcwa.log"
LAMBDAS = np.arange(1000.0, 1500.1, 50.0, dtype=np.float32)
THETAS = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)


def log(path, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def phy_kwargs(lam, theta):
    return {
        "periodicity": 500.0,
        "h": 500.0,
        "lam": float(lam),
        "tet": float(theta),
        "phi": 0.0,
        "angle_unit": "deg",
        "angle_layer": "input",
        "input_medium": "air",
        "output_medium": "SiO2",
        "structure": "Si",
    }


def simulate_one(structure, device, orders):
    layer = torch.from_numpy(structure.astype(np.float32)).to(device)
    real = np.full((len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    imag = np.full_like(real, np.nan)
    failures = []

    for i, lam in enumerate(LAMBDAS):
        for j, theta in enumerate(THETAS):
            try:
                out = torcwa_simulation(
                    phy_kwargs(lam, theta),
                    layer,
                    rcwa_orders=orders,
                    project=False,
                    device=device,
                )
                value = complex(out["tpp"].detach().cpu().item())
                real[i, j], imag[i, j] = value.real, value.imag
            except Exception as exc:
                failures.append({"lambda_nm": float(lam), "theta_deg": float(theta), "error": str(exc)})

    return real, imag, failures


def save_npz(path, structures, real, imag):
    np.savez(
        path,
        structures=structures,
        tpp_real=real,
        tpp_imag=imag,
        tpp_mag=np.sqrt(real**2 + imag**2),
        lambdas=LAMBDAS,
        thetas=THETAS,
    )


def main():
    if torcwa_simulation is None:
        raise ModuleNotFoundError(f"无法导入 RCWA 依赖，请先安装 torcwa。原始错误: {IMPORT_ERROR}")

    p = argparse.ArgumentParser(description="批量仿真 structures.npy")
    p.add_argument("--structures")
    p.add_argument("--out")
    p.add_argument("--log")
    p.add_argument("--rcwa_orders", type=int, default=9)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    structures_path = Path(a.structures) if a.structures else DEFAULT_STRUCTURES
    out_path = Path(a.out) if a.out else DEFAULT_OUT
    log_path = Path(a.log) if a.log else DEFAULT_LOG
    structures = np.load(structures_path).astype(np.uint8)

    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures.npy 应为 [N, 64, 64]，实际得到 {structures.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    real = np.full((len(structures), len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    imag = np.full_like(real, np.nan)
    failed = []

    log(log_path, f"开始 RCWA 批量仿真，样本数={len(structures)}，device={a.device}，orders={a.rcwa_orders}")
    for idx, structure in enumerate(structures):
        real[idx], imag[idx], failures = simulate_one(structure, a.device, a.rcwa_orders)
        if failures:
            failed.append({"index": idx, "failures": failures})
            log(log_path, f"样本 {idx + 1}/{len(structures)} 失败点数={len(failures)}")
        else:
            log(log_path, f"样本 {idx + 1}/{len(structures)} 完成")
        if (idx + 1) % max(1, a.save_every) == 0 or idx + 1 == len(structures):
            save_npz(out_path, structures, real, imag)
            log(log_path, f"已保存中间结果: {out_path}")

    with out_path.with_name(f"{out_path.stem}_failures.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
