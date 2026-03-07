# -*- coding: utf-8 -*-
"""读取 structures.npy，批量做 RCWA 仿真，并输出适配 model 的 train_data.npz。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from rcwa import torcwa_simulation
except ModuleNotFoundError as exc:
    torcwa_simulation = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


LAMBDAS = np.arange(1000.0, 1500.0 + 1e-6, 50.0, dtype=np.float32)
THETAS = np.arange(-40.0, 40.0 + 1e-6, 5.0, dtype=np.float32)


def append_log(log_path, message):
    """同时写文件和终端，方便长任务排查。"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_phy_kwargs(lam, theta):
    """集中管理物理参数，避免脚本里散落常数。"""
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


def simulate_one_structure(structure, device, rcwa_orders):
    """返回单个结构的 tpp 复数图，shape 为 [11, 17] = [lambda, theta]。"""
    layer = torch.from_numpy(structure.astype(np.float32)).to(device)
    tpp = np.full((len(LAMBDAS), len(THETAS)), np.nan + 1j * np.nan, dtype=np.complex64)
    ok = True
    failures = []

    for i, lam in enumerate(LAMBDAS):
        for j, theta in enumerate(THETAS):
            try:
                out = torcwa_simulation(
                    build_phy_kwargs(lam, theta),
                    layer,
                    rcwa_orders=rcwa_orders,
                    project=False,
                    device=device,
                )
                tpp[i, j] = complex(out["tpp"].detach().cpu().item())
            except Exception as exc:
                ok = False
                failures.append(
                    {
                        "lambda_nm": float(lam),
                        "theta_deg": float(theta),
                        "error": str(exc),
                    }
                )

    return tpp, ok, failures


def save_partial(save_path, structures, tpp_real, tpp_imag):
    """定期落盘，避免长任务中断后完全丢失。"""
    np.savez(
        save_path,
        structures=structures,
        tpp_real=tpp_real,
        tpp_imag=tpp_imag,
        tpp_mag=np.sqrt(tpp_real**2 + tpp_imag**2),
        lambdas=LAMBDAS,
        thetas=THETAS,
    )


def build_dataset(structures_path, save_path, log_path, rcwa_orders, save_every, device):
    if torcwa_simulation is None:
        raise ModuleNotFoundError(f"无法导入 RCWA 依赖，请先安装 torcwa。原始错误: {IMPORT_ERROR}")

    structures = np.load(structures_path).astype(np.uint8)
    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures.npy 应为 [N, 64, 64]，实际得到 {structures.shape}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tpp_real = np.full((len(structures), len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    tpp_imag = np.full((len(structures), len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    failed_samples = []

    append_log(log_path, f"开始 RCWA 批量仿真，样本数={len(structures)}，device={device}，orders={rcwa_orders}")

    for idx, structure in enumerate(structures):
        tpp, ok, failures = simulate_one_structure(structure, device=device, rcwa_orders=rcwa_orders)
        tpp_real[idx] = tpp.real.astype(np.float32)
        tpp_imag[idx] = tpp.imag.astype(np.float32)

        if ok:
            append_log(log_path, f"样本 {idx + 1}/{len(structures)} 完成")
        else:
            failed_samples.append({"index": idx, "failures": failures})
            append_log(log_path, f"样本 {idx + 1}/{len(structures)} 存在失败点，失败数={len(failures)}")

        if (idx + 1) % save_every == 0 or idx + 1 == len(structures):
            save_partial(save_path, structures, tpp_real, tpp_imag)
            append_log(log_path, f"已保存中间结果: {save_path}")

    failure_path = save_path.with_name(save_path.stem + "_failures.json")
    with failure_path.open("w", encoding="utf-8") as f:
        json.dump(failed_samples, f, ensure_ascii=False, indent=2)

    append_log(log_path, f"主数据保存完成: {save_path}")
    append_log(log_path, f"失败日志保存完成: {failure_path}")


def main():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="批量仿真 structures.npy，并生成 model 可直接读取的 train_data.npz")
    parser.add_argument(
        "--structures",
        type=str,
        default=str(repo_root / "data" / "structures" / "structures.npy"),
        help="结构数组路径，默认 data/structures/structures.npy",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(repo_root / "data" / "train_data.npz"),
        help="输出 npz 路径，默认 data/train_data.npz",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=str(repo_root / "data" / "rcwa.log"),
        help="日志路径，默认 data/rcwa.log",
    )
    parser.add_argument("--rcwa_orders", type=int, default=9, help="RCWA 截断阶数，默认 9")
    parser.add_argument("--save_every", type=int, default=10, help="每多少个样本保存一次中间结果，默认 10")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="仿真设备，默认自动选择 cuda:0 或 cpu",
    )
    args = parser.parse_args()

    build_dataset(
        structures_path=Path(args.structures),
        save_path=Path(args.out),
        log_path=Path(args.log),
        rcwa_orders=args.rcwa_orders,
        save_every=max(1, args.save_every),
        device=args.device,
    )


if __name__ == "__main__":
    main()
