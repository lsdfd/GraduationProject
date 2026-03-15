# -*- coding: utf-8 -*-
"""读取 structures.npy，批量做 RCWA 仿真并输出 |tpp|、|tss|。"""

import argparse
import json
import time
from datetime import datetime
from multiprocessing import get_context
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
LAMBDAS = np.arange(800.0, 1300.1, 50.0, dtype=np.float32)
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
    tpp_mag = np.full((len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    tss_mag = np.full_like(tpp_mag, np.nan)
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
                tpp_mag[i, j] = float(out["tpp_mag"].detach().cpu().item())
                tss_mag[i, j] = float(out["tss_mag"].detach().cpu().item())
            except Exception as exc:
                failures.append({"lambda_nm": float(lam), "theta_deg": float(theta), "error": str(exc)})

    return tpp_mag, tss_mag, failures


def save_npz(path, structures, tpp_mag, tss_mag):
    np.savez(
        path,
        structures=structures,
        tpp_mag=tpp_mag,
        tss_mag=tss_mag,
        lambdas=LAMBDAS,
        thetas=THETAS,
    )


def parse_devices(devices_arg, device_arg):
    if devices_arg:
        devices = [d.strip() for d in devices_arg.split(",") if d.strip()]
        if not devices:
            raise ValueError("--devices 为空，请传入类似 cuda:0,cuda:1")
        return devices
    if device_arg:
        return [device_arg]
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]


def _run_indices(structures, indices, device, orders):
    tpp_part = np.full((len(indices), len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    tss_part = np.full_like(tpp_part, np.nan)
    failed = []

    for local_i, idx in enumerate(indices):
        t0 = time.perf_counter()
        tpp_i, tss_i, failures = simulate_one(structures[idx], device, orders)
        tpp_part[local_i] = tpp_i
        tss_part[local_i] = tss_i
        dt = time.perf_counter() - t0
        if failures:
            failed.append({"index": int(idx), "failures": failures})
            print(
                f"[worker {device}] 样本 {local_i + 1}/{len(indices)} (global={idx}) 失败点数={len(failures)}，用时={dt:.2f}s",
                flush=True,
            )
        else:
            print(
                f"[worker {device}] 样本 {local_i + 1}/{len(indices)} (global={idx}) 完成，用时={dt:.2f}s",
                flush=True,
            )

    return tpp_part, tss_part, failed


def _worker_entry(structures, indices, device, orders, queue):
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        tpp_part, tss_part, failed = _run_indices(structures, indices, device, orders)
        queue.put(
            {
                "ok": True,
                "device": device,
                "indices": np.asarray(indices, dtype=np.int64),
                "tpp": tpp_part,
                "tss": tss_part,
                "failed": failed,
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "device": device, "error": str(exc)})


def main():
    if torcwa_simulation is None:
        raise ModuleNotFoundError(f"无法导入 RCWA 依赖，请先安装 torcwa。原始错误: {IMPORT_ERROR}")

    p = argparse.ArgumentParser(description="批量仿真 structures.npy")
    p.add_argument("--structures")
    p.add_argument("--out")
    p.add_argument("--log")
    p.add_argument("--max_samples", type=int, default=5000, help="最多仿真的样本数；默认取前 5000 个结构")
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--device", default=None, help="单设备模式；默认自动使用全部可见 GPU，无 GPU 时回退到 cpu")
    p.add_argument("--devices", default=None, help="逗号分隔设备列表，如: cuda:0,cuda:1")
    a = p.parse_args()

    structures_path = Path(a.structures) if a.structures else DEFAULT_STRUCTURES
    out_path = Path(a.out) if a.out else DEFAULT_OUT
    log_path = Path(a.log) if a.log else DEFAULT_LOG
    structures = np.load(structures_path).astype(np.uint8)
    if a.max_samples is not None:
        structures = structures[: max(1, a.max_samples)]

    if structures.ndim != 3 or structures.shape[1:] != (64, 64):
        raise ValueError(f"structures.npy 应为 [N, 64, 64]，实际得到 {structures.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tpp_mag = np.full((len(structures), len(LAMBDAS), len(THETAS)), np.nan, dtype=np.float32)
    tss_mag = np.full_like(tpp_mag, np.nan)
    failed = []
    devices = parse_devices(a.devices, a.device)
    num_workers = len(devices)

    if num_workers == 1:
        log(log_path, f"开始 RCWA 批量仿真，样本数={len(structures)}，device={devices[0]}，orders={a.rcwa_orders}")
        for idx in range(len(structures)):
            t0 = time.perf_counter()
            tpp_mag[idx], tss_mag[idx], failures = simulate_one(structures[idx], devices[0], a.rcwa_orders)
            dt = time.perf_counter() - t0
            if failures:
                failed.append({"index": idx, "failures": failures})
                log(log_path, f"样本 {idx + 1}/{len(structures)} 失败点数={len(failures)}，用时={dt:.2f}s")
            else:
                log(log_path, f"样本 {idx + 1}/{len(structures)} 完成，用时={dt:.2f}s")
            if (idx + 1) % max(1, a.save_every) == 0 or idx + 1 == len(structures):
                save_npz(out_path, structures, tpp_mag, tss_mag)
                log(log_path, f"已保存中间结果: {out_path}")
    else:
        all_indices = np.arange(len(structures), dtype=np.int64)
        split_indices = [chunk.tolist() for chunk in np.array_split(all_indices, num_workers) if len(chunk) > 0]
        active_devices = devices[: len(split_indices)]
        ctx = get_context("spawn")
        queue = ctx.Queue()
        procs = []

        log(
            log_path,
            f"开始 RCWA 多卡并行，样本数={len(structures)}，devices={active_devices}，orders={a.rcwa_orders}",
        )
        for dev, idxs in zip(active_devices, split_indices):
            log(log_path, f"分配 {dev}: {len(idxs)} 个样本（index {idxs[0]}..{idxs[-1]}）")
            proc = ctx.Process(
                target=_worker_entry,
                args=(structures, idxs, dev, a.rcwa_orders, queue),
            )
            proc.start()
            procs.append(proc)

        received = 0
        while received < len(procs):
            msg = queue.get()
            received += 1
            if not msg.get("ok", False):
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(f"worker {msg.get('device')} 失败: {msg.get('error')}")

            idxs = msg["indices"]
            tpp_mag[idxs] = msg["tpp"]
            tss_mag[idxs] = msg["tss"]
            failed.extend(msg["failed"])
            log(log_path, f"worker {msg['device']} 完成，回收 {len(idxs)} 个样本")
            save_npz(out_path, structures, tpp_mag, tss_mag)
            log(log_path, f"已保存中间结果: {out_path}")

        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"子进程异常退出，exitcode={proc.exitcode}")

    with out_path.with_name(f"{out_path.stem}_failures.json").open("w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
