# -*- coding: utf-8 -*-
"""批量生成 64x64 超表面结构。"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from generate_one import N_COARSE, N_FINE, SIGMA1, SIGMA2, MIN_FEATURE_PX, generate_structure


FILL_LEVELS = (0.4, 0.5, 0.6, 0.7, 0.8)


def log(path, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_fill_schedule(num_samples, rng):
    repeats = num_samples // len(FILL_LEVELS)
    remainder = num_samples % len(FILL_LEVELS)
    fills = np.array(FILL_LEVELS * repeats + FILL_LEVELS[:remainder], dtype=np.float32)
    rng.shuffle(fills)
    return fills


def sample_params(rng):
    return {
        "N_COARSE": int(rng.choice([8, 10, 12], p=[0.40, 0.35, 0.25])),
        "N_FINE": N_FINE,
        "SIGMA1": float(rng.uniform(SIGMA1 - 0.2, SIGMA1 + 0.25)),
        "SIGMA2": float(rng.uniform(SIGMA2 - 0.1, SIGMA2 + 0.2)),
        "MIN_FEATURE_PX": int(rng.choice([5, 6, 7, 8], p=[0.18, 0.24, 0.34, 0.24])),
    }


def sample(rng, fill):
    params = sample_params(rng)
    result = generate_structure(
        RNG_SEED=int(rng.integers(0, 10**9)),
        N_COARSE=params["N_COARSE"],
        N_FINE=params["N_FINE"],
        SIGMA1=params["SIGMA1"],
        SIGMA2=params["SIGMA2"],
        TARGET_FILL=float(fill),
        MIN_FEATURE_PX=params["MIN_FEATURE_PX"],
        SAVE_FIG=False,
        SAVE_NPY=False,
    )
    arr = np.asarray(result["final"], dtype=np.uint8)
    if arr.shape != (64, 64):
        raise ValueError(f"生成结果尺寸错误，应为 (64, 64)，实际得到 {arr.shape}")
    return arr, result["fill_ratio"], params


def build_dataset(num_samples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    log_path, save_path = out_dir / "generate.log", out_dir / "structures.npy"
    structures = np.empty((num_samples, 64, 64), dtype=np.uint8)
    fill_schedule = build_fill_schedule(num_samples, rng)
    log(log_path, f"开始生成结构，样本数={num_samples}")
    log(log_path, f"基准参数: N_COARSE={N_COARSE}, N_FINE={N_FINE}, SIGMA1={SIGMA1}, SIGMA2={SIGMA2}, MIN_FEATURE_PX={MIN_FEATURE_PX}")
    log(log_path, f"扰动范围: N_COARSE in {{8, 10, 12}}, SIGMA1 in [{SIGMA1 - 0.2:.2f}, {SIGMA1 + 0.25:.2f}], SIGMA2 in [{SIGMA2 - 0.1:.2f}, {SIGMA2 + 0.2:.2f}], MIN_FEATURE_PX in {{5, 6, 7, 8}}")
    log(log_path, f"TARGET_FILL 档位: {FILL_LEVELS}，按样本数尽量平均分配")
    for i in range(num_samples):
        structures[i], fill, params = sample(rng, float(fill_schedule[i]))
        if (i + 1) % 50 == 0 or i + 1 == num_samples:
            log(
                log_path,
                f"进度 {i + 1}/{num_samples}，fill={fill:.4f}，N_COARSE={params['N_COARSE']}，SIGMA1={params['SIGMA1']:.2f}，SIGMA2={params['SIGMA2']:.2f}，MIN_FEATURE_PX={params['MIN_FEATURE_PX']}",
            )
    np.save(save_path, structures)
    log(log_path, f"结构保存完成: {save_path}")
    log(log_path, f"数组形状: {structures.shape}，约定 1=材料，0=空气")


def main():
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="批量生成 64x64 超表面结构")
    p.add_argument("num_samples", nargs="?", type=int, default=1000, help="默认 1000")
    p.add_argument("--num_samples", dest="num_samples_flag", type=int, default=None, help="样本数；优先级高于位置参数")
    p.add_argument("--out_dir", default=str(root / "data" / "structures"), help="默认 data/structures")
    a = p.parse_args()
    build_dataset(a.num_samples_flag if a.num_samples_flag is not None else a.num_samples, a.out_dir)


if __name__ == "__main__":
    main()
