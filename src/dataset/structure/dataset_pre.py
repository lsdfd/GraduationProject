# -*- coding: utf-8 -*-
"""批量生成 64x64 超表面结构。"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from generate_one import generate_structure


def log(path, msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def sample(rng):
    fill = rng.uniform(0.4, 0.6) if (r := rng.random()) < 0.6 else rng.uniform(0.3, 0.4) if r < 0.8 else rng.uniform(0.6, 0.7)
    result = generate_structure(
        RNG_SEED=int(rng.integers(0, 10**9)),
        N_COARSE=int(rng.choice([24, 32])),
        N_FINE=64,
        SIGMA1=float(rng.uniform(1.2, 1.8)),
        SIGMA2=float(rng.uniform(0.8, 1.2)),
        TARGET_FILL=float(fill),
        MIN_FEATURE_PX=int(rng.choice([3, 4, 5], p=[0.2, 0.6, 0.2])),
        SAVE_FIG=False,
        SAVE_NPY=False,
    )
    arr = np.asarray(result["final"], dtype=np.uint8)
    if arr.shape != (64, 64):
        raise ValueError(f"生成结果尺寸错误，应为 (64, 64)，实际得到 {arr.shape}")
    return arr, result["fill_ratio"]


def build_dataset(num_samples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    log_path, save_path = out_dir / "generate.log", out_dir / "structures.npy"
    structures = np.empty((num_samples, 64, 64), dtype=np.uint8)
    log(log_path, f"开始生成结构，样本数={num_samples}")
    for i in range(num_samples):
        structures[i], fill = sample(rng)
        if (i + 1) % 50 == 0 or i + 1 == num_samples:
            log(log_path, f"进度 {i + 1}/{num_samples}，fill={fill:.4f}")
    np.save(save_path, structures)
    log(log_path, f"结构保存完成: {save_path}")
    log(log_path, f"数组形状: {structures.shape}，约定 1=材料，0=空气")


def main():
    root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="批量生成 64x64 超表面结构")
    p.add_argument("num_samples", nargs="?", type=int, default=1000, help="默认 1000")
    p.add_argument("--out_dir", default=str(root / "data" / "structures"), help="默认 data/structures")
    a = p.parse_args()
    build_dataset(a.num_samples, a.out_dir)


if __name__ == "__main__":
    main()
