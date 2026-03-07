# -*- coding: utf-8 -*-
"""批量生成 64x64 超表面结构，并保存为后续 RCWA 使用的 structures.npy。"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from generate_one import generate_structure


def sample_fill(rng):
    """按预设分布采样目标占空比。"""
    r = rng.random()
    if r < 0.6:
        return rng.uniform(0.4, 0.6)
    if r < 0.8:
        return rng.uniform(0.3, 0.4)
    return rng.uniform(0.6, 0.7)


def sample_params(rng):
    """每个样本使用独立随机种子，并采样结构参数。"""
    return {
        "RNG_SEED": int(rng.integers(0, 10**9)),
        "N_COARSE": int(rng.choice([24, 32])),
        "N_FINE": 64,
        "SIGMA1": float(rng.uniform(1.2, 1.8)),
        "SIGMA2": float(rng.uniform(0.8, 1.2)),
        "TARGET_FILL": float(sample_fill(rng)),
        "MIN_FEATURE_PX": int(rng.choice([3, 4, 5], p=[0.2, 0.6, 0.2])),
    }


def append_log(log_path, message):
    """同时写文件和终端，方便长任务排查。"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def call_generator(params):
    """调用单样本生成器，并强制校验输出尺寸。"""
    result = generate_structure(**params, SAVE_FIG=False, SAVE_NPY=False)
    arr = np.asarray(result["final"], dtype=np.uint8)

    if arr.shape != (64, 64):
        raise ValueError(f"生成结果尺寸错误，应为 (64, 64)，实际得到 {arr.shape}")

    return arr, result


def build_dataset(num_samples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "generate.log"
    save_path = out_dir / "structures.npy"
    rng = np.random.default_rng()

    append_log(log_path, f"开始生成结构，样本数={num_samples}")

    structures = np.empty((num_samples, 64, 64), dtype=np.uint8)

    for i in range(num_samples):
        params = sample_params(rng)
        arr, result = call_generator(params)
        structures[i] = arr

        if (i + 1) % 50 == 0 or i + 1 == num_samples:
            append_log(
                log_path,
                f"进度 {i + 1}/{num_samples}，seed={params['RNG_SEED']}，fill={result['fill_ratio']:.4f}",
            )

    np.save(save_path, structures)

    append_log(log_path, f"结构保存完成: {save_path}")
    append_log(log_path, f"数组形状: {structures.shape}，约定 1=材料，0=空气")


def main():
    parser = argparse.ArgumentParser(description="批量生成 64x64 超表面结构")
    parser.add_argument("num_samples", nargs="?", type=int, default=1000, help="要生成的样本数量，默认 1000")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data" / "structures"),
        help="输出目录，默认保存到仓库 data/structures",
    )
    args = parser.parse_args()
    build_dataset(args.num_samples, args.out_dir)


if __name__ == "__main__":
    main()
