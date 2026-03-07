# -*- coding: utf-8 -*-
"""批量生成 64x64 超表面结构数据集。"""

import argparse
from pathlib import Path

import numpy as np

# 依赖同目录下的 generate_one.py
from generate_one import generate_structure


def sample_fill(rng):
    """按分布采样目标占空比。"""
    r = rng.random()
    if r < 0.6:
        return rng.uniform(0.4, 0.6)
    if r < 0.8:
        return rng.uniform(0.3, 0.4)
    return rng.uniform(0.6, 0.7)


def sample_params(rng):
    """采样单次结构生成参数。"""
    return {
        "RNG_SEED": int(rng.integers(0, 10**9)),
        "N_COARSE": int(rng.choice([24, 32])),
        "N_FINE": 64,
        "SIGMA1": float(rng.uniform(1.2, 1.8)),
        "SIGMA2": float(rng.uniform(0.8, 1.2)),
        "TARGET_FILL": float(sample_fill(rng)),
        "MIN_FEATURE_PX": int(rng.choice([3, 4, 5], p=[0.2, 0.6, 0.2])),
    }


def call_generator(params):
    """调用 generate_structure 并返回 uint8 的 64x64 数组。"""
    result = generate_structure(
        RNG_SEED=params["RNG_SEED"],
        N_COARSE=params["N_COARSE"],
        N_FINE=params["N_FINE"],
        SIGMA1=params["SIGMA1"],
        SIGMA2=params["SIGMA2"],
        TARGET_FILL=params["TARGET_FILL"],
        MIN_FEATURE_PX=params["MIN_FEATURE_PX"],
        SAVE_FIG=False,
        SAVE_NPY=False,
    )

    arr = result["final"] if isinstance(result, dict) else result
    arr = np.asarray(arr, dtype=np.uint8)

    # 保底校验：确保每个样本都是 64x64
    if arr.shape != (64, 64):
        raise ValueError(f"生成结果尺寸错误，应为 (64,64)，实际得到 {arr.shape}")

    return arr


def build_dataset(num_samples, out_dir):
    """批量生成样本并保存为 structures.npy。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()
    structures = []

    # 主循环：按目标数量逐个生成，不做去重
    for i in range(num_samples):
        params = sample_params(rng)
        arr = call_generator(params)
        structures.append(arr)

        done = i + 1
        if done % 50 == 0 or done == num_samples:
            print(f"[{done}/{num_samples}] 已完成")

    structures = np.stack(structures, axis=0).astype(np.uint8)

    save_path = out_dir / "structures.npy"
    np.save(save_path, structures)

    print("\n数据集生成完成")
    print(f"保存路径: {save_path}")
    print(f"数组形状: {structures.shape}")
    print("数组约定: 1 = 材料, 0 = 空气")


def main():
    parser = argparse.ArgumentParser(description="批量生成 64x64 超表面结构数据集（仅保存 structures.npy）")
    parser.add_argument("num_samples", type=int, help="要生成的样本数量")
    parser.add_argument("--out_dir", type=str, default="dataset_out", help="输出目录")
    args = parser.parse_args()

    build_dataset(args.num_samples, args.out_dir)


if __name__ == "__main__":
    # 运行示例：python dataset_pre.py 1000 --out_dir ./dataset_out
    main()
