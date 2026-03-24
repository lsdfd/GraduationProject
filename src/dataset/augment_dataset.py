# -*- coding: utf-8 -*-
"""
数据增强脚本：大批量生成结构 → 代理模型快速筛选 → 输出高质量训练子集。

流程：
  1. 生成 --num_pool 个结构（或加载已有文件）→ data/structures/structures_pool.npy
  2. 用 ForwardSurrogate 批量推理，得到预测频谱
  3. 按 target_lambda 处二阶得分降序排列
  4. 方案 B：取 top-K 高分 + num_random 个随机（混合，避免分布偏斜）
  5. 保存到 data/structures/structures_augmented.npy

用法：
  # 全流程（生成 + 筛选）
  python src/dataset/augment_dataset.py --num_pool 100000 --topk 2000 --num_random 1000

  # 跳过生成，直接对已有结构池筛选
  python src/dataset/augment_dataset.py --skip_generate --topk 2000 --num_random 1000

  # 然后对筛出的结构跑 RCWA
  python src/dataset/rcwa/rcwa_all.py --structures data/structures/structures_augmented.npy

注意：
  - 代理模型需要已训练好的 checkpoints/forward_best.pt 和 checkpoints/cond_stats.npz
  - 生成 10w 个结构纯 CPU 约需 20-40 分钟，代理推理 GPU 约需 1-2 分钟
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "dataset" / "structure"))
sys.path.insert(0, str(ROOT / "src" / "model"))

from dataset_pre import sample, build_fill_schedule  # noqa: E402
from models import ForwardSurrogate  # noqa: E402
from infer.common import lambda_theta_grid, second_order_score_row, load_stats  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# 1. 生成结构池
# ─────────────────────────────────────────────────────────────────────────────

def generate_pool(num_samples: int, out_path: Path) -> np.ndarray:
    """生成 num_samples 个结构，保存至 out_path，返回 [N, 64, 64] uint8 数组。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    fill_schedule = build_fill_schedule(num_samples, rng)
    structures = np.empty((num_samples, 64, 64), dtype=np.uint8)
    print(f"[augment] 开始生成 {num_samples} 个结构（纯 CPU）...")
    for i in range(num_samples):
        structures[i], _, _ = sample(rng, float(fill_schedule[i]))
        if (i + 1) % 5000 == 0 or i + 1 == num_samples:
            print(f"[augment] 生成进度 {i + 1}/{num_samples}", flush=True)
    np.save(out_path, structures)
    print(f"[augment] 结构池已保存: {out_path}  shape={structures.shape}")
    return structures


# ─────────────────────────────────────────────────────────────────────────────
# 2. 代理推理 + 打分
# ─────────────────────────────────────────────────────────────────────────────

def score_with_surrogate(
    structures: np.ndarray,
    surrogate: ForwardSurrogate,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """
    批量推理并打分，返回每个结构在 1000nm 处的二阶得分 [N]。

    代理模型输出归一化频谱，先反归一化再打分，保证物理量纲正确。
    """
    _, thetas = lambda_theta_grid()
    n = len(structures)
    scores = np.zeros(n, dtype=np.float32)

    surrogate.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = (
                torch.from_numpy(structures[start:end].astype(np.float32))
                .unsqueeze(1)   # [B, 1, 64, 64]，值域 0/1
                .to(device)
            )
            pred_norm = surrogate(batch).cpu().numpy()          # [B, 2, 17]
            pred_raw = pred_norm * std + mean                   # 反归一化

            for i in range(end - start):
                tpp_row = pred_raw[i, 0]                       # [17] tpp @ target_lambda
                s = second_order_score_row(tpp_row, thetas)
                scores[start + i] = float(s["score"])

            if end % 20000 < batch_size or end == n:
                print(f"[augment] 打分进度 {end}/{n}", flush=True)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 3. 方案 B：top-K 高分 + num_random 随机混合
# ─────────────────────────────────────────────────────────────────────────────

def select_mixed(
    scores: np.ndarray,
    topk: int,
    num_random: int,
    rng_seed: int = 42,
) -> np.ndarray:
    """
    返回选中结构的索引（已排序）。

    top-K 高分保证质量，num_random 随机样本保证分布覆盖，
    两部分去重后合并，避免扩散模型只见过高分结构而丧失泛化能力。
    """
    n = len(scores)
    sorted_idx = np.argsort(scores)[::-1]
    top_idx = sorted_idx[:min(topk, n)]
    top_set = set(top_idx.tolist())

    rng = np.random.default_rng(rng_seed)
    remaining = np.array([i for i in range(n) if i not in top_set], dtype=np.int64)
    num_random = min(num_random, len(remaining))
    random_idx = rng.choice(remaining, size=num_random, replace=False)

    selected = np.sort(np.concatenate([top_idx, random_idx]))
    return selected.astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="数据增强：大批量生成结构 → 代理筛选 → 输出高质量训练子集"
    )
    p.add_argument("--num_pool",     type=int,   default=100000,
                   help="生成的结构总数，默认 100000")
    p.add_argument("--topk",         type=int,   default=2000,
                   help="保留的高分结构数，默认 2000")
    p.add_argument("--num_random",   type=int,   default=1000,
                   help="混入的随机结构数（方案 B），默认 1000")
    p.add_argument("--batch_size",   type=int,   default=512,
                   help="代理模型推理 batch size，默认 512")
    p.add_argument("--skip_generate",action="store_true",
                   help="跳过生成步骤，直接加载 --pool_path")
    p.add_argument("--pool_path",
                   default=str(ROOT / "data" / "structures" / "structures_pool.npy"),
                   help="结构池路径（生成时的输出 / 跳过生成时的输入）")
    p.add_argument("--out_path",
                   default=str(ROOT / "data" / "structures" / "structures_augmented.npy"),
                   help="筛选结果输出路径")
    p.add_argument("--scores_path",
                   default=str(ROOT / "data" / "structures" / "augmented_scores.npy"),
                   help="全量打分结果保存路径（方便事后分析）")
    p.add_argument("--forward_ckpt",
                   default=str(ROOT / "checkpoints" / "forward_best.pt"))
    p.add_argument("--stats",
                   default=str(ROOT / "checkpoints" / "cond_stats.npz"))
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    pool_path   = Path(args.pool_path)
    out_path    = Path(args.out_path)
    scores_path = Path(args.scores_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. 生成或加载结构池 ────────────────────────────────────────────────
    if args.skip_generate:
        print(f"[augment] 跳过生成，加载结构池: {pool_path}")
        structures = np.load(pool_path)
    else:
        structures = generate_pool(args.num_pool, pool_path)
    print(f"[augment] 结构池大小: {structures.shape}")

    # ── 2. 加载代理模型 ────────────────────────────────────────────────────
    mean, std = load_stats(args.stats)
    cond_ch = int(mean.shape[1])
    surrogate = ForwardSurrogate(out_ch=cond_ch).to(args.device)
    ckpt = torch.load(args.forward_ckpt, map_location=args.device)
    surrogate.load_state_dict(ckpt["model"])
    print(f"[augment] 代理模型已加载: {args.forward_ckpt}  device={args.device}")

    # ── 3. 批量打分 ────────────────────────────────────────────────────────
    scores = score_with_surrogate(
        structures, surrogate, mean, std, args.device,
        batch_size=args.batch_size,
    )
    np.save(scores_path, scores)
    print(
        f"[augment] 打分完成  "
        f"min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}"
    )
    print(f"[augment] 全量分数已保存: {scores_path}")

    # ── 4. 方案 B：top-K 高分 + num_random 随机混合 ────────────────────────
    selected = select_mixed(scores, args.topk, args.num_random)

    # 分别统计 top-K 和 random 部分的分数
    sorted_idx = np.argsort(scores)[::-1]
    topk_idx   = sorted_idx[:min(args.topk, len(scores))]
    topk_scores = scores[topk_idx]

    print(
        f"[augment] 选出 {len(selected)} 个结构 "
        f"（top-{args.topk} 高分 + {args.num_random} 随机）"
    )
    print(f"[augment] top-{args.topk} 高分部分:")
    print(f"          min={topk_scores.min():.4f}  max={topk_scores.max():.4f}  mean={topk_scores.mean():.4f}")
    print(f"[augment] 混合后全部 {len(selected)} 个样本:")
    print(f"          min={scores[selected].min():.4f}  max={scores[selected].max():.4f}  mean={scores[selected].mean():.4f}")

    # ── 5. 保存 ────────────────────────────────────────────────────────────
    np.save(out_path, structures[selected])
    print(f"[augment] 筛选结果已保存: {out_path}  shape={structures[selected].shape}")
    print()
    print("下一步运行 RCWA：")
    print(f"  python src/dataset/rcwa/rcwa_all.py --structures {out_path}")


if __name__ == "__main__":
    main()
