"""
所有对比实验的指标计算函数。

指标：
  - best_score         最优候选的 second_order_score
  - mean_score         全部候选的平均 score（衡量整体生成质量）
  - top3_score         top-3 候选的平均 score（衡量稳定性）
  - top1_success_rate  score > 0.5 的比例
  - spectrum_mae       最优候选预测光谱 vs 目标的 L1 误差（物理空间）
  - binary_rate        结构的二值化程度（均值距0.5的距离）
  - diversity          N个候选之间的平均汉明距离（归一化到[0,1]）
  - inference_time     单次推理耗时（秒）
"""

import numpy as np
import sys
import os

# ── RCWA 评分函数：复用 laplas2 里的逻辑 ──────────────────────────
def _try_import_rcwa_score():
    """尝试 import src/infer 里的评分工具，返回 score 函数。"""
    infer_dir = os.path.join(os.path.dirname(__file__), "..", "..", "infer")
    if infer_dir not in sys.path:
        sys.path.insert(0, infer_dir)
    try:
        from common import second_order_score_map
        return second_order_score_map
    except ImportError:
        return None

_rcwa_score_fn = _try_import_rcwa_score()


def compute_second_order_score(tpp: np.ndarray, tss: np.ndarray,
                                lambdas: np.ndarray, thetas: np.ndarray,
                                target_lambda: float,
                                floor: float = 0.0,
                                off_target: float = 0.9) -> float:
    """
    调用 src/infer/common.py 里的 second_order_score。
    tpp/tss: [n_lambda, n_theta]
    返回 float score
    """
    if _rcwa_score_fn is None:
        raise ImportError("无法 import src/infer/common.py 里的 second_order_score")
    return float(_rcwa_score_fn(
        tpp, tss, lambdas, thetas,
        target_lambda=target_lambda,
        floor=floor,
        off_target=off_target,
    ))


# ── 主指标 ─────────────────────────────────────────────────────────

def best_score(scores: list) -> float:
    """N 个候选里最高的 score。"""
    return float(np.max(scores))


def mean_score(scores: list) -> float:
    """全部 N 个候选的平均 score，衡量整体生成质量（不只是偶然命中最优）。"""
    return float(np.mean(scores))


def top3_score(scores: list) -> float:
    """top-3 候选的平均 score，介于 best 和 mean 之间，衡量一致性。"""
    arr = np.array(scores)
    k = min(3, len(arr))
    return float(np.mean(np.sort(arr)[::-1][:k]))


def top1_success_rate(scores: list, threshold: float = 0.5) -> float:
    """score > threshold 的候选占比。"""
    arr = np.array(scores)
    return float((arr > threshold).mean())


def spectrum_mae(pred_cond: np.ndarray, target_cond: np.ndarray) -> float:
    """
    pred_cond:   [n_lambda, n_theta, 2] 或 [2, n_lambda, n_theta]  预测光谱（物理空间）
    target_cond: 同上，目标光谱（物理空间）
    返回 L1 MAE（transmission 单位，0~1）
    """
    return float(np.mean(np.abs(pred_cond - target_cond)))


def binary_rate(structures: np.ndarray) -> float:
    """
    structures: [N, H, W]，值域 [0,1]
    衡量整体二值化程度：x*(1-x) 越小越接近 0/1
    返回 1 - 4*mean(x*(1-x))，完全二值时=1，全0.5时=0
    """
    s = structures.astype(np.float32)
    return float(1.0 - 4.0 * np.mean(s * (1.0 - s)))


def diversity(structures: np.ndarray) -> float:
    """
    structures: [N, H, W]，值为 0 或 1（已二值化）
    计算 N 个结构两两之间的平均归一化汉明距离。
    汉明距离 = 不同像素数 / 总像素数，范围 [0,1]
    """
    N = structures.shape[0]
    if N <= 1:
        return 0.0
    flat = structures.reshape(N, -1).astype(np.float32)
    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(float(np.mean(np.abs(flat[i] - flat[j]))))
    return float(np.mean(dists))


# ── 汇总函数 ────────────────────────────────────────────────────────

def summarize(
    candidates: np.ndarray,      # [N, H, W] 二值结构
    scores: list,                 # 每个候选的 second_order_score
    best_pred_cond: np.ndarray,   # 最优候选的预测光谱（物理空间）
    target_cond: np.ndarray,      # 目标光谱（物理空间）
    infer_time: float,            # 推理总耗时（秒）
    threshold: float = 0.5,
) -> dict:
    """返回所有指标的字典。"""
    return {
        "best_score"        : best_score(scores),
        "mean_score"        : mean_score(scores),
        "top3_score"        : top3_score(scores),
        "top1_success_rate" : top1_success_rate(scores, threshold),
        "spectrum_mae"      : spectrum_mae(best_pred_cond, target_cond),
        "binary_rate"       : binary_rate(candidates),
        "diversity"         : diversity(candidates),
        "inference_time_s"  : infer_time,
        "all_scores"        : list(scores),
    }
