"""
统一评估脚本：对所有方法跑相同的测试 case，输出 results.json。

评估流程：
  生成 N 个候选 → 前向代理批量打分 → 挑最优 → 计算指标
  （不跑 RCWA，代理模型足够快且一致）

用法：
  cd /data/GraduationProject
  python src/baselines/eval/run_eval.py \
      --data_path      data/train_data.npz \
      --forward_ckpt   checkpoints/forward_best.pt \
      --stats_path     checkpoints/cond_stats.npz \
      --cvae_ckpt      checkpoints/cvae/cvae_best.pt \
      --cgan_ckpt      checkpoints/cgan/cgan_best.pt \
      --diffusion_ckpt checkpoints/diffusion_best.pt \
      --n_test         15 \
      --n_samples      16 \
      --target_lambda  1000.0 \
      --save_dir       samples/eval_compare
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in [
    os.path.join(_ROOT, "src", "model"),
    os.path.join(_ROOT, "src", "infer"),
    os.path.join(_ROOT, "src", "baselines", "eval"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset import RCWADataset
from models import ForwardSurrogate
from common import lambda_theta_grid, second_order_score_map
from infer_all import load_all_models, timed_generate
from metrics import summarize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",      default="data/train_data.npz")
    p.add_argument("--forward_ckpt",   default="checkpoints/forward_best.pt",
                   help="前向代理 checkpoint 路径")
    p.add_argument("--stats_path",     default="checkpoints/cond_stats.npz",
                   help="归一化统计文件（train_forward.py 输出）")
    p.add_argument("--cvae_ckpt",      default="checkpoints/cvae/cvae_best.pt")
    p.add_argument("--cgan_ckpt",      default="checkpoints/cgan/cgan_best.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion_best.pt")
    p.add_argument("--n_test",         type=int,   default=15,
                   help="从验证集均匀采样的测试 case 数量")
    p.add_argument("--n_samples",      type=int,   default=16,
                   help="每个 case 每个方法生成的候选数量")
    p.add_argument("--target_lambda",  type=float, default=1000.0)
    p.add_argument("--save_dir",       default="samples/eval_compare")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def load_surrogate(forward_ckpt: str, stats_path: str, device: str):
    """加载前向代理模型和归一化统计量。"""
    ckpt = torch.load(forward_ckpt, map_location=device)
    cond_ch = ckpt.get("cond_ch", 2)
    surrogate = ForwardSurrogate(cond_out_ch=cond_ch).to(device)
    surrogate.load_state_dict(ckpt["model"])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    stats = np.load(stats_path)
    cond_mean = stats["mean"].astype(np.float32)   # [1,2,11,17] 或 [2,11,17]
    cond_std  = stats["std"].astype(np.float32)

    print(f"[run_eval] surrogate loaded from {forward_ckpt}")
    return surrogate, cond_mean, cond_std


def pick_test_cases(dataset, n_test: int, seed: int, train_ratio: float = 0.7):
    """从验证集均匀抽取 n_test 个 case，返回索引列表。"""
    rng     = np.random.default_rng(seed)
    n_all   = len(dataset)
    n_train = int(n_all * train_ratio)
    val_indices = list(range(n_train, n_all))
    chosen = rng.choice(val_indices, size=min(n_test, len(val_indices)),
                        replace=False)
    return sorted(chosen.tolist())


def eval_one_case(
    method_name: str,
    method_info: dict,
    cond_norm: torch.Tensor,   # [1,2,11,17] 归一化条件
    cond_raw: np.ndarray,      # [2,11,17]   物理空间条件（MAE 用）
    n_samples: int,
    device: str,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    target_lambda: float,
    surrogate: torch.nn.Module,
    cond_mean: np.ndarray,     # [1,2,11,17] 或 [2,11,17]
    cond_std: np.ndarray,
) -> dict:
    """
    对单个测试 case 运行一种方法，返回指标字典。

    评估路径：
      生成 N 个候选 → 代理批量预测光谱 → second_order_score_map 打分
      → 挑最优候选 → summarize 所有指标
    """

    # ── 生成候选结构 ──────────────────────────────────────────────────
    structs, infer_time = timed_generate(
        method_info, cond_norm, n_samples, device)   # [N, 64, 64]

    # ── 代理批量打分（一次前向，无RCWA）────────────────────────────────
    x_batch = torch.from_numpy(structs[:, None]).float().to(device)  # [N,1,64,64]
    with torch.no_grad():
        pred_norm = surrogate(x_batch)   # [N, 2, 11, 17] 归一化预测

    pred_raw = pred_norm.cpu().numpy() * cond_std + cond_mean   # [N, 2, 11, 17]

    scores     = []
    best_score = -1.0
    best_pred_cond = pred_raw[0]

    for i, pr in enumerate(pred_raw):
        sc = second_order_score_map(
            pr[0],                  # tpp [L, T]
            lambdas, thetas,
            target_lambda=target_lambda,
        )["score"]
        scores.append(sc)
        if sc > best_score:
            best_score     = sc
            best_pred_cond = pr

    # ── 汇总指标 ──────────────────────────────────────────────────────
    result = summarize(
        candidates     = structs,
        scores         = scores,
        best_pred_cond = best_pred_cond,
        target_cond    = cond_raw,
        infer_time     = infer_time,
    )
    result["method"] = method_name
    return result


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    lambdas, thetas = lambda_theta_grid()

    # ── 数据 ─────────────────────────────────────────────────────────
    dataset      = RCWADataset(args.data_path)
    test_indices = pick_test_cases(dataset, args.n_test, args.seed)
    print(f"[run_eval] {len(test_indices)} test cases, "
          f"n_samples={args.n_samples}, device={device}")

    # ── 加载前向代理 ──────────────────────────────────────────────────
    surrogate, cond_mean, cond_std = load_surrogate(
        args.forward_ckpt, args.stats_path, device)

    # ── 加载所有生成模型 ──────────────────────────────────────────────
    all_models = load_all_models(
        cvae_ckpt      = args.cvae_ckpt,
        cgan_ckpt      = args.cgan_ckpt,
        diffusion_ckpt = args.diffusion_ckpt,
        device         = device,
    )
    print(f"[run_eval] loaded methods: {list(all_models.keys())}")

    # ── 逐 case 评估 ─────────────────────────────────────────────────
    all_results = {name: [] for name in all_models}

    for case_num, idx in enumerate(test_indices):
        x01, cond_norm_t = dataset[idx]
        cond_norm = cond_norm_t.unsqueeze(0)          # [1,2,11,17]
        cond_raw  = (cond_norm_t.numpy()
                     * dataset.cond_std.squeeze(0)
                     + dataset.cond_mean.squeeze(0))  # [2,11,17] 物理空间

        print(f"\n[run_eval] case {case_num+1}/{len(test_indices)} "
              f"(dataset idx={idx})")

        for method_name, method_info in all_models.items():
            print(f"  → {method_name} ...", end=" ", flush=True)
            try:
                res = eval_one_case(
                    method_name    = method_name,
                    method_info    = method_info,
                    cond_norm      = cond_norm,
                    cond_raw       = cond_raw,
                    n_samples      = args.n_samples,
                    device         = device,
                    lambdas        = lambdas,
                    thetas         = thetas,
                    target_lambda  = args.target_lambda,
                    surrogate      = surrogate,
                    cond_mean      = cond_mean,
                    cond_std       = cond_std,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                res = {"method": method_name, "error": str(e)}

            all_results[method_name].append(res)
            score = res.get("best_score", float("nan"))
            time_ = res.get("inference_time_s", 0)
            print(f"best_score={score:.4f}  time={time_:.1f}s")

    # ── 保存结果 ─────────────────────────────────────────────────────
    out_path = os.path.join(args.save_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[run_eval] saved → {out_path}")

    # ── 打印汇总表格 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{'Method':<12} {'BestScore':>10} {'SuccessRate':>12} "
          f"{'SpecMAE':>9} {'Diversity':>10} {'Time(s)':>8}")
    print("-"*60)
    for name, results in all_results.items():
        valid = [r for r in results if "error" not in r]
        if not valid:
            print(f"{name:<12}  (no valid results)")
            continue
        bs  = np.mean([r["best_score"]        for r in valid])
        sr  = np.mean([r["top1_success_rate"]  for r in valid])
        mae = np.mean([r["spectrum_mae"]       for r in valid])
        div = np.mean([r["diversity"]          for r in valid])
        t   = np.mean([r["inference_time_s"]   for r in valid])
        print(f"{name:<12} {bs:>10.4f} {sr:>12.3f} "
              f"{mae:>9.4f} {div:>10.4f} {t:>8.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
