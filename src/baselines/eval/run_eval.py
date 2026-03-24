"""
统一评估脚本：所有方法在同一个物理目标上生成候选结构，输出 results.json。

目标构建方式与 laplas2.py 完全一致：
  topk CSV → top-1 数据集样本 → build_physics_target (sin²θ 混合) → 归一化

所有方法（random / CVAE / cGAN / diffusion）拿到完全相同的目标条件，
生成 n_samples 个候选，用前向代理打分后比较各指标。

用法：
  cd /data/GraduationProject
  python src/baselines/eval/run_eval.py \
      --data_path      data/train_data.npz \
      --topk_csv       data/second_order_scores/tpp_mag_top5_per_lambda.csv \
      --forward_ckpt   checkpoints/forward_best.pt \
      --stats_path     checkpoints/cond_stats.npz \
      --cvae_ckpt      checkpoints/cvae/cvae_best.pt \
      --cgan_ckpt      checkpoints/cgan/cgan_best.pt \
      --diffusion_ckpt checkpoints/diffusion_best.pt \
      --n_samples      32 \
      --target_lambda  1000.0 \
      --target_rank    1 \
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

from models import ForwardSurrogate
from common import lambda_theta_grid, second_order_score_map, second_order_target
from infer_all import METHODS, load_all_models, timed_generate
from metrics import summarize

# 复用当前 laplas.py 里的目标构建函数，和主推理流程保持一致
sys.path.insert(0, os.path.join(_ROOT, "src", "infer"))
from laplas import build_target


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",      default="data/train_data.npz")
    p.add_argument("--topk_csv",       default="data/second_order_scores/tpp_mag_top5_per_lambda.csv",
                   help="score_second_order.py 输出的 top-k CSV")
    p.add_argument("--forward_ckpt",   default="checkpoints/forward_best.pt")
    p.add_argument("--stats_path",     default="checkpoints/cond_stats.npz")
    p.add_argument("--cvae_ckpt",      default="checkpoints/cvae/cvae_best.pt")
    p.add_argument("--cgan_ckpt",      default="checkpoints/cgan/cgan_best.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion_best.pt")
    p.add_argument("--n_samples",      type=int,   default=32,
                   help="每个方法生成的候选数量")
    p.add_argument("--target_lambda",  type=float, default=1000.0)
    p.add_argument("--target_rank",    type=int,   default=1,
                   help="使用数据集中二阶得分第 N 名的样本作为目标模板")
    p.add_argument("--band_sigma_nm",  type=float, default=25.0)
    p.add_argument("--save_dir",       default="samples/eval_compare")
    p.add_argument("--methods",        default="all",
                   help=f"逗号分隔方法子集，候选：{','.join(METHODS)}；默认 all")
    return p.parse_args()


def parse_methods(arg: str) -> list[str]:
    if arg.strip().lower() == "all":
        return list(METHODS)
    methods = [m.strip() for m in arg.split(",") if m.strip()]
    invalid = [m for m in methods if m not in METHODS]
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}. Available: {list(METHODS)}")
    return methods


def load_surrogate(forward_ckpt: str, stats_path: str, device: str):
    ckpt = torch.load(forward_ckpt, map_location=device, weights_only=False)
    cond_ch = ckpt.get("cond_channels", ckpt.get("cond_ch", 2))
    surrogate = ForwardSurrogate(out_ch=cond_ch).to(device)
    surrogate.load_state_dict(ckpt["model"])
    surrogate.eval()
    for param in surrogate.parameters():
        param.requires_grad_(False)
    stats     = np.load(stats_path)
    cond_mean = stats["mean"].astype(np.float32)
    cond_std  = stats["std"].astype(np.float32)
    print(f"[run_eval] surrogate loaded: {forward_ckpt}")
    return surrogate, cond_mean, cond_std


def eval_one_method(
    method_name: str,
    method_info: dict,
    cond_norm: torch.Tensor,   # [1,2,11,17] 归一化目标条件
    cond_raw: np.ndarray,      # [2,11,17]   物理空间目标条件（MAE 用）
    n_samples: int,
    device: str,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    target_lambda: float,
    surrogate: torch.nn.Module,
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
) -> dict:
    """生成 n_samples 个候选，代理打分，返回指标字典。"""

    # ── 生成候选 ──────────────────────────────────────────────────────
    structs, infer_time = timed_generate(
        method_info, cond_norm, n_samples, device)   # [N, 64, 64]

    # ── 代理批量打分 ───────────────────────────────────────────────────
    x_batch = torch.from_numpy(structs[:, None]).float().to(device)  # [N,1,64,64]
    with torch.no_grad():
        pred_norm = surrogate(x_batch)   # [N, 2, 11, 17]

    pred_raw = pred_norm.cpu().numpy() * cond_std + cond_mean   # [N, 2, 11, 17]

    scores         = []
    best_score_val = -1.0
    best_pred_cond = pred_raw[0]
    best_idx = 0

    for idx, pr in enumerate(pred_raw):
        sc = second_order_score_map(
            pr[0], lambdas, thetas, target_lambda=target_lambda,
        )["score"]
        scores.append(sc)
        if sc > best_score_val:
            best_score_val = sc
            best_pred_cond = pr
            best_idx = idx

    lam_idx = int(np.argmin(np.abs(lambdas - float(target_lambda))))
    target_theta_curve = second_order_target(thetas).astype(np.float32)

    result = summarize(
        candidates     = structs,
        scores         = scores,
        best_pred_cond = best_pred_cond,
        target_cond    = cond_raw,
        infer_time     = infer_time,
    )
    result["method"] = method_name
    result["best_idx"] = int(best_idx)
    result["target_lambda_nm"] = float(target_lambda)
    result["thetas_deg"] = thetas.astype(np.float32).tolist()
    result["target_theta_curve"] = target_theta_curve.tolist()
    result["best_tpp_theta_curve"] = best_pred_cond[0, lam_idx].astype(np.float32).tolist()
    result["best_tss_theta_curve"] = best_pred_cond[1, lam_idx].astype(np.float32).tolist()
    return result


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    methods = parse_methods(args.methods)
    if args.target_rank != 1:
        raise ValueError("Current baseline target builder is aligned with laplas.py and only supports --target_rank 1.")

    lambdas, thetas = lambda_theta_grid()

    # ── 构建目标（与当前 laplas.py 保持一致）──────────────────────────
    from pathlib import Path
    cond_raw, _, _, sample_idx = build_target(
        cond_ch=2,
        target_lambda=args.target_lambda,
        band_sigma_nm=args.band_sigma_nm,
        train_npz_path=Path(args.data_path),
        topk_csv_path=Path(args.topk_csv),
    )   # [2, 11, 17] 物理空间目标

    print(f"[run_eval] target: lambda={args.target_lambda}nm  "
          f"rank={args.target_rank}  dataset_idx={sample_idx}")

    # ── 加载前向代理和生成模型 ────────────────────────────────────────
    surrogate, cond_mean, cond_std = load_surrogate(
        args.forward_ckpt, args.stats_path, device)

    all_models = load_all_models(
        cvae_ckpt      = args.cvae_ckpt,
        cgan_ckpt      = args.cgan_ckpt,
        diffusion_ckpt = args.diffusion_ckpt,
        device         = device,
        methods        = methods,
    )

    # ── 归一化目标条件 ────────────────────────────────────────────────
    cond_norm_np = (cond_raw - cond_mean.squeeze(0)) / cond_std.squeeze(0)
    cond_norm    = torch.from_numpy(cond_norm_np).unsqueeze(0).float()  # [1,2,11,17]

    # ── 为各方法注入额外参数 ──────────────────────────────────────────
    if "topo_opt" in all_models:
        all_models["topo_opt"]["target_lambda"] = args.target_lambda
    if "diffusion+guide" in all_models:
        all_models["diffusion+guide"]["surrogate"]   = surrogate
        all_models["diffusion+guide"]["target_norm"] = cond_norm.to(device)

    print(f"[run_eval] methods: {list(all_models.keys())}")

    # ── 每个方法评估 ──────────────────────────────────────────────────
    all_results = {}

    for method_name, method_info in all_models.items():
        print(f"\n[run_eval] → {method_name} ...", end=" ", flush=True)
        try:
            res = eval_one_method(
                method_name   = method_name,
                method_info   = method_info,
                cond_norm     = cond_norm,
                cond_raw      = cond_raw,
                n_samples     = args.n_samples,
                device        = device,
                lambdas       = lambdas,
                thetas        = thetas,
                target_lambda = args.target_lambda,
                surrogate     = surrogate,
                cond_mean     = cond_mean,
                cond_std      = cond_std,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            res = {"method": method_name, "error": str(e)}

        all_results[method_name] = res
        score = res.get("best_score", float("nan"))
        time_ = res.get("inference_time_s", 0.0)
        print(f"best_score={score:.4f}  time={time_:.1f}s")

    # ── 保存结果 ──────────────────────────────────────────────────────
    out_path = os.path.join(args.save_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[run_eval] saved → {out_path}")

    # ── 打印汇总表格 ──────────────────────────────────────────────────
    print("\n" + "="*62)
    print(f"target: lambda={args.target_lambda}nm  rank={args.target_rank}  "
          f"n_samples={args.n_samples}")
    print("="*62)
    print(f"{'Method':<12} {'BestScore':>10} {'SuccessRate':>12} "
          f"{'SpecMAE':>9} {'Diversity':>10} {'Time(s)':>8}")
    print("-"*62)
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<12}  ERROR: {res['error']}")
            continue
        print(f"{name:<12} {res['best_score']:>10.4f} "
              f"{res['top1_success_rate']:>12.3f} "
              f"{res['spectrum_mae']:>9.4f} "
              f"{res['diversity']:>10.4f} "
              f"{res['inference_time_s']:>8.1f}")
    print("="*62)


if __name__ == "__main__":
    main()
