"""
统一评估脚本：所有方法在同一个物理目标上生成候选结构，输出 results.json。

目标构建方式与 laplas.py 保持一致：
  topk CSV → top-1 数据集样本 → 真实目标光谱 cond_raw

所有方法（topo_opt / CVAE / cGAN / diffusion）拿到完全相同的目标条件，
生成 n_samples 个候选，并对每个候选运行真实 RCWA 全图仿真后比较各指标。

用法：
  cd /data/GraduationProject
  python src/baselines/eval/run_eval.py \
      --data_path      data/train_data.npz \
      --topk_csv       data/second_order_scores/tpp_mag_top5_per_lambda.csv \
      --n_samples      32 \
      --target_lambda  1000.0 \
      --target_rank    1 \
      --rcwa_orders    7 \
      --save_dir       samples/eval_compare
"""

import argparse
import json
import os
import sys
from pathlib import Path

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
from train_utils import resolve_latest_run
from common import (
    lambda_theta_grid,
    rcwa_eval_target_lambda,
    second_order_score_row,
    second_order_target,
)
from infer_all import (
    METHODS,
    load_all_models,
    resolve_default_cgan_ckpt,
    resolve_default_cvae_ckpt,
    resolve_default_diffusion_ckpt,
    timed_generate,
)
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
    p.add_argument("--stats_path",     default=None)
    p.add_argument("--cvae_ckpt",      default=None)
    p.add_argument("--cgan_ckpt",      default=None)
    p.add_argument("--diffusion_ckpt", default=None)
    p.add_argument("--n_samples",      type=int,   default=32,
                   help="每个方法生成的候选数量")
    p.add_argument("--target_lambda",  type=float, default=1000.0)
    p.add_argument("--target_rank",    type=int,   default=1,
                   help="使用数据集中二阶得分第 N 名的样本作为目标模板")
    p.add_argument("--band_sigma_nm",  type=float, default=25.0)
    p.add_argument("--rcwa_orders",    type=int,   default=7)
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


def normalize_target(cond_raw: np.ndarray, cond_mean: np.ndarray, cond_std: np.ndarray) -> torch.Tensor:
    cond_norm_np = (cond_raw - cond_mean.squeeze(0)) / cond_std.squeeze(0)
    return torch.from_numpy(cond_norm_np).unsqueeze(0).float()


def resolve_default_stats_path(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path

    latest_forward_run = resolve_latest_run("runs", "forward")
    if latest_forward_run is not None:
        candidate = Path(latest_forward_run) / "cond_stats.npz"
        if candidate.exists():
            return str(candidate)

    fallback = Path("checkpoints/cond_stats.npz")
    if fallback.exists():
        return str(fallback)
    return "runs/forward_runs/cond_stats.npz"


def load_surrogate(forward_ckpt: str, stats_path: str, device: str):
    ckpt = torch.load(forward_ckpt, map_location=device, weights_only=False)
    cond_ch = ckpt.get("cond_channels", ckpt.get("cond_ch", 2))
    surrogate = ForwardSurrogate(out_ch=cond_ch).to(device)
    surrogate.load_state_dict(ckpt["model"])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    stats = np.load(stats_path)
    cond_mean = stats["mean"].astype(np.float32)
    cond_std = stats["std"].astype(np.float32)
    return surrogate, cond_mean, cond_std


def eval_one_method(
    method_name: str,
    method_info: dict,
    cond_norm: torch.Tensor,   # [1,2,11,17] 归一化目标条件
    cond_raw: np.ndarray,      # [2,11,17]   物理空间目标条件（MAE 用）
    n_samples: int,
    device: str,
    thetas: np.ndarray,
    target_lambda: float,
    rcwa_orders: int,
) -> dict:
    """生成 n_samples 个候选，仅在目标波长做真实 RCWA 角度扫描并打分。"""

    # ── 生成候选 ──────────────────────────────────────────────────────
    structs, infer_time = timed_generate(
        method_info, cond_norm, n_samples, device)   # [N, 64, 64]

    # ── 真实 RCWA 打分 ────────────────────────────────────────────────
    pred_rows = []
    scores = []
    best_score_val = -1.0
    best_pred_cond = None
    best_idx = 0
    target_row = cond_raw[:, int(np.argmin(np.abs(lambda_theta_grid()[0] - float(target_lambda))))]

    for idx, struct in enumerate(structs):
        print(f"[run_eval]   RCWA {idx + 1}/{len(structs)}", flush=True)
        struct_t = torch.from_numpy(struct[None, None]).float().to(device)
        out = rcwa_eval_target_lambda(
            struct_t,
            cond_raw,
            cond_ch=int(cond_raw.shape[0]),
            device=device,
            target_lambda=target_lambda,
            rcwa_orders=rcwa_orders,
        )
        if out is None:
            raise RuntimeError("RCWA backend unavailable during baseline evaluation.")
        _, pr = out
        pred_rows.append(pr.astype(np.float32))
        sc = second_order_score_row(pr[0], thetas)["score"]
        scores.append(sc)
        if sc > best_score_val:
            best_score_val = sc
            best_pred_cond = pr
            best_idx = idx

    pred_raw = np.stack(pred_rows, axis=0)
    if best_pred_cond is None:
        raise RuntimeError("No valid RCWA predictions were produced.")

    target_theta_curve = second_order_target(thetas).astype(np.float32)

    result = summarize(
        candidates     = structs,
        scores         = scores,
        best_pred_cond = best_pred_cond,
        target_cond    = target_row,
        infer_time     = infer_time,
    )
    result["method"] = method_name
    result["best_idx"] = int(best_idx)
    result["target_lambda_nm"] = float(target_lambda)
    result["thetas_deg"] = thetas.astype(np.float32).tolist()
    result["target_theta_curve"] = target_theta_curve.tolist()
    result["best_tpp_theta_curve"] = best_pred_cond[0].astype(np.float32).tolist()
    result["best_tss_theta_curve"] = best_pred_cond[1].astype(np.float32).tolist()
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
    cond_raw, _, _, sample_idx = build_target(
        cond_ch=2,
        target_lambda=args.target_lambda,
        band_sigma_nm=args.band_sigma_nm,
        train_npz_path=Path(args.data_path),
        topk_csv_path=Path(args.topk_csv),
    )   # [2, 11, 17] 物理空间目标

    args.cvae_ckpt = resolve_default_cvae_ckpt(args.cvae_ckpt)
    args.cgan_ckpt = resolve_default_cgan_ckpt(args.cgan_ckpt)
    args.diffusion_ckpt = resolve_default_diffusion_ckpt(args.diffusion_ckpt)
    args.stats_path = resolve_default_stats_path(args.stats_path)

    print(f"[run_eval] target: lambda={args.target_lambda}nm  "
          f"rank={args.target_rank}  dataset_idx={sample_idx}")
    print(f"[run_eval] forward_ckpt={args.forward_ckpt}")
    print(f"[run_eval] stats_path={args.stats_path}")
    print(f"[run_eval] cvae_ckpt={args.cvae_ckpt}")
    print(f"[run_eval] cgan_ckpt={args.cgan_ckpt}")
    print(f"[run_eval] diffusion_ckpt={args.diffusion_ckpt}")
    print(f"[run_eval] rcwa_orders={args.rcwa_orders}")

    # ── 加载生成模型 ──────────────────────────────────────────────────
    all_models = load_all_models(
        cvae_ckpt      = args.cvae_ckpt,
        cgan_ckpt      = args.cgan_ckpt,
        diffusion_ckpt = args.diffusion_ckpt,
        device         = device,
        methods        = methods,
    )

    dataset = np.load(args.data_path)
    cond_stack = np.stack([dataset["tpp_mag"], dataset["tss_mag"]], axis=1).astype(np.float32)
    valid_mask = np.isfinite(cond_stack).all(axis=(1, 2, 3))
    cond_stack = cond_stack[valid_mask]
    default_mean = cond_stack.mean(axis=0, keepdims=True)
    default_std = cond_stack.std(axis=0, keepdims=True) + 1e-6
    guide_surrogate = None
    guide_mean = None
    guide_std = None
    if "diffusion+guide" in methods:
        guide_surrogate, guide_mean, guide_std = load_surrogate(
            args.forward_ckpt,
            args.stats_path,
            device,
        )

    # ── 为不同方法准备各自匹配的目标归一化 ────────────────────────────
    cond_norm_by_method = {}
    for method_name, method_info in all_models.items():
        method_mean = guide_mean if method_name == "diffusion+guide" and guide_mean is not None else method_info.get("cond_mean")
        method_std = guide_std if method_name == "diffusion+guide" and guide_std is not None else method_info.get("cond_std")
        if method_mean is None or method_std is None:
            method_mean = default_mean
            method_std = default_std
        cond_norm_by_method[method_name] = normalize_target(cond_raw, method_mean, method_std)
    if "topo_opt" in all_models:
        all_models["topo_opt"]["target_lambda"] = args.target_lambda
    if "diffusion+guide" in all_models:
        all_models["diffusion+guide"]["surrogate"] = guide_surrogate
        all_models["diffusion+guide"]["target_norm"] = cond_norm_by_method["diffusion+guide"].to(device)

    print(f"[run_eval] methods: {list(all_models.keys())}")

    # ── 每个方法评估 ──────────────────────────────────────────────────
    all_results = {}

    for method_name, method_info in all_models.items():
        print(f"\n[run_eval] → {method_name} ...", end=" ", flush=True)
        try:
            res = eval_one_method(
                method_name   = method_name,
                method_info   = method_info,
                cond_norm     = cond_norm_by_method[method_name],
                cond_raw      = cond_raw,
                n_samples     = args.n_samples,
                device        = device,
                thetas        = thetas,
                target_lambda = args.target_lambda,
                rcwa_orders   = args.rcwa_orders,
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
