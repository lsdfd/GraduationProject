# -*- coding: utf-8 -*-
"""Multi-start RCWA topology optimization with density filtering and continuation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset.rcwa.rcwa import torcwa_simulation  # noqa: E402
from infer.common import (  # noqa: E402
    lambda_theta_grid,
    plot_structure,
    second_order_score_row,
    second_order_score_row_torch,
    second_order_target,
)


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def iter_laplas_case_dirs() -> list[Path]:
    samples_root = ROOT / "samples"
    if not samples_root.exists():
        return []
    roots = []
    for child in samples_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name == "laplas" or (name.startswith("laplas_") and name != "laplas2"):
            roots.append(child)

    case_dirs = []
    for root in roots:
        for summary in root.glob("**/summary.json"):
            case_dir = summary.parent
            if (case_dir / "target_cond_raw.npy").exists():
                case_dirs.append(case_dir)
    return sorted(case_dirs, key=lambda p: p.stat().st_mtime)


def latest_laplas_case_dir() -> Path:
    case_dirs = iter_laplas_case_dirs()
    if not case_dirs:
        raise FileNotFoundError("未找到有效的 laplas 推理结果，请先运行 python src/infer/laplas.py")
    return case_dirs[-1]


def load_laplas_summary(case_dir: Path) -> dict:
    summary_path = case_dir / "summary.json"
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_laplas_artifacts(case_dir: Path) -> tuple[Path, Path, dict]:
    target_path = case_dir / "target_cond_raw.npy"
    topk_second = case_dir / "topk_second_samples.npy"
    topk = case_dir / "topk_samples.npy"
    if topk_second.exists():
        init_path = topk_second
    elif topk.exists():
        init_path = topk
    else:
        raise FileNotFoundError(f"未找到初始化样本文件: {case_dir}")
    return target_path, init_path, load_laplas_summary(case_dir)


def load_target_raw(path: str) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    return x[None] if x.ndim == 3 else x


def load_init_batch(path: str, device: str, max_inits: int) -> torch.Tensor:
    x = np.load(path).astype(np.float32)
    if x.ndim == 4:
        arr = x
    elif x.ndim == 3:
        arr = x[:, None, :, :]
    elif x.ndim == 2:
        arr = x[None, None, :, :]
    else:
        raise ValueError(f"Unsupported init shape: {x.shape}")
    arr = arr[: max(1, int(max_inits))]
    return torch.from_numpy(arr).to(device).clamp(0.0, 1.0)


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    rots = [torch.rot90(x, k, (-2, -1)) for k in range(4)]
    flips = [t.flip(-2) for t in rots]
    return sum(rots + flips) / 8.0


def density_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    kernel = 2 * radius + 1
    return F.avg_pool2d(x, kernel, stride=1, padding=radius)


def project_density(rho: torch.Tensor, beta: float, eta: float = 0.5) -> torch.Tensor:
    num = torch.tanh(torch.tensor(beta * eta, device=rho.device, dtype=rho.dtype)) + torch.tanh(beta * (rho - eta))
    den = torch.tanh(torch.tensor(beta * eta, device=rho.device, dtype=rho.dtype)) + torch.tanh(
        torch.tensor(beta * (1.0 - eta), device=rho.device, dtype=rho.dtype)
    )
    return num / den.clamp_min(1e-8)


def finalize_binary(x: torch.Tensor) -> torch.Tensor:
    return (symmetrize(x) > 0.5).float()


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def theta_grid() -> np.ndarray:
    return np.arange(-40.0, 40.1, 5.0, dtype=np.float32)


def target_row_tensor(device: str) -> torch.Tensor:
    return torch.as_tensor(second_order_target(theta_grid()), device=device, dtype=torch.float32).unsqueeze(0)


def outer_monotonic_penalty(y_norm: torch.Tensor, thetas: np.ndarray) -> torch.Tensor:
    if y_norm.ndim == 1:
        y_norm = y_norm.unsqueeze(0)
    abs_thetas = np.abs(np.asarray(thetas, dtype=np.float64))
    idx30 = int(np.argmin(np.abs(abs_thetas - 30.0)))
    idx35 = int(np.argmin(np.abs(abs_thetas - 35.0)))
    idx40 = int(np.argmin(np.abs(abs_thetas - 40.0)))
    p1 = F.relu(y_norm[:, idx30] - y_norm[:, idx35])
    p2 = F.relu(y_norm[:, idx35] - y_norm[:, idx40])
    return (p1 + p2).mean()


def rcwa_physics_kwargs(target_lambda: float, theta: float) -> dict:
    return {
        "periodicity": 500.0,
        "h": 500.0,
        "lam": float(target_lambda),
        "tet": float(theta),
        "phi": 0.0,
        "angle_unit": "deg",
        "angle_layer": "input",
        "input_medium": "air",
        "output_medium": "SiO2",
        "structure": "Si",
    }


def rcwa_tpp_tss_row(x: torch.Tensor, target_lambda: float, device: str, rcwa_orders: int) -> tuple[torch.Tensor, torch.Tensor]:
    thetas = theta_grid()
    layer = x.squeeze(0).squeeze(0)
    tpp_vals = []
    tss_vals = []
    for theta in thetas:
        out = torcwa_simulation(
            rcwa_physics_kwargs(target_lambda, float(theta)),
            layer,
            rcwa_orders=rcwa_orders,
            project=False,
            device=device,
        )
        tpp_vals.append(out["tpp_mag"].real.float())
        tss_vals.append(out["tss_mag"].real.float())
    return torch.stack(tpp_vals, dim=0), torch.stack(tss_vals, dim=0)


def plot_tpp_row(path: Path, row: np.ndarray, thetas: np.ndarray, title: str) -> None:
    target = second_order_target(thetas)
    y = row.astype(np.float64)
    yn = y / max(float(np.max(y)), 1e-8)
    plt.figure(figsize=(5, 3.6))
    plt.plot(thetas, target, "k--", lw=1.7, label="target ~ |sin(theta)|^2")
    plt.plot(thetas, yn, lw=1.9, color="#1f77b4", label="candidate (normalized)")
    plt.xlabel("theta (deg)")
    plt.ylabel("normalized |tpp|")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="lower right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def evaluate_binary_candidate(x_cont: torch.Tensor, args) -> tuple[torch.Tensor, dict]:
    thetas = theta_grid()
    x_bin = finalize_binary(x_cont)
    tpp_row, tss_row = rcwa_tpp_tss_row(x_bin, args.target_lambda, args.device, args.rcwa_orders)
    tpp_np = tpp_row.detach().cpu().numpy()
    tss_np = tss_row.detach().cpu().numpy()
    score = second_order_score_row(tpp_np, thetas)
    return x_bin, {
        "tpp_row": tpp_np,
        "tss_row": tss_np,
        "score": float(score["score"]),
        "center": float(score["center"]),
        "shape": float(score["shape"]),
        "edge": float(score["edge"]),
        "outer": float(score["outer"]),
        "r2": float(score["r2"]),
    }


def beta_for_step(step: int, total_steps: int, beta_start: float, beta_end: float) -> float:
    if total_steps <= 1:
        return beta_end
    alpha = step / float(total_steps - 1)
    return float(beta_start + alpha * (beta_end - beta_start))


def optimize_one(init: torch.Tensor, args, candidate_idx: int) -> tuple[torch.Tensor, torch.Tensor, dict, dict, list[dict], float]:
    thetas = theta_grid()
    target_row = target_row_tensor(args.device)
    rho_param = init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([rho_param], lr=args.lr)
    hist: list[dict] = []
    best_loss = float("inf")
    best_cont_state: tuple[torch.Tensor, dict] | None = None
    best_bin_score = -float("inf")
    best_bin_state: tuple[torch.Tensor, torch.Tensor, dict] | None = None
    t_start = time.perf_counter()

    raw_init_bin, raw_init_metrics = evaluate_binary_candidate(init.detach(), args)
    best_bin_score = raw_init_metrics["score"]
    best_bin_state = (init.detach().clone(), raw_init_bin.detach().clone(), raw_init_metrics)
    hist.append(
        {
            "step": -1,
            "raw_init_score": float(raw_init_metrics["score"]),
            "raw_init_center": float(raw_init_metrics["center"]),
            "raw_init_shape": float(raw_init_metrics["shape"]),
            "raw_init_edge": float(raw_init_metrics["edge"]),
            "raw_init_outer": float(raw_init_metrics["outer"]),
        }
    )
    print(
        f"[opt {candidate_idx:02d}] raw_init binchk={raw_init_metrics['score']:.4f} "
        f"center={raw_init_metrics['center']:.4f} shape={raw_init_metrics['shape']:.4f} "
        f"edge={raw_init_metrics['edge']:.4f} outer={raw_init_metrics['outer']:.4f}",
        flush=True,
    )

    for step in range(args.steps):
        beta = beta_for_step(step, args.steps, args.beta_start, args.beta_end)
        rho = symmetrize(rho_param).clamp(0.0, 1.0)
        rho_f = density_filter(rho, args.filter_radius)
        x = project_density(rho_f, beta=beta, eta=args.proj_eta)

        tpp_row, tss_row = rcwa_tpp_tss_row(x, args.target_lambda, args.device, args.rcwa_orders)
        main_pack = second_order_score_row_torch(tpp_row, thetas)
        row_max = tpp_row.amax(dim=-1, keepdim=True).clamp_min(1e-8)
        y_norm = tpp_row / row_max
        loss_fit = 1.0 - main_pack["score"].mean()
        loss_row = (y_norm - target_row).abs().mean()
        loss_outer = outer_monotonic_penalty(y_norm, thetas)
        loss_bin = (x * (1.0 - x)).mean()
        loss_tv = tv_loss(rho_f)
        loss = loss_fit + 0.10 * loss_row + 0.10 * loss_outer + 0.06 * loss_bin + 0.02 * loss_tv

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            rho_param.clamp_(0.0, 1.0)

        item = {
            "step": step,
            "beta": beta,
            "loss": float(loss.item()),
            "fit": float(loss_fit.item()),
            "center": float(main_pack["center"].mean().item()),
            "shape": float(main_pack["shape"].mean().item()),
            "edge": float(main_pack["edge"].mean().item()),
            "outer": float(main_pack["outer"].mean().item()),
            "row": float(loss_row.item()),
            "mono": float(loss_outer.item()),
            "bin": float(loss_bin.item()),
            "tv": float(loss_tv.item()),
        }
        hist.append(item)

        if item["loss"] < best_loss:
            best_loss = item["loss"]
            best_cont_state = (
                x.detach().clone(),
                {
                    "tpp_row": tpp_row.detach().cpu().numpy(),
                    "tss_row": tss_row.detach().cpu().numpy(),
                    "score": float(main_pack["score"].mean().item()),
                    "center": float(main_pack["center"].mean().item()),
                    "shape": float(main_pack["shape"].mean().item()),
                    "edge": float(main_pack["edge"].mean().item()),
                    "outer": float(main_pack["outer"].mean().item()),
                    "r2": float(main_pack["r2"].mean().item()),
                },
            )

        should_eval_bin = step == 0 or (step + 1) % args.binary_eval_every == 0 or step + 1 == args.steps
        if should_eval_bin:
            x_bin, bin_metrics = evaluate_binary_candidate(x.detach(), args)
            item["bin_eval_score"] = float(bin_metrics["score"])
            item["bin_eval_center"] = float(bin_metrics["center"])
            item["bin_eval_shape"] = float(bin_metrics["shape"])
            item["bin_eval_edge"] = float(bin_metrics["edge"])
            item["bin_eval_outer"] = float(bin_metrics["outer"])
            if bin_metrics["score"] > best_bin_score:
                best_bin_score = bin_metrics["score"]
                best_bin_state = (x.detach().clone(), x_bin.detach().clone(), bin_metrics)

        if step % args.log_every == 0 or step + 1 == args.steps:
            elapsed = time.perf_counter() - t_start
            step_time = elapsed / max(step + 1, 1)
            eta = max(args.steps - step - 1, 0) * step_time
            extra = ""
            if "bin_eval_score" in item:
                extra = f" binchk={item['bin_eval_score']:.4f}"
            print(
                f"[opt {candidate_idx:02d}] step={step:03d} beta={beta:.1f} loss={item['loss']:.4f} fit={item['fit']:.4f} "
                f"center={item['center']:.4f} shape={item['shape']:.4f} edge={item['edge']:.4f} outer={item['outer']:.4f} "
                f"row={item['row']:.4f} mono={item['mono']:.4f} bin={item['bin']:.4f} tv={item['tv']:.4f}"
                f"{extra} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    if best_cont_state is None or best_bin_state is None:
        raise RuntimeError("Optimization did not produce any valid state.")

    return best_cont_state[0], best_bin_state[1], best_cont_state[1], best_bin_state[2], hist, time.perf_counter() - t_start


def run_candidate(idx: int, init: torch.Tensor, target_raw: np.ndarray, args, save_dir: Path) -> dict:
    thetas = theta_grid()
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    cand_dir = save_dir / f"candidate_{idx:02d}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    best_x_cont, best_bin, best_rcwa_cont, best_rcwa_bin, hist, opt_elapsed = optimize_one(init, args, idx)
    bin_tpp_np = best_rcwa_bin["tpp_row"]
    bin_tss_np = best_rcwa_bin["tss_row"]

    lambdas, _ = lambda_theta_grid()
    lam_idx = int(np.argmin(np.abs(lambdas - float(args.target_lambda))))
    target_row = target_raw[0, 0, lam_idx]
    out = {
        "candidate_idx": idx,
        "target_lambda_nm": float(args.target_lambda),
        "optimization_time_sec": float(opt_elapsed),
        "rcwa_second_order_score": float(best_rcwa_bin["score"]),
        "rcwa_center_score": float(best_rcwa_bin["center"]),
        "rcwa_shape_score": float(best_rcwa_bin["shape"]),
        "rcwa_edge_score": float(best_rcwa_bin["edge"]),
        "rcwa_outer_score": float(best_rcwa_bin["outer"]),
        "rcwa_r2": float(best_rcwa_bin["r2"]),
        "rcwa_tpp_at_40": float(bin_tpp_np[t40_idx]),
        "target_tpp_at_40": float(target_row[t40_idx]),
        "best_continuous_score": float(best_rcwa_cont["score"]),
        "best_continuous_center": float(best_rcwa_cont["center"]),
        "best_continuous_shape": float(best_rcwa_cont["shape"]),
        "best_continuous_edge": float(best_rcwa_cont["edge"]),
        "best_continuous_outer": float(best_rcwa_cont["outer"]),
    }

    np.save(cand_dir / "optimized_continuous.npy", best_x_cont.cpu().numpy())
    np.save(cand_dir / "optimized_binary.npy", best_bin.cpu().numpy())
    np.save(cand_dir / "optimized_rcwa_tpp_row.npy", bin_tpp_np)
    np.save(cand_dir / "optimized_rcwa_tss_row.npy", bin_tss_np)
    np.save(cand_dir / "optimized_rcwa_continuous_tpp_row.npy", best_rcwa_cont["tpp_row"])
    np.save(cand_dir / "optimized_rcwa_continuous_tss_row.npy", best_rcwa_cont["tss_row"])
    np.save(cand_dir / "target_cond_raw.npy", target_raw)
    plot_structure(cand_dir / "optimized_continuous.png", best_x_cont.cpu().numpy(), f"Optimized continuous {idx:02d}")
    plot_structure(cand_dir / "optimized_binary.png", best_bin.cpu().numpy(), f"Optimized binary {idx:02d}")
    plot_tpp_row(cand_dir / "optimized_second_order_curve.png", bin_tpp_np, thetas, f"{args.target_lambda:.0f}nm fit {idx:02d}")
    with (cand_dir / "optimization_log.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "init": args.init,
                "laplas_case_dir": args.laplas_case_dir,
                "laplas_summary": args.laplas_summary,
                "candidate_idx": idx,
                "metrics": out,
                "history_tail": hist[-40:],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out


def main():
    p = argparse.ArgumentParser(description="Multi-start RCWA topology optimization from laplas top-k samples.")
    p.add_argument("--laplas_dir", help="指定某次 laplas 输出的 case 目录；不传则自动选最新有效结果")
    p.add_argument("--target")
    p.add_argument("--init")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "optimized"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target_lambda", type=float, default=None)
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--binary_eval_every", type=int, default=10)
    p.add_argument("--max_inits", type=int, default=5)
    p.add_argument("--filter_radius", type=int, default=1)
    p.add_argument("--proj_eta", type=float, default=0.5)
    p.add_argument("--beta_start", type=float, default=4.0)
    p.add_argument("--beta_end", type=float, default=16.0)
    p.add_argument("--log_every", type=int, default=10)
    args = p.parse_args()

    laplas_summary: dict = {}
    if args.laplas_dir:
        laplas_case_dir = resolve_from_root(args.laplas_dir)
    elif args.target or args.init:
        laplas_case_dir = None
    else:
        laplas_case_dir = latest_laplas_case_dir()

    if laplas_case_dir is not None:
        auto_target, auto_init, laplas_summary = resolve_laplas_artifacts(laplas_case_dir)
        args.target = args.target or str(auto_target)
        args.init = args.init or str(auto_init)

    if args.target:
        args.target = str(resolve_from_root(args.target))
    if args.init:
        args.init = str(resolve_from_root(args.init))
    args.save_dir = str(resolve_from_root(args.save_dir))
    args.laplas_case_dir = None if laplas_case_dir is None else str(laplas_case_dir)
    args.laplas_summary = laplas_summary

    if args.target_lambda is None:
        inferred_lambda = laplas_summary.get("target_lambda_nm")
        args.target_lambda = float(inferred_lambda) if inferred_lambda is not None else 1000.0

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    target_raw = load_target_raw(args.target)
    init_batch = load_init_batch(args.init, args.device, args.max_inits)

    rows = []
    total_start = time.perf_counter()
    for idx in range(init_batch.shape[0]):
        print(f"[opt] candidate {idx + 1}/{init_batch.shape[0]}", flush=True)
        rows.append(run_candidate(idx, init_batch[idx: idx + 1], target_raw, args, save_dir))

    rows = sorted(rows, key=lambda x: x["rcwa_second_order_score"], reverse=True)
    with (save_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "laplas_case_dir": args.laplas_case_dir,
                "laplas_summary": args.laplas_summary,
                "target": args.target,
                "init": args.init,
                "target_lambda_nm": float(args.target_lambda),
                "candidates": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("saved_to:", save_dir)
    print("laplas_case_dir:", args.laplas_case_dir)
    print("target_from:", args.target)
    print("init_from:", args.init)
    print("target_lambda_nm:", args.target_lambda)
    print(f"total_time_sec: {time.perf_counter() - total_start:.2f}")
    print(
        "ranking:",
        [
            {
                "candidate_idx": r["candidate_idx"],
                "score": r["rcwa_second_order_score"],
                "tpp_at_40": r["rcwa_tpp_at_40"],
            }
            for r in rows
        ],
    )


if __name__ == "__main__":
    main()
