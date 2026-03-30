from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset.rcwa.rcwa import torcwa_simulation  # noqa: E402
from infer.common import plot_structure  # noqa: E402
from infer.task_library import TASK_INDEX, original_input_score, task_score_details  # noqa: E402


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def iter_task_case_dirs() -> list[Path]:
    root = ROOT / "samples" / "task_infer"
    if not root.exists():
        return []
    case_dirs = []
    for summary_path in root.glob("**/summary.json"):
        if (summary_path.parent / "task_weight.npy").exists():
            case_dirs.append(summary_path.parent)
    return sorted(case_dirs, key=lambda p: p.stat().st_mtime)


def latest_task_case_dir() -> Path:
    case_dirs = iter_task_case_dirs()
    if not case_dirs:
        raise FileNotFoundError("未找到 task_infer 结果，请先运行 python src/infer/run_diffusion_tasks.py")
    return case_dirs[-1]


def load_case_bundle(case_dir: Path) -> dict:
    with (case_dir / "summary.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    selector = f"{summary['task_key']}:{summary['case_label']}"
    bundle = {
        "case": TASK_INDEX[selector],
        "summary": summary,
        "target_raw": np.load(case_dir / "target_cond_raw.npy").astype(np.float32),
        "weight": np.load(case_dir / "task_weight.npy").astype(np.float32),
        "topk_samples": np.load(case_dir / "topk_samples.npy").astype(np.float32),
        "lambdas": np.load(case_dir / "lambdas.npy").astype(np.float32),
        "thetas": np.load(case_dir / "thetas.npy").astype(np.float32),
    }
    return bundle


def load_init_batch(samples: np.ndarray, device: str, max_inits: int) -> torch.Tensor:
    arr = samples[: max(1, int(max_inits))]
    if arr.ndim == 3:
        arr = arr[:, None]
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


def normalize_rows_torch(spec: torch.Tensor) -> torch.Tensor:
    row_max = spec.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return spec / row_max


def task_guided_loss(case, pred: torch.Tensor, target_raw_t: torch.Tensor, weight_t: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    denom = weight_t.sum().clamp_min(1e-8)
    pred_norm = normalize_rows_torch(pred)
    target_norm = normalize_rows_torch(target_raw_t)
    loss_fit = ((pred - target_raw_t).abs() * weight_t).sum() / denom
    loss_shape = ((pred_norm - target_norm).abs() * weight_t).sum() / denom

    if case.task_key == "p_second_order":
        loss = 0.40 * loss_fit + 0.60 * loss_shape
        return loss, {"fit": loss_fit, "shape": loss_shape}

    if case.task_key == "polarization_independent":
        loss_balance = (pred_norm[0, :, :] - pred_norm[1, :, :]).abs().mean()
        loss = 0.35 * loss_fit + 0.40 * loss_shape + 0.25 * loss_balance
        return loss, {"fit": loss_fit, "shape": loss_shape, "balance": loss_balance}

    if case.task_key == "polarization_multiplexed":
        active_shape = ((pred_norm[0:1] - target_norm[0:1]).abs() * weight_t[0:1]).sum() / weight_t[0:1].sum().clamp_min(1e-8)
        passive_silent = (pred[1:2] * weight_t[1:2]).sum() / weight_t[1:2].sum().clamp_min(1e-8)
        loss = 0.30 * loss_fit + 0.45 * active_shape + 0.25 * passive_silent
        return loss, {"fit": loss_fit, "shape": active_shape, "silent": passive_silent}

    if case.task_key == "fourth_order":
        loss = 0.35 * loss_fit + 0.65 * loss_shape
        return loss, {"fit": loss_fit, "shape": loss_shape}

    if case.task_key == "lowpass":
        loss_center = (pred[:, :, :, pred.shape[-1] // 2] - target_raw_t[:, :, :, target_raw_t.shape[-1] // 2]).abs().mean()
        loss = 0.35 * loss_fit + 0.40 * loss_shape + 0.25 * loss_center
        return loss, {"fit": loss_fit, "shape": loss_shape, "center": loss_center}

    if case.task_key == "st2":
        loss = 0.55 * loss_fit + 0.45 * loss_shape
        return loss, {"fit": loss_fit, "shape": loss_shape}

    return loss_fit, {"fit": loss_fit, "shape": loss_shape}


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


def selected_pairs_from_weight(weight: np.ndarray) -> list[tuple[int, int]]:
    active = np.any(weight > 0, axis=0)
    return [(i, j) for i, j in zip(*np.where(active))]


def simulate_selected_map(
    structure: torch.Tensor,
    cond_ch: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    selected_pairs: list[tuple[int, int]],
    device: str,
    rcwa_orders: int,
) -> torch.Tensor:
    layer = structure.squeeze(0).squeeze(0)
    pred = torch.zeros((cond_ch, len(lambdas), len(thetas)), device=device, dtype=layer.dtype)
    for lam_idx, theta_idx in selected_pairs:
        out = torcwa_simulation(
            rcwa_physics_kwargs(float(lambdas[lam_idx]), float(thetas[theta_idx])),
            layer,
            rcwa_orders=rcwa_orders,
            project=False,
            device=device,
        )
        pred[0, lam_idx, theta_idx] = out["tpp_mag"].real.float()
        if cond_ch > 1:
            pred[1, lam_idx, theta_idx] = out["tss_mag"].real.float()
    return pred


def weighted_mae_numpy(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    denom = float(np.sum(weight))
    if denom <= 1e-8:
        return float(np.mean(np.abs(pred - target)))
    return float(np.sum(np.abs(pred - target) * weight) / denom)


def evaluate_candidate(
    case,
    x_cont: torch.Tensor,
    target_raw_t: torch.Tensor,
    target_raw_np: np.ndarray,
    weight_t: torch.Tensor,
    weight_np: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    selected_pairs: list[tuple[int, int]],
    args,
) -> tuple[torch.Tensor, dict]:
    x_bin = finalize_binary(x_cont)
    pred = simulate_selected_map(x_bin, target_raw_np.shape[0], lambdas, thetas, selected_pairs, args.device, args.rcwa_orders)
    pred_np = pred.detach().cpu().numpy()
    weighted_raw_mae = weighted_mae_numpy(pred_np, target_raw_np, weight_np)
    weighted_norm_mae = weighted_mae_numpy(
        pred_np / np.maximum(np.max(pred_np, axis=-1, keepdims=True), 1e-8),
        target_raw_np / np.maximum(np.max(target_raw_np, axis=-1, keepdims=True), 1e-8),
        weight_np,
    )

    lam_idx = int(np.argmin(np.abs(lambdas - float(args.target_lambda))))
    t40_idx = int(np.argmin(np.abs(thetas - 40.0)))
    metrics = {
        "weighted_raw_mae": float(weighted_raw_mae),
        "weighted_norm_mae": float(weighted_norm_mae),
        "match_score": float(1.0 / (1.0 + weighted_raw_mae)),
        "pred_selected_map": pred_np,
    }
    metrics.update(task_score_details(case, pred_np, lambdas, thetas))
    if lam_idx >= 0:
        metrics["target_tpp_at_40"] = float(target_raw_np[0, lam_idx, t40_idx])
        metrics["pred_tpp_at_40"] = float(pred_np[0, lam_idx, t40_idx])
        if target_raw_np.shape[0] > 1:
            metrics["target_tss_at_40"] = float(target_raw_np[1, lam_idx, t40_idx])
            metrics["pred_tss_at_40"] = float(pred_np[1, lam_idx, t40_idx])
    return x_bin, metrics


def better_metrics(candidate: dict, incumbent: dict | None) -> bool:
    if incumbent is None:
        return True
    cand_score = float(candidate.get("task_score", -float("inf")))
    best_score = float(incumbent.get("task_score", -float("inf")))
    if cand_score > best_score + 1e-8:
        return True
    if cand_score < best_score - 1e-8:
        return False
    return float(candidate.get("weighted_raw_mae", float("inf"))) < float(incumbent.get("weighted_raw_mae", float("inf")))


def beta_for_step(step: int, total_steps: int, beta_start: float, beta_end: float) -> float:
    if total_steps <= 1:
        return beta_end
    alpha = step / float(total_steps - 1)
    return float(beta_start + alpha * (beta_end - beta_start))


def optimize_one(init: torch.Tensor, bundle: dict, args, candidate_idx: int) -> tuple[torch.Tensor, torch.Tensor, dict, list[dict], float]:
    target_raw_np = bundle["target_raw"]
    target_raw_t = torch.from_numpy(target_raw_np).to(args.device)
    weight_np = bundle["weight"]
    weight_t = torch.from_numpy(weight_np).to(args.device)
    selected_pairs = selected_pairs_from_weight(weight_np)

    rho_param = init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([rho_param], lr=args.lr)
    hist: list[dict] = []
    best_loss = float("inf")
    best_task_score = -float("inf")
    best_metrics: dict | None = None
    best_state: tuple[torch.Tensor, torch.Tensor, dict] | None = None
    t_start = time.perf_counter()

    for step in range(args.steps):
        beta = beta_for_step(step, args.steps, args.beta_start, args.beta_end)
        rho = symmetrize(rho_param).clamp(0.0, 1.0)
        rho_f = density_filter(rho, args.filter_radius)
        x = project_density(rho_f, beta=beta, eta=args.proj_eta)

        pred = simulate_selected_map(x, target_raw_np.shape[0], bundle["lambdas"], bundle["thetas"], selected_pairs, args.device, args.rcwa_orders)
        loss_task, task_terms = task_guided_loss(bundle["case"], pred, target_raw_t, weight_t)
        loss_bin = (x * (1.0 - x)).mean()
        loss_tv = tv_loss(rho_f)
        loss = loss_task + 0.05 * loss_bin + 0.02 * loss_tv

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            rho_param.clamp_(0.0, 1.0)

        item = {
            "step": int(step),
            "beta": float(beta),
            "loss": float(loss.item()),
            "fit": float(task_terms["fit"].item()),
            "shape": float(task_terms["shape"].item()),
            "bin": float(loss_bin.item()),
            "tv": float(loss_tv.item()),
        }
        for name, value in task_terms.items():
            if name not in {"fit", "shape"}:
                item[name] = float(value.item())

        need_eval = item["loss"] < best_loss or step % args.log_every == 0 or step + 1 == args.steps
        if item["loss"] < best_loss:
            best_loss = item["loss"]
        if need_eval:
            x_bin, metrics = evaluate_candidate(bundle["case"], x.detach(), target_raw_t, target_raw_np, weight_t, weight_np, bundle["lambdas"], bundle["thetas"], selected_pairs, args)
            item["best_weighted_raw_mae"] = float(metrics["weighted_raw_mae"])
            item["best_task_score"] = float(metrics.get("task_score", np.nan))
            if better_metrics(metrics, best_metrics):
                best_metrics = metrics
                best_task_score = float(metrics.get("task_score", best_task_score))
                best_state = (x.detach().clone(), x_bin.detach().clone(), metrics)

        hist.append(item)
        if step % args.log_every == 0 or step + 1 == args.steps:
            elapsed = time.perf_counter() - t_start
            print(
                f"[task_opt {candidate_idx:02d}] step={step:03d} beta={beta:.1f} loss={item['loss']:.4f} "
                f"fit={item['fit']:.4f} shape={item['shape']:.4f} "
                f"{' '.join(f'{k}={item[k]:.4f}' for k in item if k not in {'step','beta','loss','fit','shape','bin','tv','best_weighted_raw_mae','best_task_score'})} "
                f"bin={item['bin']:.4f} tv={item['tv']:.4f} "
                f"best_task={best_task_score:.4f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("Optimization did not produce a valid state.")
    return best_state[0], best_state[1], best_state[2], hist, time.perf_counter() - t_start


def run_candidate(idx: int, init: torch.Tensor, bundle: dict, args, save_dir: Path) -> dict:
    cand_dir = save_dir / f"candidate_{idx:02d}"
    cand_dir.mkdir(parents=True, exist_ok=True)

    best_cont, best_bin, metrics, hist, elapsed = optimize_one(init, bundle, args, idx)
    np.save(cand_dir / "optimized_continuous.npy", best_cont.cpu().numpy())
    np.save(cand_dir / "optimized_binary.npy", best_bin.cpu().numpy())
    np.save(cand_dir / "optimized_selected_rcwa.npy", metrics["pred_selected_map"])
    plot_structure(cand_dir / "optimized_continuous.png", best_cont.cpu().numpy(), f"optimized continuous {idx:02d}")
    plot_structure(cand_dir / "optimized_binary.png", best_bin.cpu().numpy(), f"optimized binary {idx:02d}")

    out = {
        "candidate_idx": int(idx),
        "optimization_time_sec": float(elapsed),
        "task_score": float(metrics.get("task_score", np.nan)),
        "weighted_raw_mae": float(metrics["weighted_raw_mae"]),
        "weighted_norm_mae": float(metrics["weighted_norm_mae"]),
        "match_score": float(metrics["match_score"]),
    }
    for key, value in metrics.items():
        if key in {"pred_selected_map"}:
            continue
        if key not in out and isinstance(value, (float, int, np.floating)):
            out[key] = float(value)
    for key in ("target_tpp_at_40", "pred_tpp_at_40", "target_tss_at_40", "pred_tss_at_40"):
        if key in metrics:
            out[key] = float(metrics[key])

    with (cand_dir / "optimization_log.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_summary": bundle["summary"],
                "candidate_idx": idx,
                "metrics": out,
                "history_tail": hist[-40:],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RCWA topology optimization for diffusion task cases.")
    parser.add_argument("--case_dir", default=None, help="task_infer 的具体 case 目录；不传则自动选最近一次")
    parser.add_argument("--save_dir", default=None, help="默认保存在 case_dir 下的 optimization_TIMESTAMP")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--rcwa_orders", type=int, default=7)
    parser.add_argument("--max_inits", type=int, default=3)
    parser.add_argument("--filter_radius", type=int, default=1)
    parser.add_argument("--proj_eta", type=float, default=0.5)
    parser.add_argument("--beta_start", type=float, default=4.0)
    parser.add_argument("--beta_end", type=float, default=16.0)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    case_dir = latest_task_case_dir() if args.case_dir is None else resolve_from_root(args.case_dir)
    bundle = load_case_bundle(case_dir)
    args.target_lambda = float(bundle["summary"]["target_lambda_nm"])

    init_batch = load_init_batch(bundle["topk_samples"], args.device, args.max_inits)
    save_root = resolve_from_root(args.save_dir) if args.save_dir else case_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_root.mkdir(parents=True, exist_ok=True)

    rows = []
    total_start = time.perf_counter()
    input_score = original_input_score(bundle["case"], bundle["target_raw"], bundle["lambdas"], bundle["thetas"])
    for idx in range(init_batch.shape[0]):
        print(f"[task_opt] candidate {idx + 1}/{init_batch.shape[0]}", flush=True)
        rows.append(run_candidate(idx, init_batch[idx: idx + 1], bundle, args, save_root))

    rows = sorted(rows, key=lambda x: (float(x.get("task_score", -float("inf"))), -float(x.get("weighted_raw_mae", float("inf")))), reverse=True)
    with (save_root / "optimization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_dir": str(case_dir),
                "case_summary": bundle["summary"],
                "original_input_score": input_score,
                "candidates": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"saved_to: {save_root}")
    print(f"total_time_sec: {time.perf_counter() - total_start:.2f}")
    print(
        "ranking:",
        [
            {
                "candidate_idx": row["candidate_idx"],
                "task_score": row["task_score"],
                "weighted_raw_mae": row["weighted_raw_mae"],
                "match_score": row["match_score"],
            }
            for row in rows
        ],
    )


if __name__ == "__main__":
    main()
