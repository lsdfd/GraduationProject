# -*- coding: utf-8 -*-
"""Multi-start RCWA topology optimization with density filtering and continuation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dataset.rcwa.rcwa import torcwa_simulation  # noqa: E402
except ModuleNotFoundError:
    # Avoid collisions with unrelated top-level `dataset` modules.
    sys.path.insert(0, str(ROOT / "src" / "dataset" / "rcwa"))
    from rcwa import torcwa_simulation  # type: ignore  # noqa: E402
from infer.common import (  # noqa: E402
    lambda_theta_grid,
    plot_structure,
    second_order_score_row,
    second_order_score_row_torch,
    second_order_target,
)


def parse_devices(devices_arg: str | None, device_arg: str | None) -> list[str]:
    if devices_arg:
        devices = [d.strip() for d in devices_arg.split(",") if d.strip()]
        if not devices:
            raise ValueError("--devices 为空，请传入类似 cuda:0,cuda:1")
        return devices
    if device_arg:
        return [device_arg]
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]


def resolve_from_root(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else ROOT / path


def latest_laplas_file(name: str) -> str:
    root = ROOT / "samples" / "laplas"
    files = sorted([p for p in root.glob(f"**/{name}") if p.is_file()], key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("未找到 samples/laplas 下的推理结果，请先运行 python src/infer/laplas.py")
    return str(files[-1])


def latest_laplas_file_from_dirs(search_roots: list[Path], name: str) -> str:
    candidates = []
    for root in search_roots:
        if root.exists():
            candidates.extend([p for p in root.glob(f"**/{name}") if p.is_file()])
    if not candidates:
        roots_text = ", ".join(str(p) for p in search_roots)
        raise FileNotFoundError(f"未找到 {name}，搜索目录: {roots_text}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


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
    return np.arange(-60.0, 60.1, 10.0, dtype=np.float32)


def target_row_tensor(device: str) -> torch.Tensor:
    return torch.as_tensor(second_order_target(theta_grid()), device=device, dtype=torch.float32).unsqueeze(0)


def outer_monotonic_penalty(y_norm: torch.Tensor, thetas: np.ndarray) -> torch.Tensor:
    if y_norm.ndim == 1:
        y_norm = y_norm.unsqueeze(0)
    abs_thetas = np.abs(np.asarray(thetas, dtype=np.float64))
    uniq = np.unique(abs_thetas)
    if len(uniq) < 3:
        return torch.zeros((), device=y_norm.device, dtype=y_norm.dtype)
    vals = uniq[-3:]
    idx_lo = int(np.argmin(np.abs(abs_thetas - vals[0])))
    idx_mid = int(np.argmin(np.abs(abs_thetas - vals[1])))
    idx_hi = int(np.argmin(np.abs(abs_thetas - vals[2])))
    p1 = F.relu(y_norm[:, idx_lo] - y_norm[:, idx_mid])
    p2 = F.relu(y_norm[:, idx_mid] - y_norm[:, idx_hi])
    return (p1 + p2).mean()


def main_objective_loss(
    objective_mode: str,
    y_norm: torch.Tensor,
    target_row: torch.Tensor,
    main_pack: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, str]:
    if objective_mode == "pointwise":
        return (y_norm - target_row).abs().mean(), "point"
    if objective_mode == "pointwise_l2":
        return ((y_norm - target_row) ** 2).mean(), "point"
    return 1.0 - main_pack["score"].mean(), "fit"


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


def plot_candidate_summary(
    path: Path,
    x_bin: np.ndarray,
    tpp_row: np.ndarray,
    thetas: np.ndarray,
    title: str,
    score: float,
    tpp_at_edge: float,
) -> None:
    """合并结构图和 tpp 曲线到一张图，并标注最大角度处的透过率"""
    target = second_order_target(thetas)
    y = tpp_row.astype(np.float64)
    yn = y / max(float(np.max(y)), 1e-8)
    edge_idx = int(np.argmax(np.abs(thetas)))

    fig, (ax_struct, ax_curve) = plt.subplots(1, 2, figsize=(10, 4))

    # 左：二值化结构
    ax_struct.imshow(x_bin.squeeze(), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax_struct.set_title("Binary structure", fontsize=10)
    ax_struct.axis("off")

    # 右：tpp 曲线
    ax_curve.plot(thetas, target, "k--", lw=1.7, label="ideal ~ |sin(θ)|²")
    ax_curve.plot(thetas, yn, lw=1.9, color="#1f77b4", label="optimized (normalized)")

    # 标注最大角度处的透过率
    ax_curve.axvline(x=float(thetas[edge_idx]), color="r", ls=":", lw=1.2, alpha=0.6)
    ax_curve.annotate(
        f"|tpp|@{float(thetas[edge_idx]):.0f}°={tpp_at_edge:.3f}",
        xy=(float(thetas[edge_idx]), float(yn[edge_idx])),
        xytext=(float(thetas[edge_idx]) - 16, float(yn[edge_idx]) + 0.15),
        fontsize=9,
        color="red",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "red", "alpha": 0.8},
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.0},
    )

    ax_curve.set_xlabel("theta (deg)", fontsize=9)
    ax_curve.set_ylabel("normalized |tpp|", fontsize=9)
    ax_curve.set_ylim(-0.05, 1.15)
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=8, loc="lower right")
    ax_curve.set_title(f"2nd-order score={score:.3f}", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


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
        loss_main, main_label = main_objective_loss(args.objective_mode, y_norm, target_row, main_pack)
        loss_row = (y_norm - target_row).abs().mean()
        loss_outer = outer_monotonic_penalty(y_norm, thetas)
        loss_bin = (x * (1.0 - x)).mean()
        loss_tv = tv_loss(rho_f)
        loss = loss_main + 0.10 * loss_row + 0.10 * loss_outer + 0.06 * loss_bin + 0.02 * loss_tv

        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            rho_param.clamp_(0.0, 1.0)

        item = {
            "step": step,
            "beta": beta,
            "loss": float(loss.item()),
            main_label: float(loss_main.item()),
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
                f"[opt {candidate_idx:02d}] step={step:03d} beta={beta:.1f} loss={item['loss']:.4f} {main_label}={item[main_label]:.4f} "
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
    edge_idx = int(np.argmax(np.abs(thetas)))
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
        "rcwa_tpp_at_edge": float(bin_tpp_np[edge_idx]),
        "target_tpp_at_edge": float(target_row[edge_idx]),
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
    plot_candidate_summary(
        cand_dir / "optimized_summary.png",
        best_bin.cpu().numpy(),
        bin_tpp_np,
        thetas,
        f"Candidate {idx:02d} @ {args.target_lambda:.0f}nm",
        float(best_rcwa_bin["score"]),
        float(bin_tpp_np[edge_idx]),
    )
    with (cand_dir / "optimization_log.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target": args.target,
                "init": args.init,
                "candidate_idx": idx,
                "metrics": out,
                "history_tail": hist[-40:],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out


def _optimization_worker(indices, init_np, target_raw, args_dict, save_dir_str, device, queue):
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(device)
        torch.set_num_threads(1)
        args = SimpleNamespace(**args_dict)
        args.device = device
        save_dir = Path(save_dir_str)
        rows = []
        for idx in indices:
            print(f"[opt {device}] candidate {idx + 1}/{len(init_np)}", flush=True)
            init_t = torch.from_numpy(init_np[idx: idx + 1]).to(device)
            rows.append(run_candidate(idx, init_t, target_raw, args, save_dir))
        queue.put({"ok": True, "device": device, "rows": rows})
    except Exception as exc:
        queue.put({"ok": False, "device": device, "error": str(exc)})


def _run_with_args(args) -> None:
    """Core execution logic; accepts a pre-parsed args namespace.

    Called both by main() (direct execution) and by band-specific wrapper
    scripts (e.g. band_900nm/optimization.py) that only override default values.
    """
    if args.target:
        args.target = str(resolve_from_root(args.target))
    if args.init:
        args.init = str(resolve_from_root(args.init))
    args.save_dir = str(resolve_from_root(args.save_dir))
    devices = parse_devices(args.devices, args.device)
    args.device = devices[0]

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir.mkdir(parents=True, exist_ok=True)
    if not args.target:
        search_roots = [ROOT / "samples" / "laplas"]
        band_dir = getattr(args, "laplas_dir", None)
        if band_dir:
            search_roots.insert(0, resolve_from_root(band_dir))
        args.target = latest_laplas_file_from_dirs(search_roots, "target_cond_raw.npy")
    if not args.init:
        search_roots = [ROOT / "samples" / "laplas"]
        band_dir = getattr(args, "laplas_dir", None)
        if band_dir:
            search_roots.insert(0, resolve_from_root(band_dir))
        try:
            args.init = latest_laplas_file_from_dirs(search_roots, "topk_second_samples.npy")
        except FileNotFoundError:
            args.init = latest_laplas_file_from_dirs(search_roots, "topk_samples.npy")

    target_raw = load_target_raw(args.target)
    init_batch = load_init_batch(args.init, "cpu", args.max_inits)

    rows = []
    total_start = time.perf_counter()
    if len(devices) == 1:
        init_batch = init_batch.to(args.device)
        for idx in range(init_batch.shape[0]):
            print(f"[opt] candidate {idx + 1}/{init_batch.shape[0]}", flush=True)
            rows.append(run_candidate(idx, init_batch[idx: idx + 1], target_raw, args, save_dir))
    else:
        all_indices = np.arange(init_batch.shape[0], dtype=np.int64)
        split_indices = [chunk.tolist() for chunk in np.array_split(all_indices, len(devices)) if len(chunk) > 0]
        active_devices = devices[: len(split_indices)]
        ctx = get_context("spawn")
        queue = ctx.Queue()
        procs = []
        args_dict = vars(args).copy()
        init_np = init_batch.cpu().numpy().astype(np.float32)
        for dev, idxs in zip(active_devices, split_indices):
            proc = ctx.Process(
                target=_optimization_worker,
                args=(idxs, init_np, target_raw, args_dict, str(save_dir), dev, queue),
            )
            proc.start()
            procs.append(proc)

        received = 0
        while received < len(procs):
            msg = queue.get()
            received += 1
            if not msg.get("ok", False):
                for proc in procs:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(f"optimization worker {msg.get('device')} 失败: {msg.get('error')}")
            rows.extend(msg["rows"])

        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"optimization worker 异常退出，exitcode={proc.exitcode}")

    rows = sorted(rows, key=lambda x: x["rcwa_second_order_score"], reverse=True)
    with (save_dir / "optimization_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("saved_to:", save_dir)
    print("target_from:", args.target)
    print("init_from:", args.init)
    print(f"total_time_sec: {time.perf_counter() - total_start:.2f}")
    print(
        "ranking:",
        [
            {
                "candidate_idx": r["candidate_idx"],
                "score": r["rcwa_second_order_score"],
                "tpp_at_edge": r["rcwa_tpp_at_edge"],
            }
            for r in rows
        ],
    )


def main():
    p = argparse.ArgumentParser(description="Multi-start RCWA topology optimization from laplas top-k samples.")
    p.add_argument("--target")
    p.add_argument("--init")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--save_dir", default=str(ROOT / "samples" / "optimized"))
    p.add_argument("--device", default=None, help="单设备模式；默认自动使用全部可见 GPU")
    p.add_argument("--devices", default=None, help="逗号分隔设备列表，如: cuda:0,cuda:1")
    p.add_argument("--target_lambda", type=float, default=1000.0)
    p.add_argument("--rcwa_orders", type=int, default=7)
    p.add_argument("--binary_eval_every", type=int, default=10)
    p.add_argument("--max_inits", type=int, default=5)
    p.add_argument("--filter_radius", type=int, default=1)
    p.add_argument("--proj_eta", type=float, default=0.5)
    p.add_argument("--beta_start", type=float, default=4.0)
    p.add_argument("--beta_end", type=float, default=16.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--objective_mode", choices=["score", "pointwise", "pointwise_l2"], default="score")
    args = p.parse_args()
    _run_with_args(args)


if __name__ == "__main__":
    main()
