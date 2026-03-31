from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset.rcwa.rcwa import torcwa_simulation  # noqa: E402
from infer.common import (  # noqa: E402
    load_model,
    load_stats,
    plot_map,
    plot_structure,
    resolve_default_diffusion_ckpt,
    resolve_default_stats_path,
)
from infer.task_library import (  # noqa: E402
    TaskCase,
    build_case_weight,
    case_output_dir,
    default_train_npz,
    list_task_keys,
    original_input_score,
    select_cases,
    task_score_details,
)
from model.diffusion import GaussianDiffusion  # noqa: E402
from model.models import ConditionalUNet  # noqa: E402


def load_target_from_dataset(train_npz: Path, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(train_npz)
    tpp = np.asarray(data["tpp_mag"][sample_idx], dtype=np.float32)
    tss = np.asarray(data["tss_mag"][sample_idx], dtype=np.float32)
    lambdas = np.asarray(data["lambdas"], dtype=np.float32)
    thetas = np.asarray(data["thetas"], dtype=np.float32)
    return np.stack([tpp, tss], axis=0), lambdas, thetas


def selected_pairs_from_weight(weight: np.ndarray) -> list[tuple[int, int]]:
    active = np.any(weight > 0, axis=0)
    return [(i, j) for i, j in zip(*np.where(active))]


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


def simulate_selected_map(
    structure: torch.Tensor,
    cond_ch: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    selected_pairs: list[tuple[int, int]],
    device: str,
    rcwa_orders: int,
) -> np.ndarray:
    layer = structure.squeeze(0).squeeze(0)
    pred = np.zeros((cond_ch, len(lambdas), len(thetas)), dtype=np.float32)
    for lam_idx, theta_idx in selected_pairs:
        out = torcwa_simulation(
            rcwa_physics_kwargs(float(lambdas[lam_idx]), float(thetas[theta_idx])),
            layer,
            rcwa_orders=rcwa_orders,
            project=False,
            device=device,
        )
        pred[0, lam_idx, theta_idx] = float(out["tpp_mag"].detach().cpu().item())
        if cond_ch > 1:
            pred[1, lam_idx, theta_idx] = float(out["tss_mag"].detach().cpu().item())
    return pred


def simulate_full_map(
    structure: torch.Tensor,
    cond_ch: int,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    device: str,
    rcwa_orders: int,
) -> np.ndarray:
    all_pairs = [(i, j) for i in range(len(lambdas)) for j in range(len(thetas))]
    return simulate_selected_map(structure, cond_ch, lambdas, thetas, all_pairs, device, rcwa_orders)


def weighted_mae_numpy(pred: np.ndarray, target: np.ndarray, weight: np.ndarray) -> float:
    denom = float(np.sum(weight))
    if denom <= 1e-8:
        return float(np.mean(np.abs(pred - target)))
    return float(np.sum(np.abs(pred - target) * weight) / denom)


def normalized_row(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    return row / max(float(np.max(row)), 1e-8)


def plot_target_lambda_curves(
    out_png: Path,
    case: TaskCase,
    target_raw: np.ndarray,
    pred_raw: np.ndarray,
    thetas: np.ndarray,
    lambdas: np.ndarray,
    title: str,
    score_info: dict[str, float],
) -> None:
    lam_idx = int(np.argmin(np.abs(lambdas - float(case.target_lambda_nm))))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    names = [("tpp", 0, "#1f77b4"), ("tss", 1, "#ff7f0e")]
    for ax, (label, ch, color) in zip(axes, names):
        target_row = normalized_row(target_raw[ch, lam_idx])
        pred_row = normalized_row(pred_raw[ch, lam_idx])
        theta40 = float(thetas[idx40])
        target40_raw = float(target_raw[ch, lam_idx, idx40])
        pred40_raw = float(pred_raw[ch, lam_idx, idx40])
        target40 = float(target_row[idx40])
        pred40 = float(pred_row[idx40])
        ax.plot(thetas, target_row, "k--", lw=1.6, label="target")
        ax.plot(thetas, pred_row, lw=1.9, color=color, label="pred")
        ax.axvline(theta40, color="0.5", ls=":", lw=1.0)
        ax.scatter([theta40], [target40], color="k", s=24, zorder=3)
        ax.scatter([theta40], [pred40], color=color, s=24, zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("theta (deg)")
        ax.set_ylabel("normalized magnitude")
        ax.set_title(
            f"{label} @ {float(case.target_lambda_nm):.0f}nm\n"
            f"target40={target40_raw:.3f} pred40={pred40_raw:.3f}"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        ax.annotate(
            f"40° target={target40_raw:.3f}",
            xy=(theta40, target40),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=8,
            color="k",
            ha="left",
            va="bottom",
        )
        ax.annotate(
            f"40° pred={pred40_raw:.3f}",
            xy=(theta40, pred40),
            xytext=(8, -16),
            textcoords="offset points",
            fontsize=8,
            color=color,
            ha="left",
            va="top",
        )
        ax.text(
            0.02,
            0.03,
            f"score={float(score_info.get('task_score', np.nan)):.3f}",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_ranked_overview(
    out_png: Path,
    case: TaskCase,
    samples: np.ndarray,
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
    ranked_idx: np.ndarray,
    weighted_err: np.ndarray,
    global_err: np.ndarray,
    topk: int,
) -> None:
    if topk <= 0 or len(ranked_idx) == 0:
        return

    target_lambda = float(case.target_lambda_nm)
    lam_idx = int(np.argmin(np.abs(lambdas - target_lambda)))
    idx40 = int(np.argmin(np.abs(thetas - 40.0)))
    vmax_tpp = max(float(np.nanmax(target_raw[0])), float(np.nanmax(pred_raw[:, 0])) if pred_raw.size else 0.0, 1e-6)
    vmax_tss = max(float(np.nanmax(target_raw[1])), float(np.nanmax(pred_raw[:, 1])) if pred_raw.size else 0.0, 1e-6)
    extent = [float(thetas[0]), float(thetas[-1]), float(lambdas[0]), float(lambdas[-1])]

    fig, axes = plt.subplots(
        topk,
        5,
        figsize=(22.0, max(2.8 * topk, 6.0)),
        gridspec_kw={"width_ratios": [0.7, 1.0, 1.0, 1.0, 1.0]},
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    hm_tpp = hm_tss = None

    target_tpp_row = normalized_row(target_raw[0, lam_idx])
    target_tss_row = normalized_row(target_raw[1, lam_idx])

    for r, idx in enumerate(ranked_idx[:topk]):
        a_struct, a_hmtpp, a_curve_tpp, a_hmtss, a_curve_tss = axes[r]
        i = int(idx)
        a_struct.imshow(samples[i, 0], cmap="gray_r", interpolation="nearest", vmin=0.0, vmax=1.0)
        a_struct.set_title(f"id={i}", fontsize=10)
        a_struct.axis("off")

        hm_tpp = a_hmtpp.imshow(pred_raw[i, 0], cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=0.0, vmax=vmax_tpp, interpolation="bicubic")
        a_hmtpp.axhline(target_lambda, color="w", ls="--", lw=1.0)
        a_hmtpp.set_title(f"werr={float(weighted_err[i]):.4f}\ngerr={float(global_err[i]):.4f}", fontsize=9)
        a_hmtpp.set_xlabel("theta")
        a_hmtpp.set_ylabel("lambda (nm)")

        a_curve_tpp.plot(thetas, target_tpp_row, "k--", lw=1.5, label="target")
        a_curve_tpp.plot(thetas, normalized_row(pred_raw[i, 0, lam_idx]), lw=1.8, color="#1f77b4", label="pred")
        a_curve_tpp.set_ylim(-0.05, 1.05)
        a_curve_tpp.set_xlabel("theta")
        a_curve_tpp.set_title(
            f"tpp t40={float(target_raw[0, lam_idx, idx40]):.3f} / p40={float(pred_raw[i, 0, lam_idx, idx40]):.3f}",
            fontsize=9,
        )
        a_curve_tpp.grid(alpha=0.25)
        if r == 0:
            a_curve_tpp.legend(fontsize=7, loc="lower right")

        hm_tss = a_hmtss.imshow(pred_raw[i, 1], cmap="turbo", aspect="auto", origin="lower", extent=extent, vmin=0.0, vmax=vmax_tss, interpolation="bicubic")
        a_hmtss.axhline(target_lambda, color="w", ls="--", lw=1.0)
        a_hmtss.set_title("tss", fontsize=9)
        a_hmtss.set_xlabel("theta")

        a_curve_tss.plot(thetas, target_tss_row, "k--", lw=1.5, label="target")
        a_curve_tss.plot(thetas, normalized_row(pred_raw[i, 1, lam_idx]), lw=1.8, color="#ff7f0e", label="pred")
        a_curve_tss.set_ylim(-0.05, 1.05)
        a_curve_tss.set_xlabel("theta")
        a_curve_tss.set_title(
            f"tss t40={float(target_raw[1, lam_idx, idx40]):.3f} / p40={float(pred_raw[i, 1, lam_idx, idx40]):.3f}",
            fontsize=9,
        )
        a_curve_tss.grid(alpha=0.25)
        if r == 0:
            a_curve_tss.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"{case.case_label} Top-{min(topk, len(ranked_idx))} RCWA rerank", fontsize=12)
    if hm_tpp is not None:
        fig.colorbar(hm_tpp, ax=axes[:, 1].tolist(), shrink=0.9, pad=0.01, label="|tpp|")
    if hm_tss is not None:
        fig.colorbar(hm_tss, ax=axes[:, 3].tolist(), shrink=0.9, pad=0.01, label="|tss|")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_case_visuals(
    save_dir: Path,
    case: TaskCase,
    target_raw: np.ndarray,
    all_samples: np.ndarray,
    topk_samples: np.ndarray,
    topk_pred_raw: np.ndarray,
    lambdas: np.ndarray,
    thetas: np.ndarray,
) -> None:
    vmax = float(np.nanmax(target_raw)) if np.isfinite(target_raw).any() else 1.0
    vmax = max(vmax, 1e-6)
    plot_map(save_dir / "target_tpp.png", target_raw, lambdas, thetas, f"{case.case_label} target tpp", vmax=vmax, channel_idx=0)
    plot_map(save_dir / "target_tss.png", target_raw, lambdas, thetas, f"{case.case_label} target tss", vmax=vmax, channel_idx=1)
    if len(all_samples):
        plot_structure(save_dir / "sample_00.png", all_samples[0], f"{case.case_label} sample0")
    if len(topk_samples):
        plot_structure(save_dir / "top1_structure.png", topk_samples[0], f"{case.case_label} top1")
        plot_map(
            save_dir / "top1_pred_tpp.png",
            topk_pred_raw[0],
            lambdas,
            thetas,
            f"{case.case_label} top1 RCWA tpp",
            vmax=vmax,
            channel_idx=0,
        )
        plot_map(
            save_dir / "top1_pred_tss.png",
            topk_pred_raw[0],
            lambdas,
            thetas,
            f"{case.case_label} top1 RCWA tss",
            vmax=vmax,
            channel_idx=1,
        )


@torch.no_grad()
def run_case(
    case: TaskCase,
    args,
    diffusion: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    train_npz: Path,
    root_save_dir: Path,
) -> dict:
    target_raw, lambdas, thetas = load_target_from_dataset(train_npz, case.sample_idx)
    cond_channels = target_raw.shape[0]
    weight_np = build_case_weight(case, cond_channels, lambdas, thetas)
    selected_pairs = selected_pairs_from_weight(weight_np)

    target_norm = ((target_raw[None] - mean) / std).astype(np.float32)
    cond_batch = torch.from_numpy(target_norm).to(args.device).repeat(args.num_samples, 1, 1, 1)
    samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)

    rcwa_preds: list[np.ndarray] = []
    weighted_err = []
    global_err = []
    task_scores = []
    tpp_scores = []
    tss_scores = []
    task_details = []
    for idx in range(samples.shape[0]):
        print(f"[task_infer-rcwa] {case.case_label} sample {idx + 1}/{samples.shape[0]}", flush=True)
        if args.eval_mode == "full":
            pred = simulate_full_map(
                samples[idx : idx + 1],
                cond_channels,
                lambdas,
                thetas,
                args.device,
                args.rcwa_orders,
            )
        else:
            pred = simulate_selected_map(
                samples[idx : idx + 1],
                cond_channels,
                lambdas,
                thetas,
                selected_pairs,
                args.device,
                args.rcwa_orders,
            )
        rcwa_preds.append(pred)
        weighted_err.append(weighted_mae_numpy(pred, target_raw, weight_np))
        global_err.append(float(np.mean(np.abs(pred - target_raw))))
        details = task_score_details(case, pred, lambdas, thetas)
        task_details.append(details)
        if case.task_key == "st2":
            tpp_scores.append(float(details["tpp_score_total"]))
            tss_scores.append(float(details["tss_score_total"]))
            print(
                f"[task_infer-score] {case.case_label} sample {idx + 1}/{samples.shape[0]} "
                f"tpp_score={float(details['tpp_score_total']):.4f} "
                f"tss_score={float(details['tss_score_total']):.4f} "
                f"weighted_err={float(weighted_err[-1]):.4f}",
                flush=True,
            )
        else:
            task_scores.append(float(details["task_score"]))
            print(
                f"[task_infer-score] {case.case_label} sample {idx + 1}/{samples.shape[0]} "
                f"task_score={float(details['task_score']):.4f} weighted_err={float(weighted_err[-1]):.4f}",
                flush=True,
            )

    pred_raw = np.stack(rcwa_preds, axis=0).astype(np.float32)
    weighted_err_np = np.asarray(weighted_err, dtype=np.float32)
    global_err_np = np.asarray(global_err, dtype=np.float32)

    save_dir = case_output_dir(root_save_dir, case)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_samples = samples.cpu().numpy()

    np.save(save_dir / "target_cond_raw.npy", target_raw)
    np.save(save_dir / "target_cond_norm.npy", target_norm)
    np.save(save_dir / "task_weight.npy", weight_np)
    np.save(save_dir / "all_samples.npy", all_samples)
    np.save(save_dir / "all_pred_cond_raw.npy", pred_raw)
    np.save(save_dir / "weighted_errors.npy", weighted_err_np)
    np.save(save_dir / "global_errors.npy", global_err_np)
    np.save(save_dir / "lambdas.npy", lambdas)
    np.save(save_dir / "thetas.npy", thetas)
    input_score = original_input_score(case, target_raw, lambdas, thetas)
    summary = {
        "task_key": case.task_key,
        "task_label": case.task_label,
        "objective_key": case.objective_key,
        "score_source": case.score_source,
        "case_label": case.case_label,
        "sample_idx": int(case.sample_idx),
        "target_lambda_nm": float(case.target_lambda_nm),
        "note": case.note,
        "train_npz": str(train_npz),
        "diffusion_ckpt": str(args.diffusion_ckpt),
        "stats_path": str(args.stats_path),
        "rcwa_orders": int(args.rcwa_orders),
        "eval_mode": args.eval_mode,
        "num_samples": int(args.num_samples),
        "original_input_score": input_score,
    }

    if case.task_key == "st2":
        tpp_scores_np = np.asarray(tpp_scores, dtype=np.float32)
        tss_scores_np = np.asarray(tss_scores, dtype=np.float32)
        tpp_rank_np = np.argsort(tpp_scores_np)[::-1]
        tss_rank_np = np.argsort(tss_scores_np)[::-1]
        tpp_topk_idx_np = tpp_rank_np[: min(args.topk, args.num_samples)]
        tss_topk_idx_np = tss_rank_np[: min(args.topk, args.num_samples)]

        np.save(save_dir / "tpp_scores.npy", tpp_scores_np)
        np.save(save_dir / "tss_scores.npy", tss_scores_np)
        np.save(save_dir / "tpp_rank_indices.npy", tpp_rank_np)
        np.save(save_dir / "tss_rank_indices.npy", tss_rank_np)
        np.save(save_dir / "tpp_topk_indices.npy", tpp_topk_idx_np)
        np.save(save_dir / "tss_topk_indices.npy", tss_topk_idx_np)
        np.save(save_dir / "tpp_topk_samples.npy", all_samples[tpp_topk_idx_np])
        np.save(save_dir / "tss_topk_samples.npy", all_samples[tss_topk_idx_np])
        np.save(save_dir / "tpp_topk_pred_cond_raw.npy", pred_raw[tpp_topk_idx_np])
        np.save(save_dir / "tss_topk_pred_cond_raw.npy", pred_raw[tss_topk_idx_np])

        if len(tpp_topk_idx_np):
            plot_target_lambda_curves(
                save_dir / "target_vs_top1_tpp_curves.png",
                case,
                target_raw,
                pred_raw[int(tpp_topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} target vs top1 tpp-ranked",
                task_details[int(tpp_topk_idx_np[0])],
            )
            plot_ranked_overview(
                save_dir / f"tpp_top{len(tpp_topk_idx_np)}_overview.png",
                case,
                all_samples,
                pred_raw,
                target_raw,
                lambdas,
                thetas,
                tpp_rank_np,
                weighted_err_np,
                global_err_np,
                min(args.topk, len(tpp_rank_np)),
            )
        if len(tss_topk_idx_np):
            plot_target_lambda_curves(
                save_dir / "target_vs_top1_tss_curves.png",
                case,
                target_raw,
                pred_raw[int(tss_topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} target vs top1 tss-ranked",
                task_details[int(tss_topk_idx_np[0])],
            )
            plot_ranked_overview(
                save_dir / f"tss_top{len(tss_topk_idx_np)}_overview.png",
                case,
                all_samples,
                pred_raw,
                target_raw,
                lambdas,
                thetas,
                tss_rank_np,
                weighted_err_np,
                global_err_np,
                min(args.topk, len(tss_rank_np)),
            )

        summary.update(
            {
                "topk": int(min(args.topk, args.num_samples)),
                "best_tpp_score": float(tpp_scores_np[tpp_topk_idx_np[0]]) if len(tpp_topk_idx_np) else None,
                "best_tpp_weighted_error": float(weighted_err_np[tpp_topk_idx_np[0]]) if len(tpp_topk_idx_np) else None,
                "best_tpp_global_error": float(global_err_np[tpp_topk_idx_np[0]]) if len(tpp_topk_idx_np) else None,
                "best_tss_score": float(tss_scores_np[tss_topk_idx_np[0]]) if len(tss_topk_idx_np) else None,
                "best_tss_weighted_error": float(weighted_err_np[tss_topk_idx_np[0]]) if len(tss_topk_idx_np) else None,
                "best_tss_global_error": float(global_err_np[tss_topk_idx_np[0]]) if len(tss_topk_idx_np) else None,
                "tpp_rankings": [
                    {
                        "rank": int(r + 1),
                        "sample_rank_idx": int(tpp_topk_idx_np[r]),
                        "tpp_score": float(tpp_scores_np[tpp_topk_idx_np[r]]),
                        "weighted_error": float(weighted_err_np[tpp_topk_idx_np[r]]),
                        "global_error": float(global_err_np[tpp_topk_idx_np[r]]),
                        "task_details": task_details[int(tpp_topk_idx_np[r])],
                    }
                    for r in range(len(tpp_topk_idx_np))
                ],
                "tss_rankings": [
                    {
                        "rank": int(r + 1),
                        "sample_rank_idx": int(tss_topk_idx_np[r]),
                        "tss_score": float(tss_scores_np[tss_topk_idx_np[r]]),
                        "weighted_error": float(weighted_err_np[tss_topk_idx_np[r]]),
                        "global_error": float(global_err_np[tss_topk_idx_np[r]]),
                        "task_details": task_details[int(tss_topk_idx_np[r])],
                    }
                    for r in range(len(tss_topk_idx_np))
                ],
            }
        )
    else:
        task_scores_np = np.asarray(task_scores, dtype=np.float32)
        rank_np = np.argsort(task_scores_np)[::-1]
        topk_idx_np = rank_np[: min(args.topk, args.num_samples)]
        topk_samples = all_samples[topk_idx_np]
        topk_pred_raw = pred_raw[topk_idx_np]

        np.save(save_dir / "task_scores.npy", task_scores_np)
        np.save(save_dir / "rank_indices.npy", rank_np)
        np.save(save_dir / "topk_indices.npy", topk_idx_np)
        np.save(save_dir / "topk_samples.npy", topk_samples)
        np.save(save_dir / "topk_pred_cond_raw.npy", topk_pred_raw)

        save_case_visuals(save_dir, case, target_raw, all_samples, topk_samples, topk_pred_raw, lambdas, thetas)
        if len(topk_idx_np):
            best_details = task_details[int(topk_idx_np[0])]
            plot_target_lambda_curves(
                save_dir / "target_vs_top1_curves.png",
                case,
                target_raw,
                pred_raw[int(topk_idx_np[0])],
                thetas,
                lambdas,
                f"{case.case_label} target vs top1",
                best_details,
            )
        plot_ranked_overview(
            save_dir / f"top{len(topk_idx_np)}_overview.png",
            case,
            all_samples,
            pred_raw,
            target_raw,
            lambdas,
            thetas,
            rank_np,
            weighted_err_np,
            global_err_np,
            min(args.topk, len(rank_np)),
        )

        summary.update(
            {
                "topk": int(len(topk_idx_np)),
                "best_task_score": float(task_scores_np[topk_idx_np[0]]) if len(topk_idx_np) else None,
                "best_weighted_error": float(weighted_err_np[topk_idx_np[0]]) if len(topk_idx_np) else None,
                "best_global_error": float(global_err_np[topk_idx_np[0]]) if len(topk_idx_np) else None,
                "selected_rankings": [
                    {
                        "rank": int(r + 1),
                        "sample_rank_idx": int(topk_idx_np[r]),
                        "task_score": float(task_scores_np[topk_idx_np[r]]),
                        "weighted_error": float(weighted_err_np[topk_idx_np[r]]),
                        "global_error": float(global_err_np[topk_idx_np[r]]),
                        "task_details": task_details[int(topk_idx_np[r])],
                    }
                    for r in range(len(topk_idx_np))
                ],
            }
        )
    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Run diffusion inference for the curated high-score task cases with RCWA reranking.")
    parser.add_argument("--train_npz", default=str(default_train_npz(ROOT)))
    parser.add_argument("--stats_path", default=str(resolve_default_stats_path(ROOT)))
    parser.add_argument("--diffusion_ckpt", default=str(resolve_default_diffusion_ckpt(ROOT)))
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale; 1.0 means disabled")
    parser.add_argument("--save_dir", default=str(ROOT / "samples" / "task_infer"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rcwa_orders", type=int, default=7)
    parser.add_argument("--eval_mode", choices=["target_only", "full"], default="target_only")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=f"Task selectors: task keys {list_task_keys()} or explicit case labels like 1100nm_id11339",
    )
    args = parser.parse_args()

    train_npz = Path(args.train_npz)
    mean, std = load_stats(args.stats_path)
    cond_channels = int(mean.shape[1])

    diffusion = GaussianDiffusion(ConditionalUNet(cond_channels).to(args.device), timesteps=1000, image_size=64).to(args.device)
    diffusion = load_model(args.diffusion_ckpt, diffusion, "diffusion", args.device)

    selected_cases = select_cases(args.tasks)
    run_root = Path(args.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)

    results = []
    for case in selected_cases:
        print(
            f"[task_infer] {case.task_key} {case.case_label} sample_idx={case.sample_idx} "
            f"target_lambda={case.target_lambda_nm:.1f}nm cfg_scale={args.cfg_scale:.2f}",
            flush=True,
        )
        results.append(run_case(case, args, diffusion, mean, std, train_npz, run_root))

    with (run_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_npz": str(train_npz),
                "num_cases": len(results),
                "cases": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved_to: {run_root}")


if __name__ == "__main__":
    main()
