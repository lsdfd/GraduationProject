"""
统一推理入口：所有方法用同一个接口生成候选结构。

每个 generate_xxx(cond_norm, n_samples, device) 返回:
  np.ndarray [n_samples, 64, 64]  二值结构 {0,1}

用法示例（被 run_eval.py 调用，不直接运行）:
  from infer_all import load_all_models, METHODS
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ── 路径设置 ──────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in [
    _SRC,
    os.path.join(_ROOT, "src", "dataset", "structure"),
    os.path.join(_ROOT, "src", "baselines", "model"),
]:
    if _p not in sys.path:
        sys.path.append(_p)

from model.models import ForwardSurrogate, ConditionalUNet
from model.diffusion import GaussianDiffusion
from model.train_utils import resolve_latest_run
from infer.task_library import build_case_weight
from cvae import CVAE
from cgan import Generator
from generate_one import generate_structure

METHODS = ("topo_opt", "cvae", "cgan", "diffusion", "diffusion+guide")

_TOPO_OPT_FNS = None


def _resolve_latest_ckpt(explicit_path: str | None, base_dir: str, run_name: str, ckpt_name: str) -> str:
    if explicit_path:
        return explicit_path

    latest_run = resolve_latest_run(base_dir, run_name)
    if latest_run is None:
        return str(Path(base_dir) / ckpt_name)

    candidate = Path(latest_run) / ckpt_name
    if candidate.exists():
        return str(candidate)
    return str(Path(base_dir) / ckpt_name)


def resolve_default_cvae_ckpt(explicit_path: str | None = None) -> str:
    return _resolve_latest_ckpt(explicit_path, "checkpoints/cvae", "cvae", "cvae_best.pt")


def resolve_default_cgan_ckpt(explicit_path: str | None = None) -> str:
    return _resolve_latest_ckpt(explicit_path, "checkpoints/cgan", "cgan", "cgan_best.pt")


def resolve_default_diffusion_ckpt(explicit_path: str | None = None) -> str:
    if explicit_path:
        return explicit_path
    # 当前扩散训练仍把 best/last checkpoint 写在 checkpoints/ 根目录，
    # runs/diffusion 里只保存日志和 summary，不保存可直接加载的权重。
    return "checkpoints/diffusion_best.pt"


def _get_topo_opt_fns():
    global _TOPO_OPT_FNS
    if _TOPO_OPT_FNS is None:
        sys.path.insert(0, os.path.join(_ROOT, "src", "infer"))
        from optimization import (  # noqa: WPS433
            symmetrize, density_filter, project_density, finalize_binary,
            rcwa_tpp_tss_row, second_order_score_row_torch, theta_grid,
            target_row_tensor, outer_monotonic_penalty, tv_loss,
        )
        _TOPO_OPT_FNS = {
            "symmetrize": symmetrize,
            "density_filter": density_filter,
            "project_density": project_density,
            "finalize_binary": finalize_binary,
            "rcwa_tpp_tss_row": rcwa_tpp_tss_row,
            "second_order_score_row_torch": second_order_score_row_torch,
            "theta_grid": theta_grid,
            "target_row_tensor": target_row_tensor,
            "outer_monotonic_penalty": outer_monotonic_penalty,
            "tv_loss": tv_loss,
        }
    return _TOPO_OPT_FNS


# ══════════════════════════════════════════════════════════════════════
# 1. 拓扑优化（multistart，每个起点随机初始化）
# ══════════════════════════════════════════════════════════════════════

def normalize_rows_torch(spec: torch.Tensor) -> torch.Tensor:
    row_max = spec.amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return spec / row_max


def task_guided_surrogate_loss(task_case, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    denom = weight.sum().clamp_min(1e-8)
    pred_norm = normalize_rows_torch(pred)
    target_norm = normalize_rows_torch(target)
    loss_fit = ((pred - target).abs() * weight).sum() / denom
    loss_shape = ((pred_norm - target_norm).abs() * weight).sum() / denom

    if task_case is None:
        return loss_fit, {"fit": loss_fit, "shape": loss_shape}

    if task_case.task_key == "p_second_order":
        loss = 0.40 * loss_fit + 0.60 * loss_shape
        return loss, {"fit": loss_fit, "shape": loss_shape}

    if task_case.task_key == "polarization_independent":
        loss_balance = (pred_norm[:, 0] - pred_norm[:, 1]).abs().mean()
        loss = 0.35 * loss_fit + 0.40 * loss_shape + 0.25 * loss_balance
        return loss, {"fit": loss_fit, "shape": loss_shape, "balance": loss_balance}

    if task_case.task_key == "polarization_multiplexed":
        active_weight = weight[:, 0:1]
        passive_weight = weight[:, 1:2]
        active_shape = ((pred_norm[:, 0:1] - target_norm[:, 0:1]).abs() * active_weight).sum() / active_weight.sum().clamp_min(1e-8)
        passive_silent = (pred[:, 1:2] * passive_weight).sum() / passive_weight.sum().clamp_min(1e-8)
        loss = 0.30 * loss_fit + 0.45 * active_shape + 0.25 * passive_silent
        return loss, {"fit": loss_fit, "shape": active_shape, "silent": passive_silent}

    if task_case.task_key == "fourth_order":
        loss = 0.35 * loss_fit + 0.65 * loss_shape
        return loss, {"fit": loss_fit, "shape": loss_shape}

    if task_case.task_key == "lowpass":
        center_idx = pred.shape[-1] // 2
        loss_center = (pred[:, :, :, center_idx] - target[:, :, :, center_idx]).abs().mean()
        loss = 0.35 * loss_fit + 0.40 * loss_shape + 0.25 * loss_center
        return loss, {"fit": loss_fit, "shape": loss_shape, "center": loss_center}

    if task_case.task_key == "st2":
        theta_axis = torch.linspace(-40.0, 40.0, steps=pred.shape[-1], device=pred.device, dtype=pred.dtype)
        lambda_axis = torch.linspace(800.0, 1300.0, steps=pred.shape[-2], device=pred.device, dtype=pred.dtype)
        theta_zero_idx = torch.argmin(theta_axis.abs())
        lambda_zero_idx = torch.argmin((lambda_axis - float(task_case.target_lambda_nm)).abs())
        loss_zero_theta = pred[:, :, :, theta_zero_idx].abs().mean()
        loss_zero_lambda = pred[:, :, lambda_zero_idx, :].abs().mean()
        loss = 0.45 * loss_fit + 0.35 * loss_shape + 0.10 * loss_zero_theta + 0.10 * loss_zero_lambda
        return loss, {"fit": loss_fit, "shape": loss_shape, "zero_theta": loss_zero_theta, "zero_lambda": loss_zero_lambda}

    return loss_fit, {"fit": loss_fit, "shape": loss_shape}


def generate_topo_opt(cond_norm, n_samples: int = 16, device: str = "cpu",
                      target_lambda: float = 1000.0,
                      steps: int = 200, lr: float = 0.02,
                      filter_radius: int = 1,
                      beta_start: float = 1.0, beta_end: float = 8.0,
                      proj_eta: float = 0.5,
                      rcwa_orders: int = 7,
                      **kwargs) -> np.ndarray:
    """
    多起点拓扑优化：每次从随机初始化出发独立优化，取最优二值结构。
    优化阶段使用前向代理模型和任务目标；最终质量由 run_eval.py 的 RCWA 统一打分。
    返回 [n_samples, 64, 64] 二值结构。
    """
    topo = _get_topo_opt_fns()
    surrogate = kwargs.get("surrogate")
    target_norm = kwargs.get("target_norm")
    task_case = kwargs.get("task_case")
    lambdas = kwargs.get("lambdas")
    thetas = kwargs.get("thetas")
    if surrogate is None or target_norm is None:
        raise ValueError("topo_opt 现在需要 forward surrogate 和 target_norm 才能运行。")
    if lambdas is None or thetas is None:
        lambdas_np = np.arange(800.0, 1300.1, 50.0, dtype=np.float32)
        thetas_np = np.arange(-40.0, 40.1, 5.0, dtype=np.float32)
    else:
        lambdas_np = np.asarray(lambdas, dtype=np.float32)
        thetas_np = np.asarray(thetas, dtype=np.float32)
    cond_ch = int(target_norm.shape[1])
    if task_case is not None:
        weight_np = build_case_weight(task_case, cond_ch, lambdas_np, thetas_np)
    else:
        weight_np = np.zeros((cond_ch, len(lambdas_np), len(thetas_np)), dtype=np.float32)
        lam_idx = int(np.argmin(np.abs(lambdas_np - float(target_lambda))))
        weight_np[0, lam_idx, :] = 1.0
    weight_t = torch.from_numpy(weight_np).to(device).unsqueeze(0)
    target_t = target_norm.to(device)
    results  = []

    for i in range(n_samples):
        s    = generate_structure(RNG_SEED=None, N_COARSE=8, N_FINE=64,
                                  SAVE_FIG=False, SAVE_NPY=False)
        init = torch.from_numpy(s["final"].astype(np.float32)
                               ).unsqueeze(0).unsqueeze(0).to(device)

        rho_param      = init.clone().detach().requires_grad_(True)
        opt            = torch.optim.Adam([rho_param], lr=lr)
        best_bin       = topo["finalize_binary"](init.detach())
        best_score_val = -1.0

        for step in range(steps):
            beta  = beta_start + step / max(steps - 1, 1) * (beta_end - beta_start)
            rho   = topo["symmetrize"](rho_param).clamp(0.0, 1.0)
            rho_f = topo["density_filter"](rho, filter_radius)
            x     = topo["project_density"](rho_f, beta=beta, eta=proj_eta)
            pred = surrogate(x)
            loss_task, terms = task_guided_surrogate_loss(task_case, pred, target_t, weight_t)
            loss = loss_task + 0.06 * (x * (1.0 - x)).mean() + 0.02 * topo["tv_loss"](rho_f)

            opt.zero_grad(); loss.backward()
            opt.step()
            with torch.no_grad():
                rho_param.clamp_(0.0, 1.0)

            if (step + 1) % 50 == 0 or step + 1 == steps:
                xb = topo["finalize_binary"](x.detach())
                with torch.no_grad():
                    pred_b = surrogate(xb)
                    eval_loss, _ = task_guided_surrogate_loss(task_case, pred_b, target_t, weight_t)
                    sc = float(1.0 / (1.0 + eval_loss.item()))
                if sc > best_score_val:
                    best_score_val = sc
                    best_bin = xb.detach().clone()
                extra = " ".join(f"{k}={float(v.item()):.4f}" for k, v in terms.items() if k not in {"fit", "shape"})
                print(
                    f"[topo_opt {i+1}/{n_samples}] step={step+1:03d}/{steps} "
                    f"proxy_score={sc:.4f} fit={float(terms['fit'].item()):.4f} shape={float(terms['shape'].item()):.4f} "
                    f"{extra}".rstrip(),
                    flush=True,
                )

        results.append(best_bin.squeeze().cpu().numpy().astype(np.float32))
        print(f"[topo_opt {i+1}/{n_samples}] final_proxy_score={best_score_val:.4f}", flush=True)

    return np.stack(results, axis=0)


# ══════════════════════════════════════════════════════════════════════
# 2（原1）. 随机初始化结构（保留供调试，不进入 load_all_models）
# ══════════════════════════════════════════════════════════════════════

def generate_random(cond_norm, n_samples: int = 16, device: str = "cpu",
                    **kwargs) -> np.ndarray:
    """随机生成 n_samples 个符合 C4+σx 对称的二值结构。"""
    structs = []
    for _ in range(n_samples):
        out = generate_structure(
            RNG_SEED=None,
            N_COARSE=8,
            N_FINE=64,
            SAVE_FIG=False,
            SAVE_NPY=False,
        )
        structs.append(out["final"].astype(np.float32))
    return np.stack(structs, axis=0)   # [N, 64, 64]


# ══════════════════════════════════════════════════════════════════════
# 2. CVAE
# ══════════════════════════════════════════════════════════════════════

def load_cvae(ckpt_path: str, device: str) -> tuple:
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    cond_ch     = ckpt["cond_ch"]
    latent_dim  = ckpt.get("latent_dim", 128)
    model  = CVAE(cond_in_ch=cond_ch, latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("cond_mean"), ckpt.get("cond_std")


def generate_cvae(cond_norm, n_samples: int = 16, device: str = "cpu",
                  model=None, **kwargs) -> np.ndarray:
    """
    cond_norm: torch.Tensor [1, 2, 11, 17]（已归一化）或 np.ndarray
    """
    if isinstance(cond_norm, np.ndarray):
        cond_norm = torch.from_numpy(cond_norm).float()
    cond_norm = cond_norm.to(device)
    out = model.sample(cond_norm, n_samples=n_samples)   # [N,1,64,64]
    return out.squeeze(1).cpu().numpy()                   # [N,64,64]


# ══════════════════════════════════════════════════════════════════════
# 3. cGAN
# ══════════════════════════════════════════════════════════════════════

def load_cgan(ckpt_path: str, device: str) -> tuple:
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    cond_ch    = ckpt["cond_ch"]
    latent_dim = ckpt.get("latent_dim", 128)
    G = Generator(latent_dim=latent_dim, cond_dim=256).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G, ckpt.get("cond_mean"), ckpt.get("cond_std")


def generate_cgan(cond_norm, n_samples: int = 16, device: str = "cpu",
                  model=None, **kwargs) -> np.ndarray:
    if isinstance(cond_norm, np.ndarray):
        cond_norm = torch.from_numpy(cond_norm).float()
    cond_norm = cond_norm.to(device)
    out = model.sample(cond_norm, n_samples=n_samples)   # [N,1,64,64]
    return out.squeeze(1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 4. 扩散模型
# ══════════════════════════════════════════════════════════════════════

def load_diffusion(ckpt_path: str, device: str):
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=False)
    cond_ch  = ckpt["cond_channels"]
    cfg = ckpt.get("cfg", {})
    unet = ConditionalUNet(cond_ch).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=int(cfg.get("timesteps", 1000)), image_size=64).to(device)

    state = ckpt["diffusion"]
    diffusion.load_state_dict(state)
    diffusion.eval()
    return diffusion, None, None   # 归一化统计从 dataset 读，此处不存


def generate_diffusion(cond_norm, n_samples: int = 16, device: str = "cpu",
                       model=None, cfg_scale: float = 3.0, **kwargs) -> np.ndarray:
    """纯 CFG 扩散，无物理引导（baseline：diffusion w/o guidance）。"""
    if isinstance(cond_norm, np.ndarray):
        cond_norm = torch.from_numpy(cond_norm).float()
    cond_rep = cond_norm.expand(n_samples, -1, -1, -1).to(device)
    with torch.no_grad():
        out = model.sample(cond_rep, cfg_scale=cfg_scale)  # [N,1,64,64]
    return out.squeeze(1).cpu().numpy()


def generate_diffusion_guided(cond_norm, n_samples: int = 16, device: str = "cpu",
                               model=None, cfg_scale: float = 3.0,
                               surrogate=None, target_norm=None,
                               guidance_scale: float = 0.1,
                               guide_start_t: int = 300,
                               guide_every: int = 1,
                               **kwargs) -> np.ndarray:
    """CFG + DPS 物理引导扩散（Ours）。"""
    if isinstance(cond_norm, np.ndarray):
        cond_norm = torch.from_numpy(cond_norm).float()
    cond_rep   = cond_norm.expand(n_samples, -1, -1, -1).to(device)
    target_rep = target_norm.to(device) if target_norm is not None else None
    out = model.sample_guided(
        cond_rep, cfg_scale=cfg_scale,
        surrogate=surrogate, target_norm=target_rep,
        guidance_scale=guidance_scale,
        guide_start_t=guide_start_t,
        guide_every=guide_every,
    )   # [N,1,64,64]
    return out.squeeze(1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 统一加载入口
# ══════════════════════════════════════════════════════════════════════

def load_all_models(
    cvae_ckpt:      str | None = None,
    cgan_ckpt:      str | None = None,
    diffusion_ckpt: str | None = None,
    device:         str = "cuda",
    methods:        list[str] | None = None,
) -> dict:
    """
    返回字典：method_name → {"model": ..., "fn": generate_xxx}
    如果某个 checkpoint 不存在则跳过该方法，不报错。
    """
    models = {}
    selected = list(methods) if methods else list(METHODS)
    cvae_ckpt = resolve_default_cvae_ckpt(cvae_ckpt)
    cgan_ckpt = resolve_default_cgan_ckpt(cgan_ckpt)
    diffusion_ckpt = resolve_default_diffusion_ckpt(diffusion_ckpt)

    # 拓扑优化：无需模型，target_lambda 由 run_eval.py 填入
    if "topo_opt" in selected:
        models["topo_opt"] = {"model": None, "fn": generate_topo_opt}

    # CVAE
    if "cvae" in selected and os.path.exists(cvae_ckpt):
        m, mean, std = load_cvae(cvae_ckpt, device)
        models["cvae"] = {"model": m, "fn": generate_cvae,
                          "cond_mean": mean, "cond_std": std}
    elif "cvae" in selected:
        print(f"[infer_all] CVAE checkpoint not found: {cvae_ckpt}")

    # cGAN
    if "cgan" in selected and os.path.exists(cgan_ckpt):
        m, mean, std = load_cgan(cgan_ckpt, device)
        models["cgan"] = {"model": m, "fn": generate_cgan,
                          "cond_mean": mean, "cond_std": std}
    elif "cgan" in selected:
        print(f"[infer_all] cGAN checkpoint not found: {cgan_ckpt}")

    # 扩散模型（纯 CFG）
    need_diffusion = any(m in selected for m in ("diffusion", "diffusion+guide"))
    if need_diffusion and os.path.exists(diffusion_ckpt):
        m, _, _ = load_diffusion(diffusion_ckpt, device)
        if "diffusion" in selected:
            models["diffusion"] = {"model": m, "fn": generate_diffusion}
        if "diffusion+guide" in selected:
            models["diffusion+guide"] = {"model": m, "fn": generate_diffusion_guided}
        # surrogate / target_norm 由 run_eval.py 在注册后填入 method_info
    elif need_diffusion:
        print(f"[infer_all] Diffusion checkpoint not found: {diffusion_ckpt}")

    return models


# ══════════════════════════════════════════════════════════════════════
# 带计时的统一生成接口
# ══════════════════════════════════════════════════════════════════════

def timed_generate(method_info: dict, cond_norm, n_samples: int,
                   device: str) -> tuple[np.ndarray, float]:
    """
    调用对应方法生成结构，返回 (structures [N,64,64], elapsed_sec)。
    method_info 里除 "fn"/"model" 外的所有键都作为 kwargs 透传给生成函数
    （供 diffusion+guide 传 surrogate / target_norm 等）。
    """
    extra = {k: v for k, v in method_info.items() if k not in ("fn", "model")}
    t0 = time.time()
    structs = method_info["fn"](
        cond_norm,
        n_samples=n_samples,
        device=device,
        model=method_info["model"],
        **extra,
    )
    return structs, time.time() - t0
