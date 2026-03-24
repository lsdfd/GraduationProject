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
import numpy as np
import torch

# ── 路径设置 ──────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in [
    os.path.join(_ROOT, "src", "model"),
    os.path.join(_ROOT, "src", "infer"),
    os.path.join(_ROOT, "src", "dataset", "structure"),
    os.path.join(_ROOT, "src", "baselines", "model"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion
from cvae import CVAE
from cgan import Generator
from generate_one import generate_structure

METHODS = ("topo_opt", "cvae", "cgan", "diffusion", "diffusion+guide")

_TOPO_OPT_FNS = None


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
    不使用任何学习模型，纯 RCWA 梯度驱动。
    返回 [n_samples, 64, 64] 二值结构。
    """
    topo = _get_topo_opt_fns()
    thetas   = topo["theta_grid"]()
    target_t = topo["target_row_tensor"](device)
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

            tpp_row, _ = topo["rcwa_tpp_tss_row"](x, target_lambda, device, rcwa_orders)
            pack       = topo["second_order_score_row_torch"](tpp_row, thetas)
            row_max    = tpp_row.amax(dim=-1, keepdim=True).clamp_min(1e-8)
            y_norm     = tpp_row / row_max

            loss = ((1.0 - pack["score"].mean())
                    + 0.10 * (y_norm - target_t).abs().mean()
                    + 0.10 * topo["outer_monotonic_penalty"](y_norm, thetas)
                    + 0.06 * (x * (1.0 - x)).mean()
                    + 0.02 * topo["tv_loss"](rho_f))

            opt.zero_grad(); loss.backward()
            opt.step()
            with torch.no_grad():
                rho_param.clamp_(0.0, 1.0)

            if (step + 1) % 50 == 0 or step + 1 == steps:
                try:
                    xb       = topo["finalize_binary"](x.detach())
                    tpp_b, _ = topo["rcwa_tpp_tss_row"](xb, target_lambda, device, rcwa_orders)
                    sc       = float(topo["second_order_score_row_torch"](tpp_b, thetas)["score"].mean())
                    if sc > best_score_val:
                        best_score_val = sc
                        best_bin = xb.detach().clone()
                except Exception:
                    pass

        results.append(best_bin.squeeze().cpu().numpy().astype(np.float32))
        print(f"[topo_opt {i+1}/{n_samples}] score={best_score_val:.4f}", flush=True)

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
    cond_norm: torch.Tensor [1, 2, 11, 13]（已归一化）或 np.ndarray
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
    unet     = ConditionalUNet(cond_in_ch=cond_ch).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=1000, image_size=64).to(device)

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
    cvae_ckpt:      str = "checkpoints/cvae/cvae_best.pt",
    cgan_ckpt:      str = "checkpoints/cgan/cgan_best.pt",
    diffusion_ckpt: str = "checkpoints/diffusion_best.pt",
    device:         str = "cuda",
    methods:        list[str] | None = None,
) -> dict:
    """
    返回字典：method_name → {"model": ..., "fn": generate_xxx}
    如果某个 checkpoint 不存在则跳过该方法，不报错。
    """
    models = {}
    selected = list(methods) if methods else list(METHODS)

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
