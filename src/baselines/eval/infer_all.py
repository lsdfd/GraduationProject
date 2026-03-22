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


# ══════════════════════════════════════════════════════════════════════
# 1. 随机初始化结构（不需要模型）
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
    ckpt   = torch.load(ckpt_path, map_location=device)
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
    ckpt       = torch.load(ckpt_path, map_location=device)
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
    ckpt     = torch.load(ckpt_path, map_location=device)
    cond_ch  = ckpt["cond_channels"]
    unet     = ConditionalUNet(cond_in_ch=cond_ch).to(device)
    diffusion = GaussianDiffusion(unet, timesteps=1000, image_size=64).to(device)

    state = ckpt["diffusion"]
    diffusion.load_state_dict(state)
    diffusion.eval()
    return diffusion, None, None   # 归一化统计从 dataset 读，此处不存


def generate_diffusion(cond_norm, n_samples: int = 16, device: str = "cpu",
                       model=None, cfg_scale: float = 3.0, **kwargs) -> np.ndarray:
    if isinstance(cond_norm, np.ndarray):
        cond_norm = torch.from_numpy(cond_norm).float()
    cond_rep = cond_norm.expand(n_samples, -1, -1, -1).to(device)
    with torch.no_grad():
        out = model.sample(cond_rep, cfg_scale=cfg_scale)  # [N,1,64,64]
    return out.squeeze(1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════
# 统一加载入口
# ══════════════════════════════════════════════════════════════════════

def load_all_models(
    cvae_ckpt:      str = "checkpoints/cvae/cvae_best.pt",
    cgan_ckpt:      str = "checkpoints/cgan/cgan_best.pt",
    diffusion_ckpt: str = "checkpoints/diffusion_best.pt",
    device:         str = "cuda",
) -> dict:
    """
    返回字典：method_name → {"model": ..., "fn": generate_xxx}
    如果某个 checkpoint 不存在则跳过该方法，不报错。
    """
    models = {}

    # 随机方法：无需模型
    models["random"] = {"model": None, "fn": generate_random}

    # CVAE
    if os.path.exists(cvae_ckpt):
        m, mean, std = load_cvae(cvae_ckpt, device)
        models["cvae"] = {"model": m, "fn": generate_cvae,
                          "cond_mean": mean, "cond_std": std}
    else:
        print(f"[infer_all] CVAE checkpoint not found: {cvae_ckpt}")

    # cGAN
    if os.path.exists(cgan_ckpt):
        m, mean, std = load_cgan(cgan_ckpt, device)
        models["cgan"] = {"model": m, "fn": generate_cgan,
                          "cond_mean": mean, "cond_std": std}
    else:
        print(f"[infer_all] cGAN checkpoint not found: {cgan_ckpt}")

    # 扩散模型
    if os.path.exists(diffusion_ckpt):
        m, _, _ = load_diffusion(diffusion_ckpt, device)
        models["diffusion"] = {"model": m, "fn": generate_diffusion}
    else:
        print(f"[infer_all] Diffusion checkpoint not found: {diffusion_ckpt}")

    return models


# ══════════════════════════════════════════════════════════════════════
# 带计时的统一生成接口
# ══════════════════════════════════════════════════════════════════════

def timed_generate(method_info: dict, cond_norm, n_samples: int,
                   device: str) -> tuple[np.ndarray, float]:
    """
    调用对应方法生成结构，返回 (structures [N,64,64], elapsed_sec)
    """
    t0 = time.time()
    structs = method_info["fn"](
        cond_norm,
        n_samples=n_samples,
        device=device,
        model=method_info["model"],
    )
    elapsed = time.time() - t0
    return structs, elapsed
