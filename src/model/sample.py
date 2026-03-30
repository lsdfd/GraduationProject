import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models import ForwardSurrogate, build_conditional_unet
from diffusion import GaussianDiffusion
from parallel_utils import load_state_dict_flexible
from train_utils import resolve_latest_checkpoint, resolve_latest_run


def load_target_cond(target_path, stats_path, device):
    target = np.load(target_path).astype(np.float32)   # [C,11,17], C order: [tpp_mag, tss_mag]
    stats = np.load(stats_path)
    target = (target[None, ...] - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)
    return torch.from_numpy(target).to(device)


def load_model(ckpt_path, model, key, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    load_state_dict_flexible(model, ckpt[key])
    return model.eval()


def resolve_default_stats_path():
    latest_forward_run = resolve_latest_run("runs", "forward")
    if latest_forward_run is not None:
        stats_path = os.path.join(latest_forward_run, "cond_stats.npz")
        if os.path.exists(stats_path):
            return stats_path
    return "runs/forward_runs/cond_stats.npz"


def resolve_default_ckpt_path(prefix):
    latest = resolve_latest_checkpoint("checkpoints", prefix)
    if latest is not None:
        return str(latest)
    return f"checkpoints/{prefix}_best.pt"


@torch.no_grad()
def main():
    cfg = {
        "target_cond_path": "data/target_cond.npy",
        "stats_path": resolve_default_stats_path(),
        "forward_ckpt": resolve_default_ckpt_path("forward"),
        "diffusion_ckpt": resolve_default_ckpt_path("diffusion"),
        "num_samples": 32,
        "cfg_scale": 3.0,
        "save_dir": "samples",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    os.makedirs(cfg["save_dir"], exist_ok=True)

    target_cond = load_target_cond(cfg["target_cond_path"], cfg["stats_path"], cfg["device"])
    cond_channels = target_cond.shape[1]

    surrogate = load_model(cfg["forward_ckpt"], ForwardSurrogate(cond_channels).to(cfg["device"]), "model", cfg["device"])
    diffusion_ckpt = torch.load(cfg["diffusion_ckpt"], map_location=cfg["device"], weights_only=False)
    diffusion_cfg = diffusion_ckpt.get("cfg", {})
    diffusion = GaussianDiffusion(
        build_conditional_unet(cond_channels, diffusion_cfg).to(cfg["device"]),
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        image_size=64,
    ).to(cfg["device"])
    load_state_dict_flexible(diffusion, diffusion_ckpt["diffusion"])
    diffusion.eval()

    cond_batch = target_cond.repeat(cfg["num_samples"], 1, 1, 1)   # [K,C,11,17]
    samples = diffusion.sample(cond_batch, cfg_scale=cfg["cfg_scale"])  # [K,1,64,64]
    pred_cond = surrogate(samples)                                     # [K,C,11,17]
    err = F.l1_loss(pred_cond, cond_batch, reduction="none").mean(dim=(1, 2, 3))
    topk = torch.topk(err, k=min(5, cfg["num_samples"]), largest=False).indices

    np.save(os.path.join(cfg["save_dir"], "all_samples.npy"), samples.cpu().numpy())
    np.save(os.path.join(cfg["save_dir"], "all_pred_cond.npy"), pred_cond.cpu().numpy())
    np.save(os.path.join(cfg["save_dir"], "target_cond.npy"), target_cond.cpu().numpy())
    np.save(os.path.join(cfg["save_dir"], "all_errors.npy"), err.cpu().numpy())
    np.save(os.path.join(cfg["save_dir"], "topk_indices.npy"), topk.cpu().numpy())
    np.save(os.path.join(cfg["save_dir"], "topk_samples.npy"), samples[topk].cpu().numpy())
    np.save(os.path.join(cfg["save_dir"], "topk_pred_cond.npy"), pred_cond[topk].cpu().numpy())

    print("best errors:", err[topk].cpu().numpy())


if __name__ == "__main__":
    main()
