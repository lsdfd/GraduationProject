import os
import numpy as np
import torch
import torch.nn.functional as F
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion


def load_target_cond(target_path, stats_path, device):
    target = np.load(target_path).astype(np.float32)   # [C,11,17]
    stats = np.load(stats_path)

    mean = stats["mean"].astype(np.float32)  # [1,C,1,1]
    std = stats["std"].astype(np.float32)

    target = (target[None, ...] - mean) / std  # [1,C,11,17]
    target = torch.from_numpy(target).to(device)
    return target


@torch.no_grad()
def main():
    cfg = {
        "target_cond_path": "data/target_cond.npy",   # [C,11,17]
        "stats_path": "checkpoints/cond_stats.npz",
        "forward_ckpt": "checkpoints/forward_best.pt",
        "diffusion_ckpt": "checkpoints/diffusion_best.pt",
        "num_samples": 32,
        "cfg_scale": 3.0,
        "save_dir": "samples",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    os.makedirs(cfg["save_dir"], exist_ok=True)

    target_cond = load_target_cond(cfg["target_cond_path"], cfg["stats_path"], cfg["device"])
    cond_channels = target_cond.shape[1]

    surrogate = ForwardSurrogate(out_ch=cond_channels).to(cfg["device"])
    surrogate.load_state_dict(torch.load(cfg["forward_ckpt"], map_location=cfg["device"])["model"])
    surrogate.eval()

    unet = ConditionalUNet(cond_in_ch=cond_channels).to(cfg["device"])
    diffusion = GaussianDiffusion(unet, timesteps=1000, image_size=64).to(cfg["device"])
    diffusion.load_state_dict(torch.load(cfg["diffusion_ckpt"], map_location=cfg["device"])["diffusion"])
    diffusion.eval()

    cond_batch = target_cond.repeat(cfg["num_samples"], 1, 1, 1)
    samples = diffusion.sample(cond_batch, cfg_scale=cfg["cfg_scale"])  # [K,1,64,64], 0/1

    pred_cond = surrogate(samples)
    err = F.l1_loss(pred_cond, cond_batch, reduction="none").mean(dim=(1, 2, 3))
    topk = torch.topk(err, k=min(5, cfg["num_samples"]), largest=False).indices

    samples_np = samples.cpu().numpy()
    err_np = err.cpu().numpy()

    np.save(os.path.join(cfg["save_dir"], "all_samples.npy"), samples_np)
    np.save(os.path.join(cfg["save_dir"], "all_errors.npy"), err_np)
    np.save(os.path.join(cfg["save_dir"], "topk_indices.npy"), topk.cpu().numpy())

    print("best errors:", err[topk].cpu().numpy())


if __name__ == "__main__":
    main()