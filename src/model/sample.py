import os
import numpy as np
import torch
import torch.nn.functional as F
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion


def load_target_cond(target_path, stats_path, device):
    target = np.load(target_path).astype(np.float32)   # [C,11,17]
    stats = np.load(stats_path)
    target = (target[None, ...] - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)
    return torch.from_numpy(target).to(device)


def load_model(ckpt_path, model, key, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device)[key])
    return model.eval()


@torch.no_grad()
def main():
    cfg = {
        "target_cond_path": "data/target_cond.npy",
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

    surrogate = load_model(cfg["forward_ckpt"], ForwardSurrogate(cond_channels).to(cfg["device"]), "model", cfg["device"])
    diffusion = GaussianDiffusion(ConditionalUNet(cond_channels).to(cfg["device"]), timesteps=1000, image_size=64).to(cfg["device"])
    diffusion = load_model(cfg["diffusion_ckpt"], diffusion, "diffusion", cfg["device"])

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
