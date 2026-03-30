import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from infer.common import resolve_default_diffusion_ckpt, resolve_default_forward_ckpt, resolve_default_stats_path


def load_target_cond(target_path, stats_path, device):
    target = np.load(target_path).astype(np.float32)   # [C,11,17], C order: [tpp_mag, tss_mag]
    stats = np.load(stats_path)
    target = (target[None, ...] - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)
    return torch.from_numpy(target).to(device)


def load_target_cond_from_dataset(train_npz_path, sample_idx, stats_path, device):
    data = np.load(train_npz_path)
    tpp = np.asarray(data["tpp_mag"][sample_idx], dtype=np.float32)
    if "tss_mag" in data.files:
        tss = np.asarray(data["tss_mag"][sample_idx], dtype=np.float32)
        target = np.stack([tpp, tss], axis=0)
    else:
        target = tpp[None]
    stats = np.load(stats_path)
    target = (target[None, ...] - stats["mean"].astype(np.float32)) / stats["std"].astype(np.float32)
    return torch.from_numpy(target).to(device)


def load_model(ckpt_path, model, key, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device)[key])
    return model.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Sample structures from diffusion using a target spectrum.")
    parser.add_argument("--target_cond_path", default="data/target_cond.npy")
    parser.add_argument("--train_npz", default="data/train_data.npz")
    parser.add_argument("--target_sample_idx", type=int, default=None, help="直接从数据集读取该 sample 的原始 tpp/tss 光谱")
    parser.add_argument("--stats_path", default=str(resolve_default_stats_path(ROOT)))
    parser.add_argument("--forward_ckpt", default=str(resolve_default_forward_ckpt(ROOT)))
    parser.add_argument("--diffusion_ckpt", default=str(resolve_default_diffusion_ckpt(ROOT)))
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=3.0)
    parser.add_argument("--save_dir", default="samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.target_sample_idx is not None:
        target_cond = load_target_cond_from_dataset(args.train_npz, args.target_sample_idx, args.stats_path, args.device)
    else:
        target_cond = load_target_cond(args.target_cond_path, args.stats_path, args.device)
    cond_channels = target_cond.shape[1]

    surrogate = load_model(args.forward_ckpt, ForwardSurrogate(cond_channels).to(args.device), "model", args.device)
    diffusion = GaussianDiffusion(ConditionalUNet(cond_channels).to(args.device), timesteps=1000, image_size=64).to(args.device)
    diffusion = load_model(args.diffusion_ckpt, diffusion, "diffusion", args.device)

    cond_batch = target_cond.repeat(args.num_samples, 1, 1, 1)   # [K,C,11,17]
    samples = diffusion.sample(cond_batch, cfg_scale=args.cfg_scale)  # [K,1,64,64]
    pred_cond = surrogate(samples)                                     # [K,C,11,17]
    err = F.l1_loss(pred_cond, cond_batch, reduction="none").mean(dim=(1, 2, 3))
    topk = torch.topk(err, k=min(5, args.num_samples), largest=False).indices

    np.save(os.path.join(args.save_dir, "all_samples.npy"), samples.cpu().numpy())
    np.save(os.path.join(args.save_dir, "all_pred_cond.npy"), pred_cond.cpu().numpy())
    np.save(os.path.join(args.save_dir, "target_cond.npy"), target_cond.cpu().numpy())
    np.save(os.path.join(args.save_dir, "all_errors.npy"), err.cpu().numpy())
    np.save(os.path.join(args.save_dir, "topk_indices.npy"), topk.cpu().numpy())
    np.save(os.path.join(args.save_dir, "topk_samples.npy"), samples[topk].cpu().numpy())
    np.save(os.path.join(args.save_dir, "topk_pred_cond.npy"), pred_cond[topk].cpu().numpy())

    print("best errors:", err[topk].cpu().numpy())


if __name__ == "__main__":
    main()
