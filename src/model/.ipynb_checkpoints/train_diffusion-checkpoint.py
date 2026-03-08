import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import RCWADataset
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion
from train_utils import TrainLogger


def main():
    cfg = {
        "data_path": "data/train_data.npz",
        "forward_ckpt": "checkpoints/forward_best.pt",
        "batch_size": 32,
        "epochs": 300,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "save_dir": "checkpoints",
        "train_ratio": 0.7,
        "num_workers": 4,
        "timesteps": 1000,
        "lambda_phys": 0.5,
        "lambda_bin": 0.05,
        "cond_drop_prob": 0.1,
        "preview_every": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    os.makedirs(cfg["save_dir"], exist_ok=True)
    logger = TrainLogger("diffusion", cfg["save_dir"], ["epoch", "train_loss", "val_loss", "train_diff", "train_phys", "train_bin"])

    dataset = RCWADataset(cfg["data_path"])
    cond_channels = dataset[0][1].shape[0]

    n_train = int(len(dataset) * cfg["train_ratio"])
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    surrogate = ForwardSurrogate(out_ch=cond_channels).to(cfg["device"])
    forward_ckpt = torch.load(cfg["forward_ckpt"], map_location=cfg["device"])
    surrogate.load_state_dict(forward_ckpt["model"])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    unet = ConditionalUNet(cond_in_ch=cond_channels).to(cfg["device"])
    diffusion = GaussianDiffusion(unet, timesteps=cfg["timesteps"], image_size=64).to(cfg["device"])

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = 1e9

    for epoch in range(cfg["epochs"]):
        diffusion.train()
        train_loss = 0.0
        train_diff = 0.0
        train_phys = 0.0
        train_bin = 0.0

        for x01, cond in train_loader:
            x01 = x01.to(cfg["device"])   # [0,1]
            cond = cond.to(cfg["device"]) # normalized
            x0 = x01 * 2.0 - 1.0          # [-1,1]

            loss, log_dict = diffusion.p_losses(
                x0=x0,
                cond=cond,
                surrogate=surrogate,
                lambda_phys=cfg["lambda_phys"],
                lambda_bin=cfg["lambda_bin"],
                cond_drop_prob=cfg["cond_drop_prob"],
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * x01.size(0)
            train_diff += log_dict.get("loss_diff", 0.0) * x01.size(0)
            train_phys += log_dict.get("loss_phys", 0.0) * x01.size(0)
            train_bin += log_dict.get("loss_bin", 0.0) * x01.size(0)

        train_loss /= len(train_loader.dataset)
        train_diff /= len(train_loader.dataset)
        train_phys /= len(train_loader.dataset)
        train_bin /= len(train_loader.dataset)

        diffusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x01, cond in val_loader:
                x01 = x01.to(cfg["device"])
                cond = cond.to(cfg["device"])
                x0 = x01 * 2.0 - 1.0

                loss, _ = diffusion.p_losses(
                    x0=x0,
                    cond=cond,
                    surrogate=surrogate,
                    lambda_phys=cfg["lambda_phys"],
                    lambda_bin=cfg["lambda_bin"],
                    cond_drop_prob=0.0,
                )
                val_loss += loss.item() * x01.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"[Diffusion] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
        logger.log_scalars(
            epoch,
            [epoch, train_loss, val_loss, train_diff, train_phys, train_bin],
            {
                "loss/train": train_loss,
                "loss/val": val_loss,
                "loss_diff/train": train_diff,
                "loss_phys/train": train_phys,
                "loss_bin/train": train_bin,
            },
        )

        ckpt = {
            "diffusion": diffusion.state_dict(),
            "cond_channels": cond_channels,
        }
        torch.save(ckpt, os.path.join(cfg["save_dir"], "diffusion_last.pt"))

        if (epoch + 1) % cfg["preview_every"] == 0:
            preview_cond = val_set[0][1].unsqueeze(0).to(cfg["device"])
            preview = diffusion.sample(preview_cond, cfg_scale=3.0).cpu()
            logger.save_preview(epoch, preview)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(cfg["save_dir"], "diffusion_best.pt"))

    logger.close()


if __name__ == "__main__":
    main()
