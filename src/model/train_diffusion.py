import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import RCWADataset
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion
from train_utils import TrainLogger


def main():
    cfg = {
        "data_path": str(PROJECT_ROOT / "data" / "train_data_20000.npz"),
        "forward_ckpt": str(PROJECT_ROOT / "checkpoints" / "forward_best.pt"),
        "batch_size": 64,
        "epochs": 600,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "save_dir": str(PROJECT_ROOT / "checkpoints"),
        "train_ratio": 0.9,
        "num_workers": 8,
        "timesteps": 1000,
        "lambda_phys": 0.5,
        "lambda_bin": 0.05,
        "cond_drop_prob": 0.1,
        "preview_every": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    use_cuda = cfg["device"].startswith("cuda")
    use_amp = use_cuda
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"[Diffusion] device={cfg['device']} gpu={torch.cuda.get_device_name(0)} amp={'on' if use_amp else 'off'}")
    else:
        print(f"[Diffusion] device={cfg['device']} amp=off")

    os.makedirs(cfg["save_dir"], exist_ok=True)
    logger = TrainLogger(
        "diffusion",
        cfg["save_dir"],
        [
            "epoch",
            "train_loss",
            "val_loss",
            "train_diff",
            "train_phys",
            "train_bin",
            "val_diff",
            "val_phys",
            "val_bin",
        ],
    )

    dataset = RCWADataset(cfg["data_path"])
    cond_channels = dataset[0][1].shape[0]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_train = int(len(dataset) * cfg["train_ratio"])
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=use_cuda, persistent_workers=cfg["num_workers"] > 0
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=use_cuda, persistent_workers=cfg["num_workers"] > 0
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
        epoch_start = time.time()
        diffusion.train()
        train_loss = 0.0
        train_diff = 0.0
        train_phys = 0.0
        train_bin = 0.0

        for x01, cond in train_loader:
            x01 = x01.to(cfg["device"])   # [0,1]
            cond = cond.to(cfg["device"]) # normalized
            x0 = x01 * 2.0 - 1.0          # [-1,1]

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, log_dict = diffusion.p_losses(
                    x0=x0,
                    cond=cond,
                    surrogate=surrogate,
                    lambda_phys=cfg["lambda_phys"],
                    lambda_bin=cfg["lambda_bin"],
                    cond_drop_prob=cfg["cond_drop_prob"],
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

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
        val_diff = 0.0
        val_phys = 0.0
        val_bin = 0.0
        with torch.no_grad():
            for x01, cond in val_loader:
                x01 = x01.to(cfg["device"])
                cond = cond.to(cfg["device"])
                x0 = x01 * 2.0 - 1.0

                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss, log_dict = diffusion.p_losses(
                        x0=x0,
                        cond=cond,
                        surrogate=surrogate,
                        lambda_phys=cfg["lambda_phys"],
                        lambda_bin=cfg["lambda_bin"],
                        cond_drop_prob=0.0,
                    )
                val_loss += loss.item() * x01.size(0)
                val_diff += log_dict.get("loss_diff", 0.0) * x01.size(0)
                val_phys += log_dict.get("loss_phys", 0.0) * x01.size(0)
                val_bin += log_dict.get("loss_bin", 0.0) * x01.size(0)

        val_loss /= len(val_loader.dataset)
        val_diff /= len(val_loader.dataset)
        val_phys /= len(val_loader.dataset)
        val_bin /= len(val_loader.dataset)
        epoch_time = time.time() - epoch_start

        print(
            "[Diffusion] "
            f"epoch={epoch:03d} "
            f"train={train_loss:.6f} val={val_loss:.6f} "
            f"train_diff={train_diff:.6f} train_phys={train_phys:.6f} train_bin={train_bin:.6f} "
            f"val_diff={val_diff:.6f} val_phys={val_phys:.6f} val_bin={val_bin:.6f} "
            f"time={epoch_time:.1f}s"
        )
        logger.log_scalars(
            epoch,
            [epoch, train_loss, val_loss, train_diff, train_phys, train_bin, val_diff, val_phys, val_bin],
            {
                "loss/train": train_loss,
                "loss/val": val_loss,
                "loss_diff/train": train_diff,
                "loss_phys/train": train_phys,
                "loss_bin/train": train_bin,
                "loss_diff/val": val_diff,
                "loss_phys/val": val_phys,
                "loss_bin/val": val_bin,
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
