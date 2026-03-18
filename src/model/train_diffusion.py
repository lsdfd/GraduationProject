import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import RCWADataset
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion
from train_utils import TrainLogger


def load_state_dict_flexible(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            stripped[key[len("module."):]] = value
        else:
            stripped[key] = value
    model.load_state_dict(stripped)


def main():
    cfg = {
        "data_path": "data/train_data.npz",
        "forward_ckpt": "checkpoints/forward_best.pt",
        "batch_size": 32,
        "epochs": 100,
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "save_dir": "checkpoints",
        "train_ratio": 0.7,
        "num_workers": 4,
        "timesteps": 1000,
        "lambda_diff": 0.6451612903,
        "lambda_phys": 0.3225806452,
        "lambda_bin": 0.0322580645,
        "cond_drop_prob": 0.15,
        "preview_every": 20,
        "split_seed": 20260315,
        "grad_clip": 1.0,
        "lr_patience": 6,
        "lr_factor": 0.5,
        "min_lr": 1e-6,
        "early_stop_patience": 15,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    use_cuda = cfg["device"].startswith("cuda")
    if use_cuda:
        device_name = cfg["device"]
        if device_name == "cuda":
            device_name = "cuda:0"
            cfg["device"] = device_name
        torch.cuda.set_device(device_name)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    logger = TrainLogger("diffusion", cfg["save_dir"], ["epoch", "train_loss", "val_loss", "train_diff", "train_phys", "train_bin"])

    dataset = RCWADataset(cfg["data_path"])
    cond_channels = dataset[0][1].shape[0]

    n_train = int(len(dataset) * cfg["train_ratio"])
    n_val = len(dataset) - n_train
    split_gen = torch.Generator().manual_seed(cfg["split_seed"])
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_gen)

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=use_cuda
    )

    surrogate = ForwardSurrogate(out_ch=cond_channels).to(cfg["device"])
    forward_ckpt = torch.load(cfg["forward_ckpt"], map_location=cfg["device"])
    load_state_dict_flexible(surrogate, forward_ckpt["model"])
    surrogate.eval()
    for p in surrogate.parameters():
        p.requires_grad = False

    unet = ConditionalUNet(cond_in_ch=cond_channels).to(cfg["device"])
    diffusion = GaussianDiffusion(unet, timesteps=cfg["timesteps"], image_size=64).to(cfg["device"])

    opt = torch.optim.AdamW(unet.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg["lr_factor"],
        patience=cfg["lr_patience"],
        min_lr=cfg["min_lr"],
    )

    best_val = 1e9
    best_epoch = -1
    stale_epochs = 0
    print(f"[Diffusion] train device={cfg['device']} epochs={cfg['epochs']} batch_size={cfg['batch_size']}")

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
                lambda_diff=cfg["lambda_diff"],
                lambda_phys=cfg["lambda_phys"],
                lambda_bin=cfg["lambda_bin"],
                cond_drop_prob=cfg["cond_drop_prob"],
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg["grad_clip"])
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
                    lambda_diff=cfg["lambda_diff"],
                    lambda_phys=cfg["lambda_phys"],
                    lambda_bin=cfg["lambda_bin"],
                    cond_drop_prob=0.0,
                )
                val_loss += loss.item() * x01.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(f"[Diffusion] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} lr={current_lr:.2e}")
        logger.log_scalars(
            epoch,
            [epoch, train_loss, val_loss, train_diff, train_phys, train_bin],
            {
                "loss/train": train_loss,
                "loss/val": val_loss,
                "loss_diff/train": train_diff,
                "loss_phys/train": train_phys,
                "loss_bin/train": train_bin,
                "lr": current_lr,
            },
        )

        ckpt = {
            "diffusion": diffusion.state_dict(),
            "cond_channels": cond_channels,
            "epoch": epoch,
            "best_val": best_val,
            "cfg": cfg,
        }
        torch.save(ckpt, os.path.join(cfg["save_dir"], "diffusion_last.pt"))

        if (epoch + 1) % cfg["preview_every"] == 0:
            preview_cond = val_set[0][1].unsqueeze(0).to(cfg["device"])
            preview = diffusion.sample(preview_cond, cfg_scale=3.0).cpu()
            logger.save_preview(epoch, preview)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(cfg["save_dir"], "diffusion_best.pt"))
        else:
            stale_epochs += 1

        if stale_epochs >= cfg["early_stop_patience"]:
            print(f"[Diffusion] early stop at epoch={epoch:03d}, best_epoch={best_epoch:03d}, best_val={best_val:.6f}")
            break

    logger.close()


if __name__ == "__main__":
    main()
