import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from dataset import RCWADataset
from models import ConditionalUNet, ForwardSurrogate
from diffusion import GaussianDiffusion
from train_utils import TrainLogger, prepare_run_dir, update_latest_run, resolve_latest_run, write_json


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional diffusion model for metasurface inverse design.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training npz file.")
    parser.add_argument("--forward_ckpt", type=str, default=None, help="Path to forward surrogate checkpoint.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints/logs.")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu.")
    parser.add_argument("--lambda_diff", type=float, default=None, help="Weight for diffusion denoise loss.")
    parser.add_argument("--lambda_phys", type=float, default=None, help="Weight for physics surrogate loss.")
    parser.add_argument("--lambda_bin", type=float, default=None, help="Weight for binarization loss.")
    parser.add_argument("--cond_drop_prob", type=float, default=None, help="Condition dropout probability for CFG training.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = {
        "data_path": "data/train_data.npz",
        "forward_ckpt": None,
        "batch_size": 32,
        "epochs": 100,
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "save_dir": "checkpoints",
        "train_ratio": 0.7,
        "num_workers": 4,
        "timesteps": 1000,
        # Emphasize physics consistency to improve RCWA-aligned inverse results.
        "lambda_diff": 0.6451612903,
        "lambda_phys": 0.3225806452,
        "lambda_bin": 0.0322580645,
        "cond_drop_prob": 0.10,
        "preview_every": 20,
        "split_seed": 20260315,
        "grad_clip": 1.0,
        "lr_patience": 6,
        "lr_factor": 0.5,
        "min_lr": 1e-6,
        "early_stop_patience": 15,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Allow command-line overrides for quick experiment switching.
    for key in [
        "data_path",
        "forward_ckpt",
        "save_dir",
        "epochs",
        "batch_size",
        "lr",
        "device",
        "lambda_diff",
        "lambda_phys",
        "lambda_bin",
        "cond_drop_prob",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    use_cuda = cfg["device"].startswith("cuda")
    if use_cuda:
        device_name = cfg["device"]
        if device_name == "cuda":
            device_name = "cuda:0"
            cfg["device"] = device_name
        torch.cuda.set_device(device_name)

    os.makedirs(cfg["save_dir"], exist_ok=True)
    run_dir = prepare_run_dir(cfg["save_dir"], "diffusion")
    update_latest_run(cfg["save_dir"], "diffusion", run_dir)
    logger = TrainLogger(
        "diffusion",
        str(run_dir),
        ["epoch", "train_loss", "val_loss", "train_diff", "train_phys", "train_bin", "val_diff", "val_phys", "val_bin"],
    )
    cfg["run_dir"] = str(run_dir)

    if cfg["forward_ckpt"] is None:
        latest_forward_run = resolve_latest_run(cfg["save_dir"], "forward")
        if latest_forward_run is None:
            raise FileNotFoundError("未找到最新 forward 训练目录，请先运行 train_forward.py 或显式传入 --forward_ckpt")
        cfg["forward_ckpt"] = str(latest_forward_run / "forward_best.pt")
    print(f"[Diffusion] run_dir={run_dir}")
    print(f"[Diffusion] using forward_ckpt={cfg['forward_ckpt']}")

    dataset = RCWADataset(cfg["data_path"])
    cond_channels = dataset[0][1].shape[0]

    n_train = int(len(dataset) * cfg["train_ratio"])
    n_val = len(dataset) - n_train
    split_gen = torch.Generator().manual_seed(cfg["split_seed"])
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=split_gen)
    cfg["dataset_size"] = len(dataset)
    cfg["train_size"] = n_train
    cfg["val_size"] = n_val
    cfg["cond_channels"] = cond_channels
    write_json(os.path.join(run_dir, "run_info.json"), cfg)

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=use_cuda
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=use_cuda
    )

    surrogate = ForwardSurrogate(out_ch=cond_channels).to(cfg["device"])
    forward_ckpt = torch.load(cfg["forward_ckpt"], map_location=cfg["device"], weights_only=False)
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
        val_diff = 0.0
        val_phys = 0.0
        val_bin = 0.0
        with torch.no_grad():
            for x01, cond in val_loader:
                x01 = x01.to(cfg["device"])
                cond = cond.to(cfg["device"])
                x0 = x01 * 2.0 - 1.0

                loss, log_dict = diffusion.p_losses(
                    x0=x0,
                    cond=cond,
                    surrogate=surrogate,
                    lambda_diff=cfg["lambda_diff"],
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
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(f"[Diffusion] epoch={epoch:03d} train={train_loss:.4f} val={val_loss:.4f} "
              f"diff={train_diff:.4f} phys={train_phys:.4f} bin={train_bin:.4f} "
              f"val_diff={val_diff:.4f} val_phys={val_phys:.4f} val_bin={val_bin:.4f} lr={current_lr:.2e}")
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
        torch.save(ckpt, os.path.join(run_dir, "diffusion_last.pt"))

        if (epoch + 1) % cfg["preview_every"] == 0:
            preview_cond = val_set[0][1].unsqueeze(0).to(cfg["device"])
            preview = diffusion.sample(preview_cond, cfg_scale=3.0).cpu()
            logger.save_preview(epoch, preview)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(run_dir, "diffusion_best.pt"))
        else:
            stale_epochs += 1

        if stale_epochs >= cfg["early_stop_patience"]:
            print(f"[Diffusion] early stop at epoch={epoch:03d}, best_epoch={best_epoch:03d}, best_val={best_val:.6f}")
            break

    write_json(
        os.path.join(run_dir, "summary.json"),
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "dataset_size": len(dataset),
            "train_size": n_train,
            "val_size": n_val,
            "data_path": cfg["data_path"],
            "forward_ckpt": cfg["forward_ckpt"],
            "lambda_diff": cfg["lambda_diff"],
            "lambda_phys": cfg["lambda_phys"],
            "lambda_bin": cfg["lambda_bin"],
            "cond_drop_prob": cfg["cond_drop_prob"],
            "run_dir": str(run_dir),
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
