import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import RCWADataset
from models import ForwardSurrogate, build_conditional_unet
from diffusion import GaussianDiffusion
from parallel_utils import load_state_dict_flexible, maybe_wrap_data_parallel, parse_devices, sanitize_state_dict_keys
from train_utils import TrainLogger, prepare_run_dir, update_latest_run, resolve_latest_run, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional diffusion model for metasurface inverse design.")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training npz file.")
    parser.add_argument("--forward_ckpt", type=str, default=None, help="Path to forward surrogate checkpoint.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save checkpoints/logs.")
    parser.add_argument("--runs_dir", type=str, default=None, help="Directory to save per-run logs/metadata.")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu. Defaults to cuda:0.")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1")
    parser.add_argument("--lambda_diff", type=float, default=None, help="Weight for diffusion denoise loss.")
    parser.add_argument("--lambda_phys", type=float, default=None, help="Weight for physics surrogate loss.")
    parser.add_argument("--lambda_bin", type=float, default=None, help="Weight for binarization loss.")
    parser.add_argument("--cond_drop_prob", type=float, default=None, help="Condition dropout probability for CFG training.")
    parser.add_argument("--unet_arch", type=str, default=None, help="Conditional UNet architecture.")
    parser.add_argument("--base_ch", type=int, default=None, help="UNet base channels.")
    parser.add_argument("--cond_dim", type=int, default=None, help="Condition embedding dimension.")
    parser.add_argument("--time_dim", type=int, default=None, help="Time embedding dimension.")
    parser.add_argument("--phys_start_t", type=int, default=None, help="Only apply physics loss when t <= phys_start_t.")
    parser.add_argument("--phys_bin_mode", type=str, default=None, help="Physics surrogate input mode: ste or soft.")
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
        "runs_dir": "runs",
        "train_ratio": 0.7,
        "num_workers": 4,
        "timesteps": 1000,
        "unet_arch": "cnn_cross_v1",
        "base_ch": 64,
        "cond_dim": 192,
        "time_dim": 256,
        "lambda_diff": 0.6451612903,
        "lambda_phys": 0.3225806452,
        "lambda_bin": 0.0322580645,
        "cond_drop_prob": 0.10,
        "phys_start_t": 300,
        "phys_bin_mode": "soft",
        "preview_every": 20,
        "split_seed": 20260315,
        "grad_clip": 1.0,
        "lr_patience": 6,
        "lr_factor": 0.5,
        "min_lr": 1e-6,
        "early_stop_patience": 15,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "devices": None,
    }

    # Allow command-line overrides for quick experiment switching.
    for key in [
        "data_path",
        "forward_ckpt",
        "save_dir",
        "runs_dir",
        "epochs",
        "batch_size",
        "lr",
        "device",
        "devices",
        "lambda_diff",
        "lambda_phys",
        "lambda_bin",
        "cond_drop_prob",
        "unet_arch",
        "base_ch",
        "cond_dim",
        "time_dim",
        "phys_start_t",
        "phys_bin_mode",
    ]:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value

    devices = parse_devices(cfg["devices"], cfg["device"], default_to_all_cuda=False)
    cfg["devices"] = devices
    cfg["device"] = devices[0]
    use_cuda = cfg["device"].startswith("cuda")
    if use_cuda:
        torch.cuda.set_device(cfg["device"])

    os.makedirs(cfg["save_dir"], exist_ok=True)
    os.makedirs(cfg["runs_dir"], exist_ok=True)
    run_dir = prepare_run_dir(cfg["runs_dir"], "diffusion")
    update_latest_run(cfg["runs_dir"], "diffusion", run_dir)
    logger = TrainLogger(
        "diffusion",
        str(run_dir),
        ["epoch", "train_loss", "val_loss", "train_diff", "train_phys", "train_bin", "val_diff", "val_phys", "val_bin"],
    )
    cfg["run_dir"] = str(run_dir)

    if cfg["forward_ckpt"] is None:
        default_forward_ckpt = os.path.join(cfg["save_dir"], "forward_best.pt")
        if not os.path.exists(default_forward_ckpt):
            raise FileNotFoundError("未找到 checkpoints/forward_best.pt，请先运行 train_forward.py 或显式传入 --forward_ckpt")
        cfg["forward_ckpt"] = default_forward_ckpt
    print(f"[Diffusion] run_dir={run_dir}")
    print(f"[Diffusion] using forward_ckpt={cfg['forward_ckpt']}")
    print(f"[Diffusion] devices={devices} data_parallel={'yes' if len(devices) > 1 and use_cuda else 'no'}")
    print(
        f"[Diffusion] arch={cfg['unet_arch']} base_ch={cfg['base_ch']} cond_dim={cfg['cond_dim']} "
        f"phys_start_t={cfg['phys_start_t']} phys_bin_mode={cfg['phys_bin_mode']}"
    )

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
    surrogate = maybe_wrap_data_parallel(surrogate, devices)

    unet = build_conditional_unet(cond_channels, cfg).to(cfg["device"])
    unet = maybe_wrap_data_parallel(unet, devices)
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
                phys_start_t=cfg["phys_start_t"],
                phys_bin_mode=cfg["phys_bin_mode"],
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
                    phys_start_t=cfg["phys_start_t"],
                    phys_bin_mode=cfg["phys_bin_mode"],
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
            "diffusion": sanitize_state_dict_keys(diffusion.state_dict()),
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
            "unet_arch": cfg["unet_arch"],
            "base_ch": cfg["base_ch"],
            "cond_dim": cfg["cond_dim"],
            "time_dim": cfg["time_dim"],
            "phys_start_t": cfg["phys_start_t"],
            "phys_bin_mode": cfg["phys_bin_mode"],
            "run_dir": str(run_dir),
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
