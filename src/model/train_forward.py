import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import RCWADataset
from models import ForwardSurrogate
from train_utils import TrainLogger


def main():
    cfg = {
        "data_path": "data/train_data.npz",
        "batch_size": 32,
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-3,
        "save_dir": "checkpoints",
        "train_ratio": 0.9,
        "num_workers": 4,
        "split_seed": 20260315,
        "grad_clip": 1.0,
        "lr_patience": 10,
        "lr_factor": 0.5,
        "min_lr": 1e-6,
        "early_stop_patience": 24,
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
    logger = TrainLogger("forward", cfg["save_dir"], ["epoch", "train_loss", "val_loss"])

    dataset = RCWADataset(cfg["data_path"])
    cond_channels = dataset[0][1].shape[0]

    np.savez(
        os.path.join(cfg["save_dir"], "cond_stats.npz"),
        mean=dataset.cond_mean,
        std=dataset.cond_std
    )

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

    model = ForwardSurrogate(out_ch=cond_channels).to(cfg["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
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
    print(f"[Forward] train device={cfg['device']} epochs={cfg['epochs']} batch_size={cfg['batch_size']}")

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0

        for x, cond in train_loader:
            x = x.to(cfg["device"])       # [0,1]
            cond = cond.to(cfg["device"]) # normalized

            pred = model(x)
            loss = F.l1_loss(pred, cond)  # 改用 L1 loss，对异常值更鲁棒

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, cond in val_loader:
                x = x.to(cfg["device"])
                cond = cond.to(cfg["device"])
                pred = model(x)
                loss = F.l1_loss(pred, cond)  # 验证集也用 L1 loss
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(f"[Forward] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} lr={current_lr:.2e}")
        logger.log_scalars(
            epoch,
            [epoch, train_loss, val_loss],
            {"loss/train": train_loss, "loss/val": val_loss, "lr": current_lr},
        )

        ckpt = {
            "model": model.state_dict(),
            "cond_channels": cond_channels,
            "epoch": epoch,
            "best_val": best_val,
            "cfg": cfg,
        }
        torch.save(ckpt, os.path.join(cfg["save_dir"], "forward_last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale_epochs = 0
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(cfg["save_dir"], "forward_best.pt"))
        else:
            stale_epochs += 1

        if stale_epochs >= cfg["early_stop_patience"]:
            print(f"[Forward] early stop at epoch={epoch:03d}, best_epoch={best_epoch:03d}, best_val={best_val:.6f}")
            break

    logger.close()


if __name__ == "__main__":
    main()
