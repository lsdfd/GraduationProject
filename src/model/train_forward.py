import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import RCWADataset
from models import ForwardSurrogate
from train_utils import TrainLogger, prepare_run_dir, update_latest_run, write_json


def augment_structure(x: torch.Tensor) -> torch.Tensor:
    """C4 对称结构随机翻转增强：水平/垂直翻转不改变光学响应"""
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-1])   # 水平翻转
    if torch.rand(1).item() > 0.5:
        x = torch.flip(x, dims=[-2])   # 垂直翻转
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/train_data.npz")
    parser.add_argument("--save_dir", default="checkpoints")
    args = parser.parse_args()

    cfg = {
        "data_path": args.data_path,
        "batch_size": 64,
        "epochs": 150,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "save_dir": args.save_dir,
        "train_ratio": 0.9,
        "num_workers": 4,
        "split_seed": 20260315,
        "grad_clip": 1.0,
        "lr_patience": 10,
        "lr_factor": 0.5,
        "min_lr": 1e-6,
        "early_stop_patience": 30,
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
    run_dir = prepare_run_dir(cfg["save_dir"], "forward")
    update_latest_run(cfg["save_dir"], "forward", run_dir)
    logger = TrainLogger("forward", str(run_dir), ["epoch", "train_loss", "val_loss"])
    cfg["run_dir"] = str(run_dir)
    print(f"[Forward] run_dir={run_dir}")

    dataset = RCWADataset(cfg["data_path"])
    cond_channels = dataset[0][1].shape[0]

    # 用于反归一化，计算物理单位下的误差
    cond_mean_t = torch.from_numpy(dataset.cond_mean).float()  # [1,2,11,17]
    cond_std_t  = torch.from_numpy(dataset.cond_std).float()   # [1,2,11,17]
    mean_dev = cond_mean_t.to(cfg["device"])
    std_dev  = cond_std_t.to(cfg["device"])

    np.savez(
        os.path.join(run_dir, "cond_stats.npz"),
        mean=dataset.cond_mean,
        std=dataset.cond_std
    )

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
    best_val_phys = 1e9
    best_epoch = -1
    stale_epochs = 0
    print(f"[Forward] train device={cfg['device']} epochs={cfg['epochs']} batch_size={cfg['batch_size']}")

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0
        train_mae_phys = 0.0

        for x, cond in train_loader:
            x = x.to(cfg["device"])       # [0,1]
            cond = cond.to(cfg["device"]) # normalized

            x = augment_structure(x)      # 随机翻转增强

            pred = model(x)
            loss = F.l1_loss(pred, cond)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()

            train_loss += loss.item() * x.size(0)
            with torch.no_grad():
                pred_phys = pred * std_dev + mean_dev
                cond_phys = cond * std_dev + mean_dev
                train_mae_phys += F.l1_loss(pred_phys, cond_phys).item() * x.size(0)

        train_loss     /= len(train_loader.dataset)
        train_mae_phys /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_mae_phys = 0.0
        with torch.no_grad():
            for x, cond in val_loader:
                x = x.to(cfg["device"])
                cond = cond.to(cfg["device"])
                pred = model(x)
                loss = F.l1_loss(pred, cond)
                val_loss += loss.item() * x.size(0)
                # 反归一化到物理空间 [0,1]，计算每格点平均绝对误差
                pred_phys = pred * std_dev + mean_dev
                cond_phys = cond * std_dev + mean_dev
                val_mae_phys += F.l1_loss(pred_phys, cond_phys).item() * x.size(0)
        val_loss     /= len(val_loader.dataset)
        val_mae_phys /= len(val_loader.dataset)
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]

        print(f"[Forward] epoch={epoch:03d} train_mae={train_mae_phys:.4f} val_mae={val_mae_phys:.4f} lr={current_lr:.2e}")
        logger.log_scalars(
            epoch,
            [epoch, train_mae_phys, val_mae_phys],
            {"mae_phys/train": train_mae_phys, "mae_phys/val": val_mae_phys, "lr": current_lr},
        )

        ckpt = {
            "model": model.state_dict(),
            "cond_channels": cond_channels,
            "epoch": epoch,
            "best_val": best_val,
            "cfg": cfg,
        }
        torch.save(ckpt, os.path.join(run_dir, "forward_last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            best_val_phys = val_mae_phys
            best_epoch = epoch
            stale_epochs = 0
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(run_dir, "forward_best.pt"))
        else:
            stale_epochs += 1

        if stale_epochs >= cfg["early_stop_patience"]:
            print(f"[Forward] early stop at epoch={epoch:03d}, best_epoch={best_epoch:03d}, best_mae_phys={best_val_phys:.4f}")
            break

    write_json(
        os.path.join(run_dir, "summary.json"),
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "best_mae_phys": best_val_phys,
            "dataset_size": len(dataset),
            "train_size": n_train,
            "val_size": n_val,
            "data_path": cfg["data_path"],
            "run_dir": str(run_dir),
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
