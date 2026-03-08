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
        "batch_size": 64,
        "epochs": 300,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "save_dir": "checkpoints",
        "train_ratio": 0.9,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

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
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
    )

    model = ForwardSurrogate(out_ch=cond_channels).to(cfg["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = 1e9

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0

        for x, cond in train_loader:
            x = x.to(cfg["device"])       # [0,1]
            cond = cond.to(cfg["device"]) # normalized

            pred = model(x)
            loss = F.l1_loss(pred, cond)

            opt.zero_grad()
            loss.backward()
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
                loss = F.l1_loss(pred, cond)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"[Forward] epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")
        logger.log_scalars(epoch, [epoch, train_loss, val_loss], {"loss/train": train_loss, "loss/val": val_loss})

        ckpt = {
            "model": model.state_dict(),
            "cond_channels": cond_channels,
        }
        torch.save(ckpt, os.path.join(cfg["save_dir"], "forward_last.pt"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(cfg["save_dir"], "forward_best.pt"))

    logger.close()


if __name__ == "__main__":
    main()
