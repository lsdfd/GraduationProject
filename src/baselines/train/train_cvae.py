"""
训练 CVAE。

用法：
  cd /data/GraduationProject
  python src/baselines/train/train_cvae.py \
      --data_path data/train_data.npz \
      --save_dir  checkpoints/cvae
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# ── 路径 ──────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src", "model"))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines", "model"))

from dataset import RCWADataset
from cvae import CVAE, cvae_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="data/train_data.npz")
    parser.add_argument("--save_dir",   default="checkpoints/cvae")
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int,   default=128)
    parser.add_argument("--beta",       type=float, default=1.0,
                        help="KL 权重，beta-VAE")
    parser.add_argument("--train_ratio",type=float, default=0.7,
                        help="与 run_eval.py 保持一致，确保测试集无泄漏")
    parser.add_argument("--seed",       type=int,   default=20260315)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 数据 ─────────────────────────────────────────────────────────
    dataset  = RCWADataset(args.data_path)
    n_train  = int(len(dataset) * args.train_ratio)
    n_val    = len(dataset) - n_train
    gen      = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ── 模型 ─────────────────────────────────────────────────────────
    cond_ch = dataset[0][1].shape[0]
    model   = CVAE(cond_in_ch=cond_ch, latent_dim=args.latent_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    best_val   = 1e9
    best_epoch = -1
    stale      = 0
    patience   = 20

    print(f"[CVAE] device={device}  train={n_train}  val={n_val}")

    for epoch in range(args.epochs):
        # ── 训练 ──────────────────────────────────────────────────
        model.train()
        tr_loss = tr_recon = tr_kl = 0.0
        for x, cond in train_loader:
            x, cond = x.to(device), cond.to(device)
            recon, mu, logvar = model(x, cond)
            loss, r, kl = cvae_loss(recon, x, mu, logvar, beta=args.beta)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            n = x.size(0)
            tr_loss  += loss.item() * n
            tr_recon += r * n
            tr_kl    += kl * n

        tr_loss  /= n_train
        tr_recon /= n_train
        tr_kl    /= n_train

        # ── 验证 ──────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, cond in val_loader:
                x, cond = x.to(device), cond.to(device)
                recon, mu, logvar = model(x, cond)
                loss, _, _ = cvae_loss(recon, x, mu, logvar, beta=args.beta)
                val_loss += loss.item() * x.size(0)
        val_loss /= n_val

        scheduler.step(val_loss)
        lr = opt.param_groups[0]["lr"]

        print(f"[CVAE] epoch={epoch:03d}  "
              f"train={tr_loss:.4f} (recon={tr_recon:.4f} kl={tr_kl:.4f})  "
              f"val={val_loss:.4f}  lr={lr:.2e}")

        # ── 保存 ──────────────────────────────────────────────────
        ckpt = {
            "model"      : model.state_dict(),
            "cond_ch"    : cond_ch,
            "latent_dim" : args.latent_dim,
            "cond_mean"  : dataset.cond_mean,
            "cond_std"   : dataset.cond_std,
            "epoch"      : epoch,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "cvae_last.pt"))

        if val_loss < best_val:
            best_val   = val_loss
            best_epoch = epoch
            stale      = 0
            torch.save(ckpt, os.path.join(args.save_dir, "cvae_best.pt"))
        else:
            stale += 1

        if stale >= patience:
            print(f"[CVAE] early stop at epoch={epoch}, best={best_epoch}, "
                  f"best_val={best_val:.4f}")
            break

    print(f"[CVAE] done. best_epoch={best_epoch}  best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
