"""
训练 cGAN。

用法：
  cd /data/GraduationProject
  python src/baselines/train/train_cgan.py \
      --data_path data/train_data.npz \
      --save_dir  checkpoints/cgan
"""

import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_ROOT, "src", "model"))
sys.path.insert(0, os.path.join(_ROOT, "src", "baselines", "model"))

from dataset import RCWADataset
from cgan import Generator, Discriminator, discriminator_loss, generator_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",   default="data/train_data.npz")
    parser.add_argument("--save_dir",    default="checkpoints/cgan")
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr_g",        type=float, default=1e-4)
    parser.add_argument("--lr_d",        type=float, default=4e-4,
                        help="D 学习率通常比 G 大")
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--lambda_l1",   type=float, default=10.0)
    parser.add_argument("--n_critic",    type=int,   default=2,
                        help="每训练 1 次 G，训练 D 的次数")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="与 run_eval.py 保持一致，确保测试集无泄漏")
    parser.add_argument("--seed",        type=int,   default=20260315)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # ── 数据 ─────────────────────────────────────────────────────────
    dataset = RCWADataset(args.data_path)
    n_train = int(len(dataset) * args.train_ratio)
    n_val   = len(dataset) - n_train
    gen     = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # ── 模型 ─────────────────────────────────────────────────────────
    cond_ch = dataset[0][1].shape[0]
    G = Generator(latent_dim=args.latent_dim, cond_dim=256).to(device)
    D = Discriminator(cond_dim=256).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.999))

    best_g_loss = 1e9
    print(f"[cGAN] device={device}  train={n_train}  val={n_val}")
    print(f"[cGAN] G params={sum(p.numel() for p in G.parameters()):,}  "
          f"D params={sum(p.numel() for p in D.parameters()):,}")

    for epoch in range(args.epochs):
        G.train(); D.train()
        ep_d = ep_g = ep_adv = ep_l1 = 0.0
        n_batches = 0

        for x, cond in train_loader:
            x, cond = x.to(device), cond.to(device)
            b = x.size(0)

            # ── 训练 D（n_critic 次）────────────────────────────
            for _ in range(args.n_critic):
                z       = torch.randn(b, args.latent_dim, device=device)
                fake    = G(z, cond).detach()
                d_real  = D(x, cond)
                d_fake  = D(fake, cond)
                loss_d  = discriminator_loss(d_real, d_fake)

                opt_D.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()

            # ── 训练 G（1 次）────────────────────────────────────
            z    = torch.randn(b, args.latent_dim, device=device)
            fake = G(z, cond)
            d_fake_g = D(fake, cond)
            loss_g, l_adv, l_l1 = generator_loss(
                d_fake_g, fake, x, lambda_l1=args.lambda_l1)

            opt_G.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            opt_G.step()

            ep_d   += loss_d.item()
            ep_g   += loss_g.item()
            ep_adv += l_adv
            ep_l1  += l_l1
            n_batches += 1

        ep_d   /= n_batches
        ep_g   /= n_batches
        ep_adv /= n_batches
        ep_l1  /= n_batches

        print(f"[cGAN] epoch={epoch:03d}  "
              f"D={ep_d:.4f}  G={ep_g:.4f} "
              f"(adv={ep_adv:.4f} l1={ep_l1:.4f})")

        # ── 保存（用 G_loss 作为代理指标） ────────────────────────
        ckpt = {
            "G"          : G.state_dict(),
            "D"          : D.state_dict(),
            "cond_ch"    : cond_ch,
            "latent_dim" : args.latent_dim,
            "cond_mean"  : dataset.cond_mean,
            "cond_std"   : dataset.cond_std,
            "epoch"      : epoch,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "cgan_last.pt"))

        if ep_g < best_g_loss:
            best_g_loss = ep_g
            torch.save(ckpt, os.path.join(args.save_dir, "cgan_best.pt"))

    print(f"[cGAN] done.  best_g_loss={best_g_loss:.4f}")


if __name__ == "__main__":
    main()
