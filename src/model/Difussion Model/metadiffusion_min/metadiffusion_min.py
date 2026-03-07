from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 32
COND = 55
T = 500


def timestep_emb(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half - 1, 1))
    x = t.float()[:, None] * freq[None, :]
    emb = torch.cat([x.sin(), x.cos()], dim=1)
    return emb if dim % 2 == 0 else F.pad(emb, (0, 1))


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64))
        self.c_mlp = nn.Sequential(nn.Linear(COND, 64), nn.GELU(), nn.Linear(64, 64))
        self.in_conv = nn.Conv2d(1, 64, 3, 1, 1)
        self.down = nn.Conv2d(64, 128, 4, 2, 1)
        self.mid = nn.Conv2d(128, 128, 3, 1, 1)
        self.up = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.out = nn.Conv2d(64, 1, 3, 1, 1)
        self.to_map = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        e = torch.cat([self.t_mlp(timestep_emb(t, 64)), self.c_mlp(c)], dim=1)
        e = self.to_map(e).unsqueeze(-1).unsqueeze(-1)
        h = F.gelu(self.in_conv(x))
        d = F.gelu(self.down(h))
        m = F.gelu(self.mid(d))
        u = F.gelu(self.up(m) + e)
        return self.out(u + h)


def fake_batch(bs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(bs, 1, IMG, IMG, device=device)
    y, xg = torch.meshgrid(
        torch.linspace(-1, 1, IMG, device=device),
        torch.linspace(-1, 1, IMG, device=device),
        indexing="ij",
    )
    for i in range(bs):
        cx, cy = torch.empty(2, device=device).uniform_(-0.4, 0.4)
        r = torch.empty(1, device=device).uniform_(0.2, 0.6)
        p = (((xg - cx) ** 2 + (y - cy) ** 2) < r**2).float()
        x[i, 0] = p
    c = torch.randn(bs, COND, device=device)
    return x, c


def coeffs(device: torch.device):
    beta = torch.linspace(1e-4, 2e-2, T, device=device)
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, dim=0)
    abar_prev = torch.cat([torch.ones(1, device=device), abar[:-1]], dim=0)
    return beta, alpha, abar, abar_prev


def take(v: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return v.gather(0, t).view(-1, *([1] * (len(shape) - 1)))


def q_sample(x0: torch.Tensor, t: torch.Tensor, abar: torch.Tensor):
    eps = torch.randn_like(x0)
    xt = take(abar.sqrt(), t, x0.shape) * x0 + take((1 - abar).sqrt(), t, x0.shape) * eps
    return xt, eps


def train(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = TinyUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    beta, alpha, abar, abar_prev = coeffs(device)
    global_step = 0
    for ep in range(1, args.epochs + 1):
        loss_sum = 0.0
        model.train()
        for _ in range(args.steps):
            x0, c = fake_batch(args.batch, device)
            if args.cfg_drop > 0:
                keep = (torch.rand(args.batch, device=device) > args.cfg_drop).float().unsqueeze(1)
                c = c * keep
            t = torch.randint(0, T, (args.batch,), device=device)
            xt, eps = q_sample(x0, t, abar)
            pred = model(xt, t, c)
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            loss_sum += loss.item()
        ckpt = {"model": model.state_dict(), "epoch": ep}
        torch.save(ckpt, outdir / f"ckpt_{ep:03d}.pt")
        print(f"epoch={ep} loss={loss_sum / args.steps:.6f} saved={outdir / f'ckpt_{ep:03d}.pt'}")


@torch.no_grad()
def sample(args):
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model = TinyUNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    beta, alpha, abar, abar_prev = coeffs(device)
    c = torch.randn(args.num, COND, device=device)
    x = torch.randn(args.num, 1, IMG, IMG, device=device)
    for ti in reversed(range(T)):
        t = torch.full((args.num,), ti, device=device, dtype=torch.long)
        eps_c = model(x, t, c)
        eps_u = model(x, t, torch.zeros_like(c))
        eps = (1 + args.guidance) * eps_c - args.guidance * eps_u
        at = take(alpha, t, x.shape)
        bt = take(beta, t, x.shape)
        abt = take(abar, t, x.shape)
        abpt = take(abar_prev, t, x.shape)
        mean = (1.0 / at.sqrt()) * (x - ((1.0 - at) / (1.0 - abt + 1e-8).sqrt()) * eps)
        noise = torch.zeros_like(x) if ti == 0 else torch.randn_like(x)
        sigma = (((1.0 - abpt) / (1.0 - abt + 1e-8)) * bt).sqrt()
        x = mean + sigma * noise
    x = (torch.sigmoid(x) > args.thresh).float().cpu().numpy()
    for i in range(args.num):
        np.savetxt(outdir / f"sample_{i:03d}.txt", x[i, 0], fmt="%.0f")
    print(f"saved {args.num} samples -> {outdir}")


def main():
    p = argparse.ArgumentParser(description="Minimal single-file MetaDiffusion")
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--steps", type=int, default=100)
    pt.add_argument("--batch", type=int, default=64)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--cfg-drop", type=float, default=0.1)
    pt.add_argument("--device", type=str, default="cuda")
    pt.add_argument("--outdir", type=str, default="outputs")

    pi = sub.add_parser("infer")
    pi.add_argument("--ckpt", type=str, required=True)
    pi.add_argument("--num", type=int, default=4)
    pi.add_argument("--guidance", type=float, default=6.0)
    pi.add_argument("--thresh", type=float, default=0.5)
    pi.add_argument("--device", type=str, default="cuda")
    pi.add_argument("--outdir", type=str, default="samples")

    args = p.parse_args()
    train(args) if args.mode == "train" else sample(args)


if __name__ == "__main__":
    main()
