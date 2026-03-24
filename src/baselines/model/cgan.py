"""
cGAN (Conditional GAN) for metasurface inverse design.

Architecture:
  Generator:     z [B,128] + cond [B,2,11,13] → structure [B,1,64,64]
  Discriminator: structure [B,1,64,64] + cond [B,2,11,13] → real/fake prob [B,1]

Training:
  Loss_D = BCE(D(x_real, c), 1) + BCE(D(G(z,c), c), 0)
  Loss_G = BCE(D(G(z,c), c), 1) + λ_l1 * L1(G(z,c), x_real)

Inference:
  z ~ N(0,I) → G(z, cond) → threshold → binary structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_COUNT = 11
THETA_COUNT = 13


# ── 条件嵌入（生成器和判别器共用） ────────────────────────────────────

class CondEmbed(nn.Module):
    """[B,2,11,13] → [B, cond_dim]"""
    def __init__(self, in_ch: int = 2, cond_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * LAMBDA_COUNT * THETA_COUNT, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, cond_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, cond):
        return self.net(cond)


# ── Generator ─────────────────────────────────────────────────────────

class Generator(nn.Module):
    """
    z + cond → structure [B,1,64,64]
    FC → reshape [B,256,4,4] → 转置卷积上采样
    """
    def __init__(self, latent_dim: int = 128, cond_dim: int = 256, base_ch: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_embed = CondEmbed(cond_dim=cond_dim)
        self.base_ch = base_ch

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, base_ch * 8 * 4 * 4),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            # 4×4 → 8×8
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(),
            # 8×8 → 16×16
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(),
            # 16×16 → 32×32
            nn.ConvTranspose2d(base_ch * 2, base_ch,     4, stride=2, padding=1),
            nn.BatchNorm2d(base_ch), nn.ReLU(),
            # 32×32 → 64×64
            nn.ConvTranspose2d(base_ch, 1,               4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, cond):
        c = self.cond_embed(cond)
        h = self.fc(torch.cat([z, c], dim=1))
        h = h.reshape(h.shape[0], self.base_ch * 8, 4, 4)
        return self.dec(h)   # [B, 1, 64, 64] in [0,1]

    @torch.no_grad()
    def sample(self, cond, n_samples: int = 16):
        """
        推理：从 N(0,I) 采样 n_samples 个候选结构。
        cond: [1, 2, 11, 13]
        返回: [n_samples, 1, 64, 64] 二值结构
        """
        device = next(self.parameters()).device
        cond_rep = cond.expand(n_samples, -1, -1, -1).to(device)
        z = torch.randn(n_samples, self.latent_dim, device=device)
        out = self.forward(z, cond_rep)
        return (out > 0.5).float()


# ── Discriminator ─────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """
    structure + cond → real/fake score [B, 1]
    条件通过 projection 方式注入：score = E(x)·c_emb + bias
    这比直接 concat 更稳定（参考 cGAN-projection, Miyato 2018）
    """
    def __init__(self, cond_dim: int = 256, base_ch: int = 32):
        super().__init__()
        self.cond_embed = CondEmbed(cond_dim=cond_dim)

        self.enc = nn.Sequential(
            # 64×64 → 32×32
            nn.Conv2d(1, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32×32 → 16×16
            nn.Conv2d(base_ch,     base_ch * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2), nn.LeakyReLU(0.2),
            # 16×16 → 8×8
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 4), nn.LeakyReLU(0.2),
            # 8×8 → 4×4
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 8), nn.LeakyReLU(0.2),
        )
        feat_dim = base_ch * 8 * 4 * 4   # 4096

        # unconditional score
        self.fc_uncond = nn.Linear(feat_dim, 1)
        # conditional projection
        self.fc_proj   = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        h    = self.enc(x).flatten(1)            # [B, 4096]
        c    = self.cond_embed(cond)             # [B, 256]
        proj = (h * self.fc_proj(c)).sum(dim=1, keepdim=True)  # [B,1]
        return self.fc_uncond(h) + proj          # [B, 1]


# ── Loss 函数 ─────────────────────────────────────────────────────────

def discriminator_loss(d_real, d_fake):
    """Hinge loss，比 BCE 更稳定。"""
    loss_real = F.relu(1.0 - d_real).mean()
    loss_fake = F.relu(1.0 + d_fake).mean()
    return loss_real + loss_fake


def generator_loss(d_fake, fake, real, lambda_l1: float = 10.0):
    """Hinge adversarial loss + L1 重建。"""
    loss_adv = -d_fake.mean()
    loss_l1  = F.l1_loss(fake, real)
    return loss_adv + lambda_l1 * loss_l1, loss_adv.item(), loss_l1.item()
