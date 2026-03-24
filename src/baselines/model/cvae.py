"""
CVAE (Conditional Variational Autoencoder) for metasurface inverse design.

Architecture:
  Encoder: structure [B,1,64,64] + cond [B,2,11,13] → μ, logσ² [B, latent_dim]
  Decoder: z [B, latent_dim] + cond [B,2,11,13]    → structure [B,1,64,64]

Training loss:
  L = Recon (BCE) + β * KL(q(z|x,c) || N(0,I))

Inference:
  z ~ N(0,I) → Decoder(z, cond) → threshold → binary structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LAMBDA_COUNT = 11
THETA_COUNT = 13


# ── 条件编码器（光谱 → 嵌入向量） ────────────────────────────────────

class CondEmbed(nn.Module):
    """[B,2,11,13] → [B, cond_dim]"""
    def __init__(self, in_ch: int = 2, cond_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                              # [B, 2*11*13=286]
            nn.Linear(in_ch * LAMBDA_COUNT * THETA_COUNT, 512),
            nn.SiLU(),
            nn.Linear(512, cond_dim),
            nn.SiLU(),
        )

    def forward(self, cond):
        return self.net(cond)


# ── Encoder ───────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    结构图 + 条件嵌入 → μ, logvar
    CNN 下采样：64→32→16→8→4，再 flatten + FC
    """
    def __init__(self, cond_dim: int = 256, latent_dim: int = 128, base_ch: int = 32):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, base_ch * 8)  # 注入 bottleneck

        self.enc = nn.Sequential(
            nn.Conv2d(1, base_ch,     4, stride=2, padding=1),   # 32×32
            nn.SiLU(),
            nn.Conv2d(base_ch,     base_ch * 2, 4, stride=2, padding=1),  # 16×16
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),  # 8×8
            nn.GroupNorm(8, base_ch * 4), nn.SiLU(),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),  # 4×4
            nn.GroupNorm(8, base_ch * 8), nn.SiLU(),
        )
        feat_dim = base_ch * 8 * 4 * 4   # 256 * 16 = 4096
        self.fc_mu     = nn.Linear(feat_dim + base_ch * 8, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim + base_ch * 8, latent_dim)

    def forward(self, x, cond_emb):
        h = self.enc(x).flatten(1)          # [B, 4096]
        c = self.cond_proj(cond_emb)        # [B, 256]
        h = torch.cat([h, c], dim=1)        # [B, 4352]
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ───────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    z + 条件嵌入 → 结构图 [B,1,64,64]
    FC → reshape → 转置卷积上采样：4→8→16→32→64
    """
    def __init__(self, cond_dim: int = 256, latent_dim: int = 128, base_ch: int = 32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, base_ch * 8 * 4 * 4),
            nn.SiLU(),
        )
        self.base_ch = base_ch
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1),  # 8×8
            nn.GroupNorm(8, base_ch * 4), nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),  # 16×16
            nn.GroupNorm(8, base_ch * 2), nn.SiLU(),
            nn.ConvTranspose2d(base_ch * 2, base_ch,     4, stride=2, padding=1),  # 32×32
            nn.GroupNorm(8, base_ch), nn.SiLU(),
            nn.ConvTranspose2d(base_ch, 1,               4, stride=2, padding=1),  # 64×64
            nn.Sigmoid(),
        )

    def forward(self, z, cond_emb):
        h = self.fc(torch.cat([z, cond_emb], dim=1))       # [B, 256*16]
        h = h.reshape(h.shape[0], self.base_ch * 8, 4, 4)  # [B, 256, 4, 4]
        return self.dec(h)                                   # [B, 1, 64, 64]


# ── CVAE 主模型 ───────────────────────────────────────────────────────

class CVAE(nn.Module):
    def __init__(self, cond_in_ch: int = 2, latent_dim: int = 128,
                 cond_dim: int = 256, base_ch: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_embed = CondEmbed(cond_in_ch, cond_dim)
        self.encoder    = Encoder(cond_dim, latent_dim, base_ch)
        self.decoder    = Decoder(cond_dim, latent_dim, base_ch)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x, cond):
        """训练时调用，返回重建结果和 KL 参数。"""
        c = self.cond_embed(cond)
        mu, logvar = self.encoder(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, c)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(self, cond, n_samples: int = 16):
        """
        推理：从先验 N(0,I) 采样，解码出 n_samples 个候选结构。
        cond: [1, 2, 11, 13] 单个目标条件
        返回: [n_samples, 1, 64, 64] 二值结构（0/1）
        """
        device = next(self.parameters()).device
        cond_rep = cond.expand(n_samples, -1, -1, -1).to(device)
        c = self.cond_embed(cond_rep)
        z = torch.randn(n_samples, self.latent_dim, device=device)
        out = self.decoder(z, c)                          # [N, 1, 64, 64] in [0,1]
        return (out > 0.5).float()


# ── Loss 函数 ─────────────────────────────────────────────────────────

def cvae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """
    recon:  [B,1,64,64] Sigmoid 输出
    x:      [B,1,64,64] 原始结构 {0,1}
    beta:   KL 权重（beta-VAE）
    """
    recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()
