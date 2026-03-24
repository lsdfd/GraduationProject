import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch, max_groups=8):
    for g in (max_groups, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb_scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb_scale)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConvNormAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(_groups(out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        hidden = max(ch // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_groups(out_ch), out_ch),
        )
        self.se = SqueezeExcite(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride) if (in_ch != out_ch or stride != 1) else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.se(h)
        return F.silu(h + self.skip(x))


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax((q @ k) / math.sqrt(c), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ConditionEncoder1D(nn.Module):
    """
    输入:  [B,17]
    输出:  [B,cond_dim]

    only-tpp 版本把条件视为一条一维角度响应曲线，不再做 token 化和 cross-attn。
    """

    def __init__(self, cond_dim=256):
        super().__init__()
        hidden = cond_dim // 2
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, cond):
        h = self.net(cond.unsqueeze(1))
        h = h.mean(dim=-1)
        return self.out(h)


class ForwardSurrogate(nn.Module):
    """
    输入:  [B,1,64,64]，值域 0~1
    输出:  [B,17]，归一化 tpp 角度响应
    """

    def __init__(self, out_dim=17):
        super().__init__()
        base_ch = 32
        self.encoder = nn.Sequential(
            ConvNormAct(1, base_ch, 3),
            ResidualConvBlock(base_ch, base_ch),
            ResidualConvBlock(base_ch, base_ch * 2, stride=2),
            ResidualConvBlock(base_ch * 2, base_ch * 2),
            ResidualConvBlock(base_ch * 2, base_ch * 4, stride=2),
            ResidualConvBlock(base_ch * 4, base_ch * 4),
            ResidualConvBlock(base_ch * 4, base_ch * 8, stride=2),
            ResidualConvBlock(base_ch * 8, base_ch * 8),
            AttentionBlock(base_ch * 8),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch * 8, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.head(self.encoder(x))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim + cond_dim, out_ch * 2),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb_proj(torch.cat([t_emb, c_emb], dim=1)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """
    输入:
      x_t:   [B,1,64,64]
      cond:  [B,17]
      t:     [B]
    输出:
      v_pred [B,1,64,64]

    only-tpp 版本只用一维条件全局嵌入，通过 AdaGN/FiLM 注入。
    """

    def __init__(self, cond_dim=256, base_ch=64, time_dim=256):
        super().__init__()
        self.cond_encoder = ConditionEncoder1D(cond_dim=cond_dim)
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = ConvNormAct(1, base_ch, 3)

        self.res1 = ResBlock(base_ch, base_ch, time_dim, cond_dim)
        self.down1 = Downsample(base_ch)

        self.res2 = ResBlock(base_ch, base_ch * 2, time_dim, cond_dim)
        self.down2 = Downsample(base_ch * 2)

        self.res3 = ResBlock(base_ch * 2, base_ch * 4, time_dim, cond_dim)
        self.attn3 = AttentionBlock(base_ch * 4)
        self.down3 = Downsample(base_ch * 4)

        self.res4 = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)

        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)
        self.mid_attn = AttentionBlock(base_ch * 4)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)

        self.up1 = Upsample(base_ch * 4)
        self.up_res1 = ResBlock(base_ch * 8, base_ch * 4, time_dim, cond_dim)

        self.up2 = Upsample(base_ch * 4)
        self.up_res2 = ResBlock(base_ch * 6, base_ch * 2, time_dim, cond_dim)

        self.up3 = Upsample(base_ch * 2)
        self.up_res3 = ResBlock(base_ch * 3, base_ch, time_dim, cond_dim)

        self.out_norm = nn.GroupNorm(_groups(base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def encode_cond(self, cond, batch_size, device, cond_drop_prob=0.0, force_uncond=False):
        if force_uncond or cond is None:
            return self.null_cond.expand(batch_size, -1)

        cond_emb = self.cond_encoder(cond)
        if self.training and cond_drop_prob > 0:
            drop_mask = torch.rand(batch_size, device=device) < cond_drop_prob
            if drop_mask.any():
                cond_emb = cond_emb.clone()
                cond_emb[drop_mask] = self.null_cond.expand(drop_mask.sum(), -1)
        return cond_emb

    def forward(self, x, t, cond=None, cond_drop_prob=0.0, force_uncond=False):
        b = x.shape[0]
        t_emb = self.time_mlp(t)
        c_emb = self.encode_cond(cond, b, x.device, cond_drop_prob, force_uncond)

        x0 = self.in_conv(x)
        x1 = self.res1(x0, t_emb, c_emb)
        x2 = self.res2(self.down1(x1), t_emb, c_emb)
        x3 = self.attn3(self.res3(self.down2(x2), t_emb, c_emb))
        x4 = self.res4(self.down3(x3), t_emb, c_emb)

        h = self.mid1(x4, t_emb, c_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb, c_emb)

        h = self.up1(h)
        h = self.up_res1(torch.cat([h, x3], dim=1), t_emb, c_emb)

        h = self.up2(h)
        h = self.up_res2(torch.cat([h, x2], dim=1), t_emb, c_emb)

        h = self.up3(h)
        h = self.up_res3(torch.cat([h, x1], dim=1), t_emb, c_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
