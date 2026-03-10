import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch, max_groups=8):
    for g in (max_groups, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _coord_grid(height, width):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height),
        torch.linspace(-1.0, 1.0, steps=width),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0)


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


class SpectrumCrossAttention(nn.Module):
    def __init__(self, feat_ch, cond_ch, heads=4, head_dim=32):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.norm = nn.GroupNorm(_groups(feat_ch), feat_ch)
        self.to_q = nn.Conv2d(feat_ch, inner, 1)
        self.to_k = nn.Linear(cond_ch, inner)
        self.to_v = nn.Linear(cond_ch, inner)
        self.to_out = nn.Conv2d(inner, feat_ch, 1)
        self.scale = head_dim ** -0.5

    def forward(self, x, cond_tokens):
        b, _, h, w = x.shape
        q = self.to_q(self.norm(x)).reshape(b, self.heads, self.head_dim, h * w).transpose(-1, -2)
        k = self.to_k(cond_tokens).reshape(b, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(cond_tokens).reshape(b, -1, self.heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(-1, -2).reshape(b, -1, h, w)
        return x + self.to_out(out)


class ConditionEncoder2D(nn.Module):
    """
    输入:
      cond [B,C,11,17]
    输出:
      cond_emb [B,emb_dim]
    """

    def __init__(self, in_ch, emb_dim=256, base_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_ch + 2, base_ch, 3),
            ResidualConvBlock(base_ch, base_ch),
        )
        self.down1 = ResidualConvBlock(base_ch, base_ch * 2, stride=2)   # 11x17 -> 6x9
        self.down2 = ResidualConvBlock(base_ch * 2, base_ch * 4, stride=2)  # 6x9 -> 3x5
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.token_proj = nn.Linear(base_ch * 4, emb_dim)
        self.register_buffer("coord_11x17", _coord_grid(11, 17), persistent=False)

    def encode(self, cond):
        coord = self.coord_11x17[None].to(cond.dtype).expand(cond.shape[0], -1, -1, -1)
        x0 = self.stem(torch.cat([cond, coord], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        emb = self.fc(self.pool(x2))
        tokens = self.token_proj(x2.flatten(2).transpose(1, 2))
        return emb, {"11x17": x0, "6x9": x1, "3x5": x2}, tokens

    def forward(self, cond):
        emb, _, _ = self.encode(cond)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim, cond_ch=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim + cond_dim, out_ch * 2),
        )
        self.cond_proj = nn.Conv2d(cond_ch, out_ch, 1) if cond_ch is not None else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb, cond_map=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.cond_proj is not None and cond_map is not None:
            cond_resized = F.interpolate(cond_map, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + self.cond_proj(cond_resized)

        scale, shift = self.emb_proj(torch.cat([t_emb, c_emb], dim=1)).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


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


class ForwardSurrogate(nn.Module):
    """
    输入:  [B,1,64,64]，值域 0~1
    输出:  [B,C,11,17]
    """

    def __init__(self, out_ch):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(1 + 2, 32, 3),
            ResidualConvBlock(32, 32),
        )
        self.enc1 = ResidualConvBlock(32, 64, stride=2)
        self.enc2 = ResidualConvBlock(64, 128, stride=2)
        self.enc3 = ResidualConvBlock(128, 256, stride=2)
        self.latent = nn.Sequential(
            ResidualConvBlock(256, 256),
            ResidualConvBlock(256, 256),
        )
        self.global_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.fuse = nn.Sequential(
            ConvNormAct(32 + 64 + 128 + 256 + 128 + 2, 192, 3),
            ResidualConvBlock(192, 192),
            ResidualConvBlock(192, 128),
        )
        self.head = nn.Sequential(
            ConvNormAct(128, 128, 3),
            nn.Conv2d(128, out_ch, 1),
        )
        self.register_buffer("coord_64", _coord_grid(64, 64), persistent=False)
        self.register_buffer("coord_spec", _coord_grid(11, 17), persistent=False)

    def forward(self, x):
        coord = self.coord_64[None].to(x.dtype).expand(x.shape[0], -1, -1, -1)
        x0 = self.stem(torch.cat([x, coord], dim=1))
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.latent(self.enc3(x2))

        target_size = (11, 17)
        global_feat = self.global_mlp(x3)[:, :, None, None].expand(-1, -1, *target_size)
        feat = torch.cat(
            [
                F.interpolate(x0, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(x1, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(x2, size=target_size, mode="bilinear", align_corners=False),
                F.interpolate(x3, size=target_size, mode="bilinear", align_corners=False),
                global_feat,
                self.coord_spec[None].to(x.dtype).expand(x.shape[0], -1, -1, -1),
            ],
            dim=1,
        )
        return self.head(self.fuse(feat))


class ConditionalUNet(nn.Module):
    """
    输入:
      x_t:   [B,1,64,64]
      cond:  [B,C,11,17]
      t:     [B]
    输出:
      v_pred [B,1,64,64]
    """

    def __init__(self, cond_in_ch, base_ch=64, time_dim=256, cond_dim=256):
        super().__init__()
        self.cond_encoder = ConditionEncoder2D(cond_in_ch, emb_dim=cond_dim, base_ch=48)
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))
        self.null_tokens = nn.Parameter(torch.zeros(1, 15, cond_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = ConvNormAct(1, base_ch, 3)
        self.res1 = ResBlock(base_ch, base_ch, time_dim, cond_dim, cond_ch=48)
        self.down1 = Downsample(base_ch)

        self.res2 = ResBlock(base_ch, base_ch * 2, time_dim, cond_dim, cond_ch=96)
        self.down2 = Downsample(base_ch * 2)

        self.res3 = ResBlock(base_ch * 2, base_ch * 4, time_dim, cond_dim, cond_ch=192)
        self.attn3 = AttentionBlock(base_ch * 4)
        self.cross3 = SpectrumCrossAttention(base_ch * 4, cond_dim)
        self.down3 = Downsample(base_ch * 4)

        self.res4 = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, cond_ch=192)
        self.cross4 = SpectrumCrossAttention(base_ch * 4, cond_dim)

        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, cond_ch=192)
        self.mid_attn = AttentionBlock(base_ch * 4)
        self.mid_cross = SpectrumCrossAttention(base_ch * 4, cond_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, cond_ch=192)

        self.up1 = Upsample(base_ch * 4)
        self.up_res1 = ResBlock(base_ch * 8, base_ch * 4, time_dim, cond_dim, cond_ch=192)
        self.up_cross1 = SpectrumCrossAttention(base_ch * 4, cond_dim)

        self.up2 = Upsample(base_ch * 4)
        self.up_res2 = ResBlock(base_ch * 6, base_ch * 2, time_dim, cond_dim, cond_ch=96)

        self.up3 = Upsample(base_ch * 2)
        self.up_res3 = ResBlock(base_ch * 3, base_ch, time_dim, cond_dim, cond_ch=48)

        self.out_norm = nn.GroupNorm(_groups(base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def encode_cond(self, cond, batch_size, device, cond_drop_prob=0.0, force_uncond=False):
        if force_uncond or cond is None:
            return (
                self.null_cond.expand(batch_size, -1),
                {
                    "11x17": None,
                    "6x9": None,
                    "3x5": None,
                },
                self.null_tokens.expand(batch_size, -1, -1),
            )

        cond_emb, cond_feats, cond_tokens = self.cond_encoder.encode(cond)

        if self.training and cond_drop_prob > 0:
            drop_mask = torch.rand(batch_size, device=device) < cond_drop_prob
            if drop_mask.any():
                cond_emb = cond_emb.clone()
                cond_tokens = cond_tokens.clone()
                cond_emb[drop_mask] = self.null_cond.expand(drop_mask.sum(), -1)
                cond_tokens[drop_mask] = self.null_tokens.expand(drop_mask.sum(), -1, -1)
                for key in cond_feats:
                    feat = cond_feats[key]
                    if feat is not None:
                        feat = feat.clone()
                        feat[drop_mask] = 0.0
                        cond_feats[key] = feat

        return cond_emb, cond_feats, cond_tokens

    def forward(self, x, t, cond=None, cond_drop_prob=0.0, force_uncond=False):
        b = x.shape[0]
        t_emb = self.time_mlp(t)
        c_emb, c_feats, c_tokens = self.encode_cond(cond, b, x.device, cond_drop_prob, force_uncond)

        x0 = self.in_conv(x)
        x1 = self.res1(x0, t_emb, c_emb, c_feats["11x17"])
        x2 = self.res2(self.down1(x1), t_emb, c_emb, c_feats["6x9"])
        x3 = self.res3(self.down2(x2), t_emb, c_emb, c_feats["3x5"])
        x3 = self.cross3(self.attn3(x3), c_tokens)
        x4 = self.cross4(self.res4(self.down3(x3), t_emb, c_emb, c_feats["3x5"]), c_tokens)

        h = self.mid1(x4, t_emb, c_emb, c_feats["3x5"])
        h = self.mid_cross(self.mid_attn(h), c_tokens)
        h = self.mid2(h, t_emb, c_emb, c_feats["3x5"])

        h = self.up1(h)
        h = self.up_res1(torch.cat([h, x3], dim=1), t_emb, c_emb, c_feats["3x5"])
        h = self.up_cross1(h, c_tokens)

        h = self.up2(h)
        h = self.up_res2(torch.cat([h, x2], dim=1), t_emb, c_emb, c_feats["6x9"])

        h = self.up3(h)
        h = self.up_res3(torch.cat([h, x1], dim=1), t_emb, c_emb, c_feats["11x17"])

        return self.out_conv(F.silu(self.out_norm(h)))
