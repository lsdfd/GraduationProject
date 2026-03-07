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


class ConditionEncoder2D(nn.Module):
    """
    输入:
      cond [B,C,11,17]
        B: batch size
        C: 条件通道数，当前项目里通常是 2（tpp_real, tpp_imag）
        11: 波长采样点数
        17: 角度采样点数

    输出:
      cond_emb [B,emb_dim]
        把二维条件图压成一个条件向量，供 ConditionalUNet 各层使用
    """
    def __init__(self, in_ch, emb_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),         # [B,C,11,17] -> [B,32,11,17]
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B,32,11,17] -> [B,64,6,9]
            nn.GroupNorm(_groups(64), 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # [B,64,6,9] -> [B,128,3,5]
            nn.GroupNorm(_groups(128), 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(_groups(128), 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),               # [B,128,3,5] -> [B,128,1,1]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                               # [B,128,1,1] -> [B,128]
            nn.Linear(128, emb_dim),                    # [B,128] -> [B,emb_dim]
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),                # [B,emb_dim] -> [B,emb_dim]
        )

    def forward(self, cond):
        # cond: [B,C,11,17]
        x = self.net(cond)
        # x: [B,128,1,1]
        x = self.fc(x)
        # x: [B,emb_dim]
        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_ch, groups), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(_groups(out_ch, groups), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim + cond_dim, out_ch * 2)
        )

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(F.silu(self.norm1(x)))

        emb = self.emb_proj(torch.cat([t_emb, c_emb], dim=1))  # [B, 2*out_ch]
        scale, shift = emb.chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
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
    输出:  [B,C,11,17]，和 cond 一样的 shape
    """
    def __init__(self, out_ch):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GroupNorm(_groups(32), 32),
            nn.SiLU(),
        )
        self.enc1 = nn.Sequential(nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, padding=1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(), nn.Conv2d(128, 128, 3, padding=1), nn.SiLU())
        self.head = nn.Sequential(
            nn.Conv2d(128 + 64 + 32, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_ch, 1)
        )

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        feat = torch.cat([
            F.interpolate(x2, size=(11, 17), mode="bilinear", align_corners=False),
            F.interpolate(x1, size=(11, 17), mode="bilinear", align_corners=False),
            F.interpolate(x0, size=(11, 17), mode="bilinear", align_corners=False),
        ], dim=1)
        return self.head(feat)


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
        self.cond_encoder = ConditionEncoder2D(cond_in_ch, emb_dim=cond_dim)
        self.null_cond = nn.Parameter(torch.zeros(1, cond_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv2d(1, base_ch, 3, padding=1)

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
        self.up_res1 = ResBlock(base_ch * 4 + base_ch * 4, base_ch * 4, time_dim, cond_dim)

        self.up2 = Upsample(base_ch * 4)
        self.up_res2 = ResBlock(base_ch * 4 + base_ch * 4, base_ch * 2, time_dim, cond_dim)

        self.up3 = Upsample(base_ch * 2)
        self.up_res3 = ResBlock(base_ch * 2 + base_ch, base_ch, time_dim, cond_dim)

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
        h = torch.cat([h, x3], dim=1)
        h = self.up_res1(h, t_emb, c_emb)

        h = self.up2(h)
        h = torch.cat([h, x2], dim=1)
        h = self.up_res2(h, t_emb, c_emb)

        h = self.up3(h)
        h = torch.cat([h, x1], dim=1)
        h = self.up_res3(h, t_emb, c_emb)

        h = F.silu(self.out_norm(h))
        out = self.out_conv(h)
        return out
