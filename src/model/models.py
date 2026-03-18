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


class ConditionEncoderTokens(nn.Module):
    """
    把 [B, C, 11, 17] 光谱条件展开成 187 个 token，用 Transformer 全局建模。

    设计原则：
      - 不做空间下采样，11×17 所有格点全部保留
      - 波长轴 (11) 和角度轴 (17) 分别使用独立可学习位置编码
      - CLS token 聚合全局语义 → cond_emb（用于 AdaGN scale/shift）
      - 其余 187 个位置 token → cross-attention（精细频谱查询）
      - Pre-LN Transformer 训练更稳定

    输入: [B, C, 11, 17]
    输出:
      emb    [B, emb_dim]        → 全局条件向量
      tokens [B, 187, emb_dim]   → 逐格点条件 token
    """

    N_LAMBDA = 11
    N_THETA  = 17
    N_TOKENS = 11 * 17  # 187

    def __init__(self, in_ch, emb_dim=256, n_layers=3, n_heads=4):
        super().__init__()
        # 每个 (λ,θ) 格点的 (tpp, tss) 值映射到 emb_dim 维
        self.input_proj = nn.Linear(in_ch, emb_dim)

        # 独立位置编码：波长轴 128维 + 角度轴 128维 = 256维
        self.lambda_pe = nn.Embedding(self.N_LAMBDA, emb_dim // 2)
        self.theta_pe  = nn.Embedding(self.N_THETA,  emb_dim // 2)

        # Pre-LN Transformer：让所有 (λ,θ) 格点互相交流
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * 2,
            batch_first=True,
            dropout=0.1,
            norm_first=True,   # Pre-LN 比 Post-LN 训练更稳定
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 可学习 CLS token，聚合全局频谱信息
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))

        # 预存位置索引（固定，不随输入变化）
        li = torch.arange(self.N_LAMBDA).repeat_interleave(self.N_THETA)  # [187]
        ti = torch.arange(self.N_THETA).repeat(self.N_LAMBDA)             # [187]
        self.register_buffer("lambda_idx", li)
        self.register_buffer("theta_idx",  ti)

    def encode(self, cond):
        B, C, L, T = cond.shape          # C=in_ch, L=11波长, T=17角度

        # 展开成 token 序列 [B, 187, C] → [B, 187, emb_dim]
        x = cond.permute(0, 2, 3, 1).reshape(B, L * T, C)
        x = self.input_proj(x)

        # 加物理位置编码：知道哪个 token 是 1000nm、哪个是 ±40°
        pe = torch.cat(
            [self.lambda_pe(self.lambda_idx),   # [187, emb_dim//2]
             self.theta_pe(self.theta_idx)],    # [187, emb_dim//2]
            dim=-1,
        )                                        # [187, emb_dim]
        x = x + pe                              # broadcast over batch

        # 拼 CLS token，Transformer 编码
        cls = self.cls.expand(B, -1, -1)
        x = self.transformer(torch.cat([cls, x], dim=1))   # [B, 188, emb_dim]

        emb    = x[:, 0]    # [B, emb_dim]   — 全局频谱语义
        tokens = x[:, 1:]   # [B, 187, emb_dim] — 逐格点 token

        # 兼容旧接口：cond_feats 全为 None（不再做空间插值注入）
        return emb, {"11x17": None, "6x9": None, "3x5": None}, tokens

    def forward(self, cond):
        emb, _, _ = self.encode(cond)
        return emb


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


class UNetUpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            ResidualConvBlock(out_ch + skip_ch, out_ch),
            ResidualConvBlock(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class ForwardSurrogate(nn.Module):
    """
    输入:  [B,1,64,64]，值域 0~1
    输出:  [B,C,11,17]

    纯 encoder 结构，去掉 decoder，直接池化到目标分辨率。
    参数量约 4-6M，适合 5000 个样本规模。
    """

    def __init__(self, out_ch):
        super().__init__()
        base_ch = 32
        self.stem = nn.Sequential(
            ConvNormAct(1 + 2, base_ch, 3),
            ResidualConvBlock(base_ch, base_ch),
        )
        self.enc1 = nn.Sequential(
            ResidualConvBlock(base_ch, base_ch * 2, stride=2),    # 32x32
            ResidualConvBlock(base_ch * 2, base_ch * 2),
        )
        self.enc2 = nn.Sequential(
            ResidualConvBlock(base_ch * 2, base_ch * 4, stride=2), # 16x16
            ResidualConvBlock(base_ch * 4, base_ch * 4),
        )
        self.enc3 = nn.Sequential(
            ResidualConvBlock(base_ch * 4, base_ch * 8, stride=2), # 8x8
            ResidualConvBlock(base_ch * 8, base_ch * 8),
        )
        self.enc4 = nn.Sequential(
            ResidualConvBlock(base_ch * 8, base_ch * 8, stride=2), # 4x4
            ResidualConvBlock(base_ch * 8, base_ch * 8),
        )
        self.bottleneck = nn.Sequential(
            ResidualConvBlock(base_ch * 8, base_ch * 8),
            AttentionBlock(base_ch * 8),
            ResidualConvBlock(base_ch * 8, base_ch * 8),
        )
        # 直接池化到目标光谱分辨率，不需要 decoder
        self.spectral_head = nn.Sequential(
            ConvNormAct(base_ch * 8 + 2, base_ch * 8, 3),
            nn.Dropout2d(p=0.2),
            ResidualConvBlock(base_ch * 8, base_ch * 4),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(base_ch * 4, out_ch, 1),
        )
        self.register_buffer("coord_64", _coord_grid(64, 64), persistent=False)
        self.register_buffer("coord_spec", _coord_grid(11, 17), persistent=False)

    def forward(self, x):
        coord = self.coord_64[None].to(x.dtype).expand(x.shape[0], -1, -1, -1)
        e = self.stem(torch.cat([x, coord], dim=1))  # 64x64
        e = self.enc1(e)                              # 32x32
        e = self.enc2(e)                              # 16x16
        e = self.enc3(e)                              # 8x8
        e = self.enc4(e)                              # 4x4
        e = self.bottleneck(e)                        # 4x4

        # 直接池化到 11x17，不经过 decoder
        spec_feat = F.adaptive_avg_pool2d(e, output_size=(11, 17))
        spec_coord = self.coord_spec[None].to(x.dtype).expand(x.shape[0], -1, -1, -1)
        return self.spectral_head(torch.cat([spec_feat, spec_coord], dim=1))


class ConditionalUNet(nn.Module):
    """
    输入:
      x_t:   [B,1,64,64]
      cond:  [B,C,11,17]
      t:     [B]
    输出:
      v_pred [B,1,64,64]

    条件注入改进（v3）：
      - ConditionEncoderTokens: 把光谱展成 187 个 token，保留所有 (λ,θ) 信息
      - AdaGN (scale+shift): 全局语义注入每个 ResBlock
      - Cross-Attention: 16×16, 8×8(×2), 16×16(解码器), 32×32(解码器，新增)
      - 去掉了语义错位的空间插值注入（cond_feats → 全 None）
    """

    def __init__(self, cond_in_ch, base_ch=64, time_dim=256, cond_dim=256):
        super().__init__()

        # ── 条件编码器：Token 化，187 个 (λ,θ) 格点 ──
        self.cond_encoder = ConditionEncoderTokens(cond_in_ch, emb_dim=cond_dim)
        self.null_cond    = nn.Parameter(torch.zeros(1, cond_dim))
        self.null_tokens  = nn.Parameter(torch.zeros(1, 187, cond_dim))

        # ── 时间嵌入 ──
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # ── 编码器（条件只通过 AdaGN 和 cross-attn 进入，无空间插值注入）──
        self.in_conv = ConvNormAct(1, base_ch, 3)

        self.res1  = ResBlock(base_ch,     base_ch,     time_dim, cond_dim)   # 64×64
        self.down1 = Downsample(base_ch)

        self.res2  = ResBlock(base_ch,     base_ch * 2, time_dim, cond_dim)   # 32×32
        self.down2 = Downsample(base_ch * 2)

        self.res3   = ResBlock(base_ch * 2, base_ch * 4, time_dim, cond_dim)  # 16×16
        self.attn3  = AttentionBlock(base_ch * 4)
        self.cross3 = SpectrumCrossAttention(base_ch * 4, cond_dim)           # ← cross-attn
        self.down3  = Downsample(base_ch * 4)

        self.res4   = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)  # 8×8
        self.cross4 = SpectrumCrossAttention(base_ch * 4, cond_dim)           # ← cross-attn

        # ── Bottleneck 8×8 ──
        self.mid1      = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)
        self.mid_attn  = AttentionBlock(base_ch * 4)
        self.mid_cross = SpectrumCrossAttention(base_ch * 4, cond_dim)        # ← cross-attn
        self.mid2      = ResBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim)

        # ── 解码器 ──
        self.up1       = Upsample(base_ch * 4)
        self.up_res1   = ResBlock(base_ch * 8, base_ch * 4, time_dim, cond_dim)  # 16×16
        self.up_cross1 = SpectrumCrossAttention(base_ch * 4, cond_dim)            # ← cross-attn

        self.up2       = Upsample(base_ch * 4)
        self.up_res2   = ResBlock(base_ch * 6, base_ch * 2, time_dim, cond_dim)  # 32×32
        self.up_cross2 = SpectrumCrossAttention(base_ch * 2, cond_dim)            # ← cross-attn 新增

        self.up3       = Upsample(base_ch * 2)
        self.up_res3   = ResBlock(base_ch * 3, base_ch,     time_dim, cond_dim)  # 64×64

        self.out_norm = nn.GroupNorm(_groups(base_ch), base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def encode_cond(self, cond, batch_size, device, cond_drop_prob=0.0, force_uncond=False):
        if force_uncond or cond is None:
            return (
                self.null_cond.expand(batch_size, -1),
                self.null_tokens.expand(batch_size, -1, -1),
            )

        cond_emb, _, cond_tokens = self.cond_encoder.encode(cond)

        if self.training and cond_drop_prob > 0:
            drop_mask = torch.rand(batch_size, device=device) < cond_drop_prob
            if drop_mask.any():
                cond_emb    = cond_emb.clone()
                cond_tokens = cond_tokens.clone()
                cond_emb[drop_mask]    = self.null_cond.expand(drop_mask.sum(), -1)
                cond_tokens[drop_mask] = self.null_tokens.expand(drop_mask.sum(), -1, -1)

        return cond_emb, cond_tokens

    def forward(self, x, t, cond=None, cond_drop_prob=0.0, force_uncond=False):
        b = x.shape[0]
        t_emb            = self.time_mlp(t)
        c_emb, c_tokens  = self.encode_cond(cond, b, x.device, cond_drop_prob, force_uncond)

        # 编码器
        x0 = self.in_conv(x)
        x1 = self.res1(x0, t_emb, c_emb)                                   # 64×64
        x2 = self.res2(self.down1(x1), t_emb, c_emb)                       # 32×32
        x3 = self.res3(self.down2(x2), t_emb, c_emb)                       # 16×16
        x3 = self.cross3(self.attn3(x3), c_tokens)                         # ← cross-attn
        x4 = self.res4(self.down3(x3), t_emb, c_emb)                       # 8×8
        x4 = self.cross4(x4, c_tokens)                                     # ← cross-attn

        # Bottleneck
        h = self.mid1(x4, t_emb, c_emb)
        h = self.mid_cross(self.mid_attn(h), c_tokens)                     # ← cross-attn
        h = self.mid2(h, t_emb, c_emb)

        # 解码器
        h = self.up1(h)
        h = self.up_res1(torch.cat([h, x3], dim=1), t_emb, c_emb)         # 16×16
        h = self.up_cross1(h, c_tokens)                                    # ← cross-attn

        h = self.up2(h)
        h = self.up_res2(torch.cat([h, x2], dim=1), t_emb, c_emb)         # 32×32
        h = self.up_cross2(h, c_tokens)                                    # ← cross-attn 新增

        h = self.up3(h)
        h = self.up_res3(torch.cat([h, x1], dim=1), t_emb, c_emb)         # 64×64

        return self.out_conv(F.silu(self.out_norm(h)))
