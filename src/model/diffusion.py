import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999).float()


def extract(a, t, x_shape):
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, image_size=64):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (
            extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )

    def v_target(self, x0, t, noise):
        alpha = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return alpha * noise - sigma * x0

    def predict_x0_from_v(self, x_t, t, v):
        alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return alpha * x_t - sigma * v

    def predict_eps_from_v(self, x_t, t, v):
        alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return sigma * x_t + alpha * v

    def p_losses(
        self,
        x0,
        cond,
        surrogate=None,
        lambda_diff=1.0,
        lambda_phys=0.5,
        lambda_bin=0.05,
        cond_drop_prob=0.1
    ):
        b = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)

        v_gt = self.v_target(x0, t, noise)
        v_pred = self.model(x_t, t, cond=cond, cond_drop_prob=cond_drop_prob)

        loss_diff = F.mse_loss(v_pred, v_gt)

        x0_pred = self.predict_x0_from_v(x_t, t, v_pred).clamp(-1.0, 1.0)

        total_loss = lambda_diff * loss_diff
        log_dict = {"loss_diff": loss_diff.item()}

        if surrogate is not None:
            x01_pred = (x0_pred + 1.0) / 2.0
            x01_hard = (x01_pred > 0.5).float()
            x01_ste = x01_pred + (x01_hard - x01_pred).detach()
            pred_cond = surrogate(x01_ste)
            loss_phys = F.l1_loss(pred_cond, cond)
            total_loss = total_loss + lambda_phys * loss_phys
            log_dict["loss_phys"] = loss_phys.item()

        x01_pred = (x0_pred + 1.0) / 2.0
        loss_bin = (x01_pred * (1.0 - x01_pred)).mean()
        total_loss = total_loss + lambda_bin * loss_bin
        log_dict["loss_bin"] = loss_bin.item()

        return total_loss, log_dict

    @torch.no_grad()
    def p_sample(self, x, t_scalar, cond, cfg_scale=3.0):
        b = x.shape[0]
        t = torch.full((b,), t_scalar, device=x.device, dtype=torch.long)

        if cfg_scale != 1.0:
            v_cond = self.model(x, t, cond=cond, force_uncond=False)
            v_uncond = self.model(x, t, cond=None, force_uncond=True)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = self.model(x, t, cond=cond)

        eps = self.predict_eps_from_v(x, t, v)

        beta_t = extract(self.betas, t, x.shape)
        alpha_t = extract(self.alphas, t, x.shape)
        sqrt_one_minus_ab = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        model_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_one_minus_ab) * eps)

        if t_scalar == 0:
            return model_mean

        noise = torch.randn_like(x)
        var = extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, cond, cfg_scale=3.0):
        b = cond.shape[0]
        device = cond.device
        x = torch.randn(b, 1, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t, cond, cfg_scale=cfg_scale)

        x = x.clamp(-1.0, 1.0)
        x = (x + 1.0) / 2.0
        x = (x > 0.5).float()
        return x

    def p_sample_guided(
        self, x, t_scalar, cond, cfg_scale=3.0,
        surrogate=None, target_norm=None,
        guidance_scale=0.1, guide_start_t=300, guide_every=1,
    ):
        b = x.shape[0]
        device = x.device
        t = torch.full((b,), t_scalar, device=device, dtype=torch.long)

        # ── ① 标准 CFG 去噪（不需要梯度）──────────────────────────────
        v_uncond_saved = None
        with torch.no_grad():
            if cfg_scale != 1.0:
                v_cond   = self.model(x, t, cond=cond, force_uncond=False)
                v_uncond = self.model(x, t, cond=None,  force_uncond=True)
                v_uncond_saved = v_uncond
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(x, t, cond=cond)

            eps            = self.predict_eps_from_v(x, t, v)
            beta_t         = extract(self.betas, t, x.shape)
            alpha_t        = extract(self.alphas, t, x.shape)
            sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            model_mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / sqrt_one_minus) * eps
            )

        # ── ② 物理引导梯度（低噪声阶段才做）──────────────────────────
        do_guide = (
            surrogate is not None
            and target_norm is not None
            and t_scalar < guide_start_t
            and t_scalar % guide_every == 0
        )
        if do_guide:
            x_in = x.detach().requires_grad_(True)
            with torch.enable_grad():
                # 用完整 CFG v 估计 x0_hat，和 block ① 的去噪方向一致
                v_g_cond = self.model(x_in, t, cond=cond, force_uncond=False)
                if cfg_scale != 1.0 and v_uncond_saved is not None:
                    v_g = v_uncond_saved.detach() + cfg_scale * (v_g_cond - v_uncond_saved.detach())
                else:
                    v_g = v_g_cond
                x0_hat = self.predict_x0_from_v(x_in, t, v_g).clamp(-1.0, 1.0)
                x01    = (x0_hat + 1.0) / 2.0
                # STE：前向传 {0,1}，反向梯度不断
                x01_ste    = x01 + ((x01 > 0.5).float() - x01).detach()
                pred_cond  = surrogate(x01_ste)
                loss_g     = F.l1_loss(pred_cond, target_norm.expand_as(pred_cond))
                grad       = torch.autograd.grad(loss_g, x_in)[0]
            # 梯度归一化：消除不同时间步梯度量级差异，让 guidance_scale 可解释
            # 参考 arXiv:2601.15210 (Enhanced Posterior Sampling for Metasurfaces, 2026)
            grad_norm = grad.reshape(b, -1).norm(dim=1).reshape(b, 1, 1, 1).clamp(min=1e-8)
            grad_normalized = grad / grad_norm
            model_mean = model_mean - guidance_scale * grad_normalized

        if t_scalar == 0:
            return model_mean

        noise = torch.randn_like(x)
        var   = extract(self.posterior_variance, t, x.shape)
        return model_mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_guided(
        self, cond, cfg_scale=3.0,
        surrogate=None, target_norm=None,
        guidance_scale=0.1, guide_start_t=300, guide_every=1,
    ):
        b = cond.shape[0]
        device = cond.device
        x = torch.randn(b, 1, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.timesteps)):
            x = self.p_sample_guided(
                x, t, cond, cfg_scale=cfg_scale,
                surrogate=surrogate, target_norm=target_norm,
                guidance_scale=guidance_scale,
                guide_start_t=guide_start_t,
                guide_every=guide_every,
            )

        x = x.clamp(-1.0, 1.0)
        x = (x + 1.0) / 2.0
        x = (x > 0.5).float()
        return x
