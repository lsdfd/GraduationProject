from __future__ import annotations


DIFFUSION_PRESETS: dict[str, dict] = {
    "baseline_cnn_cross": {
        "unet_arch": "cnn_cross_v1",
    },
    "cnn_cross_v2_stable": {
        "unet_arch": "cnn_cross_v2",
        "diff_loss_weight": "min_snr",
        "min_snr_gamma": 5.0,
        "ema_decay": 0.999,
        "lambda_x0": 0.02,
    },
    "token_cross_v3": {
        "unet_arch": "token_cross_v3",
        "cond_dim": 256,
        "base_ch": 64,
    },
    "token_film_v1": {
        "unet_arch": "token_film_v1",
        "cond_dim": 224,
        "base_ch": 64,
        "diff_loss_weight": "min_snr",
        "min_snr_gamma": 5.0,
    },
    "hybrid_cross_v1": {
        "unet_arch": "hybrid_cross_v1",
        "cond_dim": 224,
        "base_ch": 64,
        "diff_loss_weight": "min_snr",
        "min_snr_gamma": 5.0,
        "ema_decay": 0.999,
    },
    "hybrid_selfcond_v1": {
        "unet_arch": "hybrid_cross_v1",
        "cond_dim": 224,
        "base_ch": 64,
        "self_condition": True,
        "diff_loss_weight": "min_snr",
        "min_snr_gamma": 5.0,
        "lambda_x0": 0.05,
        "ema_decay": 0.999,
    },
    "token_lite_fast": {
        "unet_arch": "token_cross_lite",
        "base_ch": 48,
        "cond_dim": 160,
        "time_dim": 192,
        "diff_loss_weight": "p2",
        "p2_gamma": 1.0,
    },
}


def apply_preset(cfg: dict, preset_name: str | None) -> dict:
    if not preset_name:
        return cfg
    if preset_name not in DIFFUSION_PRESETS:
        raise ValueError(f"Unknown diffusion preset: {preset_name}. Available: {sorted(DIFFUSION_PRESETS)}")
    merged = dict(cfg)
    merged.update(DIFFUSION_PRESETS[preset_name])
    return merged
