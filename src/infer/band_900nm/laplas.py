# -*- coding: utf-8 -*-
"""
900 nm 波段推理入口。

将 --target_lambda 默认值设为 900，--save_dir 默认指向 samples/laplas_900nm，
其余参数和逻辑完全复用顶层 infer/laplas.py。

Usage
-----
  python src/infer/band_900nm/laplas.py
  python src/infer/band_900nm/laplas.py --num_samples 64 --guidance_scale 0.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BAND_DIR = Path(__file__).resolve().parent
_ROOT = _BAND_DIR.parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import infer.laplas as _laplas  # noqa: E402

_DEFAULT_TARGET_LAMBDA = 900.0
_DEFAULT_SAVE_DIR = str(_ROOT / "samples" / "laplas_900nm")


def main():
    p = argparse.ArgumentParser(description="Run diffusion inference targeting 900 nm second-order response.")
    p.add_argument("--stats",          default=str(_ROOT / "checkpoints" / "cond_stats.npz"))
    p.add_argument("--diffusion_ckpt", default=str(_ROOT / "checkpoints" / "diffusion_best.pt"))
    p.add_argument("--forward_ckpt",   default=str(_ROOT / "checkpoints" / "forward_best.pt"))
    p.add_argument("--num_samples",    type=int,   default=32)
    p.add_argument("--cfg_scale",      type=float, default=3.0)
    p.add_argument("--save_dir",       default=_DEFAULT_SAVE_DIR)
    p.add_argument("--device",         default=None)
    p.add_argument("--devices",        default=None)
    p.add_argument("--target_lambda",  type=float, default=_DEFAULT_TARGET_LAMBDA)
    p.add_argument("--rcwa_orders",    type=int,   default=7)
    p.add_argument("--topk_second",    type=int,   default=5)
    p.add_argument("--guidance_scale", type=float, default=0.1)
    p.add_argument("--guide_start_t",  type=int,   default=300)
    p.add_argument("--guide_every",    type=int,   default=1)
    args = p.parse_args()
    _laplas._run_with_args(args)


if __name__ == "__main__":
    main()
