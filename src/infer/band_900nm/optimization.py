# -*- coding: utf-8 -*-
"""
900 nm 波段拓扑优化入口。

将 --target_lambda 默认值设为 900，--save_dir 默认指向 samples/optimized_900nm，
其余参数和逻辑完全复用顶层 infer/optimization.py。

Usage
-----
  python src/infer/band_900nm/optimization.py
  python src/infer/band_900nm/optimization.py --steps 200 --max_inits 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BAND_DIR = Path(__file__).resolve().parent
_ROOT = _BAND_DIR.parents[2]
sys.path.insert(0, str(_ROOT / "src"))

import infer.optimization as _opt  # noqa: E402

_DEFAULT_TARGET_LAMBDA = 900.0
_DEFAULT_SAVE_DIR = str(_ROOT / "samples" / "optimized_900nm")
_DEFAULT_LAPLAS_ROOT = str(_ROOT / "samples" / "laplas_900nm")


def main():
    p = argparse.ArgumentParser(description="Multi-start topology optimization targeting 900 nm.")
    p.add_argument("--target")
    p.add_argument("--init")
    p.add_argument("--laplas_root",       default=_DEFAULT_LAPLAS_ROOT)
    p.add_argument("--steps",              type=int,   default=100)
    p.add_argument("--lr",                 type=float, default=0.005)
    p.add_argument("--save_dir",           default=_DEFAULT_SAVE_DIR)
    p.add_argument("--device",             default=None)
    p.add_argument("--devices",            default=None)
    p.add_argument("--target_lambda",      type=float, default=_DEFAULT_TARGET_LAMBDA)
    p.add_argument("--rcwa_orders",        type=int,   default=7)
    p.add_argument("--binary_eval_every",  type=int,   default=10)
    p.add_argument("--max_inits",          type=int,   default=5)
    p.add_argument("--filter_radius",      type=int,   default=1)
    p.add_argument("--proj_eta",           type=float, default=0.5)
    p.add_argument("--beta_start",         type=float, default=4.0)
    p.add_argument("--beta_end",           type=float, default=16.0)
    p.add_argument("--log_every",          type=int,   default=10)
    args = p.parse_args()
    _opt._run_with_args(args)


if __name__ == "__main__":
    main()
