# -*- coding: utf-8 -*-
"""
1100 nm 波段 k 空间传递函数扫描入口。

Usage
-----
  python src/infer/band_1100nm/imag_process/scan_kspace.py --sample 0
  python src/infer/band_1100nm/imag_process/scan_kspace.py --from_infer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from infer.imag_process.scan_kspace import (  # noqa: E402
    load_phi0_from_training, load_phi0_from_infer,
    scan_phi45, build_2d_kspace, plot_kspace,
    DATA_PATH,
)

_DEFAULT_LAMBDA_NM = 1100.0


def main():
    parser = argparse.ArgumentParser(description="k 空间传递函数 2D 图 @ 1100 nm")
    parser.add_argument("--sample",     type=int,   default=0)
    parser.add_argument("--lambda_nm",  type=float, default=_DEFAULT_LAMBDA_NM)
    parser.add_argument("--from_infer", action="store_true")
    parser.add_argument("--infer_dir",  default=None)
    parser.add_argument("--run_phi45",  action="store_true")
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--cmap",       default="hot")
    args = parser.parse_args()

    lam = args.lambda_nm

    if args.from_infer:
        if args.infer_dir is None:
            for d in [_ROOT / "samples" / "laplas_1100nm", _ROOT / "samples" / "laplas"]:
                runs = sorted(d.iterdir()) if d.exists() else []
                if runs:
                    args.infer_dir = runs[-1]; break
        if args.infer_dir is None:
            print("找不到推理结果，请用 --sample", file=sys.stderr); sys.exit(1)
        print(f"[auto] 使用推理目录: {args.infer_dir}")
        tss0, tpp0 = load_phi0_from_infer(Path(args.infer_dir), lam)
        structure_for_phi45 = None
        label = f"infer λ={lam:.0f}nm"
    else:
        tss0, tpp0 = load_phi0_from_training(args.sample, lam)
        data = np.load(DATA_PATH)
        structure_for_phi45 = data["structures"][args.sample]
        label = f"sample={args.sample} λ={lam:.0f}nm"

    if args.run_phi45 and structure_for_phi45 is not None:
        tss45, tpp45 = scan_phi45(structure_for_phi45, lam, args.device)
    else:
        tss45, tpp45 = None, None

    kx_norm, ky_norm, T_pp = build_2d_kspace(tpp0, tpp45)
    _, _, T_ss = build_2d_kspace(tss0, tss45)

    out_npz = _HERE / "kspace_result.npz"
    np.savez(out_npz, T_pp=T_pp, T_ss=T_ss,
             kx_norm=kx_norm, ky_norm=ky_norm,
             lambda_nm=np.array([lam]))
    print(f"Saved → {out_npz}")

    plot_kspace(kx_norm, ky_norm, T_pp,
                title=rf"$|t_{{pp}}|$   {label}",
                save_path=_HERE / "kspace_tpp.png", cmap=args.cmap)
    plot_kspace(kx_norm, ky_norm, T_ss,
                title=rf"$|t_{{ss}}|$   {label}",
                save_path=_HERE / "kspace_tss.png", cmap=args.cmap)


if __name__ == "__main__":
    main()
